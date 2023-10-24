from typing import Tuple, OrderedDict, Dict
import time
import logging
import joblib
import pickle

import torch
import pyro
import pyro.distributions as dist
import hydra

from models.pyro_extensions.dcc_hmc import DCCHMC, SLPInfo
from models.pyro_extensions.handlers import named_uncondition
from scripts.stacking_prototype import evaluate_stacked_lppd


class Identity:
    has_subexprs = False


class Quadratic:
    has_subexprs = False


class Sine:
    has_subexprs = True

    def __init__(self, subexpr):
        self.subexpr = subexpr


class Addition:
    has_subexpr = True

    def __init__(self, subexpr1, subexpr2):
        self.subexpr1 = subexpr1
        self.subexpr2 = subexpr2


class ChangePoint:
    has_subexpr = True

    def __init__(self, subexpr1, subexpr2):
        self.subexpr1 = subexpr1
        self.subexpr2 = subexpr2


def eval_func(
    addr_prefix: str,
    func,
    x: torch.Tensor,
    x_lims: Tuple[float, float],
    dummy_sample: bool = False,
):
    if isinstance(func, Identity):
        a = 1.0
        if dummy_sample:
            pyro.sample(f"{addr_prefix}_dummy", dist.Normal(0, 1))
        elif "sin" not in addr_prefix and "plus" not in addr_prefix:
            # Make sure we have at least one unkown parameter.
            a = pyro.sample(f"{addr_prefix}_a", dist.Normal(0, 10))
        return a * x
    elif isinstance(func, Sine):
        x1 = eval_func(f"{addr_prefix}_sin_", func.subexpr, x, x_lims, dummy_sample)
        a = pyro.sample(f"{addr_prefix}_a", dist.Normal(0, 10))
        return torch.sin(a * x1)
    elif isinstance(func, Addition):
        x1 = eval_func(f"{addr_prefix}_plus1_", func.subexpr1, x, x_lims, dummy_sample)
        x2 = eval_func(f"{addr_prefix}_plus2_", func.subexpr2, x, x_lims, dummy_sample)
        a = pyro.sample(f"{addr_prefix}_a", dist.Normal(0, 10))
        b = pyro.sample(f"{addr_prefix}_b", dist.Normal(0, 10))
        return a * x1 + b * x2
    elif isinstance(func, Quadratic):
        pyro.sample(f"{addr_prefix}_dummy_quadr", dist.Normal(0, 1))
        return torch.pow(x, 2)
    # elif isinstance(func, ChangePoint):
    #     x_min, x_max = x_lims
    #     split_x = (x_min + x_max) / 2
    #     x1 = eval_func(f"{addr_prefix}_left_split_", func.subexpr1, x, (x_min, split_x))
    #     x2 = eval_func(
    #         f"{addr_prefix}_right_split_", func.subexpr2, x, (split_x, x_max)
    #     )
    #     return torch.where(x < split_x, x1, x2)
    else:
        raise ValueError("Unknown function type.")


def sample_expr(addr_prefix, num_splits, quadratic=False):
    # splits_allowed = num_splits < 2
    # rule_probs = [0.4, 0.4, 0.1, 0.1] if splits_allowed else [0.4, 0.4, 0.2]
    rule_probs = [0.4, 0.4, 0.2] if not quadratic else [0.3, 0.2, 0.2, 0.3]
    rule_ix = pyro.sample(
        f"{addr_prefix}_rule",
        dist.Categorical(probs=torch.tensor(rule_probs)),
        infer={"branching": True},
    )
    if rule_ix == 0:
        return Identity()
    elif rule_ix == 1:
        return Sine(sample_expr(f"{addr_prefix}_sin_", num_splits))
    elif rule_ix == 2:
        sub_expr1 = sample_expr(f"{addr_prefix}_plus1_", num_splits)
        sub_expr2 = sample_expr(f"{addr_prefix}_plus2_", num_splits)
        return Addition(sub_expr1, sub_expr2)
    elif rule_ix == 3:
        return Quadratic()


def function_induction_model(
    xs, y, dummy_sample=False, classification=False, quadratic=False
):
    sampled_expr = sample_expr("", 0, quadratic=quadratic)
    means = eval_func("", sampled_expr, xs, (-10, 10), dummy_sample=dummy_sample)
    if not classification:
        std = pyro.sample("std", dist.Gamma(1.0, 1.0))
        likelihood_dist = dist.Normal(means, std)
    else:
        likelihood_dist = dist.Bernoulli(logits=means)
    with pyro.plate("data", xs.shape[0]):
        pyro.sample("ys", likelihood_dist, obs=y)


# def data_generating_f(x: torch.Tensor, a: float = 0.2, b: float = 1.0, c: float = 5):
#     return a * torch.pow(x, 2) + b * torch.sin(c * x)
def data_generating_f(x: torch.Tensor, a: float = 1.0, b: float = 2.0, c: float = 2.0):
    return -a * x + b * torch.sin(c * torch.pow(x, 2))


def compute_lppd(
    branching_sample_values: OrderedDict[str, torch.Tensor],
    samples: dict[str, torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    model_kwargs: dict,
):
    cond_model = pyro.condition(function_induction_model, data=branching_sample_values)
    predictive = pyro.infer.Predictive(named_uncondition(cond_model, ["ys"]), samples)
    vectorized_trace = predictive.get_vectorized_trace(X, y, **model_kwargs)
    pred_fn = vectorized_trace.nodes["ys"]["fn"]
    log_p = pred_fn.log_prob(y)  # shape: (num_samples, num_data)
    return log_p.logsumexp(dim=0) - torch.log(torch.tensor(log_p.shape[0]))


def pi_mais(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    model_kwargs: dict,
    slps_info: Dict[str, SLPInfo],
    bt: str,
):
    cond_model = pyro.poutine.condition(
        model, data=slps_info[bt].branching_sample_values
    )

    samples = slps_info[bt].mcmc_samples
    q_dists = {k: dist.Normal(v, 1.0) for k, v in samples.items()}

    q_samples = {k: v.sample() for k, v in q_dists.items()}
    q_log_probs = sum(d.log_prob(q_samples[k]) for k, d in q_dists.items())

    num_samples = q_log_probs.shape[0]
    log_weights = torch.zeros((num_samples,))
    # TODO: Technically, should be able to vectorize this loop.
    for ix in range(num_samples):
        sample_dict = {key: v[ix] for key, v in samples.items()}
        trace = pyro.poutine.trace(
            pyro.poutine.condition(cond_model, data=sample_dict)
        ).get_trace(X_train, y_train, **model_kwargs)
        log_weights[ix] = trace.log_prob_sum() - q_log_probs[ix]

    # Evaluate the ESS and compute log marginal likelihood estimates
    ess = torch.exp(
        2 * torch.logsumexp(log_weights, dim=0)
        - torch.logsumexp(2 * log_weights, dim=0)
    )
    log_marginal_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(
        torch.tensor(num_samples)
    )
    return log_marginal_likelihood, ess


def predictive_dist(
    samples: dict[str, torch.Tensor],
    validation_data: Tuple[torch.Tensor, torch.Tensor],
    branching_sample_values: OrderedDict[str, torch.Tensor],
):
    X_val, y_val = validation_data
    cond_model = pyro.condition(function_induction_model, data=branching_sample_values)
    predictive = pyro.infer.Predictive(named_uncondition(cond_model, ["ys"]), samples)
    vectorized_trace = predictive.get_vectorized_trace(X_val, y_val)
    pred_fn = vectorized_trace.nodes["ys"]["fn"]
    log_p = pred_fn.log_prob(y_val)
    return log_p.logsumexp(dim=0) - torch.log(torch.tensor(log_p.shape[0]))


@hydra.main(config_path="conf_dcc_hmc", config_name="config_fun_ind")
def main(cfg):
    torch.manual_seed(cfg.seed)
    pyro.set_rng_seed(cfg.seed)

    overall_start_time = time.time()
    # Generate data
    num_train, num_val, num_test = cfg.num_train, cfg.num_val, cfg.num_test
    x_train = dist.Uniform(-5, 5).sample((num_train + num_val,))
    if cfg.classification:
        y_train = dist.Bernoulli(
            logits=data_generating_f(x_train, b=10.0, c=1.0)
        ).sample()
    else:
        if cfg.varying_noise:
            std = 0.1 * torch.pow(x_train, 2)
            means = data_generating_f(x_train, c=1.0)
        else:
            std = 0.1
            means = data_generating_f(x_train)
        y_train = dist.Normal(means, std).sample()
    x_train, x_val = x_train[:num_train], x_train[num_train:]
    y_train, y_val = y_train[:num_train], y_train[num_train:]
    x_test = dist.Uniform(-5, 5).sample((num_test,))
    if cfg.classification:
        y_test = dist.Bernoulli(
            logits=data_generating_f(x_test, b=10.0, c=1.0)
        ).sample()
    else:
        if cfg.varying_noise:
            std = 0.1 * torch.pow(x_test, 2)
            means = data_generating_f(x_test, c=1.0)
        else:
            std = 0.1
            means = data_generating_f(x_test)
        y_test = dist.Normal(means, std).sample()

    model_kwargs = {
        "dummy_sample": cfg.dummy_sample,
        "classification": cfg.classification,
        "quadratic": cfg.varying_noise,
    }
    logging.info("Starting DCC HMC run..")
    dcc_hmc = hydra.utils.instantiate(
        cfg.dcc_hmc,
        model=function_induction_model,
        predictive_dist=predictive_dist,
        validation_data=(x_val, y_val),
    )
    start_time = time.time()
    slps_info = dcc_hmc.run(x_train, y_train, **model_kwargs)
    dcc_time = time.time() - start_time
    branching_traces = list(slps_info.keys())

    logging.info("Starting DCC HMC run for PI-MAIS..")
    dcc_hmc_pi_mais = hydra.utils.instantiate(
        # dcc_hmc = hydra.utils.instantiate(
        cfg.dcc_hmc,
        model=function_induction_model,
    )
    start_time = time.time()
    slps_info_pi_mais = dcc_hmc_pi_mais.run(
        # slps_info = dcc_hmc.run(
        torch.cat([x_train, x_val], dim=0),
        torch.cat([y_train, y_val], dim=0),
        **model_kwargs,
    )
    dcc_pi_mais_time = time.time() - start_time
    assert branching_traces == list(slps_info_pi_mais.keys())
    branching_traces = list(slps_info.keys())

    logging.info("Calculating PI-MAIS marginal_likelihood estimates")
    start_time = time.time()
    pi_mais_results = joblib.Parallel(n_jobs=cfg.dcc_hmc.num_parallel, verbose=2)(
        joblib.delayed(pi_mais)(
            function_induction_model,
            torch.cat([x_train, x_val], dim=0),  # x_train,
            torch.cat([y_train, y_val], dim=0),  # y_train,
            model_kwargs,
            slps_info_pi_mais,
            bt,
        )
        for bt in branching_traces
    )
    pi_mais_lml = torch.tensor([r[0] for r in pi_mais_results])
    pi_mais_time = time.time() - start_time

    bma_log_probs = torch.ones(len(slps_info))
    bma_weights = torch.exp(bma_log_probs - torch.logsumexp(bma_log_probs, dim=0))
    pi_mais_weights = torch.exp(pi_mais_lml - torch.logsumexp(pi_mais_lml, dim=0))
    stacking_weights = torch.zeros((len(branching_traces),))
    for ix, bt in enumerate(branching_traces):
        stacking_weights[ix] = slps_info[bt].stacking_weight

    stacking_loo_weights = torch.zeros((len(branching_traces),))
    for ix, bt in enumerate(branching_traces):
        stacking_loo_weights[ix] = slps_info_pi_mais[bt].stacking_weight

    logging.info("Top 10 SLPs:")
    top_k = min(10, len(branching_traces))
    top_k_ixs = torch.argsort(stacking_weights)[-top_k:]
    for ix in top_k_ixs:
        bt = branching_traces[ix]
        logging.info(
            f"{bt}: {slps_info[bt].stacking_weight:.2f} (bma={bma_weights[ix]:.2f})"
        )

    # Calculate the lppd on held out data.
    start_time = time.time()
    test_lppds = torch.zeros((len(branching_traces), x_test.shape[0]))
    for ix, bt in enumerate(branching_traces):
        test_lppds[ix, :] = compute_lppd(
            slps_info[bt].branching_sample_values,
            slps_info[bt].mcmc_samples,
            x_test,
            y_test,
            model_kwargs,
        )

    test_pi_mais_lppds = torch.zeros((len(branching_traces), x_test.shape[0]))
    for ix, bt in enumerate(branching_traces):
        test_pi_mais_lppds[ix, :] = compute_lppd(
            slps_info_pi_mais[bt].branching_sample_values,
            slps_info_pi_mais[bt].mcmc_samples,
            x_test,
            y_test,
            model_kwargs,
        )

    # Compute stacking lppd
    stacking_lppd = evaluate_stacked_lppd(stacking_weights, test_lppds).mean()
    stacking_loo_lppd = evaluate_stacked_lppd(
        stacking_loo_weights, test_pi_mais_lppds
    ).mean()
    # bma_lppd = evaluate_stacked_lppd(bma_weights, test_lppds).mean()
    # pi_mais_lppd = evaluate_stacked_lppd(pi_mais_weights, test_lppds).mean()
    bma_lppd = evaluate_stacked_lppd(bma_weights, test_pi_mais_lppds).mean()
    pi_mais_lppd = evaluate_stacked_lppd(pi_mais_weights, test_pi_mais_lppds).mean()
    eval_time = time.time() - start_time

    logging.info(f"Stacking lppd: {stacking_lppd.item():.2f}")
    logging.info(f"Stacking (LOO) lppd: {stacking_loo_lppd.item():.2f}")
    logging.info(f"BMA lppd: {bma_lppd:.2f}")
    logging.info(f"PI-MAIS lppd: {pi_mais_lppd:.2f}")

    # Save results in pickle file.
    with open("results.pickle", "wb") as f:
        pickle.dump(
            {
                "slps_info_val": slps_info,
                "slps_info": slps_info_pi_mais,
                "stacking_lppd": stacking_lppd,
                "stacking_loo_lppd": stacking_loo_lppd,
                "bma_lppd": bma_lppd,
                "pi_mais_lppd": pi_mais_lppd,
                "stacking_weights": stacking_weights,
                "stacking_loo_weights": stacking_loo_weights,
                "bma_weights": bma_weights,
                "pi_mais_weights": pi_mais_weights,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
            },
            f,
        )

    with open("data.pickle", "wb") as f:
        torch.save([x_train, y_train, x_test, y_test], f)

    overall_time = time.time() - overall_start_time
    logging.info("")
    logging.info("Timings:")
    logging.info(
        f"Overall time: {int(overall_time // 60)} m {int(overall_time % 60)} s"
    )
    timings = [
        ("DCC-HMC", dcc_time),
        ("DCC-HMC for PI-MAIS", dcc_pi_mais_time),
        # ("LML", lml_time),
        ("PI-MAIS", pi_mais_time),
        ("Eval", eval_time),
    ]
    for name, t in timings:
        logging.info(
            f"{name} time: {int(t // 60)} m {int(t % 60)} s ({t / overall_time:.2f} %)"
        )


if __name__ == "__main__":
    main()