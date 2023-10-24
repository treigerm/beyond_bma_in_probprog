from typing import Dict, List, Tuple, Any, OrderedDict
import pickle
import argparse
import time
from scipy.io import savemat

import torch
import pyro
import pyro.distributions as dist
import numpy as np
import joblib

from models.pyro_extensions.dcc_hmc import DCCHMC, SLPInfo
from models.pyro_extensions.handlers import named_uncondition

PRIOR_MEAN, PRIOR_VAR = torch.tensor(0), torch.tensor(10)


def coeffs(js: torch.Tensor, x: int, h: int) -> torch.Tensor:
    return (h - (js - x).abs()).square() * ((js - x).abs() < h)


def get_coefficients(J=15, h=5) -> torch.Tensor:
    js = torch.arange(J) + 1
    alphas = coeffs(js, 4, h) + coeffs(js, 8, h) + coeffs(js, 12, h)
    gamma = (4 / alphas.square().sum()).sqrt()
    betas = gamma * alphas
    return betas


def generate_data(N_train=200, N_val=200, N_test=200, h=5, J=15) -> List[torch.Tensor]:
    betas = get_coefficients(J=J, h=h)

    X_train = dist.Normal(5, 1).sample((N_train, J))
    y_train = X_train @ betas + dist.Normal(0, 1).sample((N_train,))

    X_test = dist.Normal(5, 1).sample((N_test, J))
    y_test = X_test @ betas + dist.Normal(0, 1).sample((N_test,))

    return [X_train, y_train, X_test, y_test]


def data_stacking_split(
    X_train: torch.Tensor, y_train: torch.Tensor, num_val: int
) -> List[torch.Tensor]:
    num_train = X_train.shape[0] - num_val
    assert num_train > 0
    return (
        X_train[:num_train],
        y_train[:num_train],
        X_train[num_train:],
        y_train[num_train:],
    )


def pyro_subset_regression(X: torch.Tensor, y: torch.Tensor, m_open: bool):
    k = pyro.sample(
        "k", dist.Categorical(logits=torch.ones(X.shape[1])), infer={"branching": True}
    )
    X = X[:, k].unsqueeze_(-1) if m_open else X[:, : k + 1]
    beta_dim = 1 if m_open else k + 1
    beta = pyro.sample(
        "beta",
        dist.Normal(PRIOR_MEAN.float(), PRIOR_VAR.sqrt().float())
        .expand([beta_dim])
        .to_event(1),
    )
    sigma = pyro.sample("sigma", dist.Gamma(0.1, 0.1))
    mean = beta @ X.T
    with pyro.plate("data", X.shape[0]):
        pyro.sample("y", dist.Normal(mean, sigma), obs=y)


def predictive_dist(
    samples: dict[str, torch.Tensor],
    validation_data: Tuple[torch.Tensor, torch.Tensor],
    branching_sample_values: OrderedDict[str, torch.Tensor],
):
    X_val, y_val = validation_data
    cond_model = pyro.condition(pyro_subset_regression, data=branching_sample_values)
    predictive = pyro.infer.Predictive(named_uncondition(cond_model, ["y"]), samples)
    vectorized_trace = predictive.get_vectorized_trace(X_val, y_val, m_open=True)
    pred_fn = vectorized_trace.nodes["y"]["fn"]
    log_p = pred_fn.log_prob(y_val)
    return log_p.logsumexp(dim=0) - torch.log(torch.tensor(log_p.shape[0]))


def vectorized_marginal_likelihood(
    x: torch.Tensor,  # (num_data, num_features)
    y: torch.Tensor,  # (num_data, )
    prior_mean: torch.Tensor,  # (1, )
    prior_var: torch.Tensor,  # (1, )
    lik_var: torch.Tensor,  # (num_samples, )
) -> torch.Tensor:
    # NOTE: The calculation below assumes that the prior_mean is 0.
    N = x.shape[0]
    D = x.shape[1]

    post_precision = (1 / lik_var)[:, None, None] * (x.T @ x) + (
        1 / prior_var
    ) * torch.eye(
        D
    )  # (num_samples, num_features, num_features)
    post_mean = (1 / lik_var)[:, None] * torch.linalg.solve(
        post_precision, (x.T @ y)
    )  # (num_samples, num_features)

    term = (y - post_mean @ x.T).square().sum(dim=1)  # (num_samples,)
    missing_term = (1 / lik_var) * term + (1 / prior_var) * post_mean.square().sum(
        dim=1
    )  # (num_samples,)
    log_ml = 0.5 * (
        -missing_term
        - torch.logdet(post_precision)
        - N * torch.log(torch.tensor(2) * torch.pi)
        - N * torch.log(lik_var)
        - D * torch.log(prior_var)
    )
    return log_ml


def is_ml_1d_estimate(
    X: torch.Tensor, y: torch.Tensor, k: int, m_open: bool, num_samples=100_000
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_sigma = dist.Gamma(0.1, 0.1)
    sigma = q_sigma.sample((num_samples,))

    # Don't need to multiply by prior because we q_sigma is the prior.
    covariates = X[:, k].unsqueeze_(-1) if m_open else X[:, : k + 1]
    log_p = vectorized_marginal_likelihood(
        covariates.double(),
        y.double(),
        PRIOR_MEAN.double(),
        PRIOR_VAR.double(),
        sigma.double().square(),
    )

    # q is the prior so prior and proposal cancel.
    log_ws = log_p
    ess = torch.exp(
        2 * torch.logsumexp(log_ws, dim=0) - torch.logsumexp(2 * log_ws, dim=0)
    )
    return torch.logsumexp(log_ws, dim=0) - torch.log(torch.tensor(num_samples)), ess


def pi_mais(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    bt: str,
    slps_info: SLPInfo,
    m_open: bool,
) -> torch.Tensor:
    cond_model = pyro.poutine.condition(
        pyro_subset_regression, data=slps_info[bt].branching_sample_values
    )
    samples = slps_info[bt].mcmc_samples

    q_dist_beta = dist.Normal(samples["beta"], 1.0).to_event(1)
    q_dist_sigma = dist.Normal(samples["sigma"], 1.0)
    q_samples = {"beta": q_dist_beta.sample(), "sigma": q_dist_sigma.sample()}

    q_log_probs = q_dist_beta.log_prob(q_samples["beta"]) + q_dist_sigma.log_prob(
        q_samples["sigma"]
    )
    num_samples = q_log_probs.shape[0]
    log_weights = torch.zeros((num_samples,))
    for ix in range(num_samples):
        sample_dict = {key: v[ix] for key, v in samples.items()}
        trace = pyro.poutine.trace(
            pyro.poutine.condition(cond_model, sample_dict)
        ).get_trace(X_train, y_train, m_open)
        log_weights[ix] = trace.log_prob_sum() - q_log_probs[ix]

    ess = torch.exp(
        2 * torch.logsumexp(log_weights, dim=0)
        - torch.logsumexp(2 * log_weights, dim=0)
    )
    log_marginal_likelihood = torch.logsumexp(log_weights, dim=0) - torch.log(
        torch.tensor(num_samples)
    )
    return log_marginal_likelihood, ess


def lppd(
    samples: Dict[str, torch.Tensor],
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    k: int,
    m_open: bool,
) -> torch.Tensor:
    betas, sigmas = samples["beta"], samples["sigma"]
    covariates = X_val[:, k].unsqueeze_(-1) if m_open else X_val[:, : k + 1]
    log_p = dist.Normal(betas @ covariates.T, sigmas[:, None]).log_prob(y_val)
    log_p = log_p.T
    return log_p.logsumexp(dim=1) - torch.log(
        torch.tensor(betas.shape[0])
    )  # (num_val_data, )


def mean_lppd(
    samples: Dict[str, torch.Tensor],
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    k: int,
    m_open: bool,
) -> float:
    return lppd(samples, X_val, y_val, k, m_open).mean().item()


def model_wise_test_lppd(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    all_samples: Dict[int, torch.Tensor],
    J: int,
    m_open: bool,
) -> torch.Tensor:
    """For each model evaluate the lppd on each data point."""
    test_lppds = torch.zeros((J, X_test.shape[0]))
    for k in range(J):
        test_lppds[k, :] = lppd(all_samples[k], X_test, y_test, k, m_open)

    return test_lppds


def evaluate_stacked_lppd(weights: torch.Tensor, lppds: torch.Tensor) -> torch.Tensor:
    # weights: (num_models,)
    # lppds: (num_models, num_data)
    log_weights = torch.log(weights)
    log_probs = lppds + log_weights[:, None]
    stacked_lppd = torch.logsumexp(log_probs, dim=0)
    return stacked_lppd  # (num_data,)


def pseudo_bma(
    lppd: torch.Tensor, num_bb_replications: int = 1000, alpha: float = 1.0
) -> torch.Tensor:
    num_data = lppd.shape[1]

    weights = dist.Dirichlet(alpha * torch.ones((num_data,))).sample(
        (num_bb_replications,)
    )  # (num_bb_replications, num_data)

    bma_weights = num_data * (weights @ lppd.T)  # (num_bb_replications, num_models)
    bma_weights = bma_weights - torch.logsumexp(bma_weights, dim=1, keepdim=True)
    bma_weights = torch.exp(
        torch.logsumexp(bma_weights, dim=0)
        - torch.log(torch.tensor(num_bb_replications))
    )
    return bma_weights


def main(
    seed=0,
    m_closed=False,
    num_replications=10,
    warmup_steps=200,
    num_post_samples=1000,
    num_elbo_particles=100,
    num_vi_iterations=1000,
    stacking_num_val=100,
    num_train=200,
    num_test=1000,
    num_parallel_jobs=10,
    regularize_stacking=False,
    **kwargs,
):
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)

    m_open = not m_closed
    J = 15
    methods = [
        "HMC Stacking",
        "HMC LOO Stacking",
        "HMC BMA",
        "HMC BMA Ground Truth",
        "HMC Pseudo-BMA",
        "HMC CV Model Selection",
        "HMC Equal",
        "VI Stacking",
        "VI ELBO",
        "VI CV Model Selection",
    ]
    method2lppd: Dict[str, torch.Tensor] = {
        m: torch.zeros((num_replications,)) for m in methods
    }
    method2weights: Dict[str, torch.Tensor] = {
        m: torch.zeros((num_replications, J)) for m in methods
    }

    pi_mais_esss = torch.zeros((num_replications, J))
    pi_mais_log_marginals = torch.zeros((num_replications, J))
    ground_truth_esss = torch.zeros((num_replications, J))
    ground_truth_log_marginals = torch.zeros((num_replications, J))
    hmc_full_model_lppd = torch.zeros((num_replications,))
    vi_full_model_lppd = torch.zeros((num_replications,))
    prev_time = 0
    generated_data = {}
    for i in range(num_replications):
        print("\n" + 100 * "=")
        print(
            f"Starting replication {i}. Previous replication took {prev_time // 60} minutes and {prev_time % 60} seconds."
        )
        print(100 * "=" + "\n")

        start_time = time.time()
        # Generate data
        X_train, y_train, X_test, y_test = generate_data(
            J=J, N_train=num_train, N_test=num_test
        )
        generated_data[str(i)] = {
            "X_train": X_train.numpy(),
            "y_train": y_train.numpy(),
            "X_test": X_test.numpy(),
            "y_test": y_test.numpy(),
        }

        X_train_stack, y_train_stack, X_val_stack, y_val_stack = data_stacking_split(
            X_train, y_train, stacking_num_val
        )

        dcc_hmc = DCCHMC(
            pyro_subset_regression,
            num_mcmc_warmup=warmup_steps,
            num_mcmc=num_post_samples,
            enumerate_branches=True,
            num_parallel=num_parallel_jobs,
            regularize_stacking=regularize_stacking,
        )
        slp_infos = dcc_hmc.run(X_train, y_train, m_open)

        # Explicit validation set
        dcc_hmc_holdout = DCCHMC(
            pyro_subset_regression,
            num_mcmc_warmup=warmup_steps,
            num_mcmc=num_post_samples,
            enumerate_branches=True,
            num_parallel=num_parallel_jobs,
            validation_data=(X_val_stack, y_val_stack),
            predictive_dist=predictive_dist,
            regularize_stacking=regularize_stacking,
        )
        slp_infos_holdout = dcc_hmc_holdout.run(X_train_stack, y_train_stack, m_open)

        all_samples_hmc = dict()
        all_samples_hmc_holdout = dict()
        for slp_ix, slp_info in slp_infos.items():
            k = int(slp_ix)
            all_samples_hmc[k] = slp_info.mcmc_samples
            all_samples_hmc_holdout[k] = slp_infos_holdout[slp_ix].mcmc_samples

            method2weights["HMC LOO Stacking"][i, k] = slp_info.stacking_weight
            method2weights["HMC Stacking"][i, k] = slp_infos_holdout[
                slp_ix
            ].stacking_weight

        branching_traces = list(slp_infos.keys())
        pi_mais_results = joblib.Parallel(n_jobs=num_parallel_jobs, verbose=2)(
            joblib.delayed(pi_mais)(X_train, y_train, bt, slp_infos, m_open)
            for bt in branching_traces
        )
        for ix, bt in enumerate(branching_traces):
            k = int(bt)
            pi_mais_log_marginals[i, k] = pi_mais_results[ix][0]
            pi_mais_esss[i, k] = pi_mais_results[ix][1]
            (
                ground_truth_log_marginals[i, k],
                ground_truth_esss[i, k],
            ) = is_ml_1d_estimate(X_train, y_train, k, m_open)

        with torch.no_grad():
            hmc_test_lppds = model_wise_test_lppd(
                X_test, y_test, all_samples_hmc, J, m_open
            )
            hmc_test_lppds_holdout = model_wise_test_lppd(
                X_test, y_test, all_samples_hmc_holdout, J, m_open
            )  # Test lppds when having an explicit train/validation split.

            method2weights["HMC BMA"][i, :] = (
                pi_mais_log_marginals[i, :]
                - torch.logsumexp(pi_mais_log_marginals[i, :], dim=0)
            ).exp()
            method2lppd["HMC BMA"][i] = evaluate_stacked_lppd(
                method2weights["HMC BMA"][i, :], hmc_test_lppds
            ).mean()
            method2weights["HMC BMA Ground Truth"][i, :] = (
                ground_truth_log_marginals[i, :]
                - torch.logsumexp(ground_truth_log_marginals[i, :], dim=0)
            ).exp()
            method2lppd["HMC BMA Ground Truth"][i] = evaluate_stacked_lppd(
                method2weights["HMC BMA Ground Truth"][i, :], hmc_test_lppds
            ).mean()
            # method2weights["HMC Pseudo-BMA"][i, :] = pseudo_bma(hmc_loo_psis)
            # method2lppd["HMC Pseudo-BMA"][i] = evaluate_stacked_lppd(
            #     method2weights["HMC Pseudo-BMA"][i, :], hmc_test_lppds
            # ).mean()
            method2weights["HMC Equal"][i, :] = torch.ones((J,)) / J
            method2lppd["HMC Equal"][i] = evaluate_stacked_lppd(
                method2weights["HMC Equal"][i, :], hmc_test_lppds
            ).mean()
            method2lppd["HMC LOO Stacking"][i] = evaluate_stacked_lppd(
                method2weights["HMC LOO Stacking"][i, :], hmc_test_lppds
            ).mean()

            method2lppd["HMC Stacking"][i] = evaluate_stacked_lppd(
                method2weights["HMC Stacking"][i, :], hmc_test_lppds_holdout
            ).mean()

        prev_time = time.time() - start_time

    fname = "_".join(
        f"{k}={v}"
        for k, v in [
            ("seed", seed),
            ("num_replications", num_replications),
            ("warmup_steps", warmup_steps),
            ("num_post_samples", num_post_samples),
            ("num_elbo_particles", num_elbo_particles),
            ("num_vi_iterations", num_vi_iterations),
            ("num_train", num_train),
            ("num_test", num_test),
            ("stacking_num_val", stacking_num_val),
            ("regularize_stacking", regularize_stacking),
        ]
    )
    prefix = "m_open_" if m_open else "m_closed_"
    fname = prefix + fname
    with open(f"scripts/stacking_results/{fname}.pickle", "wb") as f:
        pickle.dump(
            {
                "lppds": method2lppd,
                "weights": method2weights,
                "pi_mais_esss": pi_mais_esss,
                "pi_mais_log_marginals": pi_mais_log_marginals,
                "ground_truth_log_marginals": ground_truth_log_marginals,
                "ground_truth_esss": ground_truth_esss,
                "hmc_full_model_lppd": hmc_full_model_lppd,
                "vi_full_model_lppd": vi_full_model_lppd,
            },
            f,
        )

    savemat(f"scripts/stacking_results/{fname}.mat", generated_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-replications", default=10, type=int)
    parser.add_argument("--warmup-steps", default=200, type=int)
    parser.add_argument("--num-post-samples", default=1000, type=int)
    parser.add_argument("--num-elbo-particles", default=100, type=int)
    parser.add_argument("--num-vi-iterations", default=1000, type=int)
    parser.add_argument("--stacking-num-val", default=100, type=int)
    parser.add_argument("--num-train", default=200, type=int)
    parser.add_argument("--num-test", default=1000, type=int)
    parser.add_argument("--num-parallel-jobs", default=10, type=int)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--m-closed", action="store_true")
    parser.add_argument("--regularize-stacking", action="store_true")
    args = vars(parser.parse_args())
    if args["test_run"]:
        args = {
            "seed": 0,
            "num_replications": 1,
            "warmup_steps": 10,
            "num_post_samples": 10,
            "num_elbo_particles": 1,
            "num_vi_iterations": 10,
            "stacking_num_val": 100,
            "num_train": 200,
            "num_parallel_jobs": 1,
            "m_closed": args["m_closed"],
            "regularize_stacking": args["regularize_stacking"],
        }
    main(**args)