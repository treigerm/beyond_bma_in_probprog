import torch
import pyro
import pyro.distributions as dist

import pandas as pd
import numpy as np
import hydra
import joblib

import os
import logging
import time
import pickle
from collections import OrderedDict
from typing import Dict, Tuple

from models.pyro_extensions.dcc_hmc import DCCHMC, SLPInfo
from models.pyro_extensions.handlers import named_uncondition
from scripts.stacking_prototype import evaluate_stacked_lppd

DATA_DIR = "data/radon"


def radon_model(
    log_radon: torch.Tensor,
    floor_ind: torch.Tensor,
    county: torch.Tensor,
    num_counties: int,
    uranium: torch.Tensor,
):

    hierarchical_model = pyro.sample(
        "hierarchical_model", dist.Bernoulli(0.5), infer={"branching": True}
    )
    if hierarchical_model:
        uranium_context = pyro.sample(
            "uranium_context", dist.Bernoulli(0.5), infer={"branching": True}
        )
        if uranium_context:
            gamma_0 = pyro.sample("gamma_0", dist.Normal(0, 10))
            gamma_1 = pyro.sample("gamma_1", dist.Normal(0, 10))
            mean_a = gamma_0 + gamma_1 * uranium
        else:
            mean_a = pyro.sample("mean_a", dist.Normal(0, 1))

        std_a = pyro.sample("std_a", dist.Exponential(1))
        with pyro.plate("num_alpha", num_counties):
            z_a = pyro.sample("z_a", dist.Normal(0, 1))
        alpha = mean_a + std_a * z_a

        varying_slope = pyro.sample(
            "varying_intercept", dist.Bernoulli(0.5), infer={"branching": True}
        )
        if varying_slope:
            mean_b = pyro.sample("mean_b", dist.Normal(0, 1))
            std_b = pyro.sample("std_b", dist.Exponential(1))
            with pyro.plate("num_beta", num_counties):
                z_b = pyro.sample("z_b", dist.Normal(0, 1))
            beta = mean_b + std_b * z_b
            theta = alpha[..., county] + beta[..., county] * floor_ind
        else:
            beta = pyro.sample("beta", dist.Normal(0, 10))
            theta = alpha[..., county] + beta * floor_ind
    else:
        pooled_model = pyro.sample(
            "pooled_model", dist.Bernoulli(0.5), infer={"branching": True}
        )
        beta = pyro.sample("beta", dist.Normal(0, 10))
        if pooled_model:
            alpha = pyro.sample("alpha", dist.Normal(0, 10))
            theta = alpha + beta * floor_ind
        else:
            with pyro.plate("num_alpha", num_counties):
                alpha = pyro.sample("alpha", dist.Normal(0, 10))
            theta = alpha[..., county] + beta * floor_ind

    sigma = pyro.sample("sigma", dist.Exponential(5))
    with pyro.plate("data", log_radon.shape[0]):
        pyro.sample("ys", dist.Normal(theta, sigma), obs=log_radon)


def radon_model_v2(
    log_radon: torch.Tensor,
    floor_ind: torch.Tensor,
    county: torch.Tensor,
    num_counties: int,
    uranium: torch.Tensor,
):
    alpha_choice = pyro.sample(
        "alpha_choices", dist.Categorical(torch.ones(4) / 4), infer={"branching": True}
    )
    if alpha_choice == 0:
        # Pooled model
        alpha = pyro.sample("alpha", dist.Normal(0, 10))
    elif alpha_choice == 1:
        # County specific intercepts
        with pyro.plate("num_alpha", num_counties):
            alpha = pyro.sample("alpha", dist.Normal(0, 10))
        alpha = alpha[..., county]  # Shape: (num_counties,) -> (num_data,)
    elif alpha_choice == 2 or alpha_choice == 3:
        if alpha_choice == 2:
            # Partially pooled model
            mean_a = pyro.sample("mean_a", dist.Normal(0, 1))
        elif alpha_choice == 3:
            # Uranium context
            gamma_0 = pyro.sample("gamma_0", dist.Normal(0, 10))
            gamma_1 = pyro.sample("gamma_1", dist.Normal(0, 10))
            mean_a = gamma_0 + gamma_1 * uranium

        std_a = pyro.sample("std_a", dist.Exponential(1))
        with pyro.plate("num_alpha", num_counties):
            z_a = pyro.sample("z_a", dist.Normal(0, 1))
        alpha = mean_a + std_a * z_a
        alpha = alpha[..., county]  # Shape: (num_counties,) -> (num_data,)

    beta_choice = pyro.sample(
        "beta_choices", dist.Categorical(torch.ones(3) / 3), infer={"branching": True}
    )
    if beta_choice == 0:
        # Pooled model
        beta = pyro.sample("beta", dist.Normal(0, 10))
    elif beta_choice == 1:
        # County specific slopes
        with pyro.plate("num_beta", num_counties):
            beta = pyro.sample("beta", dist.Normal(0, 10))

        beta = beta[..., county]  # Shape: (num_counties,) -> (num_data,)
    elif beta_choice == 2:
        # Partially pooled model
        mean_b = pyro.sample("mean_b", dist.Normal(0, 1))
        std_b = pyro.sample("std_b", dist.Exponential(1))
        with pyro.plate("num_beta", num_counties):
            z_b = pyro.sample("z_b", dist.Normal(0, 1))
        beta = mean_b + std_b * z_b
        beta = beta[..., county]  # Shape: (num_counties,) -> (num_data,)

    theta = alpha + beta * floor_ind
    sigma = pyro.sample("sigma", dist.Exponential(5))
    with pyro.plate("data", log_radon.shape[0]):
        pyro.sample("ys", dist.Normal(theta, sigma), obs=log_radon)


def compute_lppd(
    model,
    branching_sample_values: OrderedDict[str, torch.Tensor],
    samples: dict[str, torch.Tensor],
    model_args: Tuple,
    model_kwargs: dict,
):
    cond_model = pyro.condition(model, data=branching_sample_values)
    predictive = pyro.infer.Predictive(named_uncondition(cond_model, ["ys"]), samples)
    vectorized_trace = predictive.get_vectorized_trace(*model_args, **model_kwargs)
    pred_fn = vectorized_trace.nodes["ys"]["fn"]
    log_p = pred_fn.log_prob(model_args[0])  # shape: (num_samples, num_data)
    return log_p.logsumexp(dim=0) - torch.log(torch.tensor(log_p.shape[0]))


def stratified_train_test_split(log_radon, floor_measure, county):
    # Make dictionary of counties and their indices
    unique_counties = np.unique(county)
    county_dict = {c: np.where(county == c)[0] for c in unique_counties}

    training_ixs = []
    test_ixs = []
    for _, ixs in county_dict.items():
        # stratified sampling
        n = len(ixs)
        n_train = max(int(n * 0.8), 1)
        np.random.shuffle(ixs)
        training_ixs.append(ixs[:n_train])
        test_ixs.append(ixs[n_train:])

    training_ixs = np.concatenate(training_ixs)
    test_ixs = np.concatenate(test_ixs)

    log_radon_train = log_radon[training_ixs]
    log_radon_test = log_radon[test_ixs]
    floor_measure_train = floor_measure[training_ixs]
    floor_measure_test = floor_measure[test_ixs]
    county_train = county[training_ixs]
    county_test = county[test_ixs]
    return (
        log_radon_train,
        log_radon_test,
        floor_measure_train,
        floor_measure_test,
        county_train,
        county_test,
    )


def load_data():
    """Based on PyMC3 tutorial https://www.pymc.io/projects/examples/en/latest/case_studies/multilevel_modeling.html#example-radon-contamination-gelman2006data."""
    srrs2 = pd.read_csv(
        os.path.join(hydra.utils.get_original_cwd(), DATA_DIR, "srrs2.dat")
    )
    srrs2.columns = srrs2.columns.map(str.strip)
    srrs_mn = srrs2[srrs2.state == "MN"].copy()

    cty = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), DATA_DIR, "cty.dat"))
    srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
    cty_mn = cty[cty.st == "MN"].copy()
    cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips

    srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
    srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
    uranium = np.log(srrs_mn.Uppm).unique()

    srrs_mn.county = srrs_mn.county.map(str.strip)
    county, mn_counties = srrs_mn.county.factorize()
    srrs_mn["county_code"] = county
    radon = srrs_mn.activity
    srrs_mn["log_radon"] = log_radon = np.log(radon + 0.1).values
    floor_measure = srrs_mn.floor.values

    (
        log_radon_train,
        log_radon_test,
        floor_measure_train,
        floor_measure_test,
        county_train,
        county_test,
    ) = stratified_train_test_split(log_radon, floor_measure, county)

    return (
        torch.tensor(log_radon_train),
        torch.tensor(log_radon_test),
        torch.tensor(floor_measure_train),
        torch.tensor(floor_measure_test),
        torch.tensor(county_train),
        torch.tensor(county_test),
        torch.tensor(uranium),
        mn_counties,
    )


def pi_mais(
    model,
    model_args: Tuple,
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
    # q_log_probs = sum(d.log_prob(q_samples[k]) for k, d in q_dists.items())
    q_log_probs = []
    for k, d in q_dists.items():
        lps = d.log_prob(q_samples[k])
        if len(lps.shape) == 2:
            lps = lps.sum(dim=1)
        q_log_probs.append(lps)
    q_log_probs = sum(q_log_probs)

    num_samples = q_log_probs.shape[0]
    log_weights = torch.zeros((num_samples,))
    # TODO: Technically, should be able to vectorize this loop.
    for ix in range(num_samples):
        sample_dict = {key: v[ix] for key, v in samples.items()}
        trace = pyro.poutine.trace(
            pyro.poutine.condition(cond_model, data=sample_dict)
        ).get_trace(*model_args, **model_kwargs)
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


@hydra.main(config_path="conf_dcc_hmc", config_name="config")
def main(cfg):
    pyro.set_rng_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    overall_start_time = time.time()
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data(**cfg.dataset)
    # X_trainval = torch.cat([X_train, X_val], dim=0)
    # y_trainval = torch.cat([y_train, y_val], dim=0)
    (
        log_radon_train,
        log_radon_test,
        floor_ind_train,
        floor_ind_test,
        county_train,
        county_test,
        uranium,
        mn_counties,
    ) = model_data = load_data()
    num_counties = len(mn_counties)

    # logging.info("Starting DCC-HMC Validation Set run")
    # dcc_hmc_stacked_val = hydra.utils.instantiate(
    #     cfg.dcc_hmc,
    #     model=model,
    #     predictive_dist=get_predictive(cfg.model),
    #     validation_data=(X_val, y_val),
    # )
    # start_time = time.time()
    # slps_info_stacked_val = dcc_hmc_stacked_val.run(X_train, y_train)
    # dcc_stacked_val_time = time.time() - start_time

    logging.info("Starting DCC-HMC run")
    dcc_hmc = hydra.utils.instantiate(cfg.dcc_hmc, model=radon_model_v2)
    start_time = time.time()
    slps_info = dcc_hmc.run(
        log_radon_train, floor_ind_train, county_train, num_counties, uranium
    )
    dcc_time = time.time() - start_time

    branching_traces = list(slps_info.keys())
    logging.info("Calculating analytic log marginal likelihoods")
    start_time = time.time()
    bma_log_probs = torch.ones(len(branching_traces))
    logging.info("Skipping because analytic log marginal likelihoods not implemented")
    lml_time = time.time() - start_time

    logging.info("Calculating PI-MAIS marginal_likelihood estimates")
    start_time = time.time()
    pi_mais_results = joblib.Parallel(n_jobs=cfg.dcc_hmc.num_parallel, verbose=2)(
        joblib.delayed(pi_mais)(
            radon_model_v2,  # radon_model,
            (log_radon_train, floor_ind_train, county_train, num_counties, uranium),
            dict(),
            slps_info,
            bt,
        )
        for bt in branching_traces
    )
    pi_mais_lml = torch.tensor([r[0] for r in pi_mais_results])
    pi_mais_time = time.time() - start_time

    bma_weights = torch.exp(bma_log_probs - torch.logsumexp(bma_log_probs, dim=0))
    pi_mais_weights = torch.exp(pi_mais_lml - torch.logsumexp(pi_mais_lml, dim=0))
    stacking_weights = torch.zeros((len(branching_traces),))
    for ix, bt in enumerate(branching_traces):
        stacking_weights[ix] = slps_info[bt].stacking_weight
    # stacking_val_set_weights = torch.zeros((len(branching_traces),))
    # for ix, bt in enumerate(branching_traces):
    #     stacking_val_set_weights[ix] = slps_info_stacked_val[bt].stacking_weight
    equal_weights = torch.ones((len(branching_traces),)) / len(branching_traces)

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
    test_lppds = []
    for ix, bt in enumerate(branching_traces):
        test_lppds.append(
            compute_lppd(
                radon_model_v2,  # radon_model,
                slps_info[bt].branching_sample_values,
                slps_info[bt].mcmc_samples,
                (log_radon_test, floor_ind_test, county_test, num_counties, uranium),
                dict(),
            )
        )
    test_lppds = torch.stack(test_lppds)

    # Calculate the lppd on held out data for validation set
    # start_time = time.time()
    # test_val_set_lppds = torch.zeros((len(branching_traces), X_test.shape[0]))
    # for ix, bt in enumerate(branching_traces):
    #     X_test_selected = get_included_features(
    #         X_test, slps_info_stacked_val[bt].branching_sample_values
    #     )
    #     test_val_set_lppds[ix, :] = compute_lppd_fn(
    #         slps_info_stacked_val[bt].mcmc_samples,
    #         X_test_selected,
    #         y_test,
    #     )

    # Compute stacking lppd
    stacking_lppd = evaluate_stacked_lppd(stacking_weights, test_lppds).mean()
    # stacking_val_set_lppd = evaluate_stacked_lppd(
    #     stacking_val_set_weights, test_val_set_lppds
    # ).mean()

    # Compute bma lppd
    bma_lppd = evaluate_stacked_lppd(bma_weights, test_lppds).mean()

    # Compute PI-MAIS lppd
    pi_mais_lppd = evaluate_stacked_lppd(pi_mais_weights, test_lppds).mean()
    equal_weights_lppd = evaluate_stacked_lppd(equal_weights, test_lppds).mean()
    eval_time = time.time() - start_time

    logging.info(f"Stacking lppd: {stacking_lppd.item():.2f}")
    # logging.info(f"Stacking Val Set lppd: {stacking_val_set_lppd.item():.2f}")
    logging.info(f"BMA lppd: {bma_lppd:.2f}")
    logging.info(f"PI-MAIS lppd: {pi_mais_lppd:.2f}")
    logging.info(f"Equal lppd: {equal_weights_lppd:.2f}")

    # Save results in pickle file.
    with open("results.pickle", "wb") as f:
        pickle.dump(
            {
                "slps_info": slps_info,
                "stacking_lppd": stacking_lppd,
                # "stacking_val_set_lppd": stacking_val_set_lppd,
                "bma_lppd": bma_lppd,
                "pi_mais_lppd": pi_mais_lppd,
                "equal_lppd": equal_weights_lppd,
                "stacking_weights": stacking_weights,
                # "stacking_val_set_weights": stacking_val_set_weights,
                "bma_weights": bma_weights,
                "pi_mais_weights": pi_mais_weights,
            },
            f,
        )
    with open("data.pickle", "wb") as f:
        torch.save(model_data[:-1], f)
    with open("mn_counties.pickle", "wb") as f:
        pickle.dump(list(mn_counties), f)

    overall_time = time.time() - overall_start_time

    logging.info("")
    logging.info("Timings:")
    logging.info(
        f"Overall time: {int(overall_time // 60)} m {int(overall_time % 60)} s"
    )
    timings = [
        # ("DCC-HMC Stacked", dcc_stacked_val_time),
        ("DCC-HMC", dcc_time),
        ("LML", lml_time),
        ("PI-MAIS", pi_mais_time),
        ("Eval", eval_time),
    ]
    for name, t in timings:
        logging.info(
            f"{name} time: {int(t // 60)} m {int(t % 60)} s ({t / overall_time:.2f} %)"
        )


if __name__ == "__main__":
    main()