import torch
import pyro
import pyro.distributions as dist

import pandas as pd
import numpy as np
import hydra
import joblib

import logging
import time
import pickle
from collections import OrderedDict
from typing import Dict, Tuple

from models.pyro_extensions.dcc_hmc import DCCHMC, SLPInfo, compute_stacking_weights
from models.pyro_extensions.handlers import named_uncondition
from scripts.stacking_prototype import evaluate_stacked_lppd

INCLUSION_PROB = torch.tensor(0.5)
GAMMA_CONCENTRATION = torch.tensor(2.0)
GAMMA_RATE = torch.tensor(1.0)
WEIGHT_LAMBDA = torch.tensor(1.0)


def variable_selection_model(
    X: torch.Tensor,
    y: torch.Tensor,
    inclusion_prob: float = 0.5,
    gamma_concentration: float = 2.0,
    gamma_rate: float = 1.0,
    weight_lambda: float = 1.0,
    X_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
):
    """
    X: (num_data, num_features)
    y: (num_data,)
    """
    num_features = X.shape[1]

    features_included = torch.zeros((num_features,), dtype=torch.bool)
    for ix in range(num_features):
        features_included[ix] = pyro.sample(
            f"feature_{ix}", dist.Bernoulli(inclusion_prob), infer={"branching": True}
        )

    if features_included.any():
        X_selected = X[:, features_included]
    else:
        X_selected = torch.ones((X.shape[0], 1))

    num_selected = X_selected.shape[1]
    noise_var = pyro.sample(
        "noise_var", dist.InverseGamma(gamma_concentration, gamma_rate)
    )
    with pyro.plate("features", num_selected):
        w = pyro.sample("weights", dist.Normal(0, (noise_var / weight_lambda).sqrt()))

    means = w @ X_selected.T
    with pyro.plate("data", y.shape[0]):
        pyro.sample("obs", dist.Normal(means, noise_var.sqrt()), obs=y)


def log_reg_model(
    X: torch.Tensor,
    y: torch.Tensor,
    inclusion_prob: float = 0.5,
    X_val: torch.Tensor = None,
    y_val: torch.Tensor = None,
):
    """
    X: (num_data, num_features)
    y: (num_data,)
    """
    num_features = X.shape[1]

    features_included = torch.zeros((num_features,), dtype=torch.bool)
    for ix in range(num_features):
        features_included[ix] = pyro.sample(
            f"feature_{ix}", dist.Bernoulli(inclusion_prob), infer={"branching": True}
        )

    if features_included.any():
        X_selected = X[:, features_included]
    else:
        X_selected = torch.ones((X.shape[0], 1))

    num_selected = X_selected.shape[1]
    with pyro.plate("features", num_selected):
        w = pyro.sample("weights", dist.Normal(0, 1))

    logits = w @ X_selected.T
    with pyro.plate("data", y.shape[0]):
        pyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

    # if X_val is not None and y_val is not None:
    #     if features_included.any():
    #         X_val_selected = X_val[:, features_included]
    #     else:
    #         X_val_selected = torch.ones((X_val.shape[0], 1))
    #     logits_val = w @ X_val_selected.T
    #     return dist.Bernoulli(logits=logits_val).log_prob(y_val)


def compute_lppd(
    samples: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
):
    means = X @ samples["weights"].T
    log_p = dist.Normal(means, samples["noise_var"].sqrt()).log_prob(
        y[:, None].expand_as(means)
    )
    return log_p.logsumexp(dim=1) - torch.log(torch.tensor(log_p.shape[1]))


def compute_lppd_log_reg(
    samples: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
):
    logits = X @ samples["weights"].T
    log_p = dist.Bernoulli(logits=logits).log_prob(y[:, None].expand_as(logits))
    return log_p.logsumexp(dim=1) - torch.log(torch.tensor(log_p.shape[1]))


def ground_truth_log_marginal_likelihood(
    X: torch.Tensor,
    y: torch.Tensor,
    gamma_concentration: torch.Tensor,
    gamma_rate: torch.Tensor,
    weight_lambda: torch.Tensor,
):
    N = X.shape[0]
    D = X.shape[1]

    post_prec = X.T @ X + weight_lambda * torch.eye(D)
    post_rate = (
        gamma_rate + 0.5 * y @ (torch.eye(N) - X @ torch.inverse(post_prec) @ X.T) @ y
    )
    post_concentration = gamma_concentration + 0.5 * N

    log_ml = (
        gamma_concentration * torch.log(gamma_rate)
        + torch.lgamma(post_concentration)
        + 0.5 * D * torch.log(weight_lambda)
        - 0.5 * N * torch.log(2 * torch.tensor(np.pi))
        - torch.lgamma(gamma_concentration)
        - post_concentration * torch.log(post_rate)
        - 0.5 * torch.logdet(post_prec)
    )
    return log_ml


def get_included_features(
    X: torch.Tensor, branching_sampled_values: OrderedDict[str, torch.Tensor]
) -> torch.Tensor:
    num_features = X.shape[1]
    features_included = torch.zeros((num_features,), dtype=torch.bool)
    for ix, val in enumerate(branching_sampled_values.values()):
        features_included[ix] = val

    if features_included.any():
        X_selected = X[:, features_included]
    else:
        X_selected = torch.ones((X.shape[0], 1))

    return X_selected


def data_preprocessing(X, y, num_train, num_val, num_test, transform_y=True):
    permuted_ixs = torch.randperm(X.shape[0])
    train_ixs = permuted_ixs[:num_train]
    val_ixs = permuted_ixs[num_train : num_train + num_val]
    test_ixs = permuted_ixs[num_train + num_val : num_train + num_val + num_test]
    X_train = X[train_ixs]
    y_train = y[train_ixs]
    X_val = X[val_ixs]
    y_val = y[val_ixs]
    X_test = X[test_ixs]
    y_test = y[test_ixs]

    X_train_mean, X_train_std = X_train.mean(dim=0), X_train.std(dim=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    if transform_y:
        y_train_mean, y_train_std = y_train.mean(dim=0), y_train.std(dim=0)
        y_train = (y_train - y_train_mean) / y_train_std
        y_val = (y_val - y_train_mean) / y_train_std
        y_test = (y_test - y_train_mean) / y_train_std

    return X_train, y_train, X_val, y_val, X_test, y_test, train_ixs, val_ixs, test_ixs


def load_boston_housing_data(filepath, num_train=403, num_val=0, num_test=103):
    column_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]
    data = pd.read_csv(filepath, header=None, delimiter=r"\s+", names=column_names)

    # Drop RAD column because it is a categorical variable.
    data = data.drop(columns=["RAD"])

    # Standardize the data except for the binary CHAS variable.
    feature_cols = ~data.columns.isin(["CHAS", "MEDV"])
    X = torch.tensor(data.loc[:, feature_cols].values, dtype=torch.float)
    X_chas = torch.tensor(data.loc[:, ["CHAS"]].values, dtype=torch.float)
    y = torch.tensor(data["MEDV"].values, dtype=torch.float)

    X_train, y_train, X_test, y_test, train_ixs, test_ixs = data_preprocessing(
        X, y, num_train, num_test
    )
    X_chas_train = X_chas[train_ixs]
    X_chas_test = X_chas[test_ixs]

    X_train = torch.cat((X_train, X_chas_train), dim=1)
    X_test = torch.cat((X_test, X_chas_test), dim=1)

    return X_train, y_train, X_test, y_test


def load_california_housing_data(filepath, num_train=1000, num_val=100, num_test=1000):
    df = pd.read_csv(filepath).dropna()
    feature_cols = ~df.columns.isin(["median_house_value", "ocean_proximity"])
    X = torch.tensor(df.loc[:, feature_cols].values, dtype=torch.float)
    y = torch.tensor(df["median_house_value"].values, dtype=torch.float)

    X_train, y_train, X_val, y_val, X_test, y_test, _, _, _ = data_preprocessing(
        X, y, num_train, num_val, num_test
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_diabetes_data(filepath, num_train=460, num_val=154, num_test=154):
    df = pd.read_csv(filepath).dropna()
    feature_cols = ~df.columns.isin(["Outcome"])
    X = torch.tensor(df.loc[:, feature_cols].values, dtype=torch.float)
    y = torch.tensor(df["Outcome"].values, dtype=torch.float)

    X_train, y_train, X_val, y_val, X_test, y_test, _, _, _ = data_preprocessing(
        X, y, num_train, num_val, num_test, transform_y=False
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_stroke_data(filepath, num_train=2944, num_val=982, num_test=982):
    df = pd.read_csv(filepath).dropna()
    df = df[
        df["gender"] != "Other"
    ]  #  Should not do this for a real analysis of the data.
    df = df.drop(columns=["id", "work_type", "smoking_status"])
    df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})
    df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    feature_cols_continous = df.columns.isin(["age", "avg_glucose_level", "bmi"])
    feature_cols_binary = df.columns.isin(
        ["gender", "hypertension", "heart_disease", "ever_married", "Residence_type"]
    )
    X = torch.tensor(df.loc[:, feature_cols_continous].values, dtype=torch.float)
    X_bin = torch.tensor(df.loc[:, feature_cols_binary].values, dtype=torch.float)
    y = torch.tensor(df["stroke"].values, dtype=torch.float)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        train_ixs,
        val_ixs,
        test_ixs,
    ) = data_preprocessing(X, y, num_train, num_val, num_test, transform_y=False)
    X_bin_train = X_bin[train_ixs]
    X_bin_val = X_bin[val_ixs]
    X_bin_test = X_bin[test_ixs]

    X_train = torch.cat((X_train, X_bin_train), dim=1)
    X_val = torch.cat((X_val, X_bin_val), dim=1)
    X_test = torch.cat((X_test, X_bin_test), dim=1)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data(name, filepath, num_train, num_val, num_test):
    if name == "boston":
        return load_boston_housing_data(filepath, num_train, num_val, num_test)
    elif name == "california":
        return load_california_housing_data(filepath, num_train, num_val, num_test)
    elif name == "diabetes":
        return load_diabetes_data(filepath, num_train, num_val, num_test)
    elif name == "stroke":
        return load_stroke_data(filepath, num_train, num_val, num_test)
    else:
        raise ValueError(f"Unrecognized dataset name: {name}")


def compute_analytic_lml(X_train, y_train, slps_info, bt):
    X_selected = get_included_features(X_train, slps_info[bt].branching_sample_values)
    lml = ground_truth_log_marginal_likelihood(
        X_selected,
        y_train,
        gamma_concentration=GAMMA_CONCENTRATION,
        gamma_rate=GAMMA_RATE,
        weight_lambda=WEIGHT_LAMBDA,
    )

    num_features = X_selected.shape[1]
    D = X_train.shape[1]
    return (
        lml
        + num_features * torch.log(INCLUSION_PROB)
        + (D - num_features) * torch.log(1 - INCLUSION_PROB)
    )


def pi_mais(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    slps_info: Dict[str, SLPInfo],
    bt: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Condition model so that we are in the correct SLP.
    cond_model = pyro.poutine.block(
        pyro.poutine.condition(model, data=slps_info[bt].branching_sample_values),
        hide=set(slps_info[bt].branching_sample_values.keys()),
    )

    # Construct the proposal distributions.
    samples = slps_info[bt].mcmc_samples
    q_dist_weights = dist.Normal(samples["weights"], 1.0).to_event(1)
    q_dist_noise_var = dist.Normal(samples["noise_var"], 1.0)

    # Sample from the proposal distributions.
    q_samples = {
        "weights": q_dist_weights.sample(),
        "noise_var": q_dist_noise_var.sample(),
    }
    q_log_probs = q_dist_weights.log_prob(
        q_samples["weights"]
    ) + q_dist_noise_var.log_prob(q_samples["noise_var"])

    # Evaluate the log probability of the samples under the model.
    num_samples = q_log_probs.shape[0]
    log_weights = torch.zeros((num_samples,))
    for ix in range(num_samples):
        sample_dict = {key: v[ix] for key, v in samples.items()}
        trace = pyro.poutine.trace(
            pyro.poutine.condition(cond_model, data=sample_dict)
        ).get_trace(
            X_train,
            y_train,
            inclusion_prob=INCLUSION_PROB,
            gamma_concentration=GAMMA_CONCENTRATION,
            gamma_rate=GAMMA_RATE,
            weight_lambda=WEIGHT_LAMBDA,
        )
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


def pi_mais_log_reg(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    slps_info: Dict[str, SLPInfo],
    bt: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Condition model so that we are in the correct SLP.
    cond_model = pyro.poutine.block(
        pyro.poutine.condition(model, data=slps_info[bt].branching_sample_values),
        hide=set(slps_info[bt].branching_sample_values.keys()),
    )

    # Construct the proposal distributions.
    samples = slps_info[bt].mcmc_samples
    q_dist_weights = dist.Normal(samples["weights"], 1.0).to_event(1)

    # Sample from the proposal distributions.
    q_samples = {
        "weights": q_dist_weights.sample(),
    }
    q_log_probs = q_dist_weights.log_prob(q_samples["weights"])

    # Evaluate the log probability of the samples under the model.
    num_samples = q_log_probs.shape[0]
    log_weights = torch.zeros((num_samples,))
    for ix in range(num_samples):
        sample_dict = {key: v[ix] for key, v in samples.items()}
        trace = pyro.poutine.trace(
            pyro.poutine.condition(cond_model, data=sample_dict)
        ).get_trace(
            X_train,
            y_train,
            inclusion_prob=INCLUSION_PROB,
        )
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


def get_predictive(model: str):
    model = log_reg_model if model == "log_reg" else variable_selection_model

    def predictive_dist(
        samples: dict[str, torch.Tensor],
        validation_data: Tuple[torch.Tensor, torch.Tensor],
        branching_sample_values: OrderedDict[str, torch.Tensor],
    ):
        X_val, y_val = validation_data
        cond_model = pyro.condition(model, data=branching_sample_values)
        predictive = pyro.infer.Predictive(
            named_uncondition(cond_model, ["obs"]), samples
        )
        vectorized_trace = predictive.get_vectorized_trace(X_val, y_val)
        pred_fn = vectorized_trace.nodes["obs"]["fn"]
        log_p = pred_fn.log_prob(y_val)
        return log_p.logsumexp(dim=0) - torch.log(torch.tensor(log_p.shape[0]))

    return predictive_dist


def mcmc_to_traces(model, slp_info: SLPInfo, model_args, model_kwargs):
    traces = []
    samples = slp_info.mcmc_samples
    first_key = list(samples.keys())[0]
    for ix in range(samples[first_key].shape[0]):
        sampled_dict = {k: v[ix] for k, v in samples.items()}
        trace = pyro.poutine.trace(
            pyro.poutine.condition(
                model, data=sampled_dict | slp_info.branching_sample_values
            )
        ).get_trace(*model_args, **model_kwargs)
        for k in samples.keys():
            trace.nodes[k]["is_observed"] = False
        for k in slp_info.branching_sample_values.keys():
            trace.nodes[k]["is_observed"] = False
        traces.append(trace)

    return traces


@hydra.main(config_path="conf_dcc_hmc", config_name="config")
def main(cfg):
    pyro.set_rng_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    overall_start_time = time.time()
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(**cfg.dataset)
    X_trainval = torch.cat([X_train, X_val], dim=0)
    y_trainval = torch.cat([y_train, y_val], dim=0)

    logging.info("Starting DCC-HMC Validation Set run")
    model = variable_selection_model if cfg.model == "lin_reg" else log_reg_model
    dcc_hmc_stacked_val: DCCHMC = hydra.utils.instantiate(
        cfg.dcc_hmc,
        model=model,
        predictive_dist=get_predictive(cfg.model),
        validation_data=(X_val, y_val),
    )
    start_time = time.time()
    slps_info_stacked_val = dcc_hmc_stacked_val.run(
        X_train, y_train, X_val=X_val, y_val=y_val
    )
    dcc_stacked_val_time = time.time() - start_time

    # traces = []
    # for bt, slp_info in slps_info_stacked_val.items():
    #     traces += mcmc_to_traces(
    #         model, slp_info, (X_train, y_train), {"X_val": X_val, "y_val": y_val}
    #     )

    # clustered_traces, weights = compute_stacking_weights(traces)

    logging.info("Starting DCC-HMC run")
    dcc_hmc: DCCHMC = hydra.utils.instantiate(cfg.dcc_hmc, model=model)
    start_time = time.time()
    slps_info = dcc_hmc.run(X_trainval, y_trainval)
    dcc_time = time.time() - start_time

    branching_traces = list(slps_info.keys())
    logging.info("Calculating analytic log marginal likelihoods")
    start_time = time.time()
    if cfg.model == "lin_reg":
        bma_log_probs = joblib.Parallel(n_jobs=cfg.dcc_hmc.num_parallel, verbose=2)(
            joblib.delayed(compute_analytic_lml)(X_trainval, y_trainval, slps_info, bt)
            for bt in branching_traces
        )
        bma_log_probs = torch.tensor(bma_log_probs)
    else:
        bma_log_probs = torch.ones(len(branching_traces))
        logging.info(
            "Skipping because analytic log marginal likelihoods not implemented"
        )
    lml_time = time.time() - start_time

    logging.info("Calculating PI-MAIS marginal_likelihood estimates")
    start_time = time.time()
    pi_mais_fn = pi_mais if cfg.model == "lin_reg" else pi_mais_log_reg
    pi_mais_results = joblib.Parallel(n_jobs=cfg.dcc_hmc.num_parallel, verbose=2)(
        joblib.delayed(pi_mais_fn)(model, X_trainval, y_trainval, slps_info, bt)
        for bt in branching_traces
    )
    pi_mais_lml = torch.tensor([r[0] for r in pi_mais_results])
    pi_mais_time = time.time() - start_time

    bma_weights = torch.exp(bma_log_probs - torch.logsumexp(bma_log_probs, dim=0))
    pi_mais_weights = torch.exp(pi_mais_lml - torch.logsumexp(pi_mais_lml, dim=0))
    stacking_weights = torch.zeros((len(branching_traces),))
    for ix, bt in enumerate(branching_traces):
        stacking_weights[ix] = slps_info[bt].stacking_weight
    stacking_val_set_weights = torch.zeros((len(branching_traces),))
    for ix, bt in enumerate(branching_traces):
        stacking_val_set_weights[ix] = slps_info_stacked_val[bt].stacking_weight
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
    test_lppds = torch.zeros((len(branching_traces), X_test.shape[0]))
    compute_lppd_fn = compute_lppd if cfg.model == "lin_reg" else compute_lppd_log_reg
    for ix, bt in enumerate(branching_traces):
        X_test_selected = get_included_features(
            X_test, slps_info[bt].branching_sample_values
        )
        test_lppds[ix, :] = compute_lppd_fn(
            slps_info[bt].mcmc_samples,
            X_test_selected,
            y_test,
        )

    # Calculate the lppd on held out data for validation set
    start_time = time.time()
    test_val_set_lppds = torch.zeros((len(branching_traces), X_test.shape[0]))
    for ix, bt in enumerate(branching_traces):
        X_test_selected = get_included_features(
            X_test, slps_info_stacked_val[bt].branching_sample_values
        )
        test_val_set_lppds[ix, :] = compute_lppd_fn(
            slps_info_stacked_val[bt].mcmc_samples,
            X_test_selected,
            y_test,
        )

    # Compute stacking lppd
    stacking_lppd = evaluate_stacked_lppd(stacking_weights, test_lppds).mean()
    stacking_val_set_lppd = evaluate_stacked_lppd(
        stacking_val_set_weights, test_val_set_lppds
    ).mean()

    # Compute bma lppd
    bma_lppd = evaluate_stacked_lppd(bma_weights, test_lppds).mean()

    # Compute PI-MAIS lppd
    pi_mais_lppd = evaluate_stacked_lppd(pi_mais_weights, test_lppds).mean()
    equal_weights_lppd = evaluate_stacked_lppd(equal_weights, test_lppds).mean()
    eval_time = time.time() - start_time

    logging.info(f"Stacking lppd: {stacking_lppd.item():.2f}")
    logging.info(f"Stacking Val Set lppd: {stacking_val_set_lppd.item():.2f}")
    logging.info(f"BMA lppd: {bma_lppd:.2f}")
    logging.info(f"PI-MAIS lppd: {pi_mais_lppd:.2f}")
    logging.info(f"Equal lppd: {equal_weights_lppd:.2f}")

    # Save results in pickle file.
    with open("results.pickle", "wb") as f:
        pickle.dump(
            {
                "slps_info": slps_info,
                "slps_info_val": slps_info_stacked_val,
                "stacking_lppd": stacking_lppd,
                "stacking_val_set_lppd": stacking_val_set_lppd,
                "bma_lppd": bma_lppd,
                "pi_mais_lppd": pi_mais_lppd,
                "equal_lppd": equal_weights_lppd,
                "stacking_weights": stacking_weights,
                "stacking_val_set_weights": stacking_val_set_weights,
                "bma_weights": bma_weights,
                "pi_mais_weights": pi_mais_weights,
            },
            f,
        )
    with open("data.pickle", "wb") as f:
        torch.save([X_train, y_train, X_val, y_val, X_test, y_test], f)

    overall_time = time.time() - overall_start_time

    logging.info("")
    logging.info("Timings:")
    logging.info(
        f"Overall time: {int(overall_time // 60)} m {int(overall_time % 60)} s"
    )
    timings = [
        ("DCC-HMC Stacked", dcc_stacked_val_time),
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