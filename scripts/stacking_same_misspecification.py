from typing import Dict
import argparse

import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import arviz as az
import joblib

from stacking_prototype import compute_stacking_scipy, vectorized_marginal_likelihood

torch.set_default_dtype(torch.float64)
sns.set_style("ticks")
sns.set_context("talk")

RESULTS_DIR = "stacking_same_misspecification_results"

MODEL1_COV_IXS = torch.tensor([0, 2])
MODEL2_COV_IXS = torch.tensor([0, 3])


def pyro_model(X, y, covariate_ixs):
    w_dim = covariate_ixs.shape[0]
    w = pyro.sample(
        "w",
        dist.Normal(0, 1).expand([w_dim]).to_event(1),
    )
    mean = w @ X[:, covariate_ixs].T
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Normal(mean, 1.0), obs=y)


def run_mcmc(X, y, covariate_ixs, num_samples=1000):
    kernel = pyro.infer.NUTS(pyro_model, jit_compile=True, ignore_jit_warnings=True)
    mcmc = pyro.infer.MCMC(
        kernel, num_samples=num_samples, warmup_steps=400, disable_progbar=True
    )
    mcmc.run(X, y, covariate_ixs)
    loo_psis = torch.tensor(az.loo(az.from_pyro(mcmc), pointwise=True).loo_i.data)
    return {k: v.detach() for k, v in mcmc.get_samples().items()}, loo_psis


def lppd(
    samples: Dict[str, torch.Tensor],
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    covariate_ixs: torch.Tensor,
) -> torch.Tensor:
    ws = samples["w"]
    log_p = torch.zeros((X_val.shape[0], ws.shape[0]))
    for ix in range(log_p.shape[1]):
        fs = ws[ix] @ X_val[:, covariate_ixs].T
        log_p[:, ix] = dist.Normal(fs, 1).log_prob(y_val)
    return log_p.logsumexp(dim=1) - torch.log(torch.tensor(ws.shape[0]))


def evaluate_stacked_lppd(weights, lppds):
    # weights: (num_models,)
    # lppds: (num_models, num_data)
    log_weights = torch.log(weights)
    log_probs = lppds + log_weights[:, None]
    stacked_lppd = torch.logsumexp(
        log_probs, dim=0
    )  # - torch.log(torch.tensor(lppds.shape[0]))
    return stacked_lppd


def plot_lppd(df, out_fname):
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.boxplot(data=df, x="val_log_score", y="method", width=0.6, whis=[0, 100], ax=ax)
    sns.stripplot(
        data=df,
        x="val_log_score",
        y="method",
        color="0.1",
        linewidth=0,
        size=5,
        alpha=0.3,
        ax=ax,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_fname)


def plot_model_weights(df, out_fname):
    fig, ax = plt.subplots(figsize=(8, 2))
    sns.stripplot(data=df, x="model1_weight", y="method", alpha=0.1, size=8)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_fname)


def one_replication(
    num_train: int, cov_mean: float, model_lik_var: float, coeff3: float, coeff4: float
):
    N, J = num_train + 1000, 4
    betas = torch.tensor([1.5, 1.5, coeff3, coeff4])
    X = dist.Normal(cov_mean, 1).sample((N, J))
    y = X @ betas + dist.Normal(0, 1).sample((N,))
    X_train, y_train = X[:num_train, :], y[:num_train]
    X_test, y_test = X[num_train:, :], y[num_train:]

    all_samples = dict()
    all_samples["m1"], m1_lppd = run_mcmc(X_train, y_train, MODEL1_COV_IXS)
    all_samples["m2"], m2_lppd = run_mcmc(X_train, y_train, MODEL2_COV_IXS)
    full_lppd = torch.stack([m1_lppd, m2_lppd], dim=1).T

    # Compute stacking weights
    stacking_weights = compute_stacking_scipy(full_lppd)

    # Compute BMA weights
    m1_lml = vectorized_marginal_likelihood(
        X_train[:, MODEL1_COV_IXS],
        y_train,
        torch.tensor(0),
        torch.tensor(1.0),
        torch.tensor([model_lik_var]),
    )[0]
    m2_lml = vectorized_marginal_likelihood(
        X_train[:, MODEL2_COV_IXS],
        y_train,
        torch.tensor(0),
        torch.tensor(1.0),
        torch.tensor([model_lik_var]),
    )[0]
    lmls = torch.stack([m1_lml, m2_lml], dim=0)
    bma_weights = (lmls - torch.logsumexp(lmls, dim=0)).exp()

    # Evaluation on test data
    stacking_lppd_test = torch.stack(
        [
            lppd(all_samples["m1"], X_test, y_test, MODEL1_COV_IXS),
            lppd(all_samples["m2"], X_test, y_test, MODEL2_COV_IXS),
        ],
        dim=1,
    ).T

    stacking_mean_test_lppd = evaluate_stacked_lppd(
        stacking_weights, stacking_lppd_test
    ).mean()
    bma_mean_test_lppd = evaluate_stacked_lppd(bma_weights, stacking_lppd_test).mean()

    return stacking_weights, bma_weights, stacking_mean_test_lppd, bma_mean_test_lppd


def main(
    cov_mean: float = 5.0,
    model_lik_var: float = 1.0,
    coeff3: float = 0.0,
    coeff4: float = 0.0,
    num_train: int = 200,
    num_replications: int = 5,
    num_parallel_jobs: int = 10,
    seed: int = 0,
):
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)

    stacking_weights = torch.zeros((num_replications, 2))
    bma_weights = torch.zeros((num_replications, 2))
    stacking_mean_test_lppd = torch.zeros((num_replications,))
    bma_mean_test_lppd = torch.zeros((num_replications,))

    results = joblib.Parallel(n_jobs=num_parallel_jobs, verbose=2)(
        joblib.delayed(one_replication)(
            num_train, cov_mean, model_lik_var, coeff3, coeff4
        )
        for _ in range(num_replications)
    )
    for ix, (s_ws, b_ws, s_lppd, b_lppd) in enumerate(results):
        stacking_weights[ix, :] = s_ws
        bma_weights[ix, :] = b_ws
        stacking_mean_test_lppd[ix] = s_lppd
        bma_mean_test_lppd[ix] = b_lppd

    df = pd.DataFrame(
        {
            "method": num_replications * ["Full Bayes"]
            + num_replications * ["Stacking"],
            "model1_weight": np.concatenate(
                (bma_weights[:, 0].numpy(), stacking_weights[:, 0].numpy())
            ),
            "val_log_score": np.concatenate(
                (bma_mean_test_lppd.numpy(), stacking_mean_test_lppd.numpy())
            ),
        }
    )
    prefix = "_".join(
        f"{k}={v}"
        for k, v in [
            ("seed", seed),
            ("num_replications", num_replications),
            ("num_train", num_train),
            ("cov_mean", cov_mean),
            ("model_lik_var", model_lik_var),
            ("coeff3", coeff3),
            ("coeff4", coeff4),
        ]
    )
    prefix = f"{RESULTS_DIR}/overlap_{prefix}"
    df.to_csv(f"{prefix}.csv", index=False)
    # plot_lppd(df, f"{prefix}_lppd.pdf")
    # plot_model_weights(df, f"{prefix}_model_weights.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-lik-var", type=float, default=1.0)
    parser.add_argument("--cov-mean", type=float, default=5.0)
    parser.add_argument("--coeff3", type=float, default=0.0)
    parser.add_argument("--coeff4", type=float, default=0.0)
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-replications", type=int, default=5)
    parser.add_argument("--num-parallel-jobs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))