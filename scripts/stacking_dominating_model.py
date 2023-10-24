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

from stacking_prototype import compute_stacking_scipy, evaluate_stacked_lppd

torch.set_default_dtype(torch.float64)
sns.set_style("ticks")
sns.set_context("talk")

RESULTS_DIR = "stacking_dominating_model_results"


def pyro_model(X, y, sin_transform=False):
    # prior_std = 1 if not sin_transform else 5
    w = pyro.sample("w", dist.Normal(0, 1))
    sigma = pyro.sample("sigma", dist.Gamma(1, 1))
    fs = w * X
    if sin_transform:
        fs = torch.sin(fs)
    with pyro.plate("data", X.shape[0]):
        pyro.sample("obs", dist.Normal(fs, sigma), obs=y)


def run_mcmc(X, y, num_samples=1000, sin_transform=False):
    kernel = pyro.infer.NUTS(pyro_model, jit_compile=True, ignore_jit_warnings=True)
    mcmc = pyro.infer.MCMC(
        kernel, num_samples=num_samples, warmup_steps=400, disable_progbar=True
    )
    mcmc.run(X, y, sin_transform=sin_transform)
    loo_psis = torch.tensor(az.loo(az.from_pyro(mcmc), pointwise=True).loo_i.data)
    return {k: v.detach() for k, v in mcmc.get_samples().items()}, loo_psis


def lppd(
    samples: Dict[str, torch.Tensor],
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    sin_transform: bool = False,
) -> torch.Tensor:
    ws, sigmas = samples["w"], samples["sigma"]
    log_p = torch.zeros((X_val.shape[0], ws.shape[0]))
    for ix in range(log_p.shape[1]):
        fs = X_val * ws[ix]
        if sin_transform:
            fs = torch.sin(fs)
        log_p[:, ix] = dist.Normal(fs, sigmas[ix]).log_prob(y_val)
    return log_p.logsumexp(dim=1) - torch.log(torch.tensor(ws.shape[0]))


def joint_log_prob(w, sigma, X, y, sin_transform=False):
    # Define the prior distributions
    prior_std = 1 if not sin_transform else 5
    prior_w = dist.Normal(0, prior_std)
    prior_sigma = dist.Gamma(1, 1)

    # Define the likelihood
    fs = w[:, None] * X
    if sin_transform:
        fs = torch.sin(fs)
    likelihood = dist.Normal(fs, sigma[:, None])

    # Compute the log probability of the joint distribution
    log_prob = (
        prior_w.log_prob(w)
        + prior_sigma.log_prob(sigma)
        + likelihood.log_prob(y).sum(dim=1)
    )
    return log_prob


def importance_sampling_lml(X, y, num_samples=100_000, sin_transform=False):
    q_w, q_sigma = dist.Normal(0, 1), dist.Gamma(1, 1)
    betas = q_w.sample((num_samples,))
    sigma = q_sigma.sample((num_samples,))

    log_p = joint_log_prob(betas, sigma, X, y, sin_transform=sin_transform)
    log_q = q_w.log_prob(betas) + q_sigma.log_prob(sigma)

    log_ws = log_p - log_q
    ess = torch.exp(
        2 * torch.logsumexp(log_ws, dim=0) - torch.logsumexp(2 * log_ws, dim=0)
    )
    return torch.logsumexp(log_ws, dim=0) - torch.log(torch.tensor(num_samples)), ess


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
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(out_fname)


def one_replication(num_train=200, num_test=1000):
    def f(x):
        return 2 * x + torch.sin(5 * x)

    noise_std = 1.0
    X_train = dist.Normal(0, 1).sample((num_train,))
    y_train = f(X_train) + dist.Normal(0, noise_std).sample((num_train,))

    X_test = dist.Normal(0, 1).sample((num_test,))
    y_test = f(X_test) + dist.Normal(0, noise_std).sample((num_test,))

    all_samples = dict()
    all_samples["m1"], m1_lppd = run_mcmc(X_train, y_train, sin_transform=False)
    all_samples["m2"], m2_lppd = run_mcmc(X_train, y_train, sin_transform=True)
    full_lppd = torch.stack([m1_lppd, m2_lppd], dim=1).T

    # Compute stacking weights
    stacking_weights = compute_stacking_scipy(full_lppd)

    # Compute BMA weights
    m1_lml, _ = importance_sampling_lml(
        X_train, y_train, sin_transform=False, num_samples=100_000
    )
    m2_lml, _ = importance_sampling_lml(
        X_train, y_train, sin_transform=True, num_samples=100_000
    )
    lmls = torch.stack([m1_lml, m2_lml], dim=0)
    bma_weights = (lmls - torch.logsumexp(lmls, dim=0)).exp()

    # Evaluation on test data
    stacking_lppd_test = torch.stack(
        [
            lppd(all_samples["m1"], X_test, y_test, sin_transform=False),
            lppd(all_samples["m2"], X_test, y_test, sin_transform=True),
        ],
        dim=1,
    ).T

    stacking_mean_test_lppd = evaluate_stacked_lppd(
        stacking_weights, stacking_lppd_test
    ).mean()
    bma_mean_test_lppd = evaluate_stacked_lppd(bma_weights, stacking_lppd_test).mean()

    return stacking_weights, bma_weights, stacking_mean_test_lppd, bma_mean_test_lppd


def main(
    num_replications: int = 5,
    num_parallel_jobs: int = 10,
    num_train: int = 200,
    num_test: int = 1000,
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
        joblib.delayed(one_replication)(num_train, num_test)
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
            ("num_test", num_test),
        ]
    )
    prefix = f"{RESULTS_DIR}/{prefix}"
    df.to_csv(f"{prefix}.csv", index=False)
    plot_lppd(df, f"{prefix}_lppd.pdf")
    plot_model_weights(df, f"{prefix}_model_weights.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-replications", type=int, default=5)
    parser.add_argument("--num-train", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=1000)
    parser.add_argument("--num-parallel-jobs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))