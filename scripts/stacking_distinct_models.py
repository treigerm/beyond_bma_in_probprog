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

RESULTS_DIR = "stacking_distinct_model_results"

PRIOR_MEAN, PRIOR_STD = torch.tensor(0.0), torch.tensor(1.0)
LIKELIHOOD1_STD = torch.tensor(2.0)
LIKELIHOOD2_STD = torch.tensor(0.62177)


def pyro_model(y, model1=True):
    z = pyro.sample("z", dist.Normal(PRIOR_MEAN, PRIOR_STD))
    sigma = LIKELIHOOD1_STD if model1 else LIKELIHOOD2_STD
    with pyro.plate("data", y.shape[0]):
        pyro.sample("obs", dist.Normal(z, sigma), obs=y)


def run_mcmc(y, num_samples=1000, model1=True):
    kernel = pyro.infer.NUTS(pyro_model, jit_compile=True, ignore_jit_warnings=True)
    mcmc = pyro.infer.MCMC(
        kernel, num_samples=num_samples, warmup_steps=400, disable_progbar=True
    )
    mcmc.run(y, model1=model1)
    loo_psis = torch.tensor(az.loo(az.from_pyro(mcmc), pointwise=True).loo_i.data)
    return {k: v.detach() for k, v in mcmc.get_samples().items()}, loo_psis


def lppd(
    samples: Dict[str, torch.Tensor],
    y_val: torch.Tensor,
    model1: bool = True,
) -> torch.Tensor:
    zs = samples["z"]
    log_p = torch.zeros((y_val.shape[0], zs.shape[0]))
    for ix in range(log_p.shape[1]):
        sigma = LIKELIHOOD1_STD if model1 else LIKELIHOOD2_STD
        log_p[:, ix] = dist.Normal(zs[ix], sigma).log_prob(y_val)
    return log_p.logsumexp(dim=1) - torch.log(torch.tensor(zs.shape[0]))


def log_marginal_likelihood(
    data: torch.Tensor,
    likelihood_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """Calculate the marginal likelihood of a branch.

    Taken from Section 2.5 at https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf.
    """
    num_data = torch.tensor(data.shape[0], dtype=torch.float)
    likelihood_var = torch.pow(likelihood_std, 2)
    prior_var = torch.pow(prior_std, 2)

    first_term = (
        torch.log(likelihood_std)
        - num_data * torch.log(torch.sqrt(torch.tensor(2) * torch.pi) * likelihood_std)
        + 0.5 * torch.log(num_data * prior_var + likelihood_var)
    )
    second_term = -(torch.pow(data, 2).sum() / (2 * likelihood_var)) - (
        torch.pow(prior_mean, 2) / (2 * prior_var)
    )
    third_term = (
        (
            prior_var
            * torch.pow(num_data, 2)
            * torch.pow(torch.mean(data), 2)
            / likelihood_var
        )
        + (likelihood_var * torch.pow(prior_mean, 2) / prior_var)
        + 2 * num_data * torch.mean(data) * prior_mean
    ) / (2 * (num_data * prior_var + likelihood_var))
    return first_term + second_term + third_term


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


def one_replication():
    y_train = dist.Normal(0, 1).sample((200,))
    y_test = dist.Normal(0, 1).sample((1000,))

    all_samples = dict()
    all_samples["m1"], m1_lppd = run_mcmc(y_train, model1=True)
    all_samples["m2"], m2_lppd = run_mcmc(y_train, model1=False)
    full_lppd = torch.stack([m1_lppd, m2_lppd], dim=1).T

    # Compute stacking weights
    stacking_weights = compute_stacking_scipy(full_lppd)

    # Compute BMA weights
    m1_lml = log_marginal_likelihood(y_train, LIKELIHOOD1_STD, PRIOR_MEAN, PRIOR_STD)
    m2_lml = log_marginal_likelihood(y_train, LIKELIHOOD2_STD, PRIOR_MEAN, PRIOR_STD)
    lmls = torch.stack([m1_lml, m2_lml], dim=0)
    bma_weights = (lmls - torch.logsumexp(lmls, dim=0)).exp()

    # Evaluation on test data
    stacking_lppd_test = torch.stack(
        [
            lppd(all_samples["m1"], y_test, model1=True),
            lppd(all_samples["m2"], y_test, model1=False),
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
        joblib.delayed(one_replication)() for _ in range(num_replications)
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
        ]
    )
    prefix = f"{RESULTS_DIR}/distinct_{prefix}"
    df.to_csv(f"{prefix}.csv", index=False)
    # plot_lppd(df, f"{prefix}_lppd.pdf")
    # plot_model_weights(df, f"{prefix}_model_weights.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-replications", type=int, default=5)
    parser.add_argument("--num-parallel-jobs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))