import torch
import pyro.distributions as dist
import pyro
import pandas as pd
import numpy as np
from evaluator import Evaluator
from collections import defaultdict
from dataclasses import dataclass, field
import sys
from typing import Dict, Callable, List, Tuple
import os
import pickle
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
from utils import (
    MethodResults,
    print_lppd_df,
    plot_weights,
    plot_beta_weight_sensitivity,
)

sys.path.append("..")

from models.pyro_extensions.dcc_hmc import SLPInfo

FORCE_CACHE_REGEN = False
BURN_IN_RJMCMC = 20_000
BETAS = [0.001, 0.01, 0.1, 1.0, 10.0]

PRIOR_MEAN, PRIOR_VAR = torch.tensor(0), torch.tensor(10)

# TODO: Fill out paths to result file generated from ../subset_regression.py
PYRO_REBUTTAL = ""

RENAME_DICT = (
    {
        "HMC LOO Stacking": "Stacked",
        "HMC Stacking": "Stacked (Val)",
        "HMC BMA": "BMA",
        "HMC BMA Ground Truth": "BMA (Analytic)",
        "HMC Equal": "Equal",
    }
    | {
        f"HMC LOO Stacking Beta={i}": f"Stacking Beta={i}"
        for i in [0.001, 0.01, 0.1, 1.0, 10]
    }
    | {
        f"HMC Stacking Beta={i}": f"Stacking (Val) Beta={i}"
        for i in [0.001, 0.01, 0.1, 1.0, 10]
    }
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


def convert_gen_to_pyro(traces: Dict[str, List[float]]) -> Dict[str, SLPInfo]:
    slp_infos = dict()
    for ix in range(len(traces["k"])):
        key = str(traces["k"][ix] + 1)
        if key not in slp_infos:
            branching_sample_values = dict()
            branching_sample_values["k"] = torch.tensor(traces["k"][ix]) - 1
            slp_infos[key] = SLPInfo(
                initial_trace=None,
                branching_sample_values=branching_sample_values,
                mcmc_samples=defaultdict(list),
            )

        for name in ["beta", "sigma"]:
            slp_infos[key].mcmc_samples[name].append(traces[name][ix])

    for key in slp_infos.keys():
        for k, v in slp_infos[key].mcmc_samples.items():
            slp_infos[key].mcmc_samples[k] = torch.tensor(v)

    return slp_infos


def extract_key(data, run_ix: str, key: str) -> torch.Tensor:
    return torch.tensor(data[run_ix][key][0, 0].astype(np.float64))


@dataclass
class SubsetExperiment(Evaluator):
    name: str = "Subset"
    model: Callable = pyro_subset_regression
    num_jobs: int = 1  # Unused
    pyro_multirun_dir: str = PYRO_REBUTTAL
    rjmcmc_dir: str = "../rjmcmc_gen/rjmcmc_results.pickle"
    gen_fname: str = ""  # Unused
    force_cache_regen: bool = FORCE_CACHE_REGEN
    burn_in_rjmcmc: int = BURN_IN_RJMCMC
    model_kwargs: Dict = field(default_factory=dict)
    betas: List[float] = field(default_factory=lambda: BETAS)
    num_replications: int = 10

    def evaluate(self, rjmcmc=True, post_stacking=True) -> List[MethodResults]:
        sns.set_context("talk", font_scale=1.1, rc={"text.usetex": True})
        update_rc_params = {
            "font.family": "serif",
        }
        plt.rcParams.update(update_rc_params)

        method_results: List[MethodResults] = []

        # Add RJMCMC results.
        if rjmcmc:
            rjmcmc_result, rjmcmc_slp_info = self.load(
                self.pyro_multirun_dir, self.rjmcmc_dir, self.num_replications
            )
            method_results.append(rjmcmc_result)
            method_results.append(
                self.stacking(
                    self.pyro_multirun_dir,
                    rjmcmc_slp_info,
                    beta=float("inf"),
                    prefix="RJMCMC ",
                )
            )

        method_results += self.load_script_data(self.pyro_multirun_dir)

        df_weights = pd.concat([x.weights_to_df() for x in method_results])
        df_lppds = pd.concat([x.lppd_to_df() for x in method_results])

        self.plot_weights(df_weights)

        print_lppd_df(
            df_lppds[df_lppds["Method"].str.startswith("RJMCMC")],
            reference_method="RJMCMC",
        )

        print_lppd_df(df_lppds)
        df_lppds["name"] = self.name
        df_lppds.to_csv(f"stacking_lppds/{self.name}_lppds.csv")

    def load(
        self, pyro_file_prefix: str, rjmcmc_file: str, num_runs: int
    ) -> Tuple[MethodResults, List[Dict[str, SLPInfo]]]:
        data_file = f"{pyro_file_prefix}.mat"
        with open(rjmcmc_file, "rb") as f:
            rjmcmc_results = pickle.load(f)
            samples = rjmcmc_results["samples"]
            weights = torch.tensor(rjmcmc_results["weights"])

        slp_infos = [self.convert_gen_to_pyro(s) for s in samples]
        # Calculate LPPD and SLP weights.
        lppds = torch.zeros(len(slp_infos))
        for ix, rjmcmc_samples in enumerate(slp_infos):
            _, data = self.load_data(os.path.join(data_file, str(ix)))
            test_lppds = []
            for slp_name, samples in rjmcmc_samples.items():
                # Compute lppd for each SLP.
                test_lppds.append(
                    self.compute_lppd(
                        samples.branching_sample_values,
                        samples.mcmc_samples,
                        data,
                        self.model_kwargs,
                    )  # (num_samples, num_data)
                )
                samples.psis_loo = test_lppds[-1].logsumexp(dim=0) - torch.log(
                    torch.tensor(test_lppds[-1].shape[0])
                )
            test_lppds = torch.cat(test_lppds, dim=0)
            lppds[ix] = (
                test_lppds.logsumexp(dim=0)
                - torch.log(torch.tensor(test_lppds.shape[0]))
            ).mean()

        return (
            MethodResults(
                name="RJMCMC",
                lppd=lppds,
                weights=weights,
                slp_names=[str(x) for x in range(15)],
            ),
            slp_infos,
        )

    def convert_gen_to_pyro(self, traces) -> Dict[str, SLPInfo]:
        return convert_gen_to_pyro(traces)

    def load_data(self, data_dir: str) -> Tuple:
        # This is a bit of an abuse of the interface. To call load_data the
        # PostStacking class appends the run_ix to the data_dir.
        data_file, run_ix = os.path.split(data_dir)
        data = scipy.io.loadmat(data_file)
        return [
            (
                extract_key(data, run_ix, f"X_{x}"),
                extract_key(data, run_ix, f"y_{x}"),
                True,
            )
            for x in ["train", "test"]
        ]

    def compute_lppd(self, branching_sample_values, samples, data, model_kwargs):
        X, y, _ = data
        k = branching_sample_values["k"]
        return dist.Normal(
            X[:, k] * samples["beta"].unsqueeze(1), samples["sigma"].unsqueeze(1)
        ).log_prob(
            y
        )  # (num_samples, num_data)

    def load_script_data(self, file_prefix: str) -> List[MethodResults]:
        fname = f"{file_prefix}.pickle"
        with open(fname, "rb") as f:
            results = pickle.load(f)
            method2lppd = results["lppds"]
            method2weights = results["weights"]

        method_results = []
        slp_names = [str(x) for x in range(15)]
        for k, lppds in method2lppd.items():
            if not (k in RENAME_DICT.keys()):
                continue
            method_results.append(
                MethodResults(
                    name=RENAME_DICT[k],
                    lppd=lppds.squeeze(),
                    weights=method2weights[k].squeeze(),
                    slp_names=slp_names,
                )
            )
        return method_results

    def plot_weights(self, df: pd.DataFrame):
        df["SLP Index"] = df["SLP Index"].astype(int)
        df["SLP Index"] = df["SLP Index"] + 1
        df = df[df["SLP Index"] != 16]
        # Plot only two SLPs
        plot_weights(
            df[df["SLP Index"].isin([8, 12])],
            figsize=(8, 3),
            method_names=[
                "Stacked",
                "Stacked (Val)",
                "BMA",
                "BMA (Analytic)",
                "RJMCMC",
            ],
            legend=True,
            legend_kwargs={"loc": "center left", "bbox_to_anchor": (1, 0.45)},
            fname="stacking_figures/subset_weights_k8_k12.pdf",
        )
        plot_weights(
            df,
            figsize=(20, 4),
            method_names=[
                "Stacked",
                "Stacked (Val)",
                "BMA",
                "BMA (Analytic)",
                "RJMCMC",
            ],
            legend=True,
            fname="stacking_figures/subset_weights_all.pdf",
        )

        # Plot the weights for different levels of beta.
        plot_beta_weight_sensitivity(
            df, figsize=(22, 4), fname="stacking_figures/subset_weights_betas.pdf"
        )

        # Plot the RJMCMC, RJMCMC Stacked, and the Stacked weights.
        fig, ax = plt.subplots(figsize=(20, 4))
        sns.stripplot(
            data=df,
            x="SLP Index",
            y="SLP Weight",
            hue="Method",
            hue_order=["RJMCMC", "RJMCMC Stacking Beta=inf", "Stacked"],
            alpha=0.8,
            dodge=True,
            ax=ax,
            size=10,
        )
        fig.tight_layout()
        fig.savefig("stacking_figures/subset_weights_rjmcmc.pdf")


if __name__ == "__main__":
    SubsetExperiment().evaluate()