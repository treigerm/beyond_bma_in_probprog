from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Callable
import pandas as pd
import pyro
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import torch
import os

import sys

sys.path.append("..")

from radon import radon_model_v2
from models.pyro_extensions.handlers import named_uncondition
from models.pyro_extensions.dcc_hmc import SLPInfo

from evaluator import Evaluator
from utils import plot_weights, convert_tuples_to_tensor, plot_beta_weight_sensitivity

NUM_JOBS = 13

NUM_RADON_SLPS = 12
BURN_IN_RJMCMC = 20_000
FORCE_CACHE_REGEN = False
BETAS = [0.001, 0.01, 0.1, 1.0, 10.0]

# Directory with Gen results
RADON_RESULTS_DIR = "../rjmcmc_gen/results/radon"

# TODO: Fill out paths to result directory.
RADON_DATA = ""

LOOKUP = {
    "0": "P",  # Pooling
    "1": "NP",  # No pooling
    "2": "H",  # Hierarchical
    "3": "G",  # Group-level predictor
}


def convert_gen_to_pyro(traces: List[Dict]) -> Dict[str, SLPInfo]:
    slp_infos = dict()
    for trace in traces:
        #  Need -1 because Julia uses 1-based indexing
        first_char = str(trace["alpha_choice"] - 1)
        second_char = str(trace["beta_choice"] - 1)
        key = LOOKUP[first_char] + "," + LOOKUP[second_char]

        if key not in slp_infos:
            branching_sample_values = OrderedDict()
            branching_sample_values["alpha_choices"] = (
                torch.tensor(trace["alpha_choice"]) - 1
            )
            branching_sample_values["beta_choices"] = (
                torch.tensor(trace["beta_choice"]) - 1
            )

            slp_infos[key] = SLPInfo(
                initial_trace=None,
                branching_sample_values=branching_sample_values,
                mcmc_samples=defaultdict(list),
            )

        # Alpha choices:
        if trace["alpha_choice"] == 1:
            slp_infos[key].mcmc_samples["alpha"].append(trace["alpha"])
        elif trace["alpha_choice"] == 2:
            slp_infos[key].mcmc_samples["alpha"].append(
                convert_tuples_to_tensor(trace, "alpha")
            )
        elif trace["alpha_choice"] == 3:
            slp_infos[key].mcmc_samples["mean_a"].append(trace["mean_a"])
            slp_infos[key].mcmc_samples["std_a"].append(trace["std_a"])
            slp_infos[key].mcmc_samples["z_a"].append(
                convert_tuples_to_tensor(trace, "z_a")
            )
        elif trace["alpha_choice"] == 4:
            slp_infos[key].mcmc_samples["gamma_0"].append(trace["gamma_0"])
            slp_infos[key].mcmc_samples["gamma_1"].append(trace["gamma_1"])
            slp_infos[key].mcmc_samples["std_a"].append(trace["std_a"])
            slp_infos[key].mcmc_samples["z_a"].append(
                convert_tuples_to_tensor(trace, "z_a")
            )

        # Beta choices:
        if trace["beta_choice"] == 1:
            slp_infos[key].mcmc_samples["beta"].append(trace["betas"])
        elif trace["beta_choice"] == 2:
            slp_infos[key].mcmc_samples["beta"].append(
                convert_tuples_to_tensor(trace, "beta")
            )
        elif trace["beta_choice"] == 3:
            slp_infos[key].mcmc_samples["mean_b"].append(trace["mean_b"])
            slp_infos[key].mcmc_samples["std_b"].append(trace["std_b"])
            slp_infos[key].mcmc_samples["z_b"].append(
                convert_tuples_to_tensor(trace, "z_b")
            )

        slp_infos[key].mcmc_samples["sigma"].append(trace["sigma"])

    # Append all the samples for each key
    for key in slp_infos.keys():
        for k, v in slp_infos[key].mcmc_samples.items():
            if isinstance(v[0], float):
                slp_infos[key].mcmc_samples[k] = torch.tensor(v)
            else:
                slp_infos[key].mcmc_samples[k] = torch.stack(v)

    return slp_infos


def load_radon_data(data_dir):
    data = torch.load(os.path.join(data_dir, "data.pickle"))
    with open(os.path.join(data_dir, "mn_counties.pickle"), "rb") as f:
        counties = pickle.load(f)
    return *data, counties


def compute_lppd(
    model,
    branching_sample_values: OrderedDict[str, torch.Tensor],
    samples: dict[str, torch.Tensor],
    model_args: Tuple,
    model_kwargs: dict,
) -> torch.Tensor:
    """Generic function to compute LPPD for a given model."""
    cond_model = pyro.condition(model, data=branching_sample_values)
    predictive = pyro.infer.Predictive(named_uncondition(cond_model, ["ys"]), samples)
    vectorized_trace = predictive.get_vectorized_trace(*model_args, **model_kwargs)
    pred_fn = vectorized_trace.nodes["ys"]["fn"]
    return pred_fn.log_prob(model_args[0])  # shape: (num_samples, num_data)


@dataclass
class RadonExperiment(Evaluator):
    name: str = "Radon"
    gen_fname: str = "radon_traces"
    model: Callable = radon_model_v2
    num_jobs: int = NUM_JOBS
    force_cache_regen: bool = FORCE_CACHE_REGEN
    burn_in_rjmcmc: int = BURN_IN_RJMCMC
    model_kwargs: Dict = field(default_factory=dict)
    pyro_multirun_dir: str = RADON_DATA
    rjmcmc_dir: str = RADON_RESULTS_DIR
    betas: List[float] = field(default_factory=lambda: BETAS)
    num_replications: int = 40

    def convert_gen_to_pyro(self, traces) -> Dict[str, SLPInfo]:
        return convert_gen_to_pyro(traces)

    def load_data(self, data_dir: str) -> Tuple:
        (
            log_radon_train,
            log_radon_test,
            floor_train,
            floor_test,
            county_train,
            county_test,
            uranium,
        ) = torch.load(os.path.join(data_dir, "data.pickle"))
        with open(os.path.join(data_dir, "mn_counties.pickle"), "rb") as f:
            num_counties = len(pickle.load(f))
        train_data = (log_radon_train, floor_train, county_train, num_counties, uranium)
        test_data = (log_radon_test, floor_test, county_test, num_counties, uranium)
        return train_data, test_data

    def compute_lppd(self, branching_sample_values, samples, data, model_kwargs):
        return compute_lppd(
            radon_model_v2, branching_sample_values, samples, data, model_kwargs
        )

    def plot_weights(self, df_weights: pd.DataFrame):
        df_weights = rename_slp_index(df_weights)
        plot_weights(
            df_weights,
            figsize=(13, 3),
            method_names=["Stacked", "BMA", "RJMCMC"],
            legend=True,
            legend_kwargs={"loc": "center left", "bbox_to_anchor": (0.7, 0.6)},
            fname="stacking_figures/radon_weights_rjmcmc.pdf",
        )

        plot_beta_weight_sensitivity(
            df_weights, figsize=(13, 3), fname="stacking_figures/radon_beta_weights.pdf"
        )

        fig, ax = plt.subplots(figsize=(13, 3))
        sns.stripplot(
            data=df_weights[df_weights["Method"].str.startswith("RJMCMC")],
            x="SLP Index",
            y="SLP Weight",
            hue="Method",
            alpha=0.8,
            dodge=True,
            ax=ax,
            size=10,
        )
        fig.tight_layout()
        fig.savefig("stacking_figures/radon_weights_rjmcmc_stacked.pdf")


def rename_slp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the SLP index to a more interpretable format. Default is a list of integers
    i.e. '00' or '01'. This function converts this to a string representation with mnemonic
    descriptions of the branching traces."""
    slp_index2string_representation = {}
    lookup = {
        "0": "P",  # Pooling
        "1": "NP",  # No pooling
        "2": "H",  # Hierarchical
        "3": "G",  # Group-level predictor
    }
    chars = list(lookup.keys())
    for first_char in chars:
        for second_char in chars[:-1]:
            bt = "".join([first_char, second_char])
            slp_index2string_representation[bt] = (
                lookup[first_char] + "," + lookup[second_char]
            )

    return df.replace(slp_index2string_representation)


if __name__ == "__main__":
    RadonExperiment().evaluate()