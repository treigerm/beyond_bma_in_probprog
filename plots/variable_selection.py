import torch
from typing import Tuple, Dict, List, Callable
from collections import OrderedDict, defaultdict
import os
from dataclasses import dataclass, field
import pyro.distributions as dist

import sys

sys.path.append("..")
from variable_selection import variable_selection_model, log_reg_model
from models.pyro_extensions.dcc_hmc import SLPInfo

from evaluator import Evaluator
from utils import (
    filter_threshold,
    convert_tuples_to_tensor,
    print_latex_table,
    plot_weights,
)

NUM_JOBS = 1
BURN_IN_RJMCMC = 20_000
FORCE_CACHE_REGEN = True
BETAS = [0.001, 0.01, 0.1, 1.0, 10.0]

RJMCMC_RESULTS_DIR = "../rjmcmc_gen/results/variable_selection"

# TODO: Fill out paths to result directories.
CAL_DATA = ""
DIABETES_DATA = ""
STROKE_DATA = ""


def convert_gen_to_pyro_variable_selection(traces) -> Dict[str, SLPInfo]:
    num_features = 8
    slp_infos = dict()
    for trace in traces:

        branching_sample_values = OrderedDict()
        key = ""
        for ix in range(num_features):
            branching_sample_values[f"feature_{ix}"] = torch.tensor(
                trace[("feature", ix + 1)], dtype=torch.float32
            )
            key += f"{trace[('feature', ix+1)]:.1f}"

        samples = {}
        samples["weights"] = convert_tuples_to_tensor(trace, "weight")

        if "noise_var" in trace:
            samples["noise_var"] = torch.tensor(trace["noise_var"])

        if key not in slp_infos:
            slp_infos[key] = SLPInfo(
                initial_trace=None,
                branching_sample_values=branching_sample_values,
                mcmc_samples=defaultdict(list),
            )
        for k, v in samples.items():
            slp_infos[key].mcmc_samples[k].append(v)

    # Append all the samples for each key
    for key in slp_infos.keys():
        for k, v in slp_infos[key].mcmc_samples.items():
            slp_infos[key].mcmc_samples[k] = torch.stack(v)

    return slp_infos


def compute_lppd(
    samples: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
):
    means = X @ samples["weights"].T
    log_p = dist.Normal(means, samples["noise_var"].sqrt()).log_prob(
        y[:, None].expand_as(means)
    )
    return log_p.T  # (num_samples, num_data)
    # return (log_p.logsumexp(dim=1) - torch.log(torch.tensor(log_p.shape[1]))).mean()


def compute_lppd_log_reg(
    samples: Dict[str, torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
):
    logits = X @ samples["weights"].T
    log_p = dist.Bernoulli(logits=logits).log_prob(y[:, None].expand_as(logits))
    return log_p.T  # (num_samples, num_data)
    # return (log_p.logsumexp(dim=1) - torch.log(torch.tensor(log_p.shape[1]))).mean()


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


@dataclass
class VarSelectExperiment(Evaluator):
    # Need these three fields or otherwise we get dataclass error.
    name: str = "California"
    gen_fname: str = "california"
    pyro_multirun_dir: str = CAL_DATA
    model: Callable = variable_selection_model
    num_jobs: int = NUM_JOBS
    force_cache_regen: bool = FORCE_CACHE_REGEN
    burn_in_rjmcmc: int = BURN_IN_RJMCMC
    model_kwargs: Dict = field(default_factory=lambda: dict())
    rjmcmc_dir: str = RJMCMC_RESULTS_DIR
    betas: List[float] = field(default_factory=lambda: BETAS)
    num_replications: int = 10

    def convert_gen_to_pyro(self, traces) -> Dict:
        return convert_gen_to_pyro_variable_selection(traces)

    def load_data(self, data_dir: str) -> Tuple[Tuple, Tuple]:
        (x_train, y_train, x_val, y_val, x_test, y_test) = torch.load(
            os.path.join(data_dir, "data.pickle")
        )
        x_train = torch.cat([x_train, x_val], dim=0)
        y_train = torch.cat([y_train, y_val], dim=0)
        return (x_train, y_train), (x_test, y_test)

    def compute_lppd(self, branching_sample_values, samples, data, model_kwargs):
        do_log_reg = self.gen_fname != "california"
        X, y = data
        lppd_fn = compute_lppd_log_reg if do_log_reg else compute_lppd
        X_selected = get_included_features(X, branching_sample_values)
        return lppd_fn(samples, X_selected, y)

    def plot_weights(self, df):
        df["SLP Index"] = df["SLP Index"].replace(
            {k: i for i, k in enumerate(sorted(df["SLP Index"].unique()))}
        )
        df = filter_threshold(df)
        self._plot_weights(df)

    def _plot_weights(self, df):
        raise NotImplementedError()


METHODS2PLOT = ["Stacked", "Stacked (Val)", "BMA", "BMA (Analytic)", "RJMCMC"]


@dataclass
class California(VarSelectExperiment):
    def _plot_weights(self, df):
        plot_weights(
            df,
            figsize=(8, 3),
            method_names=METHODS2PLOT,
            legend=True,
            fname="stacking_figures/california_weights.pdf",
        )


@dataclass
class Diabetes(VarSelectExperiment):
    name: str = "Diabetes"
    gen_fname: str = "diabetes"
    pyro_multirun_dir: str = DIABETES_DATA
    model: Callable = log_reg_model

    def _plot_weights(self, df):
        plot_weights(
            df,
            figsize=(20, 3),
            method_names=METHODS2PLOT,
            fname="stacking_figures/diabetes_weights.pdf",
        )


@dataclass
class Stroke(VarSelectExperiment):
    name: str = "Stroke"
    gen_fname: str = "stroke"
    pyro_multirun_dir: str = STROKE_DATA
    model: Callable = log_reg_model

    def _plot_weights(self, df):
        plot_weights(
            df,
            figsize=(12, 3),
            method_names=METHODS2PLOT,
            fname="stacking_figures/stroke_weights.pdf",
        )


def main():
    regression_results = dict()
    regression_exps = [California(), Diabetes(), Stroke()]
    for conf in regression_exps:
        print(40 * "=")
        print(conf.name.upper())
        print(40 * "=")
        regression_results[conf.name] = conf.evaluate()
        print("\n")

    print(40 * "=")
    print("LPPD RESULTS TABLE")
    print(40 * "=")
    # print_latex_table(regression_results)


if __name__ == "__main__":
    main()