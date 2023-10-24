import torch
from typing import Tuple, Dict, List, Callable
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
import pyro
import pandas as pd
import sys

sys.path.append("..")
from models.pyro_extensions.handlers import named_uncondition
from models.pyro_extensions.dcc_hmc import SLPInfo
from function_induction import function_induction_model
from evaluator import Evaluator

import os

from utils import (
    plot_weights,
    filter_threshold,
)

NUM_JOBS = 1
BURN_IN_RJMCMC = 20_000
FORCE_CACHE_REGEN = False
BETAS = [0.001, 0.01, 0.1, 1.0, 10.0]

RJMCMC_RESULTS_DIR = "../rjmcmc_gen/results/fun_ind"

# TODO: Fill out paths to result directory.
FUN_IND_DATA = ""

# NOTE: Ideally this should be read from the experiment config. For now we just hardcode it.
MODEL_KWARGS = {
    "dummy_sample": True,
    "classification": False,
    "quadratic": False,
}


def add_branching_sample_values(
    prefix: str, branching_trace: str, bsv: OrderedDict, trace: Dict
) -> str:
    # DFS through the parse tree.
    # NOTE: This is not noticeably faster than the recursive version.
    frontier = deque([prefix])
    while len(frontier) > 0:
        prefix = frontier.popleft()
        key = f"{prefix}_rule"
        bsv[key] = trace[key]
        branching_trace += str(trace[key])
        # if trace[key] == 0 we just continue
        if trace[key] == 1:
            frontier.appendleft(f"{prefix}_sin_")
        elif trace[key] == 2:
            frontier.appendleft(f"{prefix}_plus1_")
            frontier.appendleft(f"{prefix}_plus2_")

    return branching_trace


@dataclass
class FunIndExperiment(Evaluator):
    name: str = "Fun Ind"
    gen_fname: str = "traces"
    model: Callable = function_induction_model
    num_jobs: int = NUM_JOBS
    force_cache_regen: bool = FORCE_CACHE_REGEN
    burn_in_rjmcmc: int = BURN_IN_RJMCMC
    model_kwargs: Dict = field(default_factory=lambda: MODEL_KWARGS)
    pyro_multirun_dir: str = FUN_IND_DATA
    rjmcmc_dir: str = RJMCMC_RESULTS_DIR
    betas: List[float] = field(default_factory=lambda: BETAS)
    num_replications: int = 10

    def convert_gen_to_pyro(self, traces) -> Dict[str, SLPInfo]:
        slp_infos = dict()
        for trace in traces:
            branching_sample_values = OrderedDict()
            key = add_branching_sample_values("", "", branching_sample_values, trace)

            if key not in slp_infos:
                slp_infos[key] = SLPInfo(
                    initial_trace=None,
                    branching_sample_values=branching_sample_values,
                    mcmc_samples=defaultdict(list),
                )

            for k, v in trace.items():
                if "_rule" in k:
                    continue
                slp_infos[key].mcmc_samples[k].append(v)

        # Append all the samples for each key
        for key in slp_infos.keys():
            for k, v in slp_infos[key].mcmc_samples.items():
                if isinstance(v[0], float):
                    slp_infos[key].mcmc_samples[k] = torch.tensor(v)
                else:
                    slp_infos[key].mcmc_samples[k] = torch.stack(v)

        return slp_infos

    def load_data(self, data_dir: str) -> Tuple[Tuple, Tuple]:
        (x_train, y_train, x_test, y_test) = torch.load(
            os.path.join(data_dir, "data.pickle")
        )
        return (x_train, y_train), (x_test, y_test)

    def compute_lppd(self, branching_sample_values, samples, data, model_kwargs):
        X, y = data
        cond_model = pyro.condition(
            function_induction_model, data=branching_sample_values
        )
        predictive = pyro.infer.Predictive(
            named_uncondition(cond_model, ["ys"]), samples
        )
        vectorized_trace = predictive.get_vectorized_trace(X, y, **model_kwargs)
        pred_fn = vectorized_trace.nodes["ys"]["fn"]
        return pred_fn.log_prob(y)

    def plot_weights(self, df_weights: pd.DataFrame):
        df_weights["SLP Index"] = df_weights["SLP Index"].replace(
            {
                k: i
                for i, k in enumerate(sorted(df_weights["SLP Index"].unique(), key=len))
            }
        )
        df_weights = filter_threshold(df_weights, 0.3)
        plot_weights(
            df_weights,
            figsize=(15, 3),
            method_names=["Stacked", "Stacked (Val)", "BMA", "RJMCMC"],
            legend=True,
            fname="stacking_figures/fun_ind_weights_rjmcmc.pdf",
        )


if __name__ == "__main__":
    FunIndExperiment().evaluate()