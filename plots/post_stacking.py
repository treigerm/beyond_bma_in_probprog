from typing import List, Tuple, Dict
from collections import defaultdict
import torch
import os
from utils import MethodResults

import sys

sys.path.append("..")
from models.pyro_extensions.dcc_hmc import DCCHMC, SLPInfo
from scripts.stacking_prototype import evaluate_stacked_lppd


class PostStacking:
    model_kwargs: Dict = dict()

    def stacking(
        self,
        data_dir: str,
        slp_infos: List[Dict[str, SLPInfo]],
        beta: float = 1.0,
        prefix: str = "",
    ) -> MethodResults:
        stacking_lppd = torch.zeros(len(slp_infos))
        slp_weights = defaultdict(lambda: torch.zeros(len(slp_infos)))
        for run_ix, slp_info in enumerate(slp_infos):
            _, data = self.load_data(os.path.join(data_dir, str(run_ix)))
            # Under RJMCMC some SLPs might not have psis_loo computed.
            branching_traces = [
                k for k, v in slp_info.items() if v.psis_loo is not None
            ]

            lppds = torch.stack(
                [slp_info[addr_trace].psis_loo for addr_trace in branching_traces]
            )
            stacking_weights = DCCHMC.compute_stacking_scipy(lppds, beta=beta)

            test_lppds = []
            for ix, bt in enumerate(branching_traces):
                log_p = self.compute_lppd(
                    slp_info[bt].branching_sample_values,
                    slp_info[bt].mcmc_samples,
                    data,
                    self.model_kwargs,
                )
                test_lppds.append(
                    log_p.logsumexp(dim=0) - torch.log(torch.tensor(log_p.shape[0]))
                )
            test_lppds = torch.stack(test_lppds)

            stacking_lppd[run_ix] = evaluate_stacked_lppd(
                stacking_weights, test_lppds
            ).mean()
            for ix, bt in enumerate(branching_traces):
                slp_weights[bt][run_ix] = stacking_weights[ix]

        return MethodResults(
            name=f"{prefix}Stacking Beta={beta}",
            lppd=stacking_lppd,
            weights=torch.stack(list(slp_weights.values()), dim=1),
            slp_names=list(slp_weights.keys()),
        )

    def load_data(self, data_dir: str) -> Tuple[Tuple, Tuple]:
        raise NotImplementedError()

    def compute_lppd(self, branching_sample_values, samples, data, model_kwargs):
        raise NotImplementedError()