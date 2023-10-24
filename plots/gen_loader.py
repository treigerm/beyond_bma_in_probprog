import torch
from collections import defaultdict
from dataclasses import dataclass
import joblib
import time
import os
import pickle
import pyro
import arviz as az
from pyro import poutine
from typing import Tuple, Dict, List, Callable
from utils import MethodResults
import warnings

import sys

sys.path.append("..")
from models.pyro_extensions.dcc_hmc import SLPInfo


@dataclass
class GenLoader:

    gen_fname: str
    num_jobs: int
    model: Callable
    force_cache_regen: bool
    burn_in_rjmcmc: int
    model_kwargs: Dict

    def load(
        self, pyro_multirun_dir: str, results_dir: str, num_runs: int
    ) -> Tuple[MethodResults, List[Dict[str, SLPInfo]]]:
        """Load all the RJMCMC runs and compute the LPPD on held out data."""
        lppds = torch.zeros(num_runs)
        slp_weights = defaultdict(lambda: torch.zeros(num_runs))
        slp_infos = []

        parallel = joblib.Parallel(n_jobs=self.num_jobs, return_as="generator")
        sample_gen = parallel(
            joblib.delayed(self.load_and_convert_gen)(
                pyro_multirun_dir, results_dir, run_ix
            )
            for run_ix in range(num_runs)
        )
        start_t = time.time()
        for run_ix, (lppd, slp_ws, slp_info) in enumerate(sample_gen):
            lppds[run_ix] = lppd
            for slp_name, weight in slp_ws.items():
                slp_weights[slp_name][run_ix] = weight
            slp_infos.append(slp_info)

        print(f"Loading Gen LPPDs: {time.time() - start_t}\n")

        return (
            MethodResults(
                name="RJMCMC",
                lppd=lppds,
                weights=torch.stack(list(slp_weights.values()), dim=1),
                slp_names=list(slp_weights.keys()),
            ),
            slp_infos,
        )

    def load_and_convert_gen(
        self, pyro_multirun_dir: str, results_dir: str, run_ix: int
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, SLPInfo]]:
        """Load a single run of RJMCMC and compute LPPD and weights."""
        cache_fname = os.path.join(pyro_multirun_dir, f"gen_rjmcmc_{run_ix}.pkl")
        if os.path.exists(cache_fname) and not self.force_cache_regen:
            with open(cache_fname, "rb") as f:
                d = pickle.load(f)
                return d["lppd"], d["slp_weights"], d["slp_infos"]

        # Need to be able to handle .pkl or .pickle files
        fname = os.path.join(results_dir, f"{self.gen_fname}_{run_ix}.pickle")
        if not os.path.isfile(fname):
            fname = os.path.join(results_dir, f"{self.gen_fname}_{run_ix}.pkl")
        with open(fname, "rb") as f:
            traces = pickle.load(f)
            traces = traces["traces"]
        slp_infos = self.convert_gen_to_pyro(traces[self.burn_in_rjmcmc :])

        train_data, test_data = self.load_data(
            os.path.join(pyro_multirun_dir, str(run_ix))
        )

        self.gen_compute_loo(slp_infos, train_data)
        lppd, slp_ws = self.gen_compute_lppd(slp_infos, test_data)
        with open(cache_fname, "wb") as f:
            pickle.dump(
                {"lppd": lppd, "slp_weights": slp_ws, "slp_infos": slp_infos}, f
            )
        return lppd, slp_ws, slp_infos

    def gen_compute_lppd(
        self, rjmcmc_samples: Dict[str, SLPInfo], data: Tuple
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the LPPD for each SLP and return the weights."""
        slp_weights = {}
        num_samples = 0
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
            slp_weights[slp_name] = test_lppds[-1].shape[0]
            num_samples += slp_weights[slp_name]

        slp_weights = {k: v / num_samples for k, v in slp_weights.items()}

        test_lppds = torch.cat(test_lppds, dim=0)
        return (
            test_lppds.logsumexp(dim=0) - torch.log(torch.tensor(test_lppds.shape[0]))
        ).mean(), slp_weights

    def gen_compute_loo(
        self, rjmcmc_samples: Dict[str, SLPInfo], data: Tuple
    ) -> Dict[str, torch.Tensor]:
        """Compute the PSIS-LOO for each SLP."""
        for _, samples in rjmcmc_samples.items():
            bsv = samples.branching_sample_values
            num_samples = next(iter(samples.mcmc_samples.values())).shape[0]
            if num_samples < 10:
                continue
            cond_model = poutine.block(
                poutine.condition(self.model, data=bsv),
                hide=set(bsv.keys()),
            )  # Block out the branching so they don't get interpreted as likelihood.
            kernel = pyro.infer.NUTS(cond_model)
            mcmc = pyro.infer.MCMC(kernel, num_samples=num_samples)

            # Add an extra dimension to samples because we have a single chain
            for k, v in samples.mcmc_samples.items():
                samples.mcmc_samples[k] = v.unsqueeze(0)
            mcmc._samples = samples.mcmc_samples
            mcmc._args = data
            mcmc._kwargs = dict()
            mcmc._diagnostics = [[]]
            az_pyro = az.data.io_pyro.PyroConverter(posterior=mcmc)
            idata = az.data.InferenceData(
                posterior=az_pyro.posterior_to_xarray(),
                log_likelihood=az_pyro.log_likelihood_to_xarray(),
            )
            # Ignore excessive warnings from ArviZ.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples.psis_loo = torch.tensor(
                    az.loo(idata, pointwise=True).loo_i.data
                )

            # Remove the extra dimension again.
            for k, v in samples.mcmc_samples.items():
                samples.mcmc_samples[k] = v.squeeze(0)

    def convert_gen_to_pyro(self, traces) -> Dict[str, SLPInfo]:
        raise NotImplementedError()

    def load_data(self, experiment_dir: str) -> Tuple[Tuple, Tuple]:
        raise NotImplementedError()

    def compute_lppd(self, branching_sample_values, samples, data, model_kwargs):
        raise NotImplementedError()
