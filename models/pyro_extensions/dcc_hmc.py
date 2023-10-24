from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from typing import Callable, Optional, List, Tuple, Dict
from six.moves import queue
import math

import torch
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.poutine.util import site_is_subsample

import numpy as np
import scipy.optimize
import scipy.special

import arviz as az
import joblib

from .handlers import BranchingTraceMessenger
from .resource_allocation import AbstractUtility
from .util import get_sample_addresses


@dataclass
class SLPInfo:
    initial_trace: poutine.Trace
    num_proposed: int = 0
    num_selected: int = 0
    mcmc_samples: Optional[dict[str, torch.Tensor]] = None
    branching_sample_values: OrderedDict[str, torch.Tensor] = field(
        default_factory=OrderedDict
    )
    stacking_weight: torch.Tensor = torch.tensor(0.0)
    psis_loo: Optional[torch.Tensor] = None


def compute_stacking_scipy(
    lppds: torch.Tensor, log_prior: Optional[Callable] = None
) -> torch.Tensor:
    J = lppds.shape[0]

    if log_prior is None:
        log_prior = lambda _: 0.0

    def stacking_obj(log_w):
        log_w = torch.tensor(log_w)
        w_full = log_w - torch.logsumexp(log_w, dim=0)
        # Can use sum instead of mean because normalization factor does not affect
        # the optimization.
        total_sum = torch.logsumexp(
            lppds + w_full[:, None], dim=0
        ).sum()  # + log_prior(w_full)
        return -total_sum.numpy()

    theta = np.ones((J,))
    sol = scipy.optimize.minimize(
        fun=stacking_obj,
        x0=theta,
        method="L-BFGS-B",
        bounds=[(0, None) for _ in range(J)],
    )

    return torch.tensor(sol["x"] - scipy.special.logsumexp(sol["x"])).exp()


def branching_escape(trace, msg):
    return (
        (msg["type"] == "sample")
        and (not msg["is_observed"])
        and (msg["name"] not in trace)
        and ("infer" in msg)
        and (msg["infer"].get("branching", False))
    )


class DCCHMC:
    """
    Runs an adjusted version of DCC. Differences to vanilla DCC:
    - All the branches have to be determined by discrete samples and the remaining variables are continuous
    - We use the PSIS-LOO method to approximate the LOO error
    - Resource allocation is based on PSIS-LOO and not on marginal likelihood
    - Final SLP weights are determined by stacking and not marginal likelihood
    """

    def __init__(
        self,
        model,
        num_chains: int = 1,
        num_mcmc_warmup: int = 400,
        num_mcmc: int = 1000,
        enumerate_branches: bool = False,
        max_slps: Optional[int] = None,
        num_slp_samples: int = 100,
        num_parallel: int = 1,
        max_stacked: Optional[int] = None,  # None means no limit.
        predictive_dist: Optional[Callable] = None,
        validation_data: Optional[None] = None,
        beta: float = float("inf"),
    ) -> None:
        self.model = model
        self.num_chains = num_chains
        self.num_mcmc_warmup = num_mcmc_warmup
        self.num_mcmc = num_mcmc

        self.enumerate_branches = enumerate_branches
        self.max_slps = max_slps
        self.num_slp_samples = num_slp_samples

        self.num_parallel = num_parallel

        self.max_stacked = max_stacked
        self.beta = beta

        self.predictive_dist = predictive_dist
        self.validation_data = validation_data

    def find_slps(self, *args, **kwargs) -> tuple[set[str], dict[str, SLPInfo]]:
        if self.enumerate_branches:
            return self.enumerate_slps(*args, **kwargs)
        else:
            return self.sample_slps(*args, **kwargs)

    def sample_slps(self, *args, **kwargs) -> tuple[set[str], dict[str, SLPInfo]]:
        """
        Distinguish SLPs based on the sampled values at sample sites which are
        annotated with infer={'branching'=True}.
        """
        slp_traces = set()
        slp_info: dict[str, SLPInfo] = dict()
        for _ in range(self.num_slp_samples):
            # trace = poutine.trace(self.model).get_trace(*args, **kwargs)
            with torch.no_grad():
                with pyro.poutine.trace_messenger.TraceMessenger() as tmsngr:
                    with BranchingTraceMessenger() as btmsngr:
                        ret = pyro.poutine.block(self.model, hide_types=["param"])(
                            *args, **kwargs
                        )

            trace = tmsngr.get_trace()
            # Need to manually add the return node if we use trace messenger as a
            # context.
            trace.add_node("_RETURN", name="_RETURN", type="return", value=ret)

            branching_trace = btmsngr.get_trace()
            if branching_trace in slp_traces:
                slp_info[branching_trace].num_proposed += 1
            else:
                slp_traces.add(branching_trace)
                slp_info[branching_trace] = SLPInfo(
                    trace, branching_sample_values=btmsngr.get_sampled_values()
                )
                if self.max_slps and len(slp_traces) >= self.max_slps:
                    break

        return slp_traces, slp_info

    def enumerate_slps(self, *args, **kwargs) -> tuple[set[str], dict[str, SLPInfo]]:
        q = queue.Queue()
        q.put(poutine.Trace())

        enum_model = poutine.queue(self.model, queue=q, escape_fn=branching_escape)
        branching_sample_values = []
        while not q.empty():
            trace = poutine.trace(enum_model).get_trace(*args, **kwargs)
            bsv = OrderedDict()
            for name, site in trace.nodes.items():
                infer = site.get("infer", {})
                if infer.get("branching", False):
                    bsv[name] = site["value"]
            branching_sample_values.append(bsv)
            if self.max_slps and len(branching_sample_values) >= self.max_slps:
                break

        slp_traces = set()
        slp_info: dict[str, SLPInfo] = dict()
        for bsv in branching_sample_values:
            bt = "".join([str(v.item()) for v in bsv.values()])
            slp_traces.add(bt)
            slp_info[bt] = SLPInfo(trace, branching_sample_values=bsv)

        return slp_traces, slp_info

    def run(self, *args, **kwargs) -> dict[str, SLPInfo]:
        A_total, slps_info = self.find_slps(*args, **kwargs)
        A_active: set[str] = set()
        initialized_slps: set[str] = set()

        # Add models into active set
        for addr_trace, _ in slps_info.items():
            # if slp_info.num_proposed >= self.min_num_proposed:
            #     A_active.add(addr_trace)
            A_active.add(addr_trace)

        # For all new models initialise MCMC chains
        A_active = list(A_active)
        results = joblib.Parallel(n_jobs=self.num_parallel, verbose=2)(
            joblib.delayed(self.run_mcmc)(
                self.model,
                args,
                kwargs,
                slp_info=slps_info[addr_trace],
                num_mcmc_steps=self.num_mcmc,
                num_warmup=self.num_mcmc_warmup,
                num_chains=self.num_chains,
            )
            for addr_trace in A_active
        )
        for ix, (mcmc_samples, loo_vals) in enumerate(results):
            slps_info[A_active[ix]].mcmc_samples = mcmc_samples
            slps_info[A_active[ix]].psis_loo = loo_vals

        self.compute_stacking_weights(slps_info, self.max_stacked)
        return slps_info

    def run_mcmc(
        self,
        model,
        args,
        kwargs,
        slp_info: SLPInfo,
        num_mcmc_steps: int,
        num_chains: int = 1,
        num_warmup: int = 100,
        jit_compile: bool = True,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """
        Run MCMC on the given SLP.
        """
        cond_model = poutine.block(
            poutine.condition(model, data=slp_info.branching_sample_values),
            hide=set(slp_info.branching_sample_values.keys()),
        )  # Block out the branching so they don't get interpreted as likelihood.
        kernel = pyro.infer.NUTS(
            cond_model, jit_compile=jit_compile, ignore_jit_warnings=True
        )
        mcmc = pyro.infer.MCMC(
            kernel,
            num_samples=num_mcmc_steps,
            warmup_steps=num_warmup,
            disable_progbar=True,
        )
        mcmc.run(*args, **kwargs)

        log_val_scores = self.get_log_scores(mcmc, slp_info)
        samples = {k: v.detach() for k, v in mcmc.get_samples().items()}

        for _ in range(num_chains - 1):
            kernel2 = pyro.infer.NUTS(
                cond_model, jit_compile=jit_compile, ignore_jit_warnings=True
            )
            mcmc2 = pyro.infer.MCMC(
                kernel2,
                num_samples=num_mcmc_steps,
                warmup_steps=num_warmup,
                disable_progbar=True,
            )
            mcmc2.run(*args, **kwargs)
            log_scores = self.get_log_scores(mcmc2, slp_info)

            log_val_scores = torch.cat([log_val_scores, log_scores])
            samples = {
                k: torch.cat([v, mcmc2.get_samples()[k].detach()])
                for k, v in samples.items()
            }

        return samples, log_val_scores

    def get_log_scores(
        self, mcmc: pyro.infer.mcmc.api.MCMC, slp_info: SLPInfo
    ) -> torch.Tensor:
        if self.predictive_dist is None:
            log_scores = torch.tensor(
                az.loo(az.from_pyro(mcmc), pointwise=True).loo_i.data
            )
        else:
            log_scores = self.predictive_dist(
                mcmc.get_samples(),
                self.validation_data,
                slp_info.branching_sample_values,
            )

        return log_scores

    def compute_stacking_weights(
        self, slps_info: dict[str, SLPInfo], max_stacked: Optional[int]
    ):
        """
        Compute the stacking weights for the SLPs.
        """
        # Join the lppd values of all the SLPs.
        branching_traces = list(slps_info.keys())
        if not max_stacked:
            max_stacked = len(branching_traces)

        lppds = torch.stack(
            [slps_info[addr_trace].psis_loo for addr_trace in branching_traces]
        )

        if len(branching_traces) > max_stacked:
            # Remove the SLPs with the lowest mean lppd values.
            mean_lppds = lppds.mean(dim=1)
            sorted_ixs = torch.argsort(mean_lppds)
            bottom_ixs, top_k_ixs = sorted_ixs[:-max_stacked], sorted_ixs[-max_stacked:]
            for ix in bottom_ixs:
                slps_info[branching_traces[ix]].stacking_weight = torch.tensor(0.0)
            branching_traces = [branching_traces[ix] for ix in top_k_ixs]
            lppds = lppds[top_k_ixs]

        weights = self.compute_stacking_scipy(lppds, self.beta)

        # Update the SLPInfo objects with the stacking weights.
        for ix, bt in enumerate(branching_traces):
            slps_info[bt].stacking_weight = weights[ix]

    @staticmethod
    def compute_stacking_scipy(
        lppds: torch.Tensor, beta: float = float("inf")
    ) -> torch.Tensor:
        J = lppds.shape[0]

        regularizer = lambda _: 0.0
        if math.isfinite(beta):
            # Negative entropy of categorical distribution.
            regularizer = lambda log_w: (1 / beta) * (log_w.exp() @ log_w)

        def stacking_obj(log_w):
            log_w = torch.tensor(log_w)
            w_full = log_w - torch.logsumexp(log_w, dim=0)
            # Can use sum instead of mean because normalization factor does not affect
            # the optimization.
            total_sum = torch.logsumexp(
                lppds + w_full[:, None], dim=0
            ).sum() - regularizer(w_full)
            return -total_sum.numpy()

        theta = np.ones((J,))
        sol = scipy.optimize.minimize(
            fun=stacking_obj,
            x0=theta,
            method="L-BFGS-B",
            bounds=[(0, None) for _ in range(J)],
        )

        return torch.tensor(sol["x"] - scipy.special.logsumexp(sol["x"])).exp()


def compute_stacking_weights(
    post_samples: List[poutine.Trace],
) -> Tuple[List[List[poutine.Trace]], torch.Tensor]:
    # Cluster traces based on SLPs
    addr_trace2trace: Dict[str, List[poutine.Trace]] = defaultdict(list)
    for trace in post_samples:
        addr_trace = ""
        for name, site in trace.nodes.items():
            addr_trace += name
            infer = site.get("infer", {})
            if infer.get("branching", False):
                addr_trace += str(site["value"].item())

        addr_trace2trace[addr_trace].append(trace)

    addr_traces = list(addr_trace2trace.keys())
    traces = list(addr_trace2trace.values())
    val_set_len = traces[0][0].nodes["_RETURN"]["value"].shape[0]

    # Extract the return values and store the mean in a (K,M) matrix
    lppds = torch.zeros((len(addr_traces), val_set_len))
    for ix, traces_for_slps in enumerate(traces):
        num_traces = len(traces_for_slps)
        log_p = torch.zeros((num_traces, val_set_len))
        for jx, trace in enumerate(traces_for_slps):
            log_p[jx, :] = trace.nodes["_RETURN"]["value"]

        lppds[ix, :] = log_p.logsumexp(dim=0) - torch.log(torch.tensor(num_traces))

    # Compute the stacking weights based on the (K, M) matrix
    weights = compute_stacking_scipy(lppds)

    return traces, weights


def make_log_prior(slp_info: List[SLPInfo]) -> Callable:
    slp_ess = torch.ones((len(slp_info),)) / len(slp_info)
    concentration = slp_ess * 1.001  # Factor of 1.001 recommended by Yao et al.

    def log_prior(log_weights: torch.Tensor) -> torch.Tensor:
        return dirichlet_log_prob(concentration, log_weights)

    return log_prior


def dirichlet_log_prob(
    concentration: torch.Tensor, log_x: torch.Tensor
) -> torch.Tensor:
    """
    Log probability of a Dirichlet distribution as a function of the log inputs.
    """
    return (
        (log_x * (concentration - 1)).sum()
        + torch.lgamma(concentration.sum())
        - torch.lgamma(concentration).sum()
    )