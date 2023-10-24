import torch
import pyro
import copy

from collections import OrderedDict
from typing import Any, Callable, Dict
from pyro import poutine
from pyro.poutine.util import site_is_subsample

from .util import get_sample_addresses


def identity(x):
    return x


branch = poutine.runtime.effectful(identity, type="branch")


def greater_than_0(x: torch.Tensor):
    return x > 0


branch_greater_than_0 = poutine.runtime.effectful(
    greater_than_0, type="branch_greater_than"
)


class BranchingTraceMessenger(pyro.poutine.messenger.Messenger):
    def __init__(self):
        super().__init__()

    def __call__(self, fn):
        return BranchingTraceHandler(self, fn)

    def __enter__(self):
        self.branching_trace = ""
        self.sampled_values: Dict[str, torch.Tensor] = OrderedDict()
        self.guard_evaluations: list[bool] = []
        self.guard_arguments: list[torch.Tensor] = []
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        return super().__exit__(*args, **kwargs)

    def _postprocess_message(self, msg):
        if msg["type"] == "branch":
            binary_branch = str(int(msg["value"]))
            self.branching_trace += binary_branch
        elif msg["type"] == "sample" and msg["infer"].get("branching", False):
            self.branching_trace += str(msg["value"].item())
            self.sampled_values[msg["name"]] = msg["value"]
        elif msg["type"] == "branch_greater_than":
            self.guard_evaluations.append(bool(msg["value"]))
            assert len(msg["args"]) == 1 and isinstance(msg["args"][0], torch.Tensor)
            self.guard_arguments.append(msg["args"][0])

            self.branching_trace += str(int(msg["value"]))

    def get_trace(self):
        return copy.deepcopy(self.branching_trace)

    def get_sampled_values(self):
        return copy.deepcopy(self.sampled_values)


class BranchingTraceHandler:
    def __init__(self, msngr, fn):
        self.fn = fn
        self.msngr = msngr

    def __call__(self, *args, **kwargs):
        with self.msngr:
            return self.fn(*args, **kwargs)

    def get_trace(self, *args, **kwargs):
        self(*args, **kwargs)
        return self.msngr.get_trace()


_, branching_trace = pyro.poutine.handlers._make_handler(BranchingTraceMessenger)


class BranchingReplayMessenger(pyro.poutine.messenger.Messenger):
    def __init__(self, guard_evaluations: list[bool] = None):
        super().__init__()

        self.guard_evaluations = guard_evaluations

    def __enter__(self):
        self.guard_count: int = 0
        return super().__enter__()

    def _pyro_branch_greater_than(self, msg):
        # Replace whatever the guard evaluated to with the predefined value.
        guard_evaluation = self.guard_evaluations[self.guard_count]
        msg["value"] = guard_evaluation
        self.guard_count += 1
        return None


_, branch_replay = pyro.poutine.handlers._make_handler(BranchingReplayMessenger)


class LogJointBranchSmoothingHandler:
    def __init__(
        self,
        fn: Callable,
        guard_evaluations: list[bool],
        epsilon: torch.Tensor,
        sigmoid_factor: torch.Tensor = torch.tensor(1.0),
        max_plate_nesting: int = 0,
    ):
        self.fn = fn
        self.guard_evaluations = guard_evaluations
        self.epsilon = epsilon

        self.max_plate_nesting = max_plate_nesting

        # Depending on whether the guard evaluated to true or false we need to
        # change the sign in the sigmoid.
        self.sigmoid_signs = torch.tensor(
            [1 if b else -1 for b in self.guard_evaluations]
        )
        for _ in range(self.max_plate_nesting + 1):
            self.sigmoid_signs.unsqueeze_(0)
        self.sigmoid_factor = sigmoid_factor

        self.msngr = BranchingTraceMessenger()
        self.trace_msngr = pyro.poutine.trace_messenger.TraceMessenger()

    def __call__(self, *args: Any, **kwargs: Any):
        with self.trace_msngr:
            with self.msngr:
                return branch_replay(self.fn, guard_evaluations=self.guard_evaluations)(
                    *args, **kwargs
                )

    def log_prob(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self(*args, **kwargs)
        trace = self.trace_msngr.get_trace()
        trace.compute_log_prob()

        log_prob = None
        log_prob_initialized = False
        for _, site in trace.nodes.items():
            if site["type"] == "sample":
                site_log_prob = site["log_prob"]
                if self.max_plate_nesting > 0:
                    site_log_prob = site_log_prob.sum(
                        dim=tuple(range(-self.max_plate_nesting, 0))
                    )
                if not log_prob_initialized:
                    log_prob = site_log_prob
                    log_prob_initialized = True
                else:
                    log_prob = log_prob + site_log_prob

        smoothing_factor = torch.nn.functional.logsigmoid(
            self.sigmoid_signs
            * self.sigmoid_factor
            * torch.stack(self.msngr.guard_arguments, dim=-1)
        ).sum(dim=tuple(range(-(self.max_plate_nesting + 1), 0)))
        complementary_smoothing_factor = torch.log(
            1
            - torch.max(
                torch.min(torch.exp(smoothing_factor), torch.tensor(1) - 8 * 1e-16),
                torch.tensor(1e-20),
            )
        )
        log_p = torch.logsumexp(
            torch.stack(
                [
                    smoothing_factor + log_prob,
                    complementary_smoothing_factor + self.epsilon,
                ],
                dim=1,
            ),
            dim=1,
        ).sum()
        return log_p
        self(*args, **kwargs)
        trace = self.trace_msngr.get_trace()

        log_prob = trace.log_prob_sum()

        # Get smoothed guards
        guard_arguments = torch.stack(self.msngr.guard_arguments)
        smoothing_factor = torch.nn.functional.logsigmoid(
            self.sigmoid_signs * self.sigmoid_factor * guard_arguments
        ).sum()

        # Have minimum and maximum values for numerical stability.
        complementary_smoothing_factor = torch.log(
            1
            - torch.max(
                torch.min(torch.exp(smoothing_factor), torch.tensor(1) - 8 * 1e-16),
                torch.tensor(1e-20),
            )
        )
        # Make smoothed density
        smoothed_density = torch.logsumexp(
            torch.stack(
                [
                    smoothing_factor + log_prob,
                    complementary_smoothing_factor + self.epsilon,
                ],
                dim=0,
            ),
            dim=0,
        )
        return smoothed_density


class LogJointBranchingTraceHandler:
    def __init__(self, fn, branching_trace, epsilon=torch.tensor(float("-inf"))):
        self.fn = fn
        self.msngr = BranchingTraceMessenger()
        self.trace_msngr = pyro.poutine.trace_messenger.TraceMessenger()
        self.slp_identifier = branching_trace
        self.epsilon = epsilon

    def __call__(self, *args, **kwargs):
        with self.trace_msngr:
            with self.msngr:
                return self.fn(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        self(*args, **kwargs)
        trace = self.trace_msngr.get_trace()
        # NOTE: This is a bit of a hack. We should really have an argument to the
        #       handler which checks whether the SLP is fully identified by samples
        #       from discrete distributions. Here we are using the heuristic that
        #       if there is a comma in the slp_identifier we are only using the address
        #       path to distinguish between SLPs.
        if "," in self.slp_identifier:
            slp_id = ",".join(get_sample_addresses(trace))
        else:
            slp_id = self.msngr.get_trace()

        if slp_id == self.slp_identifier:
            return trace.log_prob_sum()
        else:
            return self.epsilon


class LogJointBranchingTraceHandlerv2:
    def __init__(
        self,
        fn,
        guide_trace: poutine.Trace,
        prototype_trace: poutine.Trace,
        branching_trace: str,
        epsilon=torch.tensor(float("-inf")),
    ):
        self.fn = fn
        # self.msngr = BranchingTraceMessenger()
        self.trace_msngr = pyro.poutine.trace_messenger.TraceMessenger()
        self.guide_trace = guide_trace
        self.prototype_trace = prototype_trace
        self.slp_identifier = branching_trace
        self.epsilon = epsilon

    def __call__(self, *args, **kwargs):
        with self.trace_msngr:
            # with self.msngr:
            return self.fn(*args, **kwargs)

    def log_prob(self, *args, **kwargs):
        self(*args, **kwargs)
        trace = self.trace_msngr.get_trace()
        address_trace = ",".join(get_sample_addresses(trace))
        # branching_trace = self.msngr.get_trace()

        if address_trace == self.slp_identifier:
            return trace.log_prob_sum()
        else:
            # For each sample statement check whether it is also in self.prototype_trace
            # Evaluate density under the prior distribution in prototype_trace
            # If at any point we get an error assign 0 weight (i.e. log_prob = -inf)
            log_prob_sum = torch.tensor(0.0)
            for name, site in self.guide_trace.iter_stochastic_nodes():
                if (
                    site["type"] != "sample"
                    or site["is_observed"]
                    or site_is_subsample(site)
                    or site.get("infer", dict()).get("is_auxiliary", False)
                ):
                    continue

                if not (name in self.prototype_trace.nodes):
                    print(
                        f"WARNING: Address {name} not in prototype trace (keys: {self.prototype_trace.nodes.keys()}"
                    )
                    return torch.tensor(float("-inf"))

                prior_dist = self.prototype_trace.nodes[name]["fn"]
                log_prob_sum += prior_dist.log_prob(site["value"])

            return self.epsilon + log_prob_sum


class NamedUnconditionMessenger(poutine.messenger.Messenger):
    """
    Messenger to force the value of observed nodes to be sampled from their
    distribution, ignoring observations. Only applied to nodes with specific names.
    """

    def __init__(self, node_list: list[str]):
        super().__init__()
        self.node_list = node_list

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        Samples value from distribution, irrespective of whether or not the
        node has an observed value.
        """
        if msg["name"] in self.node_list and msg["is_observed"]:
            msg["is_observed"] = False
            msg["infer"]["was_observed"] = True
            msg["infer"]["obs"] = msg["value"]
            msg["value"] = None
            msg["done"] = False
        return None


_, named_uncondition = poutine.handlers._make_handler(NamedUnconditionMessenger)