import math
import scipy.optimize

import torch
import numpy as np


def evaluate_stacked_lppd(weights: torch.Tensor, lppds: torch.Tensor) -> torch.Tensor:
    # weights: (num_models,)
    # lppds: (num_models, num_data)
    log_weights = torch.log(weights)
    log_probs = lppds + log_weights[:, None]
    stacked_lppd = torch.logsumexp(log_probs, dim=0)
    return stacked_lppd  # (num_data,)


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
        total_sum = torch.logsumexp(lppds + w_full[:, None], dim=0).sum() - regularizer(
            w_full
        )
        return -total_sum.numpy()

    theta = np.ones((J,))
    sol = scipy.optimize.minimize(
        fun=stacking_obj,
        x0=theta,
        method="L-BFGS-B",
        bounds=[(0, None) for _ in range(J)],
    )

    return torch.tensor(sol["x"] - scipy.special.logsumexp(sol["x"])).exp()