import pandas as pd
import scipy.stats
import os
import torch
from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from typing import Tuple, List, Optional, Dict, Any
import seaborn as sns
import sys

# This is necessary to load the results from the pickle files.
sys.path.append("..")
from models.pyro_extensions.dcc_hmc import SLPInfo

DIVERGING_COLORS = sns.color_palette("coolwarm")
METHOD_COLORS = {
    "Stacked": DIVERGING_COLORS[5],  # "#1f77b4",
    "Stacked (Val)": DIVERGING_COLORS[4],  # "#d62728",
    "BMA": DIVERGING_COLORS[2],  # "#ff7f0e",
    "BMA (Analytic)": DIVERGING_COLORS[1],  # "#2ca02c",
    "RJMCMC": DIVERGING_COLORS[0],  # "#9467bd",
}

LEGEND_ELEMENTS = {
    k: Line2D(
        [0], [0], marker="o", color="w", label=k, markerfacecolor=v, markersize=10
    )
    for k, v in METHOD_COLORS.items()
}


def filter_threshold(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    slps_over_threshold = df[df["SLP Weight"] > threshold]["SLP Index"].unique()
    return df[df["SLP Index"].isin(slps_over_threshold)]


@dataclass
class MethodResults:
    name: str
    lppd: torch.Tensor  # (num_runs,)
    weights: torch.Tensor  # (num_runs, num_slps)
    slp_names: List[str]  # (num_slps,)

    def weights_to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "SLP Weight": [
                    self.weights[i, j].item()
                    for i in range(self.weights.shape[0])
                    for j in range(self.weights.shape[1])
                ],
                "SLP Index": [
                    self.slp_names[i]
                    for _ in range(self.weights.shape[0])
                    for i in range(self.weights.shape[1])
                ],
                "Method": [self.name] * (self.weights.shape[0] * self.weights.shape[1]),
            }
        )
        return df

    def lppd_to_df(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "LPPD": self.lppd,
                "Method": [self.name] * self.lppd.shape[0],
                "Run Index": list(range(self.lppd.shape[0])),
            }
        )
        return df


@dataclass
class MethodKeys:
    lppd_key: str
    weights_key: Optional[str] = None


VAR_SELECT_KEYS = {
    "Stacked": MethodKeys("stacking_lppd", "stacking_weights"),
    "Stacked (Val)": MethodKeys("stacking_val_set_lppd", "stacking_val_set_weights"),
    "BMA": MethodKeys("pi_mais_lppd", "pi_mais_weights"),
    "BMA (Analytic)": MethodKeys("bma_lppd", "bma_weights"),
    "Equal": MethodKeys("equal_lppd"),
}

FUN_IND_KEYS = {
    "Stacked": MethodKeys("stacking_loo_lppd", "stacking_loo_weights"),
    "Stacked (Val)": MethodKeys("stacking_lppd", "stacking_weights"),
    "BMA": MethodKeys("pi_mais_lppd", "pi_mais_weights"),
    "BMA (Analytic)": MethodKeys("bma_lppd", "bma_weights"),
    "Equal": MethodKeys("bma_lppd"),  # Here we use the bma_lppd key for the equal lppd.
}

RADON_KEYS = {
    "Stacked": MethodKeys("stacking_lppd", "stacking_weights"),
    "BMA": MethodKeys("pi_mais_lppd", "pi_mais_weights"),
    "BMA (Analytic)": MethodKeys("bma_lppd", "bma_weights"),
    "Equal": MethodKeys("equal_lppd"),
}

METHOD_KEYS = {
    "California": VAR_SELECT_KEYS,
    "Diabetes": VAR_SELECT_KEYS,
    "Stroke": VAR_SELECT_KEYS,
    "Fun Ind": FUN_IND_KEYS,
    "Radon": RADON_KEYS,
}


def load_data(
    experiment_dir: str, name: str
) -> Tuple[List[MethodResults], List[SLPInfo]]:
    folders = os.listdir(experiment_dir)
    # Only keep directories that are digits.
    folders = [
        f
        for f in folders
        if os.path.isdir(os.path.join(experiment_dir, f)) and f.isdigit()
    ]
    folders = sorted(folders, key=int)

    method_keys = METHOD_KEYS[name]

    lppds = defaultdict(lambda: torch.zeros(len(folders)))
    weights = defaultdict(list)
    slp_infos = [None] * len(folders)

    for folder in folders:
        fname = os.path.join(experiment_dir, folder, "results.pickle")
        with open(fname, "rb") as f:
            results = pickle.load(f)

        slp_infos[int(folder)] = results["slps_info"]
        num_slps = len(results["slps_info"].keys())
        for method, keys in method_keys.items():
            lppds[method][int(folder)] = results[keys.lppd_key]
            if keys.weights_key is not None:
                weights[method].append(results[keys.weights_key])
            elif method == "Equal":
                weights[method].append(torch.ones(num_slps) / num_slps)

    for k in weights.keys():
        weights[k] = torch.stack(weights[k])

    slp_names = list(slp_infos[0].keys())
    results = []
    for method in method_keys.keys():
        results.append(MethodResults(method, lppds[method], weights[method], slp_names))
    return results, slp_infos


def print_lppd_df(df: pd.DataFrame, reference_method: str = "Stacked"):
    ref_df = df[df["Method"] == reference_method].copy()
    # Merge on 'Run Index'
    merged_df = pd.merge(
        df, ref_df[["LPPD", "Run Index"]], on="Run Index", suffixes=("", "_ref")
    )

    # Compute the LPPD difference
    merged_df["LPPD_diff"] = merged_df["LPPD"] - merged_df["LPPD_ref"]

    # No need to include difference to reference method.
    merged_df = merged_df[merged_df["Method"] != reference_method]
    # Group by 'Method' and compute the mean, standard deviation, and wilcoxon test.
    wilcoxon = lambda x: scipy.stats.wilcoxon(x).pvalue
    results = merged_df.groupby("Method")["LPPD_diff"].agg(["mean", "std", wilcoxon])
    # Give Wilcoxon p-values a more descriptive name.
    results = results.rename(columns={"<lambda_0>": "pvalue"})
    results["Significant"] = results["pvalue"] < 0.05
    print(results.to_markdown(tablefmt="simple", floatfmt=".2e"))


def print_latex_table(results):
    datasets = list(results.keys())
    # methods = list(set(method for dataset in results for method in results[dataset]))
    methods = [
        "Stacking",
        "Stacking (Val)",
        "BMA",
        "BMA (Analytic)",
        "RJMCMC",
        "Equal",
    ]

    method2latex = {
        "Stacking": "\\colorstacked",
        "Stacking (Val)": "\\colorstackedval",
        "BMA": "\\colorbma",
        "BMA (Analytic)": "\\colorbmaanalytic",
        "RJMCMC": "\\colorrjmcmc",
        "Equal": "\\textbf{Equal}",
    }
    print("\\begin{table}")
    print("\\centering")
    print("\\begin{tabular}{|c|" + "|".join(["c"] * len(datasets)) + "|}")
    print("\\hline")
    print("Method & " + " & ".join(datasets) + "\\\\")
    print("\\hline")
    for method in methods:
        row = method2latex[method] + " & "
        for dataset in datasets:
            stacking_results = results[dataset]["Stacking"]
            if method in results[dataset]:
                metric_values = results[dataset][method]
                diff_to_stacking = stacking_results - metric_values
                mean = diff_to_stacking.mean().item()
                std_dev = diff_to_stacking.std().item()
                row += f"${mean:.2e} \\pm {std_dev:.2e}$ & "
            else:
                row += " N/A & "
        print(row[:-2] + "\\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Experimental results}")
    print("\\label{tab:results}")
    print("\\end{table}")


def plot_weights(
    df: pd.DataFrame,
    figsize: Tuple[int],
    method_names: List[str],
    legend: bool = False,
    legend_kwargs: Optional[Dict[str, Any]] = None,
    fname: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=figsize)
    sns.stripplot(
        data=df,
        x="SLP Index",
        y="SLP Weight",
        hue="Method",
        hue_order=method_names,
        palette=METHOD_COLORS,
        alpha=0.8,
        dodge=True,
        ax=ax,
        size=10,
    )
    legend_elements = [LEGEND_ELEMENTS[k] for k in method_names]
    if legend and not legend_kwargs:
        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
    elif legend:
        ax.legend(handles=legend_elements, **legend_kwargs)
    else:
        ax.get_legend().remove()
    fig.tight_layout()
    if fname:
        fig.savefig(fname)


def plot_beta_weight_sensitivity(
    df: pd.DataFrame, figsize: Tuple[int], fname: Optional[str] = None
):
    rename_dict = {
        f"Stacking Beta={i}": f"beta = {i}" for i in [0.001, 0.01, 0.1, 1.0, 10]
    } | {"Stacked": "beta = inf"}
    fig, ax = plt.subplots(figsize=figsize)
    plot_df = df.replace(rename_dict)
    plot_df["beta"] = plot_df["Method"].str.extract("beta = (inf|\d+\.?\d*)")[0]
    # Converting beta values to appropriate float representation
    plot_df["beta"] = plot_df["beta"].astype(float)
    sns.stripplot(
        data=plot_df,
        x="SLP Index",
        y="SLP Weight",
        hue="beta",
        hue_order=[0.001, 0.01, 0.1, 1.0, 10, float("inf")],
        palette="rocket",
        alpha=0.8,
        dodge=True,
        ax=ax,
        size=10,
    )
    ax.set_ylim((0.0, 1.0))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), title="$\\beta$")
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)


def convert_tuples_to_tensor(trace, name) -> torch.Tensor:
    """Utility function to convert a dictionary of tuples (generated by Gen) to a tensor."""
    alphas = [(k, v) for k, v in trace.items() if isinstance(k, tuple) and k[0] == name]
    alpha_tensor = torch.zeros(len(alphas))
    for k, v in alphas:
        alpha_tensor[k[1] - 1] = v
    return alpha_tensor