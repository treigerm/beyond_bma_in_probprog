from typing import List
from dataclasses import dataclass

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from gen_loader import GenLoader
from post_stacking import PostStacking
from utils import MethodResults, load_data, print_lppd_df


@dataclass
class Evaluator(GenLoader, PostStacking):
    name: str
    pyro_multirun_dir: str
    rjmcmc_dir: str
    betas: List[float]
    num_replications: int

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
            # TODO: Allow for multiple betas.
            method_results.append(
                self.stacking(
                    self.pyro_multirun_dir,
                    rjmcmc_slp_info,
                    beta=float("inf"),
                    prefix="RJMCMC ",
                )
            )

        # Load results already computed in main run.
        main_results, slp_infos = load_data(self.pyro_multirun_dir, name=self.name)
        method_results += main_results

        # Compute the additional stacking results.
        if post_stacking:
            for beta in self.betas:
                method_results.append(
                    self.stacking(self.pyro_multirun_dir, slp_infos, beta=beta)
                )

        # Convert the results to a dataframe.
        df_weights = pd.concat([x.weights_to_df() for x in method_results])
        df_lppds = pd.concat([x.lppd_to_df() for x in method_results])

        self.plot_weights(df_weights)

        # Process the lppds.
        print_lppd_df(df_lppds)
        df_lppds["name"] = self.name
        df_lppds.to_csv(f"stacking_lppds/{self.name}_lppds.csv")
        return method_results

    def plot_weights(self, df_weights: pd.DataFrame):
        raise NotImplementedError()