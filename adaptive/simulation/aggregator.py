import pandas as pd
from collections.abc import Iterable
from dataclasses import dataclass
from tqdm.auto import tqdm

from .simulators import ExperimentSimulator
from .estimators import Estimator


@dataclass
class SimulationAggregatorResult:
    """
    Encapsulates the logic for aggregating simulation results.
    """

    df: pd.DataFrame
    identifiers: list[str]

    def means(self) -> pd.DataFrame:
        """
        Returns the mean of the simulation results.
        """
        return self.df.groupby(self.identifiers).mean().reset_index()


class SimulationAggregator:
    """
    Encapsulates the logic for aggregating simulation results.
    Assumes all simulators have the same identifier keys.
    """

    def __init__(self, simulators: Iterable[ExperimentSimulator]):
        self.simulators = list(simulators)
        if self.simulators:
            # collect the key‐set of the first simulator
            common_keys = set(self.simulators[0].identifier.keys())
            # ensure all simulators share the same identifier keys
            for sim in self.simulators[1:]:
                if set(sim.identifier.keys()) != common_keys:
                    raise ValueError(
                        f"All simulators must have the same identifier keys. "
                        f"Expected {common_keys}, got {set(sim.identifier.keys())} from {sim}"
                    )

        self.identifiers = list(
            self.simulators[0].identifier.keys() if self.simulators else []
        )

    def simulate(
        self, estimators: Iterable[Estimator], num_reps: int, alpha: float, **kwargs
    ) -> SimulationAggregatorResult:
        df = pd.DataFrame()
        for simulator in tqdm(self.simulators, desc="Simulators", leave=False):
            for estimator in tqdm(estimators, desc="Estimators", leave=False):
                new_df = None
                for _ in tqdm(range(num_reps), desc="Repetition", leave=False):
                    expt = simulator.simulate()
                    summary_df = estimator.result(expt, alpha, **kwargs).summary()

                    if new_df is None:
                        new_df = summary_df
                    else:
                        new_df = pd.concat([new_df, summary_df], ignore_index=True)

                # Add the simulator and estimators to the DataFrame
                for id in simulator.identifier:
                    new_df[id] = simulator.identifier[id]
                new_df["estimator"] = estimator.name

                # Add the new DataFrame to the main DataFrame
                df = pd.concat([df, new_df], ignore_index=True)

        return SimulationAggregatorResult(df, list(self.identifiers) + ["estimator"])
