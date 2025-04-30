from abc import ABC, abstractmethod
from cmdstanpy import CmdStanModel, CmdStanMCMC
import numpy as np

from .results import ExperimentResult


class StanFitter(ABC):
    def __init__(self, model: CmdStanModel):
        self.model = model

    def fit(self, expt: ExperimentResult, **kwargs) -> CmdStanMCMC:
        return self.model.sample(data=self.to_dict(expt), **kwargs)

    @abstractmethod
    def to_dict(self, expt: ExperimentResult) -> dict: ...


class StanFitterBasic(StanFitter):
    def to_dict(self, expt: ExperimentResult) -> dict:
        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
        }


class StanFitterBasicWithHyperparams(StanFitter):
    def to_dict(self, expt: ExperimentResult) -> dict:
        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
            "mu_b": expt.params["mu_b"],
            "mu_theta": expt.params["mu_theta"],
            "sigma_b": expt.params["sigma_b"],
            "sigma_theta": expt.params["sigma_theta"],
        }


class StanFitterWithEstimatedHyperparams(StanFitter):
    def __init__(self, model_hyper: CmdStanModel, basic_mlm_fitter: StanFitterBasic):
        super().__init__(model_hyper)
        self.basic_mlm_fitter = basic_mlm_fitter

    def to_dict(self, expt: ExperimentResult) -> dict:
        fit = self.basic_mlm_fitter.fit(expt, show_progress=False)
        mu_b_hat = np.mean(fit.mu_b)
        mu_theta_hat = np.mean(fit.mu_theta)
        sigma_b_hat = np.std(fit.sigma_b)
        sigma_theta_hat = np.std(fit.sigma_theta)

        return {
            "J": len(expt.data),
            "j": list(range(1, len(expt.data) + 1)),
            "y1_bar": list(expt.data["y1_bar"]),
            "y0_bar": list(expt.data["y0_bar"]),
            "sigma_y1_bar": list(expt.data["sigma_y1_bar"]),
            "sigma_y0_bar": list(expt.data["sigma_y0_bar"]),
            "n": list(expt.data["n"]),
            "p": list(expt.data["p"]),
            "mu_b": mu_b_hat,
            "mu_theta": mu_theta_hat,
            "sigma_b": sigma_b_hat,
            "sigma_theta": sigma_theta_hat,
        }
