"""
Bayesian uncertainty quantification via Monte Carlo Dropout.

Why uncertainty quantification matters operationally:
    "This battery has 47 cycles remaining" is less useful than
    "47 cycles remaining, 90% CI: [41, 53]"

A maintenance engineer scheduling a replacement needs to know
whether the model is confident or highly uncertain before acting.

Approach — MC Dropout (Gal & Ghahramani, 2016):
    - Dropout is kept active at inference time
    - Run N forward passes through the same input
    - Distribution of outputs approximates the posterior predictive
    - Mean = point estimate, std = epistemic uncertainty

This wraps any trained model (TCN or XGBoost) with a consistent
uncertainty interface so downstream code is model-agnostic.

Reference: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
    arXiv:1506.02142
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Protocol


class RULPredictor(Protocol):
    """Interface that all RUL models must satisfy for the uncertainty wrapper."""
    def predict(self, X) -> np.ndarray: ...


class MCDropoutWrapper:
    """
    Monte Carlo Dropout uncertainty wrapper for PyTorch models.

    Usage:
        wrapper = MCDropoutWrapper(trained_tcn, n_samples=100)
        mean, lower, upper = wrapper.predict_with_uncertainty(X, confidence=0.90)
    """

    def __init__(self, model: nn.Module, n_samples: int = 100):
        """
        Args:
            model: Trained PyTorch model with Dropout layers.
            n_samples: Number of stochastic forward passes.
                       100 is a reasonable default. More = smoother intervals,
                       diminishing returns beyond ~200.
        """
        self.model = model
        self.n_samples = n_samples

    def _enable_dropout(self) -> None:
        """Set all Dropout layers to train mode (active) while keeping rest in eval."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        X: torch.Tensor,
        confidence: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run N stochastic forward passes and compute prediction intervals.

        Args:
            X: Input tensor, shape (batch, seq_len, n_features).
            confidence: Confidence level for prediction interval (e.g. 0.90 = 90% PI).

        Returns:
            (mean, lower_bound, upper_bound): All shape (batch,), in cycles.
        """
        # TODO: implement
        raise NotImplementedError

    def calibration_score(
        self,
        X: torch.Tensor,
        y_true: np.ndarray,
        confidence: float = 0.90,
    ) -> float:
        """
        Compute empirical coverage of prediction intervals.

        A well-calibrated model with confidence=0.90 should have ~90% of
        true values fall within the predicted interval.
        Significantly below 90% = overconfident.
        Significantly above 90% = underconfident (too wide intervals).

        Args:
            X: Input tensor.
            y_true: True RUL values.
            confidence: Target confidence level.

        Returns:
            Empirical coverage rate (float between 0 and 1).
        """
        # TODO: implement
        raise NotImplementedError
