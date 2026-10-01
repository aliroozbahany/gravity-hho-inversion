"""
Evaluation and error metrics for geophysical inversion.
"""

from typing import Dict, Union
import numpy as np


def compute_rmse(d_obs: np.ndarray, d_pred: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((d_obs - d_pred) ** 2)))


def compute_mae(d_obs: np.ndarray, d_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(d_obs - d_pred)))


def compute_relative_error(true_val: float, est_val: float) -> float:
    """
    Relative Error percentage: |(true - est) / true| * 100
    If true_val is 0, returns absolute error.
    """
    if np.isclose(true_val, 0.0):
        return float(np.abs(true_val - est_val))
    return float(np.abs((true_val - est_val) / true_val) * 100.0)


def evaluate_model_errors(
    true_params: Dict[str, float], est_params: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute relative/absolute errors for all model parameters.
    """
    errors = {}
    for key in true_params:
        if key in est_params:
            errors[f"{key}_err"] = compute_relative_error(
                true_params[key], est_params[key]
            )
    return errors
