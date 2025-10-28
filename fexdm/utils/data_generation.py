"""
Utility functions for data generation and analysis.
"""

import numpy as np
from typing import Tuple, Callable, Optional


def generate_ornstein_uhlenbeck(
    theta: float = 1.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    x0: float = 0.0,
    t_span: Tuple[float, float] = (0, 10),
    dt: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time series from Ornstein-Uhlenbeck process.
    
    The process follows: dX = theta*(mu - X)*dt + sigma*dW
    
    Parameters
    ----------
    theta : float
        Mean reversion rate
    mu : float
        Long-term mean
    sigma : float
        Volatility
    x0 : float
        Initial value
    t_span : tuple
        Time span (t_start, t_end)
    dt : float
        Time step
    seed : int, optional
        Random seed
    
    Returns
    -------
    t : np.ndarray
        Time points
    X : np.ndarray
        Process values
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)
    
    X = np.zeros(n_steps)
    X[0] = x0
    
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n_steps):
        dW = np.random.randn()
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * sqrt_dt * dW
    
    return t, X


def generate_double_well(
    alpha: float = 1.0,
    beta: float = 1.0,
    sigma: float = 0.5,
    x0: float = 1.0,
    t_span: Tuple[float, float] = (0, 10),
    dt: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate time series from double-well potential system.
    
    The process follows: dX = (alpha*X - beta*X^3)*dt + sigma*dW
    
    Parameters
    ----------
    alpha : float
        Linear coefficient
    beta : float
        Cubic coefficient
    sigma : float
        Noise strength
    x0 : float
        Initial value
    t_span : tuple
        Time span (t_start, t_end)
    dt : float
        Time step
    seed : int, optional
        Random seed
    
    Returns
    -------
    t : np.ndarray
        Time points
    X : np.ndarray
        Process values
    """
    if seed is not None:
        np.random.seed(seed)
    
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt) + 1
    t = np.linspace(t_start, t_end, n_steps)
    
    X = np.zeros(n_steps)
    X[0] = x0
    
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n_steps):
        dW = np.random.randn()
        drift = alpha * X[i-1] - beta * X[i-1]**3
        X[i] = X[i-1] + drift * dt + sigma * sqrt_dt * dW
    
    return t, X


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    return np.sqrt(calculate_mse(y_true, y_pred))


def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
