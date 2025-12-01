import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import random
import torch
from pathlib import Path

# Add src directory to path to import utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, 'src'))
from utils.FEX import FEX
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

def params_init(case_name = None,
                sample:int = 10000)->dict:
    params = {
        'MC':sample, # number of Monte Carlo simulations for trajectories
        'Dt':1e-2,
        'T':1.0,
        'Nt': int(1 / 1e-02), # number of discretized time steps
        
    }
    
    if case_name == 'OU1d':  # Ou 1d case
        # Ornstein-Uhlenbeck process parameters: dX_t = th(mu-X_t)dt + sig dB_t
        params['MC'] = 15000
        params['th'] = 1.0  # theta (mean reversion rate)
        params['mu'] = 1.2  # long-term mean
        params['sig'] = 0.3  # sigma (volatility/diffusion coefficient)
        params['IC'] = 'uniform'  # initial condition type: 'uniform' 
        params['dim'] = 1  # dimension (1D case)
        params['namefig'] = 'OU1d'
    else:
        raise ValueError(f"Case name {case_name} is not supported.")
    
    return params


def std_normal(N_data, t_steps, seeds):
    """
    Generate standard normal increments for Brownian motion.
    
    Args:
        N_data: number of trajectories
        t_steps: time steps array
        seeds: random seed
    
    Returns:
        grow: array of shape (N_data, t_steps.shape[0]) with Brownian increments
    """
    np.random.seed(seeds)
    diff = t_steps[1:] - t_steps[:-1]
    grow = np.zeros([N_data, t_steps.shape[0]])
    noise = np.random.normal(0.0, np.sqrt(diff[0]), [t_steps.shape[0]-1, N_data])
    for i in range(t_steps.shape[0]-1):
        grow[:, i+1] = noise[i]
    return grow


def data_generation(params,
                    noise_level=1.0,
                    seed=SEED,
                    mean=False,
                    steady=False,
                    domain_start=None,
                    domain_end=None
                    ):
    
    # Extract parameters
    th = params['th']  # theta (mean reversion rate)
    mu = params['mu']  # long-term mean
    sig = params['sig'] * noise_level  # sigma (volatility) scaled by noise_level
    T = params['T']  # time horizon
    Nt = params['Nt']  # number of discretized time steps
    N_data = params['MC']  # number of data trajectories
    IC_ = params['IC']  # initial condition type
    
    # Time steps
    t = np.linspace(0, T, Nt + 1)
    
    # Get initial condition parameters
    start_point, end_points, initial_value = initial_condition_generation(params, domain_start=domain_start, domain_end=domain_end)
    
    # Initial condition
    if IC_ == 'uniform':
        np.random.seed(2)  # Match Gendata behavior
        xIC = np.random.uniform(start_point, end_points, N_data)
    elif IC_ == 'value':
        xIC = initial_value * np.ones(N_data)
    else:
        raise ValueError(f"Initial condition type '{IC_}' not supported. Use 'uniform' or 'value'.")
    
    # Generate data array: shape (1, Nt+1, N_data)
    data = np.zeros((1, Nt + 1, N_data))
    
    # Generate Brownian motion increments
    brownian = std_normal(N_data, t, seed)
    
    # OU process solution: X_t = xIC*exp(-th*t) + mu*(1-exp(-th*t)) + sig*exp(-th*t)*int_0^t exp(th*s)dB_s
    Ext = np.exp(-th * t)
    data[0, :, :] = (xIC[:, None] * Ext + mu * (1 - Ext) + 
                     sig * Ext * np.cumsum(np.exp(th * t) * brownian, axis=-1)).T
    
    # Handle steady state
    if steady:
        Nt = 1
        data = data[:, [0, -1], :]
        data[0][1] -= np.mean(data[0][1])
    
    # Handle mean trajectory
    if mean:
        data = (np.mean(data, axis=2).reshape([1, Nt + 1]))
    
    # Handle single trajectory
    if N_data == 1:
        data = data.reshape([1, Nt + 1])
    
    # Extract start and end points
    x_start_new = data[0, :Nt, :].reshape(-1, 1)
    x_end_new = data[0, 1:Nt+1, :].reshape(-1, 1)
    
    # Sort by starting position (matching Gendata behavior)
    sorted_indices = np.argsort(x_start_new, axis=0)
    x_start = x_start_new[sorted_indices].reshape(-1, 1)
    x_end = x_end_new[sorted_indices].reshape(-1, 1)
    
    return x_start, x_end, data
    

def initial_condition_generation(params, domain_start=None, domain_end=None):
    """
    Generate initial condition parameters.
    
    Args:
        params: Dictionary containing model parameters
        domain_start: Start point of domain (if None, uses default or from params)
        domain_end: End point of domain (if None, uses default or from params)
    
    Returns:
        start_point, end_points, initial_value
    """
    if params['namefig'] == 'OU1d':
        # Use provided domain_start and domain_end, or default values
        start_point = domain_start if domain_start is not None else params.get('domain_start', 0.0)
        end_points = domain_end if domain_end is not None else params.get('domain_end', 2.5)
        initial_value = params.get('initial_value', 1.5)
    else:
        raise ValueError(f"Model type '{params.get('namefig', 'unknown')}' not supported in initial_condition_generation.")
    
    return start_point, end_points, initial_value






