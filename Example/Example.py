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
    elif case_name == 'Trigonometric1d':
        # Trigonometric 1d case
        params['MC'] = 10000
        params['th'] = 1.0 
        params['sig'] = 0.5
        params['IC'] = 'uniform'  # initial condition type: 'uniform' 
        params['dim'] = 1  # dimension (1D case)
        params['namefig'] = 'Trigonometric1d'
    elif case_name == 'DoubleWell1d':
        # Double Well 1d case
        params['MC'] = 10000
        params['th'] = 1.0
        params['sig'] = 0.5
        params['IC'] = 'uniform'
        params['dim'] = 1
        params['namefig'] = 'DoubleWell1d'
    elif case_name == 'OL2d':
        # OL2d case: 2D potential-based SDE
        # dX = -dVdx/gamma * dt + Sigma * dB
        # V(x,y) = 0.5*(x^2-1)^2 + 5*y^2
        params['MC'] = 10000
        params['th'] = 1.0  # Not used, kept for compatibility
        params['sig'] = np.sqrt(2.0)  # Sigma scaling factor
        params['IC'] = 'uniform'
        params['dim'] = 2
        params['namefig'] = 'OL2d'
    elif case_name == 'EXP1d':
        # Exponential 1d case: dX_t = th * X_t * dt + sig * dB_t
        params['MC'] = 10000
        params['th'] = -2.0  # drift coefficient
        params['sig'] = 0.1  # diffusion coefficient
        params['IC'] = 'uniform'
        params['dim'] = 1
        params['namefig'] = 'EXP1d'
    elif case_name == 'SIR':
        # SIR case
        params['MC'] = 10000
        params['th'] = 1.0
        params['sig'] = 0.5
        params['IC'] = 'uniform'
        params['dim'] = 1
        params['namefig'] = 'SIR'
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


def std_exp(N_data, t_steps, seeds):
    """
    Generate exponential increments for EXP1d model.
    
    Args:
        N_data: number of trajectories
        t_steps: time steps array
        seeds: random seed
    
    Returns:
        grow: array of shape (N_data, t_steps.shape[0]) with exponential increments
    """
    np.random.seed(seeds)
    diff = t_steps[1:] - t_steps[:-1]
    grow = np.zeros([N_data, t_steps.shape[0]])
    for i in range(t_steps.shape[0] - 1):
        grow[:, i+1] = np.random.exponential(1.0, N_data)
    return grow
    for i in range(t_steps.shape[0]-1):
        grow[:, i+1] = noise[i]
    return grow


def std_normal2(N_data, t_steps, seeds, Sig):
    """
    Generate correlated normal increments for OL2d model.
    
    Args:
        N_data: number of trajectories
        t_steps: time steps array (1D or 2D)
        seeds: random seed
        Sig: covariance matrix (n_dim x n_dim)
    
    Returns:
        grow: array of shape (N_data, t_steps.shape[-1], n_dim) with correlated noise
    """
    np.random.seed(seeds)
    # Handle both 1D and 2D t_steps
    if t_steps.ndim == 1:
        t_steps = t_steps.reshape(1, -1)
    diff = t_steps[0, 1:] - t_steps[0, :-1]
    
    n_dim = np.shape(Sig)[0]
    grow_temp = np.zeros([N_data, t_steps.shape[1], n_dim])
    noise = np.random.normal(0.0, np.sqrt(diff[0]), [N_data, t_steps.shape[1]-1, n_dim])
    for i in range(t_steps.shape[1] - 1):
        grow_temp[:, i+1, :] = noise[:, i, :]
    # Apply covariance matrix: grow = Sig @ grow_temp.transpose(0,2,1)
    # grow_temp shape: (N_data, t_steps, n_dim)
    # For each sample and time step, apply Sig: (N_data, t_steps, n_dim) @ (n_dim, n_dim) = (N_data, t_steps, n_dim)
    # This applies Sig to each (n_dim,) vector in grow_temp
    grow = grow_temp @ Sig.T  # (N_data, t_steps, n_dim) @ (n_dim, n_dim) = (N_data, t_steps, n_dim)
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
    T = params['T']  # time horizon
    Nt = params['Nt']  # number of discretized time steps
    N_data = params['MC']  # number of data trajectories
    IC_ = params['IC']  # initial condition type
    model_name = params.get('namefig', 'OU1d')
    
    # Time steps
    t = np.linspace(0, T, Nt + 1)
    dt = t[1] - t[0]  # time step size
    
    # Get initial condition parameters
    start_point, end_points, initial_value = initial_condition_generation(params, domain_start=domain_start, domain_end=domain_end)
    
    # Initial condition
    if IC_ == 'uniform':
        np.random.seed(2)  # Match Gendata behavior
        if model_name == 'OL2d':
            # 2D case: x in [-1.5, 1.5], y in [-1, 1]
            xIC = np.zeros((N_data, 2))
            xIC[:, 0] = np.random.uniform(-1.5, 1.5, N_data)
            xIC[:, 1] = np.random.uniform(-1, 1, N_data)
        else:
            # 1D case
            xIC = np.random.uniform(start_point, end_points, N_data)
    elif IC_ == 'value':
        if model_name == 'OL2d':
            # 2D case: use [0.6, 0.6] as default
            xIC = np.array([[0.6, 0.6]] * N_data)
        else:
            # 1D case
            xIC = initial_value * np.ones(N_data)
    else:
        raise ValueError(f"Initial condition type '{IC_}' not supported. Use 'uniform' or 'value'.")
    
    # Generate data array: shape depends on dimension
    if model_name == 'OL2d':
        data = np.zeros((2, Nt + 1, N_data))  # 2D case: (2, Nt+1, N_data)
    else:
        data = np.zeros((1, Nt + 1, N_data))  # 1D case: (1, Nt+1, N_data)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    if model_name == 'OU1d':
        # Ornstein-Uhlenbeck process: dX_t = th*(mu-X_t)dt + sig*dB_t
        th = params['th']  # theta (mean reversion rate)
        mu = params['mu']  # long-term mean
        sig = params['sig'] * noise_level  # sigma (volatility) scaled by noise_level
        
        # Generate Brownian motion increments
        brownian = std_normal(N_data, t, seed)
        
        # OU process solution: X_t = xIC*exp(-th*t) + mu*(1-exp(-th*t)) + sig*exp(-th*t)*int_0^t exp(th*s)dB_s
        Ext = np.exp(-th * t)
        data[0, :, :] = (xIC[:, None] * Ext + mu * (1 - Ext) + 
                         sig * Ext * np.cumsum(np.exp(th * t) * brownian, axis=-1)).T
    
    elif model_name == 'Trigonometric1d':
        # Trigonometric SDE: dX_t = sin(2*k*pi*X_t)dt + sig*cos(2*k*pi*t)*dW_t
        k = 1  # frequency parameter
        sig = params['sig'] * noise_level  # sigma (volatility) scaled by noise_level
        
        # Initialize data with initial conditions
        data[0, 0, :] = xIC
        
        # Euler-Maruyama method for trigonometric SDE
        for i in range(Nt):
            Xt = data[0, i, :]  # Current state
            current_time = t[i]  # Current time
            # Drift: sin(2*k*pi*X_t)
            drift = np.sin(2 * k * np.pi * Xt)
            # Diffusion: sig*cos(2*k*pi*t) - function of time, not X_t
            diffusion = sig * np.cos(2 * k * np.pi * current_time)
            # Brownian increment
            dW = np.sqrt(dt) * np.random.randn(N_data)
            # Euler-Maruyama step: X_{t+1} = X_t + drift*dt + diffusion*dW
            data[0, i+1, :] = Xt + drift * dt + diffusion * dW
    
    elif model_name == 'DoubleWell1d':
        # Double Well Potential process: dX_t = (X_t - X_t^3)dt + sig*dB_t
        sig = params['sig'] * noise_level  # sigma (volatility) scaled by noise_level  
        # Define drift and diffusion functions
        def drift(x):
            return x - x**3
        def diffusion(x):
            return sig  # Constant diffusion
        # Initialize data with initial conditions
        data[0, 0, :] = xIC
        # Generate Brownian motion increments
        brownian = std_normal(N_data, t, seed)      
        # Euler-Maruyama method for Double Well SDE
        for i in range(Nt):
            Xt = data[0, i, :]  # Current state
            # Drift: X_t - X_t^3
            drift_val = drift(Xt)
            # Diffusion: sig (constant)
            diff_val = diffusion(Xt)
            # Brownian increment
            dW = brownian[:, i+1]  # Use pre-generated Brownian increments
            # Euler-Maruyama step: X_{t+1} = X_t + drift*dt + diffusion*dW
            data[0, i+1, :] = Xt + drift_val * dt + diff_val * dW
    
    elif model_name == 'EXP1d':
        # Exponential 1d process: dX_t = th * X_t * dt + sig * Exp(1) * sqrt(dt)
        th = params['th']  # drift coefficient
        sig = params['sig'] * noise_level  # diffusion coefficient scaled by noise_level
        # Initialize data with initial conditions
        data[0, 0, :] = xIC
        # Generate exponential increments
        exp_increments = std_exp(N_data, t, seed)
        # Euler-Maruyama method for EXP1d SDE with exponential noise
        for i in range(Nt):
            Xt = data[0, i, :]  # Current state
            # Drift: th * X_t
            drift_val = th * Xt
            # Diffusion: sig * Exp(1) * sqrt(dt)
            # Use exponential increment: Exp(1) * sqrt(dt)
            dExp = exp_increments[:, i+1] * np.sqrt(dt)  # Exponential increment scaled by sqrt(dt)
            # Euler-Maruyama step: X_{t+1} = X_t + drift*dt + sig*dExp
            data[0, i+1, :] = Xt + drift_val * dt + sig * dExp
    
    elif model_name == 'OL2d':
        # OL2d: 2D potential-based SDE
        # dX = -dVdx/gamma * dt + Sigma * dB
        # V(x,y) = 0.5*(x^2-1)^2 + 5*y^2
        # dVdx = [2*x*(x^2-1), 10*y]
        sig_base = params['sig'] * noise_level
        Sigma = sig_base * np.array([[1.0, 0.0], [0.0, 1.0]])  # Covariance matrix
        gamma = np.ones(2)  # Friction coefficients
        
        def V(x):
            """Potential function: V(x,y) = 0.5*(x^2-1)^2 + 5*y^2"""
            return 0.5 * (x[0, :]**2 - 1)**2 + 5 * x[1, :]**2
        
        def dVdx(x):
            """Gradient of potential: dVdx = [2*x*(x^2-1), 10*y]"""
            return np.array([2 * x[0, :] * (x[0, :]**2 - 1), 10 * x[1, :]])
        
        # Initialize data with initial conditions
        data[:, 0, :] = xIC.T  # xIC is (N_data, 2), data is (2, Nt+1, N_data)
        
        # Generate correlated Brownian motion
        t_reshaped = t.reshape(1, -1)  # Reshape for std_normal2
        brownian = std_normal2(N_data, t_reshaped, seed, Sigma)
        # brownian shape: (N_data, Nt+1, 2), transpose to (2, Nt+1, N_data) to match data shape
        brownian = brownian.transpose(2, 1, 0)  # (2, Nt+1, N_data)
        
        # Euler-Maruyama method for OL2d SDE
        for i in range(Nt):
            x = data[:, i, :]  # Current state: (2, N_data)
            # Drift: -dVdx(x) / gamma
            drift = -dVdx(x) / gamma[:, np.newaxis]  # (2, N_data)
            # Diffusion: brownian increment
            diffusion = brownian[:, i+1, :]  # (2, N_data)
            # Euler-Maruyama step: X_{t+1} = X_t + drift*dt + diffusion
            data[:, i+1, :] = x + drift * dt + diffusion
    
    else:
        raise ValueError(f"Model type '{model_name}' not supported in data_generation.")
    
    # Handle steady state
    if steady:
        Nt = 1
        data = data[:, [0, -1], :]
        if model_name == 'OL2d':
            data[0, 1, :] -= np.mean(data[0, 1, :])
            data[1, 1, :] -= np.mean(data[1, 1, :])
        else:
            data[0][1] -= np.mean(data[0][1])
    
    # Handle mean trajectory
    if mean:
        if model_name == 'OL2d':
            data = np.mean(data, axis=2)  # (2, Nt+1)
        else:
            data = (np.mean(data, axis=2).reshape([1, Nt + 1]))
    
    # Handle single trajectory
    if N_data == 1:
        if model_name == 'OL2d':
            data = data.reshape([2, Nt + 1])
        else:
            data = data.reshape([1, Nt + 1])
    
    # Extract start and end points
    if model_name == 'OL2d':
        # For 2D case, flatten both dimensions
        # data shape: (2, Nt+1, N_data)
        x_start_new = data[:, :Nt, :].transpose(1, 0, 2).reshape(-1, 2)  # (Nt*N_data, 2)
        x_end_new = data[:, 1:Nt+1, :].transpose(1, 0, 2).reshape(-1, 2)  # (Nt*N_data, 2)
        
        # Sort by first dimension (x coordinate)
        sorted_indices = np.argsort(x_start_new[:, 0], axis=0)
        x_start = x_start_new[sorted_indices]
        x_end = x_end_new[sorted_indices]
    else:
        # For 1D case
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
    elif params['namefig'] == 'Trigonometric1d':
        # Use provided domain_start and domain_end, or default values
        start_point = domain_start if domain_start is not None else params.get('domain_start', 0.0)
        end_points = domain_end if domain_end is not None else params.get('domain_end', 1.0)
        initial_value = params.get('initial_value', 0.6)
    elif params['namefig'] == 'DoubleWell1d':
        # Use provided domain_start and domain_end, or default values
        start_point = domain_start if domain_start is not None else params.get('domain_start', -2.0)
        end_points = domain_end if domain_end is not None else params.get('domain_end', 2.0)
        initial_value = params.get('initial_value', 0.6)
    elif params['namefig'] == 'EXP1d':
        # Use provided domain_start and domain_end, or default values
        start_point = domain_start if domain_start is not None else params.get('domain_start', 0.0)
        end_points = domain_end if domain_end is not None else params.get('domain_end', 2.5)
        initial_value = params.get('initial_value', 1.5)
    elif params['namefig'] == 'OL2d':
        # OL2d is 2D, so domain_start/end are not used in the same way
        # For OL2d, x in [-1.5, 1.5], y in [-1, 1]
        # We return None for start_point and end_points since OL2d handles initial conditions differently
        start_point = None  # Not used for 2D models
        end_points = None  # Not used for 2D models
        initial_value = None  # Not used for 2D models (uses [0.6, 0.6] as default)
    else:
        raise ValueError(f"Model type '{params.get('namefig', 'unknown')}' not supported in initial_condition_generation.")
    
    return start_point, end_points, initial_value






