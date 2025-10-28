# FEX-DM Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Examples](#examples)
6. [Mathematical Background](#mathematical-background)

## Introduction

FEX-DM (Finite Expression Methods for Dynamics Modeling) is a Python package designed to identify stochastic dynamical systems from time series data. The package implements symbolic regression techniques to discover interpretable mathematical expressions for both drift and diffusion functions of stochastic differential equations.

## Installation

### Requirements

- Python 3.7 or higher
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.7.0

### Install from source

```bash
git clone https://github.com/xxjan719/FEX-DM.git
cd FEX-DM
pip install -e .
```

## Quick Start

### Basic Example

```python
from fexdm import StochasticDynamicsIdentifier
from fexdm.utils import generate_ornstein_uhlenbeck

# Generate data from a known process
t, X = generate_ornstein_uhlenbeck(
    theta=1.0,  # mean reversion rate
    mu=0.0,     # long-term mean
    sigma=0.5,  # volatility
    x0=2.0,     # initial value
    t_span=(0, 20),
    dt=0.01,
    seed=42
)

# Identify the dynamics
identifier = StochasticDynamicsIdentifier(
    max_complexity=3,
    dt=0.01
)
identifier.fit(X, t, verbose=True)

# Get results
print("Drift function:", identifier.get_drift_expression())
print("Diffusion function:", identifier.get_diffusion_expression())
```

## API Reference

### Core Classes

#### `StochasticDynamicsIdentifier`

Main class for identifying stochastic dynamics from time series data.

**Constructor Parameters:**
- `max_complexity` (int, default=3): Maximum number of terms in symbolic expressions
- `dt` (float, optional): Time step between observations (auto-detected if None)
- `basis_functions` (list, optional): Custom basis functions for symbolic regression

**Methods:**

##### `fit(X, t=None, verbose=False)`
Identify drift and diffusion functions from time series data.
- `X`: Time series data (1D or 2D array)
- `t`: Time points (optional, assumes uniform spacing if None)
- `verbose`: Print progress information

Returns: self

##### `get_drift_expression()`
Returns the discovered drift function as a string formula.

##### `get_diffusion_expression()`
Returns the discovered diffusion function as a string formula.

##### `get_sde()`
Returns the identified StochasticDifferentialEquation model.

##### `predict_drift(X)`
Evaluate the drift function at given points.

##### `predict_diffusion(X)`
Evaluate the diffusion function at given points.

#### `StochasticDifferentialEquation`

Class for representing and simulating stochastic differential equations.

**Constructor Parameters:**
- `drift`: Callable drift function f(X)
- `diffusion`: Callable diffusion function g(X)
- `dimension` (int, default=1): State space dimension

**Methods:**

##### `simulate(x0, t_span, dt=0.01, n_trajectories=1, seed=None)`
Simulate trajectories using Euler-Maruyama method.
- `x0`: Initial condition
- `t_span`: Tuple (t_start, t_end)
- `dt`: Time step
- `n_trajectories`: Number of trajectories to simulate
- `seed`: Random seed for reproducibility

Returns: (t, X) where t is time array and X is trajectory array

#### `SymbolicRegressor`

Performs symbolic regression to discover functional forms.

**Constructor Parameters:**
- `basis_functions` (list, optional): Basis functions to use
- `max_complexity` (int, default=3): Maximum expression complexity
- `operators` (list, optional): Operators for expressions

**Methods:**

##### `fit(X, y, verbose=False)`
Fit regressor to data.

##### `predict(X)`
Make predictions using discovered expression.

##### `get_expression()`
Get the best discovered Expression object.

### Utility Functions

#### Data Generation

##### `generate_ornstein_uhlenbeck(theta, mu, sigma, x0, t_span, dt, seed=None)`
Generate Ornstein-Uhlenbeck process time series.

##### `generate_double_well(alpha, beta, sigma, x0, t_span, dt, seed=None)`
Generate double-well potential system time series.

#### Metrics

##### `calculate_mse(y_true, y_pred)`
Calculate mean squared error.

##### `calculate_rmse(y_true, y_pred)`
Calculate root mean squared error.

##### `calculate_r2_score(y_true, y_pred)`
Calculate R-squared score.

## Examples

### Example 1: Ornstein-Uhlenbeck Process

The Ornstein-Uhlenbeck process is a mean-reverting stochastic process:
```
dX = θ(μ - X)dt + σdW
```

Run the example:
```bash
python examples/example_ornstein_uhlenbeck.py
```

### Example 2: Double-Well Potential

A nonlinear system with bistable dynamics:
```
dX = (αX - βX³)dt + σdW
```

Run the example:
```bash
python examples/example_double_well.py
```

## Mathematical Background

### Stochastic Differential Equations

FEX-DM identifies SDEs of the form:
```
dX = f(X)dt + g(X)dW
```

where:
- `f(X)` is the **drift function** (deterministic dynamics)
- `g(X)` is the **diffusion function** (noise intensity)
- `dW` is a Wiener process (Brownian motion) increment

### Identification Method

The algorithm consists of three main steps:

1. **Local Estimation**: From discrete time series data, estimate local drift and diffusion at each point
   - Drift: `f(X) ≈ E[ΔX]/Δt`
   - Diffusion: `g(X) ≈ √(Var[ΔX]/Δt)`

2. **Symbolic Regression**: Search for best symbolic expressions
   - Test combinations of basis functions
   - Fit coefficients using least squares
   - Select based on error and complexity

3. **Model Construction**: Build SDE with discovered functions

### Basis Functions

Default basis functions include:
- Polynomial: `1, x, x², x³`
- Trigonometric: `sin(x), cos(x)`
- Gaussian: `exp(-x²)`

Custom basis functions can be specified as strings that evaluate to numpy operations.

## Advanced Usage

### Custom Basis Functions

```python
identifier = StochasticDynamicsIdentifier(
    basis_functions=['1', 'x', 'x**2', 'x**3', 'sin(x)', 'exp(-x)'],
    max_complexity=4
)
```

### Multiple Trajectories

```python
# Stack multiple trajectories
X = np.column_stack([X1, X2, X3])
identifier.fit(X, t)
```

### Validation

```python
# Simulate from identified model
sde = identifier.get_sde()
t_sim, X_sim = sde.simulate(
    x0=initial_value,
    t_span=(0, 10),
    dt=0.01,
    n_trajectories=100,
    seed=42
)

# Compare statistics
print("Original mean:", np.mean(X))
print("Simulated mean:", np.mean(X_sim))
```

## Tips and Best Practices

1. **Data Quality**: Use sufficient data (>1000 points recommended)
2. **Time Step**: Smaller dt improves estimation but requires more data
3. **Complexity**: Start with low max_complexity and increase if needed
4. **Validation**: Always validate by simulating from identified model
5. **Multiple Runs**: Try different random seeds to check stability

## Troubleshooting

### Poor Identification Results

- Increase data length or reduce time step
- Adjust basis functions to match expected dynamics
- Increase max_complexity if dynamics are complex
- Check for non-stationary behavior in data

### Numerical Issues

- Normalize data if values are very large/small
- Reduce time step in simulations
- Check for division by zero in custom basis functions

## References

For more information on the theory and applications of finite expression methods for stochastic dynamics identification, please refer to the paper:

"Identifying Stochastic Dynamics via Finite Expression Methods"
