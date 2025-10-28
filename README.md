# FEX-DM: Finite Expression Methods for Identifying Stochastic Dynamics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

FEX-DM is a Python package for identifying stochastic dynamical systems from time series data using finite expression methods. This implementation is based on the paper "Identifying Stochastic Dynamics via Finite Expression Methods".

## Overview

FEX-DM discovers the underlying drift and diffusion functions of stochastic differential equations (SDEs) from observed time series data. The method uses symbolic regression with a finite library of basis functions to find interpretable mathematical expressions for the dynamics.

## Features

- **Automatic Discovery**: Identifies drift and diffusion functions from data
- **Symbolic Expressions**: Returns interpretable mathematical formulas
- **Flexible Basis Functions**: Customizable library of candidate functions
- **SDE Simulation**: Simulate identified dynamics for validation
- **Easy to Use**: Simple API for quick experimentation

## Installation

### From source

```bash
git clone https://github.com/xxjan719/FEX-DM.git
cd FEX-DM
pip install -e .
```

### Requirements

- Python 3.7+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SciPy >= 1.7.0

## Quick Start

Here's a simple example identifying an Ornstein-Uhlenbeck process:

```python
import numpy as np
from fexdm import StochasticDynamicsIdentifier
from fexdm.utils import generate_ornstein_uhlenbeck

# Generate synthetic data
t, X = generate_ornstein_uhlenbeck(
    theta=1.0, mu=0.0, sigma=0.5,
    x0=2.0, t_span=(0, 20), dt=0.01, seed=42
)

# Identify the dynamics
identifier = StochasticDynamicsIdentifier(max_complexity=3, dt=0.01)
identifier.fit(X, t, verbose=True)

# Get discovered expressions
print(f"Drift: {identifier.get_drift_expression()}")
print(f"Diffusion: {identifier.get_diffusion_expression()}")
```

## Examples

The `examples/` directory contains several demonstration scripts:

### Ornstein-Uhlenbeck Process
```bash
python examples/example_ornstein_uhlenbeck.py
```

Identifies a mean-reverting process with linear drift.

### Double-Well Potential
```bash
python examples/example_double_well.py
```

Identifies a nonlinear system with bistable dynamics.

## Usage

### Basic Workflow

1. **Prepare your data**: Time series of a stochastic process
2. **Create identifier**: Configure complexity and basis functions
3. **Fit to data**: Discover drift and diffusion functions
4. **Validate**: Simulate from identified model and compare

### API Reference

#### `StochasticDynamicsIdentifier`

Main class for identifying stochastic dynamics.

**Parameters:**
- `max_complexity` (int): Maximum number of terms in expressions
- `dt` (float): Time step between observations
- `basis_functions` (list): Custom basis functions (optional)

**Methods:**
- `fit(X, t, verbose)`: Identify dynamics from time series
- `get_drift_expression()`: Get drift function formula
- `get_diffusion_expression()`: Get diffusion function formula
- `get_sde()`: Get identified SDE model
- `predict_drift(X)`: Evaluate drift at points
- `predict_diffusion(X)`: Evaluate diffusion at points

#### `StochasticDifferentialEquation`

Class for representing and simulating SDEs.

**Methods:**
- `simulate(x0, t_span, dt, n_trajectories, seed)`: Generate trajectories

### Utility Functions

```python
from fexdm.utils import (
    generate_ornstein_uhlenbeck,
    generate_double_well,
    calculate_mse,
    calculate_rmse,
    calculate_r2_score
)
```

## Mathematical Background

FEX-DM identifies stochastic differential equations of the form:

```
dX = f(X)dt + g(X)dW
```

where:
- `f(X)` is the drift function
- `g(X)` is the diffusion function
- `dW` is a Wiener process increment

The method:
1. Estimates local drift and diffusion from data
2. Uses symbolic regression to find expressions
3. Selects best functions based on error and complexity

## Testing

Run tests with pytest:

```bash
pip install pytest pytest-cov
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use FEX-DM in your research, please cite:

```bibtex
@article{fexdm2025,
  title={Identifying Stochastic Dynamics via Finite Expression Methods},
  author={FEX-DM Contributors},
  year={2025}
}
```

## Acknowledgments

This implementation is based on research in symbolic regression and stochastic dynamics identification.

## Contact

For questions or issues, please open an issue on GitHub.