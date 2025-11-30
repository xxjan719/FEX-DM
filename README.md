# FEX-DM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.07085-b31b1b.svg)](https://arxiv.org/abs/2504.07085)

This repository contains the code implementation for the paper **"Identifying stochastic dynamics via Finite Expression Methods"**.

FEX-DM is a machine learning two-stage framework that models stochastic differential equations (OU process, Double Well Potential, etc).

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

FEX-DM is a two-stage framework for learning stochastic differential equations (SDEs):

1. **Stage 1 (Deterministic)**: Learn the drift term using Finite Expression (FEX) methods
2. **Stage 2 (Stochastic)**: Learn the diffusion term using neural networks

The framework can handle various SDE models including Ornstein-Uhlenbeck (OU) processes, Double Well Potential, and more.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- SymPy
- FAISS (for nearest neighbor search)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FEX-DM
```

2. Set up environment:
### Stage 0: Set up env
Run `config.py` in the `src` folder. The required packages will be automatically installed.
```bash
python src/config.py
```

## Usage



### Stage 1: Deterministic Training

Train the FEX model to learn the drift term:

```bash
python src/1stage_deterministic.py
```

### Stage 2: Stochastic Training

Train the neural network to learn the diffusion term:

```bash
python src/2stage_stochastic_time_independent.py
```


## Citation

If you use FEX-DM, please cite the following paper:

```bibtex
@article{fexdm2024,
  title={Identifying Unknown Stochastic Dynamics via Finite Expression Methods},
  author={...},
  journal={arXiv preprint arXiv:2504.07085},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, or issues:
- **Email**: xingjianxu@ufl.edu, senwei.liang@ttu.edu
- **Issues**: Please use the GitHub Issues page for bug reports 