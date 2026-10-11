# Minimax Score Combination for Diffusion Sampling Toward Several Targets

This repository is the official implementation of **Minimax Score Combination for Diffusion Sampling Toward Several Targets**.

We study a DDPM-type sampler driven by a convex combination of the true scores of several target distributions, and propose choosing the weights by minimizing the worst-case excess path divergence to the targets (the *minimax weights*), which is a second-order cone program.

This repository contains the code for the experiments in the paper. They compare the minimax weights with uniformly random weights, equal weights, and other weight-selection rules on Gaussian targets, where the worst-case total variation of the sampler can be computed exactly.

## Requirements

Create a fresh environment and install the Python dependencies:

```bash
conda create -n multiobjective-diffusion python=3.11
conda activate multiobjective-diffusion
pip install -r requirements.txt
```

No GPU is needed.

## Reproducing the Results

Run the commands from the `experiments` directory:

```bash
python sanity_checks.py        # checks the closed forms, the quadrature and the solvers
python run_random_weights.py   # runs all configurations with 100 seeds each
python plot_random_weights.py  # produces the tables and the figure
```

The outputs are written to `experiments/results`. The full experiment takes about 44 minutes on the machine below, using 12 worker processes. The number of seeds and of worker processes can be changed with `--seeds` and `--workers`.

## System Specifications of Reported Runs

For all of the results we report in our paper, we use a laptop with the following specifications:

| **Component** | **Details** |
|--------------|-------------|
| **Machine** | Apple MacBook Pro (Mac16,6) |
| **CPU** | Apple M4 Max (14 cores: 10 performance, 4 efficiency) |
| **GPU** | Not used |
| **RAM** | 36 GB |
| **OS** | macOS 27.0.1 |
| **Software** | Python 3.11.17, NumPy 2.4.6, SciPy 1.17.1, CVXPY 1.9.3, Clarabel 0.11.1 |
