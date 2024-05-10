# Introduction

**PGopt** is a software for determining optimal input trajectories with probabilistic performance and constraint satisfaction guarantees for unknown systems with latent states based on input-output measurements. In order to quantify uncertainties, which is crucial for deriving formal guarantees, a Bayesian approach is employed and a prior over the unknown dynamics and the system trajectory is formulated in state-space representation. Since for practical applicability, the prior must be updated based on input-output measurements, but the corresponding posterior distribution is analytically intractable, particle Gibbs (PG) sampling is utilized to draw samples from this distribution. Based on these samples, a scenario optimal control problem (OCP) is formulated and probabilistic performance and constraint satisfaction guarantees are inferred via a greedy constraint removal.

The approach is explained in the paper "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available as a preprint on [arXiv](https://arxiv.org/abs/2303.17963).

## Versions
This document describes the Julia implementation of `PGopt`, which does not require proprietary software. The open-source solver [Altro](https://github.com/RoboticExplorationLab/Altro.jl) is used for the optimization. The results presented in the paper were generated with this version, and the software reproduces these results exactly.

**Please note that this version has several limitations: only cost functions of the form $J_H=\sum\nolimits_{t=0}^H \frac{1}{2} u_t R u_t$, measurement functions of the form $y=x_{1:n_y}$, and output constraints of the form $y_\mathrm{min} \leq y \leq y_\mathrm{max}$ are supported.**

Besides the Julia implementation, there is also a MATLAB implementation, which is more general and allows arbitrary cost functions $J_H(u_{1:H},x_{1:H},y_{1:H})$, measurement functions $y=g(x,u)$, and constraints $h(u_{1:H},x_{1:H},y_{1:H})$. Further information can be found [here](https://github.com/TUM-ITR/PGopt/tree/main/MATLAB).

## Installation
PGopt can be installed using the Julia package manager. Start a Pkg REPL (press `]` in a Julia REPL), and install PGopt via
```
pkg> add https://github.com/TUM-ITR/PGopt:Julia
```

Alternatively, to inspect the source code more easily, download the source code from [GitHub](https://github.com/TUM-ITR/PGopt). Navigate to the folder `PGopt/Julia`, start a Pkg REPL (press `]` in a Julia REPL), and install the dependencies via
```
pkg>activate . 
pkg>instantiate
```

You can then execute the scripts `autocorrelation.jl`, `PG_OCP_known_basis_functions.jl`, or `PG_generic_basis_functions.jl` in the folder `PGopt/Julia/examples`.