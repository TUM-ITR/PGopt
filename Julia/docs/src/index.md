# Introduction

**PGopt** is a software for determining optimal input trajectories with probabilistic performance and constraint satisfaction guarantees for unknown systems with latent states based on input-output measurements. In order to quantify uncertainties, which is crucial for deriving formal guarantees, a Bayesian approach is employed and a prior over the unknown dynamics and the system trajectory is formulated in state-space representation. Since for practical applicability, the prior must be updated based on input-output measurements, but the corresponding posterior distribution is analytically intractable, particle Gibbs (PG) sampling is utilized to draw samples from this distribution. Based on these samples, a scenario optimal control problem (OCP) is formulated and probabilistic performance and constraint satisfaction guarantees are inferred via a greedy constraint removal.

The approach is explained in the paper "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available on [IEEExplore](https://doi.org/10.23919/ECC64448.2024.10590972) and as a preprint on [arXiv](https://arxiv.org/abs/2303.17963).

## Versions
This document describes the Julia implementation of `PGopt`. In this version, the solvers [Altro](https://github.com/RoboticExplorationLab/Altro.jl) and [IPOPT](https://coin-or.github.io/Ipopt/) can be employed to solve the optimal control problem (OCP).

The results presented in the paper were generated with the solver Altro. Please note that this implementation has several limitations: only cost functions of the form $J_H=\sum\nolimits_{t=0}^H \frac{1}{2} u_t R u_t$, measurement functions of the form $y=x_{1:n_y}$, and output constraints of the form $y_\mathrm{min} \leq y \leq y_\mathrm{max}$ are supported.

The solver IPOPT is more general than Altro, and this implementation allows arbitrary cost functions $J_H(u_{1:H},x_{1:H},y_{1:H})$, measurement functions $y=g(x,u)$, and constraints $h(u_{1:H},x_{1:H},y_{1:H})$. However, using IPOPT together with the proprietary [HSL Linear Solvers for Julia](https://licences.stfc.ac.uk/product/libhsl) (`HSL_jll.jl`) is recommended. A license (free to academics) is required.

Besides the Julia implementation, there is also a MATLAB implementation that utilizes [CasADi](https://web.casadi.org/) and [IPOPT](https://coin-or.github.io/Ipopt/). Further information can be found [here](https://github.com/TUM-ITR/PGopt/tree/main/MATLAB).

## Erratum: Hyperparameters in Section V-C
It has come to our attention that the hyperparameters reported in the example with generic basis functions in Section V-C of the paper were unclear due to a notational inconsistency in the kernel representation. While the theoretical conclusions of the paper remain unaffected, the code for the example with generic basis functions has been updated for consistency with the general literature in version **v0.2.2** and later.

### Corrected Kernel Definition
The squared exponential kernel is now defined as
$k(z, z') = s_f \exp (-\frac{1}{2} (z - z')^\top \Lambda^{-1} (z - z'))$,
where $\Lambda = l^2 I$, and $I$ is the identity matrix. 

This updated definition ensures consistency with the general literature. Using this definition, the hyperparameters for the example in Section V-C (Table III) of the paper are:
* $l = 2\pi \approx 6.28$ 
* $s_f = \frac{100^2}{8\pi^4} \approx 12.83$

The updated definition may result in small numerical differences (e.g., due to rounding or recalculation) compared to the original results presented in the paper. These differences are minor and do not impact the overall findings or conclusions.

## Installation
`PGopt` can be installed using the Julia package manager. Start a Pkg REPL (press `]` in a Julia REPL), and install `PGopt` via
```
pkg> add https://github.com/TUM-ITR/PGopt:Julia
```

Alternatively, to inspect the source code more easily, download the source code from [GitHub](https://github.com/TUM-ITR/PGopt). Navigate to the folder `PGopt/Julia`, start a Pkg REPL (press `]` in a Julia REPL), and install the dependencies via
```
pkg>activate . 
pkg>instantiate
```

You can then execute the examples in the folder `PGopt/Julia/examples`.