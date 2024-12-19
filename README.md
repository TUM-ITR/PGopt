<p align="center">
<img width="452" height="257" src="PGopt_logo.png">
</p>

# `PGopt`: Particle Gibbs-based optimal control with performance guarantees for unknown systems with latent states
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13737628.svg)](https://doi.org/10.5281/zenodo.13737628)

`PGopt` is a software for determining optimal input trajectories with probabilistic performance and constraint satisfaction guarantees for unknown systems with latent states based on input-output measurements. In order to quantify uncertainties, which is crucial for deriving formal guarantees, a Bayesian approach is employed and a prior over the unknown dynamics and the system trajectory is formulated in state-space representation. Since for practical applicability, the prior must be updated based on input-output measurements, but the corresponding posterior distribution is analytically intractable, particle Gibbs (PG) sampling is utilized to draw samples from this distribution. Based on these samples, a scenario optimal control problem (OCP) is formulated and probabilistic performance and constraint satisfaction guarantees are inferred via a greedy constraint removal.

The approach is explained in the paper "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available on [IEEExplore](https://doi.org/10.23919/ECC64448.2024.10590972) and as a preprint on [arXiv](https://arxiv.org/abs/2303.17963).

Two versions of the algorithm are currently available: a [Julia implementation](Julia) and a [MATLAB implementation](MATLAB).

## Julia
[![Dev](https://img.shields.io/badge/docs-stable-blue?logo=Julia&logoColor=white)](https://TUM-ITR.github.io/PGopt)

In order to ensure the reproducibility of the results presented in the paper without reliance on proprietary software, a Julia implementation that utilizes the solver [Altro](https://github.com/RoboticExplorationLab/Altro.jl) to solve the OCP is provided. This version was used for the results presented in the paper. However, this version has some limitations: only cost functions of the form $J_H=\sum\nolimits_{t=0}^H \frac{1}{2} u_t R u_t$, measurement functions of the form $y=x_{1:n_y}$, and output constraints of the form $y_\mathrm{min} \leq y \leq y_\mathrm{max}$ are supported. 

Besides the Julia implementation that utilizes Altro, there is also an implementation that utilizes the solver [IPOPT](https://coin-or.github.io/Ipopt/). This implementation allows arbitrary cost functions $J_H(u_{1:H},x_{1:H},y_{1:H})$, measurement functions $y=g(x,u)$, and constraints $h(u_{1:H},x_{1:H},y_{1:H})$. However, using IPOPT together with the proprietary [HSL Linear Solvers for Julia](https://licences.stfc.ac.uk/product/libhsl) (`HSL_jll.jl`) is recommended. A license (free to academics) is required.

Further information can be found in the [PGopt Julia documentation](Julia/README.md).

## MATLAB
The MATLAB implementation allows arbitrary cost functions $J_H(u_{1:H},x_{1:H},y_{1:H})$, measurement functions $y=g(x,u)$, and constraints $h(u_{1:H},x_{1:H},y_{1:H})$. [CasADi](https://web.casadi.org/) and [IPOPT](https://coin-or.github.io/Ipopt/) are used to solve the scenario optimal control problem. In addition, the proprietary [HSL Linear Solvers](https://licences.stfc.ac.uk/product/coin-hsl) are used, which significantly accelerate the optimization. A license for the HSL Linear Solvers (free to academics) is required. 

Further information can be found in the [PGopt MATLAB documentation](MATLAB/README.md).

## Erratum: Hyperparameters in Section V-C
It has come to our attention that the hyperparameters reported in the example with generic basis functions in Section V-C of the paper were unclear due to a notational inconsistency in the kernel representation. While the theoretical conclusions of the paper remain unaffected, the code for the example with generic basis functions has been updated for consistency with the general literature in version **v0.2.2** and later.

### Corrected Kernel Definition
The squared exponential kernel is now defined as

$k(z, z') = s_f \exp (-\frac{1}{2} (z - z')^\top \Lambda^{-1} (z - z'))$,

where $\Lambda = l^2 I$, and $I$ is the identity matrix. 

This updated definition ensures consistency with the general literature. Using this definition, the hyperparameters for the example in Section V-C (Table III) of the paper are:
* $l = 2\pi \approx 6.28$ 
* $s_f = \frac{100^2}{8\pi^4} \approx 12.83$

All other parameters in Table III remain unchanged. 

The updated definition may result in small numerical differences (e.g., due to rounding or recalculation) compared to the original results presented in the paper. These differences are minor and do not impact the overall findings or conclusions.

### Reproducing Original Results
If you wish to reproduce the exact results from the paper, you can pull an earlier release of the code (before version **v0.2.2**) by checking out the following tag:
```
git checkout v0.2.1
```

## Reference
If you found this software useful for your research, consider citing us.
```
@inproceedings{lefringhausen2024,
  title={Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States},
  author={Lefringhausen, Robert and Srithasan, Supitsana and Lederer, Armin and Hirche, Sandra},
  booktitle={2024 European Control Conference (ECC)},
  pages={90--97},
  year={2024},
  organization={IEEE}
}
```