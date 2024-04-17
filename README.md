<p align = "left">
  <img src="PGopt_logo.svg" alt="PGopt_logo" style="width:60%;"/>
</p>

# `PGopt`: Particle Gibbs-based optimal control with performance guarantees

`PGopt` is a software for determining optimal input trajectories with probabilistic performance and constraint satisfaction guarantees for unknown systems with latent states based on input-output measurements. In order to quantify uncertainties, which is crucial for deriving formal guarantees, a Bayesian approach is employed and a prior over the unknown dynamics and the system trajectory is formulated in state-space representation. Since for practical applicability, the prior must be updated based on input-output measurements, but the corresponding posterior distribution is analytically intractable, particle Gibbs (PG) sampling is utilized to draw samples from this distribution. Based on these samples, a scenario optimal control problem (OCP) is formulated and probabilistic performance and constraint satisfaction guarantees are inferred via a greedy constraint removal.

The procedure is explained in the paper "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available as a preprint on [arXiv](https://arxiv.org/abs/2303.17963).

Two versions of the algorithm are currently available: a [Julia implementation](Julia) and a [MATLAB implementation](MATLAB).

## Julia

In order to ensure the reproducibility of the results presented in the paper without reliance on proprietary software, a Julia implementation using the [Altro](https://github.com/RoboticExplorationLab/Altro.jl) solver is provided. This version was used for the results presented in the paper and reproduces them exactly. However, this version has some limitations: only cost functions of the form $J_H=\sum\nolimits_{t=0}^H \frac{1}{2} u_t R u_t$, measurement functions of the form $y=x_{1:n_y}$, and output constraints of the form $y_\mathrm{min} \leq y \leq y_\mathrm{max}$ are supported. Further information can be found in the [PGopt Julia documentation](Julia/README.md).

## MATLAB

The MATLAB implementation is more general than the Julia implementation and allows arbitrary cost functions $J_H(u_{1:H},x_{1:H},y_{1:H})$, measurement functions $y=g(x,u)$, and constraints $h(u_{1:H},x_{1:H},y_{1:H})$. [CasADi](https://web.casadi.org/) and [IPOPT](https://coin-or.github.io/Ipopt/) are used to solve the scenario optimal control problem. In addition, the proprietary [HSL Linear Solvers](https://licences.stfc.ac.uk/product/coin-hsl) are used, which significantly accelerate the optimization. A license for the HSL Linear Solvers (free to academics) is required. Further information can be found in the [PGopt MATLAB documentation](MATLAB/README.md).

## Reference
If you found this software useful for your research, consider citing us.
```
@article{PMCMC_OCP_arXiv_2023,
      title={Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States},
      author={Lefringhausen, Robert and Srithasan, Supitsana and Lederer, Armin and Hirche, Sandra},
      journal={arXiv preprint arXiv:2303.17963},
      year={2023}
}
```
