# Introduction

**PGopt** is a software for determining optimal input trajectories with probabilistic performance and constraint satisfaction guarantees for unknown systems with latent states based on input-output measurements. In order to quantify uncertainties, which is crucial for deriving formal guarantees, a Bayesian approach is employed and a prior over the unknown dynamics and the system trajectory is formulated in state-space representation. Since for practical applicability, the prior must be updated based on input-output measurements, but the corresponding posterior distribution is analytically intractable, particle Gibbs (PG) sampling is utilized to draw samples from this distribution. Based on these samples, a scenario optimal control problem (OCP) is formulated and probabilistic performance and constraint satisfaction guarantees are inferred via a greedy constraint removal.

The approach is explained in the paper "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available as a preprint on [arXiv](https://arxiv.org/abs/2303.17963).

## Installation
PGopt can be installed using the Julia package manager. Inside the Julia REPL, type `]` to enter the Pkg REPL mode then run

`pkg> add https://github.com/TUM-ITR/PGopt:Julia`

# Examples