# Optimal control with generic basis functions

This example reproduces the results of the optimal control approach with generic basis functions (Figure 3) given in Section V-C of the [paper](../reference.md). Due to small adjustments to the code (see Erratum in the [Introduction](../index.md)) and resulting numerical differences, the results from **v0.2.2** and later differ slightly from those presented in the paper. If you wish to reproduce the exact results from the paper, you can pull an earlier release of the code before version **v0.2.2**.

![autocorrelation](../assets/PG_OCP_generic_basis_functions.svg)

The method presented in the paper ["A flexible state–space model for learning nonlinear dynamical systems"](https://doi.org/10.1016/j.automatica.2017.02.030) is utilized to systematically derive basis functions and priors for the parameters based on a reduced-rank GP approximation. Afterward, by calling the function `particle_Gibbs()`, samples are drawn from the posterior distribution using particle Gibbs sampling. These samples are then passed to the function `solve_PG_OCP_Altro()` (or `solve_PG_OCP_Ipopt()` in case IPOPT is used), which solves the scenario OCP using the solver Altro.

A Julia script that contains all the steps described in the following and reproduces Figure 2 of the paper can be found at `PGopt/Julia/examples/PG_OCP_generic_basis_functions_Altro.jl`. For the results in Table IV of the paper, this script is repeated with seeds 1:100. The runtime of the script is about 2 hours on a standard laptop.

A similar example that utilizes the solver IPOPT can be found at `PGopt/Julia/examples/PG_OCP_generic_basis_functions_Ipopt.jl`. Due to the different solver, the results differ slightly from the ones presented in the paper.

## Define parameters
First, load packages and initialize.
```julia
using PGopt
using LinearAlgebra
using Random
using Distributions
using Printf
using Plots

# Specify seed (for reproducible results).
Random.seed!(82)

# Time PGS algorithm.
sampling_timer = time()
```
Then, specify the parameters of the algorithm.
```julia
# Learning parameters
K = 100 # number of PG samples
k_d = 50 # number of samples to be skipped to decrease correlation (thinning)
K_b = 1000 # length of burn-in period
N = 30 # number of particles of the particle filter

# Number of states, etc.
n_x = 2 # number of states
n_u = 1 # number of control inputs
n_y = 1 # number of outputs
```
## Define basis functions
Then, generate generic basis functions and priors based on a reduced-rank GP approximation.
The approach is described in the paper ["A flexible state–space model for learning nonlinear dynamical systems"](https://doi.org/10.1016/j.automatica.2017.02.030).
The equation numbers given in the following refer to this paper.
In this example, a GP with a squared exponential kernel
$k(z, z') = s_f \exp (-\frac{1}{2} (z - z')^\top \Lambda^{-1} (z - z'))$,
where $\Lambda = l^2 I$ and $I$ is the identity matrix, is approximated.
```julia
n_phi_x = [5 5] # number of basis functions for each state
n_phi_u = 5 # number of basis functions for the control input
n_phi_dims = [n_phi_u n_phi_x] # array containing the number of basis functions for each input dimension
n_phi = prod(n_phi_dims) # total number of basis functions
l_x = 20
L_x = [l_x l_x] # interval lengths for x
L_u = 10 # interval length for u
L = zeros(1, 1, n_z) # array containing the interval lengths
L[1, 1, :] = [L_u L_x]

# Initialize.
j_vec = zeros(n_phi, 1, n_z) # contains all possible vectors j; j_vec[i, 1, :] corresponds to the vector j in eq. (5) for basis function i
lambda = zeros(n_phi, n_z) # lambda[i, :] corresponds to the vector λ in eq. (9) (right-hand side) for basis function i
```
In the following, all possible vectors ``j`` are constructed (i.e., `j_vec`). The possible combinations correspond to the Cartesian product `[1 : n_basis[1]] x ... x [1 : n_basis[end]]`.
```julia
cart_prod_sets = Array{Any}(undef, n_z) # array of arrays; cart_prod_sets[i] corresponds to the i-th set to be considered for the Cartesian product, i.e., [1 : n_basis[i]].
for i = 1:n_z
    cart_prod_sets[i] = Array(1:n_phi_dims[i])
end

subscript_values = Array{Int64}(undef, n_z) # contains the equivalent subscript values corresponding to a given single index i
variants = [1; cumprod(n_phi_dims[1:end-1])] # required to convert the single index i to the equivalent subscript value

# Construct Cartesian product and calculate spectral densities.
for i in 1:n_phi
    # Convert the single index i to the equivalent subscript values.
    remaining = i - 1
    for j in n_z:-1:1
        subscript_values[j] = floor(remaining / variants[j]) + 1
        remaining = mod(remaining, variants[j])
    end

    # Fill j_vec with the values belonging to the respective subscript indices.
    for j in 1:n_z
        j_vec[i, 1, j] = cart_prod_sets[j][subscript_values[j]]
    end

    # Calculate the eigenvalue of the Laplace operator corresponding to the vector j_vec[i, 1, :] - see eq. (9) (right-hand side).
    lambda[i, :] = (pi .* j_vec[i, 1, :] ./ (2 * dropdims(L, dims=tuple(findall(size(L) .== 1)...)))) .^ 2
end

# Reverse j_vec.
j_vec = reverse(j_vec, dims=3)
```
Then, define basis functions phi. This function evaluates ``\phi_{1 : n_x+n_u}`` according to eq. (5).
```julia
# Precompute.
L_sqrt_inv = 1 ./ sqrt.(L)
pi_j_over_2L = pi .* j_vec ./ (2 .* L)
function phi_sampling(x, u)
    # Initialize.
    z = vcat(u, x) # augmented state
    phi = ones(n_phi, size(z, 2))

    for k in axes(z, 1)
        phi .= phi .* (L_sqrt_inv[:, :, k] .* sin.(pi_j_over_2L[:, :, k] * (z[k, :] .+ L[:, :, k])'))
    end

    return phi
end
```
Since the optimization cannot deal with multithreading or in-place computations, a less efficient definition of phi is required for the subsequent optimization.
```julia
function phi_opt(x, u)
    # Initialize.
    z = vcat(u, x) # augmented state
    phi = Array{Any}(undef, n_phi, size(z, 2))

    for i in axes(z, 2)
        phi_temp = ones(n_phi)
        for k in axes(z, 1)
            phi_temp = phi_temp .* ((1 ./ (sqrt.(L[:, :, k]))) .* sin.((pi .* j_vec[:, :, k] .* (z[k, i] .+ L[:, :, k])) ./ (2 .* L[:, :, k])))
        end
        phi[:, i] = phi_temp
    end
    return phi
end
```
## Define prior
Select the parameters of the inverse Wishart prior for ``Q``.
```julia
ell_Q = 10 # degrees of freedom
Lambda_Q = 100 * I(n_x) # scale matrix
```
Determine the parameters of the matrix normal prior (with mean matrix ``0``, right covariance matrix ``Q`` (see above), and left covariance matrix ``V``) for ``A``.
``V`` is derived from the GP approximation according to eq. (8b), (9).
The spectral density of the anisotropic squared exponential kernel 
$k(z, z') = s_f \exp (-\frac{1}{2} (z - z')^\top \Lambda^{-1} (z - z'))$
is
$S(\omega) = s_f (2 \pi)^{\frac{d}{2}} |\Lambda|^{\frac{1}{2}} \exp(-\frac{\omega^\top \Lambda \omega}{2})$; 
see eq. (68) in the paper ["Hilbert space methods for reduced-rank Gaussian process regression"](https://doi.org/10.1007/s11222-019-09886-w).
```julia
V_diag = Array{Float64}(undef, size(lambda, 1)) # diagonal of V
for i in axes(lambda, 1)
    V_diag[i] = sf * ((2 * pi)^(n_z / 2)) * prod(l) * exp(-0.5 * sum((l .^ 2) .* lambda[i, :]))
end
V = Diagonal(V_diag)
```
Provide an initial guess for the parameters.
```julia
Q_init = Lambda_Q # initial Q
A_init = zeros(n_x, n_phi) # initial A
```
Choose the distribution of the initial state. Here, a normally distributed initial state is assumed.
```julia
x_init_mean = [2, 2] # mean
x_init_var = 1 * I # variance
x_init_dist = MvNormal(x_init_mean, x_init_var)
```
Define the measurement model. It is assumed to be known (without loss of generality). Make sure that ``g(x, u)`` is defined in vectorized form, i.e., `g(zeros(n_x, N), zeros(n_u, N))` should return a matrix of dimension `(n_y, N)`.
```julia
g(x, u) = [1 0] * x # observation function
R = 0.1 # variance of zero-mean Gaussian measurement noise
```
## Generate data
Generate training data.
```julia
# Parameters for data generation
T = 2000 # number of steps for training
T_test = 500 # number of steps used for testing (via forward simulation - see below)
T_all = T + T_test

# Unknown system
f_true(x, u) = [0.8 * x[1, :] - 0.5 * x[2, :] + 0.1 * cos.(3 * x[1, :]) .* x[2, :]; 0.4 * x[1, :] + 0.5 * x[2,:] + (ones(size(x, 2)) + 0.3 * sin.(2 * x[2, :])) .* u[1, :]] # true state transition function
Q_true = [0.03 -0.004; -0.004 0.01] # true process noise variance
mvn_v_true = MvNormal(zeros(n_x), Q_true) # true process noise distribution
g_true = g # true measurement function
R_true = R # true measurement noise variance
mvn_e_true = MvNormal(zeros(n_y), R_true) # true measurement noise distribution

# Input trajectory used to generate training and test data
mvn_u_training = Normal(0, 3) # training input distribution
u_training = rand(mvn_u_training, T) # training inputs
u_test = 3 * sin.(2 * pi * (1 / T_test) * (Array(1:T_test) .- 1)) # test inputs
u = reshape([u_training; u_test], 1, T_all) # training + test inputs

# Generate data by forward simulation.
x = Array{Float64}(undef, n_x, T_all + 1) # true latent state trajectory
x[:, 1] = rand(x_init_dist, 1) # random initial state
y = Array{Float64}(undef, n_y, T_all) # output trajectory (measured)
for t in 1:T_all
  x[:, t+1] = f_true(x[:, t], u[:, t]) + rand(mvn_v_true, 1)
  y[:, t] = g_true(x[:, t], u[:, t]) + rand(mvn_e_true, 1)
end

# Split data into training and test data.
u_training = u[:, 1:T]
x_training = x[:, 1:T+1]
y_training = y[:, 1:T]

u_test = u[:, T+1:end]
x_test = x[:, T+1:end]
y_test = y[:, T+1:end]
```
## Infer model
Run the particle Gibbs sampler to jointly estimate the model parameters and the latent state trajectory.
```julia
PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi_sampling, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_dist, g, R)

time_sampling = time() - sampling_timer
```

## Define and solve optimal control problem using Altro
In the following, the optimal control problem is defined and solved using the solver Altro. An example using the solver IPOPT is given in the next section.
With the Altro solver, problems of the following form can be solved

``\min \sum_{t=0}^{H} \frac{1}{2} u_t \operatorname{diag}(R_{\mathrm{cost}}) u_t``

subject to:
```math
\begin{aligned}
\forall k, \forall t \\
x_{t+1}^{[k]} &= f_{\theta^{[k]}}(x_t^{[k]}, u_t) + v_t^{[k]} \\
x_{t, 1:n_y}^{[k]} &\geq y_{\mathrm{min},\ t} - e_t^{[k]} \\
x_{t, 1:n_y}^{[k]} &\leq y_{\mathrm{max},\ t} - e_t^{[k]} \\
u_t &\geq u_{\mathrm{min},\ t} \\
u_t &\leq u_{\mathrm{max},\ t}.
\end{aligned}
```

(Note that the output constraints imply the measurement function ``y_t^{[k]} = x_{t, 1:n_y}^{[k]}``)
```julia
# Horizon
H = 41

# Define constraints for u and y.
u_max = [5] # max control input
u_min = [-5] # min control input
y_max = reshape(fill(Inf, H), (1, H)) # max system output
y_min = reshape([-fill(Inf, 20); 2 * ones(6); -fill(Inf, 15)], (1, H)) # min system output

R_cost_diag = [2] # diagonal of R_cost
```
Solve the optimal control problem using the solver Altro. In this case, no formal guarantees for the constraint satisfaction can be derived since Assumption 1 is not satisfied as the employed basis functions cannot represent the actual dynamics with arbitrary precision.
```julia
u_opt, x_opt, y_opt, J_opt, penalty_max = solve_PG_OCP_Altro(PG_samples, phi_opt, R, H, u_min, u_max, y_min, y_max, R_cost_diag; K_pre_solve=20, opts=opts)[[1, 2, 3, 4, 8]]
```
Finally, apply the input trajectory to the actual system and plot the output trajectories.
```julia
# Apply input trajectory to the actual system.
y_sys = Array{Float64}(undef, n_y, H)
x_sys = Array{Float64}(undef, n_x, H)
x_sys[:, 1] = x_training[:, end]
u_sys = [u_opt 0]
for t in 1:H
  if t >= 2
    x_sys[:, t] = f_true(x_sys[:, t-1], u_sys[:, t-1]) + rand(mvn_v_true, 1)
  end
  y_sys[:, t] = g_true(x_sys[:, t], u_sys[:, t]) + rand(mvn_e_true, 1)
end

# Plot predictions.
plot_predictions(y_opt, y_sys; plot_percentiles=false, y_min=y_min, y_max=y_max)
```

## Define and solve optimal control problem using Ipopt
Besides the solver Altro, IPOPT can be used to solve the OCP. The solver IPOPT is more general than Altro, and this implementation allows arbitrary cost functions ``J_H(u_{1:H},x_{1:H},y_{1:H})``, measurement functions ``y=g(x,u)``, and constraints ``h(u_{1:H},x_{1:H},y_{1:H})``.

First, load the necessary packages. Further information on how to install `HSL_jll` can be found [here](../index.md).
```julia
using JuMP
import HSL_jll
```
Then, set up the OCP.
```julia
# Set up OCP.
# Horizon
H = 41

# Define constraints for u.
u_max = repeat([5], 1, H) # max control input
u_min = repeat([-5], 1, H) # min control input
n_input_const = sum(isfinite.(u_min)) + sum(isfinite.(u_max))

# Define constraints for y.
y_max = reshape(fill(Inf, H), (1, H)) # max system output
y_min = reshape([-fill(Inf, 20); 2 * ones(6); -fill(Inf, 15)], (1, H)) # min system output
n_output_const = sum(isfinite.(y_min)) + sum(isfinite.(y_max))
```
The following functions define the input and output constraints. The function `bounded_input()` returns the constraint vector ``h_1(u_{1:H})`` and the function `bounded_output()` returns the constraint vector ``h_2(u_{1:H},x_{1:H}^{[k]},y_{1:H}^{[k]})``. Feasible solutions must satisfy ``h_1(u_{1:H}) \leq 0`` and ``h_2(u_{1:H},x_{1:H}^{[k]},y_{1:H}^{[k]}) \leq 0 \; \forall k``. The functions should be callable with arrays of type `VariableRef` and `<:Number`.
```julia
function bounded_input(u::Array{VariableRef})
  # Initialize constraint vector.
  h_u = Array{AffExpr}(undef, n_input_const)

  # Construct constraint vector - constraints are only considered if they are finite.
  i = 1
  for t in 1:H
    for n in 1:n_u
      if isfinite(u_min[n, t])
        h_u[i] = u_min[n, t] - u[n, t]
        i += 1
      end
      if isfinite(u_max[n, t])
        h_u[i] = u[n, t] - u_max[n, t]
        i += 1
      end
    end
  end
  return h_u
end

function bounded_input(u::Array{<:Number})
  # Initialize constraint vector.
  h_u = Array{Float64}(undef, n_input_const)

  # Construct constraint vector - constraints are only considered if they are finite.
  i = 1
  for t in 1:H
    for n in 1:n_u
      if isfinite(u_min[n, t])
        h_u[i] = u_min[n, t] - u[n, t]
        i += 1
      end
      if isfinite(u_max[n, t])
        h_u[i] = u[n, t] - u_max[n, t]
        i += 1
      end
    end
  end
  return h_u
end

function bounded_output(u::Array{VariableRef}, x::Array{VariableRef}, y::Array{VariableRef})
  # Initialize constraint vector.
  h_scenario = Array{AffExpr}(undef, n_output_const)

  # Construct constraint vector - constraints are only considered if they are finite.
  i = 1
  for t in 1:H
    for n in 1:n_y
      if isfinite(y_min[n, t])
        h_scenario[i] = y_min[n, t] - y[n, t]
        i += 1
      end
      if isfinite(y_max[n, t])
        h_scenario[i] = y[n, t] - y_max[n, t]
        i += 1
      end
    end
  end
  return h_scenario
end

function bounded_output(u::Array{<:Number}, x::Array{<:Number}, y::Array{<:Number})
  # Initialize constraint vector.
  h_scenario = Array{Float64}(undef, n_output_const)

  # Construct constraint vector - constraints are only considered if they are finite.
  i = 1
  for t in 1:H
    for n in 1:n_y
      if isfinite(y_min[n, t])
        h_scenario[i] = y_min[n, t] - y[n, t]
        i += 1
      end
      if isfinite(y_max[n, t])
        h_scenario[i] = y[n, t] - y_max[n, t]
        i += 1
      end
    end
  end
  return h_scenario
end
```
Define the cost function. In this case the objective is ``\min \sum_{t=0}^{H} u_t^2``.
```julia
function cost_function(u) 
  cost = sum(u.^2)
  return cost
end
```
Then, set the solver's options. The option `"linear_solver" => "ma57"` requires the proprietary [HSL Linear Solvers for Julia](https://licences.stfc.ac.uk/product/libhsl).
```julia
Ipopt_options = Dict("max_iter" => 10000, "tol" => 1e-8, "hsllib" => HSL_jll.libhsl_path, "linear_solver" => "ma57")
```
Solve the optimal control problem using the solver IPOPT. In this case, no formal guarantees for the constraint satisfaction can be derived since Assumption 1 is not satisfied as the employed basis functions cannot represent the actual dynamics with arbitrary precision.
Since the cost function depends only on the control inputs ``u_{0:H}``, the optional argument `J_u` is set to `true`.
```julia
u_opt, x_opt, y_opt, J_opt, solve_successful, iterations, mu = solve_PG_OCP_Ipopt(PG_samples, phi_opt, g, R, H, cost_function, bounded_output, bounded_input; J_u=true, K_pre_solve=20, solver_opts=Ipopt_options)
```
Finally, apply the input trajectory to the actual system and plot the output trajectories.
```julia
# Apply input trajectory to the actual system.
y_sys = Array{Float64}(undef, n_y, H)
x_sys = Array{Float64}(undef, n_x, H)
x_sys[:, 1] = x_training[:, end]
u_sys = [u_opt 0]
for t in 1:H
  if t >= 2
    x_sys[:, t] = f_true(x_sys[:, t-1], u_sys[:, t-1]) + rand(mvn_v_true, 1)
  end
  y_sys[:, t] = g_true(x_sys[:, t], u_sys[:, t]) + rand(mvn_e_true, 1)
end

# Plot predictions.
plot_predictions(y_opt, y_sys; plot_percentiles=false, y_min=y_min, y_max=y_max)
```