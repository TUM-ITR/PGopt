# This code reproduces the results of the optimal control approach with generic basis functions given in Section V-C of the paper 
# "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available as pre-print on arXiv: https://arxiv.org/abs/2303.17963.
# This script reproduces Figure 3. For the results given in Table IV, this script is repeated with seeds 1:100.
# Please note that the results depend heavily on random numbers and that changing the order of the generated random numbers (e.g., by executing commented-out code parts) changes the results.

using LinearAlgebra
using Random
using Distributions
using Printf
using Plots
include("particle_Gibbs.jl")
include("optimal_control_Altro.jl")

# Specify seed (for reproducible results).
Random.seed!(82)

# Time sampling algorithm.
sampling_timer = time()

# Learning parameters
K = 100 # number of PG samples
k_d = 50 # number of samples to be skipped to decrease correlation (thinning)
K_b = 1000 # length of burn-in period
N = 30 # number of particles of the particle filter

# Number of states, etc.
n_x = 2 # number of states
n_u = 1 # number of control inputs
n_y = 1 # number of outputs
n_z = n_x + n_u # number of augmented states

# Generate generic basis functions and priors based on a reduced-rank GP approximation.
# See the paper
#   A. Svensson and T. B. Schön, "A flexible state–space model for learning nonlinear dynamical systems," Automatica, vol. 80, pp. 189–199, 2017.
# and the code provided in the supplementary material for further explanations. The equation numbers given in the following refer to this paper.
n_phi_x = [5 5] # number of basis functions for each state
n_phi_u = 5 # number of basis functions for the control input
n_phi_dims = [n_phi_u n_phi_x] # array containing the number of basis functions for each input dimension
n_phi = prod(n_phi_dims) # total number of basis functions
l_x = 20
L_x = [l_x l_x] # interval lengths for x
L_u = 10 # interval length for u
L = zeros(1, 1, n_z) # array containing the interval lengths
L[1, 1, :] = [L_u L_x]

# Hyperparameters of the squared exponential kernel
l = [2] # length scale
sf = 100 # scale factor

# Initialize.
j_vec = zeros(n_phi, 1, n_z) # contains all possible vectors j; j_vec[i, 1, :] corresponds to the vector j in eq. (5) for basis function i
lambda = zeros(n_phi, n_z) # lambda[i, :] corresponds to the vector λ in eq. (9) (right-hand side) for basis function i

# In the following, all possible vectors j are constructed (i.e., j_vec). The possible combinations correspond to the Cartesian product [1 : n_basis[1]] x ... x [1 : n_basis[end]].
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

# Define basis functions phi.
# This function evaluates Φ_{1 : n_x+n_u} according to eq. (5).
# Multithreading is used to speed up the sampling.
function phi_sampling(x, u)
    # Initialize
    z = vcat(u, x) # augmented state
    phi = Array{Float64}(undef, n_phi, size(z, 2))

    Threads.@threads for i in 1:size(z, 2)
        phi_temp = ones(n_phi)
        for k in 1:size(z, 1)
            phi_temp = phi_temp .* ((1 ./ (sqrt.(L[:, :, k]))) .* sin.((pi .* j_vec[:, :, k] .* (z[k, i] .+ L[:, :, k])) ./ (2 .* L[:, :, k])))
        end
        phi[:, i] = phi_temp
    end
    return phi
end

# Prior for Q - inverse Wishart distribution
ell_Q = 10 # degrees of freedom
Lambda_Q = 100 * I(n_x) # scale matrix

# Prior for A - matrix normal distribution (mean matrix = 0, right covariance matrix = Q (see above), left covariance matrix = V)
# V is derived from the GP approximation according to eq. (8b), (11a), and (9).
V_diagonal = Array{Float64}(undef, size(lambda, 1)) # diagonal of V
for i in 1:size(lambda, 1)
    V_diagonal[i] = sf^2 * sqrt(opnorm(2 * pi * Diagonal(repeat(l, trunc(Int, n_z / size(l, 1))) .^ 2))) * exp.(-(pi^2 * transpose(sqrt.(lambda[i, :])) * Diagonal(repeat(l, trunc(Int, n_z / size(l, 1))) .^ 2) * sqrt.(lambda[i, :])) / 2)
end
V = Diagonal(V_diagonal)

# Initial guess for model parameters
Q_init = Lambda_Q # initial Q
A_init = zeros(n_x, n_phi) # initial A

# Normally distributed initial state
x_init_mean = [2, 2] # mean
x_init_var = 1 * I # variance
x_init_dist = MvNormal(x_init_mean, x_init_var)

# Define measurement model - assumed to be known (without loss of generality).
# Make sure that g(x,u) is defined in vectorized form, i.e., g(zeros(n_x,N), zeros(n_u, N)) should return a matrix of dimension (n_y, N).
g(x, u) = [1 0] * x # observation function
R = 0.1 # variance of zero-mean Gaussian measurement noise

# Parameters for data generation
T = 2000 # number of steps for training
T_test = 500  # number of steps for testing
T_all = T + T_test

# Generate training data.
# Choose the actual system (to be learned) and generate input-output data of length T_all.
# The system is of the form
# x_t+1 = f_true(x_t, u_t) + N(0, Q_true)
# y_t = g_true(x_t, u_t) + N(0, R_true).

# Unknown system
f_true(x, u) = [0.8 * x[1, :] - 0.5 * x[2, :] + 0.1 * cos.(3 * x[1, :]) .* x[2, :]; 0.4 * x[1, :] + 0.5 * x[2, :] + (ones(size(x, 2)) + 0.3 * sin.(2 * x[2, :])) .* u[1, :]] # true state transition function
Q_true = [0.03 -0.004; -0.004 0.01] # true process noise variance
mvn_v_true = MvNormal(zeros(n_x), Q_true) # true process noise distribution
g_true = g # true measurement function
R_true = R # true measurement noise variance
mvn_e_true = MvNormal(zeros(n_y), R_true) # true measurement noise distribution

# Input trajectory used to generate training and test data
mvn_u_training = Normal(0, 3) # training input distribution
u_training = rand(mvn_u_training, (1, T)) # training inputs
u_test = 3 * sin.(2 * pi * (1 / T_test) * (Array(1:T_test)' .- 1)) # test inputs
u = reshape([u_training u_test], (n_u, T_all)) # training + test inputs

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

# Plot data.
# plot(Array(1:T_all), u[1,:], label="input", lw=2, legend=:topright);
# plot!(Array(1:T_all), y[1,:], label="output", lw=2);
# xlabel!("t");
# ylabel!("u | y");

# Learn models.
# Result: K models of the type
# x_t+1 = PG_samples[i].A*phi(x_t, u_t) + N(0, PG_samples[i].Q),
# where phi are the basis functions defined above.
PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi_sampling, Lambda_Q, ell_Q, Lambda_Q, V, A_init, x_init_dist, g, R)

time_sampling = time() - sampling_timer

# Test the models with the test data by simulating it forward in time.
# test_prediction(PG_samples, phi_sampling, g, R, 10, u_test, y_test)

# Plot autocorrelation.
# plot_autocorrelation(PG_samples; max_lag=K-1)

# Set up OCP.
# Horizon
H = 41

# Define constraints for u and y.
u_max = [5] # max control input
u_min = [-5] # min control input
y_max = reshape(fill(Inf, H), (1, H)) # max system output
y_min = reshape([-fill(Inf, 20); 2 * ones(6); -fill(Inf, 15)], (1, H)) # min system output

# Define cost function.
# Objective: min ∑_{∀t} 1/2 * u_t * Diagonal(R_cost_diag) * u_t.
R_cost_diag = [2] # diagonal of R_cost

# Redefine phi - the optimization cannot deal with multithreading.
# This function evaluates Φ_{1 : n_x+n_u} according to eq. (5).
function phi_opt(x, u)
    # Initialize.
    z = vcat(u, x) # augmented state
    phi = Array{Any}(undef, n_phi, size(z, 2))

    for i in 1:size(z, 2)
        phi_temp = ones(n_phi)
        for k in 1:size(z, 1)
            phi_temp = phi_temp .* ((1 ./ (sqrt.(L[:, :, k]))) .* sin.((pi .* j_vec[:, :, k] .* (z[k, i] .+ L[:, :, k])) ./ (2 .* L[:, :, k])))
        end
        phi[:, i] = phi_temp
    end
    return phi
end

# Optimization settings
opts = SolverOptions()
opts.constraint_tolerance = 1e-5
opts.cost_tolerance = 1e-3
opts.cost_tolerance_intermediate = 10 * opts.cost_tolerance
opts.projected_newton_tolerance = sqrt(opts.constraint_tolerance)
opts.penalty_scaling = 25
opts.penalty_initial = 1e4
opts.iterations = 5000
opts.max_cost_value = 1e12
opts.static_bp = false
opts.square_root = true

# Solve PG OCP.
x_opt, u_opt, y_opt, J_opt, penalty_max = solve_PG_OCP(PG_samples, phi_opt, R, H, u_min, u_max, y_min, y_max, R_cost_diag; K_pre_solve=20, opts=opts)[[1, 2, 3, 4, 8]]

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