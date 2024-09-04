# This code reproduces the normalized auto-correlation function plot given in Section V-B (Fig. 1) of the paper 
# "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available on IEEExplore: https://doi.org/10.23919/ECC64448.2024.10590972.

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

# Learning parameters
K = 10000 # number of PG samples
k_d = 0 # number of samples to be skipped to decrease correlation (thinning)
K_b = 1000 # length of burn-in period
N = 30 # number of particles of the particle filter

# Number of states, etc.
n_x = 2 # number of states
n_u = 1 # number of control inputs
n_y = 1 # number of outputs

# Define basis functions - assumed to be known in this example.
# Make sure that phi(x,u) is defined in vectorized form, i.e., phi(zeros(n_x,N), zeros(n_u, N)) should return a matrix of dimension (n_phi, N).
n_phi = 5 # number of basis functions
phi(x, u) = [0.1 * x[1, :] 0.1 * x[2, :] u[1, :] 0.01 * cos.(3 * x[1, :]) .* x[2, :] 0.1 * sin.(2 * x[2, :]) .* u[1, :]]' # basis functions

# Prior for Q - inverse Wishart distribution
ell_Q = 10 # degrees of freedom
Lambda_Q = 100 * I(n_x) # scale matrix

# Prior for A - matrix normal distribution (mean matrix = 0, right covariance matrix = Q (see above), left covariance matrix = V)
V = Diagonal(10 * ones(n_phi)) # left covariance matrix

# Initial guess
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
T_test = 500  # number of steps used for testing (via forward simulation - see below)
T_all = T + T_test

# Generate training data.
# Choose the actual system (to be learned) and generate input-output data of length T_all.
# The system is of the form
# x_t+1 = f_true(x_t, u_t) + N(0, Q_true),
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

# Plot data.
# plot(Array(1:T_all), u[1,:], label="input", lw=2, legend=:topright)
# plot!(Array(1:T_all), y[1,:], label="output", lw=2)
# xlabel!("t")
# ylabel!("u | y")

# Learn models.
# Result: K models of the type
# x_t+1 = PG_samples[i].A*phi(x_t, u_t) + N(0, PG_samples[i].Q),
# where phi are the basis functions defined above.
PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_dist, g, R)

time_sampling = time() - sampling_timer

# Test the models with the test data by simulating it forward in time.
# test_prediction(PG_samples, phi, R, 10, u_test, y_test)

# Plot autocorrelation.
plot_autocorrelation(PG_samples; max_lag=100)