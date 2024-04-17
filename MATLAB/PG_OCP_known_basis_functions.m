% This code produces results for the optimal control approach with known basis functions similar to the ones given in Section V-B (Fig. 2) of the paper
% "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available as pre-print on arXiv: https://arxiv.org/abs/2303.17963.
% Since the Julia implementation was used for the results in the paper, the results are not exactly reproduced due to different random numbers.

% Clear
clear;
clc;
close all;

% Specify seed (for reproducible results).
rng(5);

% Import CasADi
addpath('C:\Users\Robert Lefringhausen\Documents\CasADi')
import casadi.*

%% Learning parameters
K = 200; % number of PG samples
k_d = 50; % number of samples to be skipped to decrease correlation (thinning)
K_b = 1000; % length of burn-in period
N = 30; % number of particles of the particle filter

%% Number of states, etc.
n_x = 2; % number of states
n_u = 1; % number of control inputs
n_y = 1; % number of outputs

%% State-space prior
% Define basis functions - assumed to be known in this example.
% Make sure that phi(x,u) is defined in vectorized form, i.e., phi(zeros(n_x,N), zeros(n_u, N)) should return a matrix of dimension (n_phi, N).
n_phi = 5; % number of basis functions
phi = @(x, u) [0.1 * x(1, :); 0.1 * x(2, :); u(1, :); 0.01 * cos(3*x(1, :)) .* x(2, :); 0.1 * sin(2*x(2, :)) .* u(1, :)]; % basis functions

% Prior for Q - inverse Wishart distribution
ell_Q = 10; % degrees of freedom
Lambda_Q = 100 * eye(n_x); % scale matrix

% Prior for A - matrix normal distribution (mean matrix = 0, right covariance matrix = Q (see above), left covariance matrix = V)
V = diag(10*ones(n_phi, 1)); % left covariance matrix

% Initial guess for model parameters
Q_init = Lambda_Q; % initial Q
A_init = zeros(n_x, n_phi); % initial A

% Normally distributed initial state
x_init_mean = [2; 2]; % mean
x_init_var = 1 * ones(n_x, 1); % variance

%% Measurement model
% Define measurement model - assumed to be known (without loss of generality).
% Make sure that g(x,u) is defined in vectorized form, i.e., g(zeros(n_x,N), zeros(n_u, N)) should return a matrix of dimension (n_y, N).
g = @(x, u) [1, 0] * x; % observation function
R = 0.1; % variance of zero-mean Gaussian measurement noise

%% Parameters for data generation
T = 2000; % number of steps for training
T_test = 500; % number of steps used for testing (via forward simulation - see below)
T_all = T + T_test;

%% Generate training data.
% Choose the actual system (to be learned) and generate input-output data of length T_all.
% The system is of the form
% x_t+1 = f_true(x_t, u_t) + N(0, Q_true),
% y_t = g_true(x_t, u_t) + N(0, R_true).

% Unknown system
f_true = @(x, u) [0.8 * x(1, :) - 0.5 * x(2, :) + 0.1 * cos(3*x(1, :)) * x(2, :); 0.4 * x(1, :) + 0.5 * x(2, :) + (1 + 0.3 * sin(2*x(2, :))) * u(1, :)]; % true state transition function
Q_true = [0.03, -0.004; -0.004, 0.01]; % true process noise variance
g_true = g; % true measurement function
R_true = R; % true measurement noise variance

% Input trajectory used to generate training and test data
u_training = mvnrnd(0, 3, T)'; % training inputs
u_test = 3 * sin(2*pi*(1 / T_test)*((1:T_test) - 1)); % test inputs
u = [u_training, u_test]; %  training + test inputs

% Generate data by forward simulation.
x = zeros(n_x, T_all+1); % true latent state trajectory
x(:, 1) = normrnd(x_init_mean, x_init_var); % random initial state
y = zeros(n_y, T_all); % output trajectory (measured)
for t = 1:T_all
    x(:, t+1) = f_true(x(:, t), u(t)) + mvnrnd(zeros(n_x, 1), Q_true)';
    y(:, t) = g_true(x(:, t)) + mvnrnd(0, R_true, n_y);
end

% Split data into training and test data
u_training = u(:, 1:T);
x_training = x(:, 1:T+1);
y_training = y(:, 1:T);

u_test = u(:, T+1:end);
x_test = x(:, T+1:end);
y_test = y(:, T+1:end);

%% Plot data.
% figure()
% hold on
% u_plot = plot(u, 'linewidth', 1);
% y_plot = plot(y, 'linewidth', 1);
% legend([u_plot, y_plot], 'input', 'output')
% uistack(u_plot, 'top')
% ylabel('u | y')
% xlabel('t')

%% Learn models.
% Result: K models of the type
% x_t+1 = PG_samples{i}.A*phi(x_t,u_t) + N(0,PG_samples{i}.Q),
% where phi are the basis functions defined above.
PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_mean, x_init_var, g, R);

%% Test the learned models.
% Test the models with the test data by simulating it forward in time.
% test_prediction(PG_samples, phi, g, R, 10, u_test, y_test);

%% Plot autocorrelation.
% plot_autocorrelation(PG_samples, 'max_lag', 100)

%% Set up OCP.
% Horizon
H = 41;

% Define constraints for u.
u_max = 5; % max control input
u_min = -5; % min control input
input_constraints = @(u) bounded_input(u, u_min, u_max); % generate a function that returns the constraint vector h_u(u)

% Define constraints for y.
y_min = [-Inf * ones(20, 1); 2 * ones(6, 1); -Inf * ones(15, 1)]'; % min system output
y_max = Inf * ones(H, 1)'; % max system output
scenario_constraints = @(u, x, y) bounded_output(u, x, y, y_min, y_max); % generate a function that returns the constraint vector h_scenario(u, x, y)

% Define cost function.
% Objective: min sum u_t^2
cost_function = @(u) sum(u.^2);

% Confidence parameter for the theoretical guarantees
beta = 0.01;

% Solve the PG OCP.
% [u_opt, x_opt, y_opt, J_opt, solve_successful, ~, ~] = solve_PG_OCP(PG_samples, phi, g, R, H, cost_function, scenario_constraints, input_constraints, 'J_u', true, 'K_pre_solve', 10);

% Solve the PG OCP and determine complexity s and max constraint violation probability via greedy algorithm.
[u_opt, x_opt, y_opt, J_opt, solve_successful, s, epsilon_prob, epsilon_perc, time_first_solve, time_guarantees, num_failed_optimizations] = solve_PG_OCP_greedy_guarantees(PG_samples, phi, g, R, H, cost_function, scenario_constraints, input_constraints, beta, 'J_u', true, 'K_pre_solve', 10);

%% Test solution
if solve_successful
    % Apply input trajectory to the actual system.
    y_sys = zeros(n_y, H);
    x_sys = zeros(n_x, H+1);
    x_sys(:, 1) = x_training(:, end);
    for t = 1:H
        x_sys(:, t+1) = f_true(x_sys(:, t), u_opt(t)) + mvnrnd(zeros(n_x, 1), Q_true)';
        y_sys(:, t) = g_true(x_sys(:, t), u_opt(t)) + mvnrnd(zeros(n_y, 1), R_true);
    end

    % Plot predictions.
    plot_predictions(y_opt, y_sys, 'y_min', y_min)
end