% This code produces results for the optimal control approach with generic basis functions similar to the ones given in Section V-B (Fig. 3) of the paper
% "Learning-Based Optimal Control with Performance Guarantees for Unknown Systems with Latent States", available as pre-print on arXiv: https://arxiv.org/abs/2303.17963.
% Since the Julia implementation was used for the results in the paper, the results are not exactly reproduced due to different random numbers.

% Clear
clear;
clc;
close all;

% Specify seed (for reproducible results).
rng(5);

% Import CasADi - insert your path here.
addpath('<yourpath>/casadi-3.6.5-windows64-matlab2018b')
import casadi.*

%% Learning parameters
K = 100; % number of PG samples
k_d = 50; % number of samples to be skipped to decrease correlation (thinning)
K_b = 1000; % length of burn-in period
N = 30; % number of particles of the particle filter

%% Number of states, etc.
n_x = 2; % number of states
n_u = 1; % number of control inputs
n_y = 1; % number of outputs
n_z = n_x + n_u; % number of augmented states

%% State-space prior
% Generate generic basis functions and priors based on a reduced-rank GP approximation.
% See the paper
%   A. Svensson and T. B. Schön, "A flexible state–space model for learning nonlinear dynamical systems," Automatica, vol. 80, pp. 189–199, 2017.
% and the code provided in the supplementary material for further explanations. The equation numbers given in the following refer to this paper.
n_phi_x = [5, 5]; % number of basis functions for each state
n_phi_u = 5; % number of basis functions for the control input
n_phi_dims = [n_phi_u, n_phi_x]; % array containing the number of basis functions for each input dimension
n_phi = prod(n_phi_dims); % total number of basis functions
l_x = 20;
L_x = [l_x, l_x]; % interval lengths for x
L_u = 10; % interval length for u
L = zeros(1, 1, n_z); % array containing the interval lengths
L(:) = [L_u, L_x];

% Hyperparameters of the squared exponential kernel
l = 2; % length scale
sf = 100; % scale factor

% Initialize.
j_vec = zeros(n_phi, 1, n_z); %contains all possible vectors j; j_vec(i, 1, :) corresponds to the vector j in eq. (5) for basis function i
lambda = zeros(n_phi, n_z); % lambda(i, :) corresponds to the vector lambda in eq. (9) (right-hand side) for basis function i

% In the following, all possible vectors j are constructed (i.e., j_vec). The possible combinations correspond to the Cartesian product [1 : n_basis(1)] x ... x [1 : n_basis(end)].
cart_prod_sets = cell(1, n_z); % cell array; cart_prod_sets{i} corresponds to the i-th set to be considered for the Cartesian product, i.e., [1 : n_basis(i)].
for i = 1:n_z
    cart_prod_sets{i} = 1:n_phi_dims(i);
end

subscript_values = zeros(n_z, 1); % contains the equivalent subscript values corresponding to a given single index i
variants = [1, cumprod(n_phi_dims(1:end-1))]; % required to convert the single index i to the equivalent subscript value

% Construct Cartesian product and calculate spectral densities.
for i = 1:n_phi
    % Convert the single index i to the equivalent subscript values.
    remaining = i - 1;
    for j = n_z:-1:1
        subscript_values(j) = floor(remaining/variants(j)) + 1;
        remaining = rem(remaining, variants(j));
    end

    % Fill j_vec with the values belonging to the respective subscript indices.
    for j = 1:n_z
        j_vec(i, 1, j) = cart_prod_sets{j}(subscript_values(j));
    end

    % Calculate the eigenvalue of the Laplace operator corresponding to the vector j_vec[i, 1, :] - see eq. (9) (right-hand side).
    lambda(i, :) = (pi .* squeeze(j_vec(i, 1, :)) ./ (2 * squeeze(L))).^2;
end

% Reverse j_vec.
j_vec = flip(j_vec, 3);

% Define basis functions phi.
% This function evaluates phi_{1 : n_x+n_u} according to eq. (5).
phi_sampling = @(x, u) prod(repelem(L.^(-1 / 2), n_phi, size(u, 2), 1).*sin(pi*repelem(j_vec, 1, size(u, 2), 1).*repelem((permute([u; x], [3, 2, 1]) + L)./repelem(2*L, 1, size(u, 2), 1), n_phi, 1, 1)), 3);

% Prior for Q - inverse Wishart distribution
ell_Q = 10; % degrees of freedom
Lambda_Q = 100 * eye(n_x); % scale matrix

% Prior for A - matrix normal distribution (mean matrix = 0, right covariance matrix = Q (see above), left covariance matrix = V)
% V is derived from the GP approximation according to eq. (8b), (11a), and (9).
V_diag = zeros(length(lambda), 1); % diagonal of V
for i = 1:length(lambda)
    V_diag(i) = sf^2 * sqrt(norm(2*pi*diag(l.^2))) * exp(-(pi^2 * sqrt(lambda(i, :)) * diag(l.^2) * sqrt(lambda(i, :)'))/2);
end
V = diag(V_diag);

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
PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi_sampling, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_mean, x_init_var, g, R);

%% Test the learned models.
% Test the models with the test data by simulating it forward in time.
test_prediction(PG_samples, phi_sampling, g, R, 10, u_test, y_test);

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

% Redefine phi.
% This step is necessary as the highly vectorized function phi_sampling (defined inline above) cannot be used with CasADi MX objects.
% This function is only used for the optimization as it is less efficient.
phi = @(x, u) phi_opt(n_phi, n_z, L, j_vec, x, u);

% Solve the PG OCP.
[u_opt, x_opt, y_opt, J_opt, solve_successful, ~, ~] = solve_PG_OCP(PG_samples, phi, g, R, H, cost_function, scenario_constraints, input_constraints, 'J_u', true, 'K_pre_solve', 10);

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