function [u_opt, x_opt, y_opt, J_opt, solve_successful, s, epsilon_prob, epsilon_perc, time_first_solve, time_guarantees, num_failed_optimizations] = solve_PG_OCP_greedy_guarantees(PG_samples, phi, g, R, H, J, h_scenario, h_u, beta, varargin)
%SOLVE_PG_OCP_GREEDY_GUARANTEES Solve the sample-based optimal control problem and determine a support sub-sample with cardinality s via a greedy constraint removal.
% Based on the cardinality s, a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system is calculated.
%
%   Inputs:
%       PG_samples: PG samples
%       phi: basis functions
%       g: observation function
%       R: variance of zero-mean Gaussian measurement noise
%       H: horizon of the OCP
%       J: function with input arguments (u_1:H, x_1:H, y_1:H) (or (u_1:H) if J_u is set to true) that returns the cost to be minimized
%       h_scenario: function with input arguments (u_1:H, x_1:H, y_1:H) that returns the constraints belonging to a scenario; a feasible solution must yield h_scenario <= 0 for all scenarios.
%       h_u: function with input argument u_1:H that returns constraints that only dependent on u; a feasible solution must yield h_u <= 0
%       beta: confidence parameter
%
%   Variable-length input argument list:
%       J_u: set to true if cost depends only on inputs u. This accelerates the optimization
%       x_vec_0: vector with K*n_x elements containing the initial state of all models - if not provided, the initial states are sampled based on the PGS samples
%       v_vec: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled based on the PGS samples
%       e_vec: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled based on the provided R
%       u_init: initial guess for the optimal trajectory
%       K_pre_solve: if K_pre_solve > 0, an initial guess for the optimal trajectory is obtained by solving the OCP with only K_pre_solve < K models
%       casadi_opts: CasADi options; see CasADi documentation
%       solver_opts: solver options; see IPOPT documentation
%       print_progress: if set to true, the progress is printed
%
%   Outputs:
%       u_opt: optimal u
%       x_opt: state predictions for input u_opt
%       y_opt: output predictions for input u_opt
%       J_opt: worst-case prediction for input u_opt
%       solve_successful: true if a feasible solution was found, false otherwise
%       s: cardinality of support sub-sample
%       epsilon_prob: probability epsilon
%       epsilon_perc: probability epsilon in percent
%       time_first_solve: time to compute optimal u
%       time_guarantees: time to compute guarantees
%       num_failed_optimizations: number of failed optimizations during the computation of guarantees

% Time first optimization.
first_solve_timer = tic();

% Get number of states, etc.
K = length(PG_samples);
n_u = size(PG_samples{1}.u_m1, 1);
n_x = size(PG_samples{1}.x_m1(:, 1), 1);
n_y = size(R, 1);

%% Read variable-length input argument list.
% Default values
J_u = false;
x_vec_0 = [];
v_vec = [];
e_vec = [];
u_init = [];
K_pre_solve = 0;
active_scenarios = 1:K;
casadi_opts = struct('expand', 1);
solver_opts = struct('linear_solver', 'ma57', 'max_iter', 10000);
print_progress = true;

% Read variable-length input argument list.
for i = 1:2:length(varargin)
    if strcmp('J_u', varargin{i})
        J_u = varargin{i+1};
    elseif strcmp('x_vec_0', varargin{i})
        x_vec_0 = varargin{i+1};
    elseif strcmp('v_vec', varargin{i})
        v_vec = varargin{i+1};
    elseif strcmp('e_vec', varargin{i})
        e_vec = varargin{i+1};
    elseif strcmp('u_init', varargin{i})
        u_init = varargin{i+1};
    elseif strcmp('K_pre_solve', varargin{i})
        K_pre_solve = varargin{i+1};
    elseif strcmp('active_constraints', varargin{i})
        active_scenarios = varargin{i+1};
    elseif strcmp('casadi_opts', varargin{i})
        casadi_opts = varargin{i+1};
    elseif strcmp('solver_opts', varargin{i})
        solver_opts = varargin{i+1};
    elseif strcmp('print_progress', varargin{i})
        print_progress = varargin{i+1};
    end
end

% Sample initial state x_vec_0 if not provided.
if isempty(x_vec_0)
    x_vec_0 = zeros(n_x*K, 1);
    for i = 1:K
        % Get model.
        A = PG_samples{i}.A;
        Q = PG_samples{i}.Q;
        f = @(x, u) A * phi(x, u);

        % Sample state at t=-1.
        star = systematic_resampling(PG_samples{i}.w_m1, 1);
        x_m1 = PG_samples{i}.x_m1(:, star);

        % Propagate.
        x_vec_0((i - 1)*n_x+1:i*n_x) = f(x_m1, PG_samples{i}.u_m1) + mvnrnd(zeros(1, n_x), Q)';
    end
end

% Sample process noise array v_vec if not provided.
if isempty(v_vec)
    v_vec = zeros(n_x, H, K);
    for i = 1:K
        Q = PG_samples{i}.Q;
        v_vec(:, :, i) = mvnrnd(zeros(1, n_x), Q, H)';
    end
end

% Sample measurement noise array e_vec if not provided.
if isempty(e_vec)
    e_vec = zeros(n_y, H, K);
    for i = 1:K
        e_vec(:, :, i) = mvnrnd(zeros(1, n_y), R, H)';
    end
end

if ~print_progress
    solver_opts.print_level = 0;
end

num_failed_optimizations = 0; % number of failed optimizations during the computation of the guarantees

%% Solve OCP.
% The following optimization is only done to obtain an initialization for all following optimizations.
fprintf("### Started optimization of fully constrained problem\n");

[u_init, ~, ~, ~, ~, ~, mu] = solve_PG_OCP(PG_samples, phi, g, R, H, J, h_scenario, h_u, 'J_u', J_u, 'x_vec_0', x_vec_0, 'v_vec', v_vec, 'e_vec', e_vec, 'u_init', u_init, 'K_pre_solve', K_pre_solve, 'casadi_opts', casadi_opts, 'solver_opts', solver_opts, 'print_progress', print_progress);

% The OCP is solved again with initialization to obtain the optimal input u_opt.
solver_opts.max_iter = 200; % reduce number of iterations
solver_opts.mu_init = mu;
disp("### Started optimization of fully constrained problem with initialization");
[u_opt, x_opt, y_opt, J_opt, solve_successful, iter, ~] = solve_PG_OCP(PG_samples, phi, g, R, H, J, h_scenario, h_u, 'J_u', J_u, 'x_vec_0', x_vec_0, 'v_vec', v_vec, 'e_vec', e_vec, 'u_init', u_init, 'casadi_opts', casadi_opts, 'solver_opts', solver_opts, 'print_progress', print_progress);

%% Determine guarantees.
% If a feasible u_opt is found, probabilistic constraint satisfaction guarantees are derived by greedily removing constraints to determine a support sub-sample S.
if solve_successful
    time_first_solve = toc(first_solve_timer);

    % Time computation of guarantees.
    guarantees_timer = tic();

    % Determine support sub-samples and guarantees for the generalization of the resulting input trajectory.
    disp("### Started search for support sub-sample");

    % Reduce number of iterations - if the number of iterations of the original OCP is exceeded, the solution will likely be different, and the optimization can be stopped.
    solver_opts.max_iter = 2 * iter;

    % Sort scenarios according to the distance to the constraint boundary - removing the scenarios with the largest distance to the constraint boundary first usually yields better results.
    h_scenario_max = zeros(K, 1); % minimum distance of the scenarios to the constraint boundary
    for i = 1:K
        h_scenario_max(i) = max(h_scenario(u_opt, x_opt(:, :, i), y_opt(:, :, i)));
    end
    [~, scenarios_sorted] = sort(h_scenario_max); % sort scenarios

    % Pre-allocate.
    active_scenarios = scenarios_sorted;

    % Greedily remove constraints and check whether the solution changes to determine a support sub-sample.
    for i = 1:K
        % Print progress.
        fprintf("Started optimization with new constraint set\nIteration: %i/%i\n", i, K);

        % Temporarily remove the constraints corresponding to the PG samples with index i from the constraint set.
        temp_constraints = active_scenarios(active_scenarios ~= scenarios_sorted(i));

        % Get the initial states of the active scenarios.
        x_vec_0_temp = zeros(n_x*length(temp_constraints), 1);
        for k = 1:length(temp_constraints)
            x_vec_0_temp((k - 1)*n_x+1:n_x*k) = x_vec_0((temp_constraints(k) - 1)*n_x+1:n_x*temp_constraints(k));
        end

        % Solve the OCP with reduced constraint set.
        [u_opt_temp, ~, ~, J_opt_temp, solve_successful_temp, ~, ~] = solve_PG_OCP(PG_samples(temp_constraints), phi, g, R, H, J, h_scenario, h_u, 'J_u', J_u, 'x_vec_0', x_vec_0_temp, 'v_vec', v_vec(:, :, temp_constraints), 'e_vec', e_vec(:, :, temp_constraints), 'u_init', u_init, 'casadi_opts', casadi_opts, 'solver_opts', solver_opts, 'print_progress', print_progress);

        % If the optimization is successful and the solution does not change, permanently remove the constraints corresponding to the PG samples with index i from the constraint set.
        % A valid subsample has the same local minimum. However, since the numerical solver does not reach this minimum exactly, a threshold value is used here to check whether the solutions are the same.
        if solve_successful_temp && all(abs(u_opt_temp-u_opt) < 1e-6) && (abs(J_opt_temp-J_opt) < 1e-6)
            active_scenarios = temp_constraints;
        elseif ~solve_successful_temp
            warning("Optimization with temporarily removed constraints failed. Proceeding with next candidate for a support sub-sample.");
            num_failed_optimizations = num_failed_optimizations + 1;
        end
    end

    % Determine the cardinality of the support sub-sample.
    s = length(active_scenarios);

    % Based on the cardinality of the support sub-sample, determine the parameter epsilon.
    % 1-epsilon corresponds to a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system.
    epsilon_prob = epsilon(s, K, beta);
    epsilon_perc = epsilon_prob * 100;

    % Print s, epsilon, and runtime.
    time_guarantees = toc(guarantees_timer);
    fprintf("### Support sub sample found\nCardinality of the support sub-sample (s): %i\nMax. constraint violation probability (1-epsilon): %.2f %%\nTime to compute u*: %.2f s\nTime to compute 1-epsilon: %.2f s\n", s, 100-epsilon_perc, time_first_solve, time_guarantees)
else
    % In case the initial problem is infeasible, skip the computation of guarantees.
    warning("No feasible solution found for the initial problem. Skipping computation of guarantees.");
    u_opt = [];
    x_opt = [];
    y_opt = [];
    J_opt = [];
    J_opt = [];
    epsilon_prob = NaN;
    epsilon_perc = NaN;
    time_guarantees = NaN;
    time_first_solve = NaN;
    num_failed_optimizations = 0;
end
end