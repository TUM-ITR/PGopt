function [u_opt, x_opt, y_opt, J_opt, solve_successful, iterations, mu] = solve_PG_OCP(PG_samples, phi, g, R, H, J, h_scenario, h_u, varargin)
%SOLVE_PG_OCP Solve the sample-based optimal control problem.
%
%   Inputs:
%       PG_samples: PG samples
%       phi: basis functions
%       g: observation function
%       R: variance of zero-mean Gaussian measurement noise
%       H: horizon of the OCP
%       J: function with input arguments (u_1:H, x_1:H, y_1:H) (or (u_1:H) if J_u is set true) that returns the cost to be minimized
%       h_scenario: function with input arguments (u_1:H, x_1:H, y_1:H) that returns the constraint vector belonging to a scenario; a feasible solution must satisfy h_scenario <= 0 for all scenarios.
%       h_u: function with input argument u_1:H that returns the constraint vector for u; a feasible solution satisfy yield h_u <= 0
%
%   Variable-length input argument list:
%       J_u: set to true if cost depends only on inputs u. This accelerates the optimization
%       x_vec_0: vector with K*n_x elements containing the initial state of all models - if not provided, the initial states are sampled based on the PGS samples
%       v_vec: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled based on the PGS samples
%       e_vec: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled based on the provided R
%       u_init: initial guess for the optimal trajectory
%       lam_g0: initial guess for the dual variables
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
%       iterations: number of iterations of the solver
%       mu: final mu - parameter of the solver

% Time optimization
optimization_timer = tic;

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
    for k = 1:K
        % Get model.
        A = PG_samples{k}.A;
        Q = PG_samples{k}.Q;
        f = @(x, u) A * phi(x, u);

        % Sample state at t=-1.
        star = systematic_resampling(PG_samples{k}.w_m1, 1);
        x_m1 = PG_samples{k}.x_m1(:, star);

        % Propagate.
        x_vec_0((k - 1)*n_x+1:k*n_x) = f(x_m1, PG_samples{k}.u_m1) + mvnrnd(zeros(1, n_x), Q)';
    end
end

% Sample process noise array v_vec if not provided.
if isempty(v_vec)
    v_vec = zeros(n_x, H, K);
    for k = 1:K
        Q = PG_samples{k}.Q;
        v_vec(:, :, k) = mvnrnd(zeros(1, n_x), Q, H)';
    end
end

% Sample measurement noise array e_vec if not provided.
if isempty(e_vec)
    e_vec = zeros(n_y, H, K);
    for k = 1:K
        e_vec(:, :, k) = mvnrnd(zeros(1, n_y), R, H)';
    end
end

if ~print_progress
    solver_opts.print_level = 0;
end

% Determine initialization.
% If K_pre_solve is provided, a problem that only considers K_pre_solve randomly selected scenarios is solved first to obtain an initialization for the problem with all K scenarios.
% With a good initialization the runtime of the optimization is reduced.
if isempty(u_init)
    if (0 < K_pre_solve) && (K_pre_solve < K)
        % Sample the K_pre_solve scenarios that are considered for the initialization.
        k_pre_solve = randsample(1:K, K_pre_solve);

        % Get the initial states of the considered scenarios.
        x_vec_0_pre_solve = zeros(n_x*K_pre_solve, 1);
        for k = 1:K_pre_solve
            x_vec_0_pre_solve((k - 1)*n_x+1:n_x*k) = x_vec_0((k_pre_solve(k) - 1)*n_x+1:n_x*k_pre_solve(k));
        end

        if print_progress
            fprintf('###### Startet pre-solving step\n')
        end

        % Solve problem that only contains K_pre_solve randomly selected scenarios.
        [u_init, ~, ~, ~, solve_successful, ~, ~] = solve_PG_OCP(PG_samples(k_pre_solve), phi, g, R, H, J, h_scenario, h_u, 'J_u', J_u, 'x_vec_0', x_vec_0_pre_solve, 'v_vec', v_vec(:, :, k_pre_solve), 'e_vec', e_vec(:, :, k_pre_solve), 'K_pre_solve', 0, 'casadi_opts', casadi_opts, 'solver_opts', solver_opts, 'print_progress', print_progress);

        if ~solve_successful
            u_init = zeros(n_u, H);
        end

        if print_progress
            fprintf('###### Pre-solving step complete, switching back to the original problem\n');
        end
    else
        u_init = zeros(n_u, H);
    end
end

% Determine initial guess for X and Y.
X_init = [x_vec_0, zeros(n_x*K, H)]; % initial guess for X
Y_init = zeros(n_y*K, H); % initial guess for Y
for k = 1:K
    % Get model.
    A = PG_samples{k}.A;
    Q = PG_samples{k}.Q;
    f = @(x, u) A * phi(x, u);
    for t = 1:H
        X_init(n_x*(k - 1)+1:n_x*k, t+1) = f(X_init(n_x*(k - 1)+1:n_x*k, t), u_init(:, t)) + v_vec(:, t, k);
        Y_init(n_y*(k - 1)+1:n_y*k, t) = g(X_init(n_x*(k - 1)+1:n_x*k, t), u_init(:, t)) + e_vec(:, t, k);
    end
end

%% Set up OCP.
opti = casadi.Opti();
U = opti.variable(n_u, H);
X = opti.variable(n_x*K, H+1);
Y = opti.variable(n_y*K, H);

% Set the initial state.
opti.subject_to(X(:, 1) == x_vec_0);

% Define objective.
if J_u % cost function depends only on u, i.e., J_max = J
    opti.minimize(J(U));
else
    J_max = opti.variable(1, 1);
    opti.minimize(J_max);
    for k = 1:K
        opti.subject_to(J(U, X(n_x*(k - 1)+1:n_x*k, :), Y(n_y*(k - 1)+1:n_y*k, :)) <= J_max);
    end
end

% Add dynamic and additional constraints for all scenarios.
for k = 1:K
    % Get model.
    A = PG_samples{k}.A;
    Q = PG_samples{k}.Q;
    f = @(x, u) A * phi(x, u);

    for t = 1:H
        % Add dynamic constraints.
        opti.subject_to(X(n_x*(k - 1)+1:n_x*k, t+1) == f(X(n_x*(k - 1)+1:n_x*k, t), U(:, t))+v_vec(:, t, k));
        opti.subject_to(Y(n_y*(k - 1)+1:n_y*k, t) == g(X(n_x*(k - 1)+1:n_x*k, t), U(:, t))+e_vec(:, t, k));
    end

    % Add scenario constraints.
    opti.subject_to(h_scenario(U, X(n_x*(k - 1)+1:n_x*k, :), Y(n_y*(k - 1)+1:n_y*k, :)) <= 0);
end

% Add constraints for the input.
opti.subject_to(h_u(U) <= 0);

% Initialize primal variables.
opti.set_initial(U, u_init);
opti.set_initial(X, X_init);
opti.set_initial(Y, Y_init);

% Set numerical backend.
opti.solver('ipopt', casadi_opts, solver_opts);

% Solve OCP.
if print_progress
    fprintf('### Started optimization algorithm\n');
end

try
    sol = opti.solve();
    time_optimization = toc(optimization_timer);

    if print_progress
        fprintf('### Optimization complete\nRuntime: %.2f s\n', time_optimization);
    end

    % Extract the solution and convert to matrices.
    u_opt = sol.value(U);
    x_opt = reshape(sol.value(X)', [n_x, H + 1, K]);
    y_opt = reshape(sol.value(Y)', [n_y, H, K]);
    J_opt = sol.stats.iterations.obj(end);
    solve_successful = sol.stats.success;
    iterations = sol.stats.iter_count;
    mu = sol.stats.iterations.mu(end);
catch
    time_optimization = toc(optimization_timer);
    warning('Optimization not sucessful!\nRuntime: %.2f s\n', time_optimization);
    u_opt = [];
    x_opt = [];
    y_opt = [];
    J_opt = [];
    solve_successful = false;
    iterations = [];
    mu = [];
end
end