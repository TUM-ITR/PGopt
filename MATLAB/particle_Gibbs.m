function PG_samples = particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_mean, x_init_var, g, R, varargin)
%PARTICLE_GIBBS Run particle Gibbs sampler with ancestor sampling to obtain samples (A, Q, x_T:-1) from the joint parameter and state posterior distribution p(A, Q, x_T:-1 | D=(u_training, y_training)).
%
%   Inputs:
%       u_training: training input trajectory
%       y_training: training output trajectory
%       K: number of models/scenarios to be sampled
%       K_b: length of the burn in period
%       k_d: number of models/scenarios to be skipped to decrease correlation (thinning)
%       N: number of particles
%       phi: basis functions
%       Lambda_Q: scale matrix of IW prior on Q
%       ell_Q: degrees of freedom of IW prior on Q
%       Q_init: initial value of Q
%       V: left covariance matrix of MN prior on A
%       A_init: initial value of A
%       x_init_mean: mean of normally distributed initial state
%       x_init_var: variance of normally distributed initial state
%       g: observation function
%       R: variance of zero-mean Gaussian measurement noise
%
%   Variable-length input argument list:
%       x_prim: prespecified trajectory for the first iteration
%
%   Outputs:
%       PG_samples: cell array containing the different PG samples
%
% This function is based on the papers
%   F. Lindsten and M. I. Jordan T. B. Schön, "Ancestor sampling for Particle Gibbs", Proceedings of the 2012 Conference on Neural Information Processing Systems, Lake Taho, USA, 2012.
%   A. Svensson and T. B. Schön, "A flexible state–space model for learning nonlinear dynamical systems", Automatica, vol. 80, pp. 189– 199, 2017.
% and the code provided in the supplementary material.

%% Initialization
% Total number of models
K_total = K_b + 1 + (K - 1) * (k_d + 1);

% Get number of states, etc.
n_x = size(A_init, 1);
n_u = size(u_training, 1);
n_y = size(y_training, 1);
T = size(y_training, 2);

% Initialize.
PG_samples = cell(K, 1);
w = zeros(T, N);
x_pf = zeros(n_x, N, T);
a = zeros(T, N);
PG_samples{1}.A = A_init;
PG_samples{1}.Q = Q_init;

% Define the prespecified trajectory for the first iteration if not provided.
x_prim = zeros(n_x, 1, T);
% Read variable-length input argument list.
for i = 1:2:length(varargin)
    if strcmp('x_prim', varargin{i})
        x_prim = varargin{i+1};
    end
end

%% Particle Gibbs Sampling
% Time PGS sampling.
learning_timer = tic;

fprintf('### Started learning algorithm\n')

for k = 1:K_total
    % Get current model.
    A = PG_samples{k}.A;
    Q = PG_samples{k}.Q;
    f = @(x, u) A * phi(x, u);

    % Initialize particle filter with ancestor sampling.
    x_pf(:, end, :) = x_prim;

    % Sample initial states.
    for n = 1:N - 1
        x_pf(:, n, 1) = normrnd(x_init_mean, x_init_var);
    end

    % Particle filter (resampling, propagation, and ancestor sampling)
    for t = 1:T
        if t >= 2
            if k > 1 % Run the conditional PF with ancestor sampling.
                % Resample particles (= sample ancestors).
                a(t, 1:N-1) = systematic_resampling(w(t-1, :), N-1);

                % Propagate resampled particles.
                x_pf(:, 1:N-1, t) = f(x_pf(:, a(t, 1:N-1), t-1), repmat(u_training(:, t-1), 1, N-1)) + mvnrnd(zeros(n_x, 1), Q, N-1)';

                % Sample ancestors of prespecified trajectory x_prim.
                waN = w(t-1, :) .* mvnpdf(f(x_pf(:, :, t-1), repmat(u_training(:, t-1), 1, N))', x_pf(:, N, t)', Q)';
                waN = waN ./ sum(waN);
                a(t, N) = systematic_resampling(waN, 1);

            else % Run a standard PF on first iteration.
                % Resample particles.
                a(t, :) = systematic_resampling(w(t-1, :), N);

                % Propagate resampled particles.
                x_pf(:, :, t) = f(x_pf(:, a(t, :), t-1), repmat(u_training(:, t-1), 1, N)) + mvnrnd(zeros(n_x, 1), Q, N)';
            end
        end

        % PF weight update based on measurement model (logarithms are used for numerical reasons).
        if n_y == 1 % scalar output
            log_w = -(g(x_pf(:, :, t), repmat(u_training(:, t), 1, N)) - y_training(t)).^2 / 2 / R; % logarithm of normal distribution pdf (ignoring scaling factors)
        else % vector-valued output
            log_w = -sum((y_training(:, t) - g(x_pf(:, :, t), repmat(u_training(:, t), 1, N))).*(R \ (y_training(:, t) - g(x_pf(:, :, t)))), 1) / 2; % logarithm of multivariate normal distribution pdf (ignoring scaling factors)
        end
        w(t, :) = exp(log_w-max(log_w));
        w(t, :) = w(t, :) / sum(w(t, :));
    end

    % Store weights and corresponding states in the last timestep of the training dataset (t=-1).
    % All particles are stored to increase the robustness in case the forward simulation is repeated multiple times.
    PG_samples{k}.w_m1 = squeeze(w(end, :));
    PG_samples{k}.x_m1 = squeeze(x_pf(:, :, end));

    % Store the control input in the last timestep of the training dataset (t=-1).
    PG_samples{k}.u_m1 = u_training(:, end);

    % Sample state trajectory x_T:-1 to condition on.
    star = systematic_resampling(w(end, :), 1);
    x_prim(:, 1, T) = x_pf(:, star, T);
    for t = T - 1:-1:1
        star = a(t+1, star);
        x_prim(:, 1, t) = x_pf(:, star, t);
    end

    % Sample new model parameters conditional on sampled trajectory, i.e., sample from p(A, Q | x_T:-1).
    zeta = squeeze(x_prim(:, 1, 2:T));
    z = phi(squeeze(x_prim(:, 1, 1:T-1)), u_training(:, 1:T-1));
    Phi = zeta * zeta'; % statistic; see paper "A flexible state-space model for learning nonlinear dynamical systems"
    Psi = zeta * z'; % statistic
    Sigma = z * z'; % statistic
    PG_samples{k+1} = MNIW_sample(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T-1); % sample new model parameters

    % Print progress.
    fprintf('Iteration %i/%i\n', k, K_total);
end

% Print runtime.
time_learning = toc(learning_timer);
fprintf('### Learning complete\nRuntime: %.2f s\n', time_learning)

% Discard burn-in period and perform thinning.
model_indices = (K_b + 1):(k_d + 1):K_total; % indices of "valid" samples
PG_samples = PG_samples(model_indices);
end