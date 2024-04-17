function test_prediction(PG_samples, phi, g, R, k_n, u_test, y_test)
%TEST_PREDICTION Simulate the PGS samples forward in time and compare the predictions to the test data.
%
%   Inputs:
%       PG_samples: PG samples
%       phi: basis functions
%       g: observation function
%       R: variance of zero-mean Gaussian measurement noise
%       k_n: each model is simulated k_n times
%       u_test: test input
%       y_test: test output

fprintf('### Testing model\n')

% Get number of models, etc.
K = size(PG_samples, 1);
n_x = size(PG_samples{1}.A, 1);
n_y = size(y_test, 1);
T_test = size(y_test, 2);

% Pre-allocate.
x_test_sim = zeros(n_x, T_test+1, K, k_n);
y_test_sim = zeros(n_y, T_test, K, k_n); % zeros(T_test,n_y,K,k_n);

%% Simulate models forward.
% If the Parallel Computing Toolbox is available, the for loop can be replaced with a parfor loop to decrease the runtime.
for k = 1:K
    % Get current model.
    A = PG_samples{k}.A;
    Q = PG_samples{k}.Q;
    f = @(x, u) A * phi(x, u);

    % Simulate each model k_n times.
    for kn = 1:k_n
        % Pre-allocate.
        x_loop = zeros(n_x, T_test+1);
        y_loop = zeros(n_y, T_test);

        % Sample initial state.
        star = systematic_resampling(PG_samples{k}.w_m1, 1);
        x_m1 = PG_samples{k}.x_m1(:, star);
        x_loop(:, 1) = f(x_m1, PG_samples{k}.u_m1) + mvnrnd(zeros(1, n_x), Q)';

        % Simulate model forward.
        for t = 1:T_test
            x_loop(:, t+1) = f(x_loop(:, t), u_test(t)) + mvnrnd(zeros(1, n_x), Q)';
            y_loop(:, t) = g(x_loop(:, t), u_test(t)) + mvnrnd(zeros(1, n_y), R)';
        end

        % Store trajectory.
        x_test_sim(:, :, k, kn) = x_loop;
        y_test_sim(:, :, k, kn) = y_loop;
    end
end

% Reshape.
x_test_sim = reshape(x_test_sim, [n_x, T_test + 1, K * k_n]);
y_test_sim = reshape(y_test_sim, [n_y, T_test, K * k_n]);

fprintf('### Testing complete\n')

% Plot results.
plot_predictions(y_test_sim, y_test, 'plot_percentiles', true)

% Compute and print RMSE.
mean_rmse = sqrt(mean((squeeze(y_test_sim) - y_test').^2, 'all'));
fprintf('Mean rmse: %.2f\n', mean_rmse);
end