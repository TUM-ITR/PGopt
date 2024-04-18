function plot_autocorrelation(PG_samples, varargin)
% PLOT_AUTOCORRELATION Plot the autocorrelation function (ACF) of the PG samples.
% This might be helpful when adjusting the thinning parameter k_d.
%
%   Inputs:
%       PG_samples: PG samples
%
%   Variable-length input argument list:
%       max_lag: maximum lag at which to calculate the ACF

% Get number of models.
K = size(PG_samples, 1);

% Default values
max_lag = K - 1;

% Read variable-length input argument list.
for i = 1:2:length(varargin)
    if strcmp('max_lag', varargin{i})
        max_lag = varargin{i+1};
    end
end

% Get number of parameters of the PG samples.
number_of_variables = numel(PG_samples{1}.A) + numel(PG_samples{1}.Q) + size(PG_samples{1}.x_m1, 1);

% Fill matrix with the series of the parameters of the PG samples.
signal_matrix = zeros(K, number_of_variables);
for i = 1:K
    % Sample initial state.
    star = systematic_resampling(PG_samples{i}.w_m1, 1);
    x_m1 = PG_samples{i}.x_m1(:, star);
    signal_matrix(i, :) = [PG_samples{i}.A(:); PG_samples{i}.Q(:); x_m1(:)];
end

% Calculate the autocorrelation.
autocorrelation = zeros(max_lag+1, number_of_variables);
for i = 1:number_of_variables
    [autocorrelation(:, i), ~] = autocorr(signal_matrix(:, i), 'NumLags', max_lag);
end


% Plot the ACF.
figure();
hold on;
for i = 1:number_of_variables
    % Plot the ACF of the elements of A.
    if i == 1
        plot(0:max_lag, autocorrelation(:, i), 'r', 'LineWidth', 2, 'DisplayName', 'A');
    elseif 1 < i && i <= numel(PG_samples{1}.A)
        plot(0:max_lag, autocorrelation(:, i), 'r', 'LineWidth', 2, 'HandleVisibility', 'off');
        % Plot the ACF of the elements of Q.
    elseif i == numel(PG_samples{1}.A) + 1
        plot(0:max_lag, autocorrelation(:, i), 'b', 'LineWidth', 2, 'DisplayName', 'Q');
    elseif numel(PG_samples{1}.A) + 1 < i && i <= numel(PG_samples{1}.A) + numel(PG_samples{1}.Q)
        plot(0:max_lag, autocorrelation(:, i), 'b', 'LineWidth', 2, 'HandleVisibility', 'off');
        % Plot the ACF of the elements of x_t-1
    elseif i == numel(PG_samples{1}.A) + numel(PG_samples{1}.Q) + 1
        plot(0:max_lag, autocorrelation(:, i), 'Color', '#77AC30', 'LineWidth', 2, 'DisplayName', 'x');
    elseif numel(PG_samples{1}.A) + numel(PG_samples{1}.Q) + 1 < i
        plot(0:max_lag, autocorrelation(:, i), 'Color', '#77AC30', 'LineWidth', 2, 'HandleVisibility', 'off');
    end
end

title('Autocorrelation Function (AFC)');
xlabel('Lag');
ylabel('AFC');
legend('show');
grid on;
end