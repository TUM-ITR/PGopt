function plot_predictions(y_pred, y_test, varargin)
%PLOT_PREDITIONS Plot the predictions and the test data.
%
%   Inputs:
%       y_pred: matrix containing the output predictions
%       y_test: test output trajectory
%
%   Variable-length input argument list:
%       plot_percentiles: if set to true, percentiles are plotted
%       y_min: min output to be plotted as constraint
%       y_max: max output to be plotted as constraint

% Default values
plot_percentiles = false;
y_min = [];
y_max = [];

% Read variable-length input argument list.
for i = 1:2:length(varargin)
    if strcmp('plot_percentiles', varargin{i})
        plot_percentiles = varargin{i+1};
    elseif strcmp('y_min', varargin{i})
        y_min = varargin{i+1};
    elseif strcmp('y_max', varargin{i})
        y_max = varargin{i+1};
    end
end

% Get prediction horizon and number of outputs.
[n_y, T_pred] = size(y_test);
t_pred = 0:T_pred - 1;

% Plot the predictions and the test data for all output dimensions.
for i = 1:n_y
    % Calculate median, mean, maximum, and minimum prediction.
    y_pred_med = median(y_pred(i, :, :), 3);
    y_pred_mean = mean(y_pred(i, :, :), 3);

    y_pred_max = max(y_pred(i, :, :), [], 3);
    y_pred_min = min(y_pred(i, :, :), [], 3);

    % Calculate percentiles.
    y_pred_09 = quantile(y_pred(i, :, :), 0.9, 3);
    y_pred_01 = quantile(y_pred(i, :, :), 0.1, 3);

    % Plot range of predictions.
    figure;
    fill([t_pred, flip(t_pred)], [y_pred_min, flip(y_pred_max)], 0.7*[1, 1, 1], 'linestyle', 'none', 'DisplayName', 'all predictions');
    hold on;

    % Plot percentiles.
    if plot_percentiles
        fill([t_pred, flip(t_pred)], [y_pred_01, flip(y_pred_09)], 0.5*[1, 1, 1], 'linestyle', 'none', 'DisplayName', '10% perc. - 90% perc.');
    end

    % Plot true output.
    plot(t_pred, y_test(i, :), 'linewidth', 2, 'DisplayName', 'true output');

    % Plot median/mean prediction.
    % plot(0:T_pred-1, y_pred_med, 'linewidth', 2, 'DisplayName', 'median prediction');
    plot(t_pred, y_pred_mean, 'linewidth', 2, 'DisplayName', 'mean prediction');

    % Plot constraints.
    if ~isempty(y_min)
        non_inf_idx = ~isinf(y_min);
        fill([t_pred(non_inf_idx), flip(t_pred(non_inf_idx))], [y_min(non_inf_idx), flip(min(min(y_pred_min, y_test))*ones(1, sum(non_inf_idx)))], 'r', 'linestyle', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'constraints');
    end
    if ~isempty(y_max)
        non_inf_idx = ~isinf(y_min);
        fill([t_pred(non_inf_idx), flip(t_pred(non_inf_idx))], [y_max(non_inf_idx), flip(max(max(y_pred_max, y_test))*ones(1, sum(non_inf_idx)))], 'r', 'linestyle', 'none', 'FaceAlpha', 0.35, 'DisplayName', 'constraints');
    end

    % Add title, labels...
    if n_y > 1
        title(['y_', num2str(i), ': predicted output vs. true output']);
        ylabel(['y_', num2str(i)]);
    else
        title('predicted output vs. true output');
        ylabel('y');
    end
    xlabel('t');
    legend('Location', 'northwest');
    ylim([min(min(y_pred_min, y_test)), max(max(y_pred_max, y_test))]);
    grid on;
    hold off;
end
end