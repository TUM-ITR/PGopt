function h_scenario = bounded_output(u, x, y, y_min, y_max)
% BOUNDED_OUTPUT This function returns the constraint vector of a single scenario for constraints of the form y_min <= y <= y_max.
%
%   Inputs:
%       y: vector of dimension (n_y, H) containing the outputs of the considered scenario for all timesteps
%       y_min: lower bound for the output y
%       y_max: upper bound for the output y
%
%   Outputs:
%       h_scenario: constraint vector for the scenario
%
% If you define your own constraint function, make sure it can be called with casadi.MX vectors as well as double vectors.

% Initialize.
n_y = size(y, 1);
H = size(y, 2);

% Extend vectors if necessary.
if size(y_min, 2) == 1
    y_min = repelem(y_min, 1, H);
end
if size(y_max, 2) == 1
    y_max = repelem(y_max, 1, H);
end

% Initialize constraint vector.
if isa(u, 'casadi.MX') || isa(x, 'casadi.MX') || isa(y, 'casadi.MX')
    h_scenario = casadi.MX(sum(isfinite(y_min), 'all')+sum(isfinite(y_max), 'all'), 1);
else
    h_scenario = zeros(sum(isfinite(y_min), 'all')+sum(isfinite(y_max), 'all'), 1);
end

% Construct constraint vector - constraints are only considered if they are finite.
i = 1;
for t = 1:H
    for n = 1:n_y
        if isfinite(y_min(n, t))
            h_scenario(i) = y_min(n, t) - y(n, t);
            i = i + 1;
        end
        if isfinite(y_max(n, t))
            h_scenario(i) = y(n, t) - y_max(n, t);
            i = i + 1;
        end
    end
end
end