function h_u = bounded_input(u, u_min, u_max)
% BOUNDED_INPUT This function returns the constraint vector for constraints of the form u_min <= u <= u_max.
%
%   Inputs:
%       u: vector of dimension (n_u, H) containing the control inputs for all timesteps
%       u_min: lower bound for the control input u
%       u_max: upper bound for the control input u
%
%   Outputs:
%       h_u: constraint vector
%
% If you define your own constraint function, make sure it can be called with casadi.MX vectors as well as double vectors.

% Initialize.
n_u = size(u, 1);
H = size(u, 2);

% Extend vectors if necessary.
if size(u_min, 2) == 1
    u_min = repelem(u_min, 1, H);
end
if size(u_max, 2) == 1
    u_max = repelem(u_max, 1, H);
end

% Initialize constraint vector.
if isa(u, 'casadi.MX')
    h_u = casadi.MX(sum(isfinite(u_min), 'all')+sum(isfinite(u_max), 'all'), 1);
else
    h_u = zeros(sum(isfinite(u_min), 'all')+sum(isfinite(u_max), 'all'), 1);
end

% Construct constraint vector - constraints are only considered if they are finite.
i = 1;
for t = 1:H
    for n = 1:n_u
        if isfinite(u_min(n, t))
            h_u(i) = u_min(n, t) - u(n, t);
            i = i + 1;
        end
        if isfinite(u_max(n, t))
            h_u(i) = u(n, t) - u_max(n, t);
            i = i + 1;
        end
    end
end
end