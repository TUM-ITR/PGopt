function phi = phi_opt(n_phi, n_z, L, j_vec, x, u)
% PHI_OPT Evaluate basis functions phi at at (x, u).
% This function can be used with CasADi MX objects.
% This function should only be used for the optimization and not during learning since it is less efficient.
%
%   Inputs:
%       n_basis_tot: total number of basis functions
%       n_z: number of augmented states
%       L: interval lengths
%       jv: matrix containing all elements of the Cartesian product of the
%           vectors 1:n_basis_1, ..., 1:n_basis_n_z, where n_basis_i is the
%           number of basis functions for augmented state dimension i.
%       x: state vector
%       u: input vector
%
%   Outputs:
%       phi: value of the basis functions evaluated at (x,u)
%
% This function is based on the paper
%   A. Svensson and T. B. Schön, "A flexible state–space model for learning nonlinear dynamical systems", Automatica, vol. 80, pp. 189– 199, 2017.
% and the code provided in the supplementary material.

% Initialize.
z = [u; x]; % augmented state
phi = ones(n_phi, 1);

for k = 1:n_z
    phi = phi .* (1 ./ (sqrt(L(:, :, k))) * sin((pi * j_vec(:, :, k) .* (z(k) + L(:, :, k)))./(2 * L(:, :, k))));
end
end