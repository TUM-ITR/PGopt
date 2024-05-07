function [model_param] = MNIW_sample(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T)
%MNIW_SAMPLE Sample new model parameters (A, Q) from the conditional distribution p(A, Q | x_T:-1), which is a matrix normal inverse Wishart (MNIW) distribution.
%
%   Inputs:
%       Phi: statistic; see paper below for definition
%       Psi: statistic; see paper below for definition
%       Sigma: statistic; see paper below for definition
%       V: left covariance matrix of MN prior on A
%       Lambda_Q: scale matrix of IW prior on Q
%       ell_Q: degrees of freedom of IW prior on Q
%       T: length of the training trajectory
%
%   Outputs:
%       model_param: struct containing the parameters A and Q
%
% This function is based on the paper
%   A. Svensson and T. B. Schön, "A flexible state–space model for learning nonlinear dynamical systems", Automatica, vol. 80, pp. 189– 199, 2017.
% and the code provided in the supplementary material.

n_x = size(Phi, 1); % number of states
n_basis = size(V, 1); % number of basis functions

% Update statistics (only relevant if mean matrix M of MN prior on A is not zero).
Phibar = Phi; % + (M / V) * M';
Psibar = Psi; % + M / V;

% Calculate the components of the posterior MNIW distribution over model parameters.
Sigbar = Sigma + eye(n_basis) / V;
cov_M = Lambda_Q + Phibar - (Psibar / Sigbar) * Psibar'; % scale matrix of posterior IW distribution
cov_M_sym = 0.5 * (cov_M + cov_M'); % to ensure matrix is symmetric
post_mean = Psibar / Sigbar; % mean matrix of posterior MN distribution
left_cov = eye(n_basis) / Sigbar; % left covariance matrix of posterior MN distribution
left_cov_sym = 0.5 * (left_cov + left_cov'); % to ensure matrix is symmetric

% Sample Q from posterior IW distribution.
Q = iwishrnd(cov_M_sym, ell_Q+T*n_x);

% Sample A from posterior MN distribution.
% See Wikipedia entry "Matrix normal distribution".
X = randn(n_x, n_basis);
A = post_mean + chol(Q) * X * chol(left_cov_sym);

% Return parameters A and Q as a struct.
model_param.A = A;
model_param.Q = Q;
end
