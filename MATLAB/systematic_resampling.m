function idx = systematic_resampling(W, N)
%SYSTEMATIC_RESAMPLING Sample N indices according to the weights W.
%
%   Inputs:
%       W: vector containing the weights of the particles
%       N: number of indices to be sampled
%
%   Outputs:
%       idx: vector containing the indices of the resampled indices
%
% This function is based on the paper
%   A. Svensson and T. B. Schön, "A flexible state–space model for learning nonlinear dynamical systems", Automatica, vol. 80, pp. 189– 199, 2017.
% and the code provided in the supplementary material.

% Normalize weights.
W = W / sum(W);

u = 1 / N * rand;
idx = zeros(N, 1); % array containing the sampled indices
q = 0;
n = 0;
for i = 1:N
    while q < u
        n = n + 1;
        q = q + W(n);
    end
    idx(i) = n;
    u = u + 1 / N;
end