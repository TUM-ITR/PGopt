function epsilon = epsilon(s, K, beta)
%EPSILON Determine the parameter epsilon.
% 1-epsilon corresponds to a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system.
% epsilon is the unique solution over the interval (0,1) of the polynomial equation in the v variable:
% binomial(K, s) * (1 - v)^(K - s) - (beta / K) * ∑_{m = s}^{K - 1} binomial(m, s) * (1 - v)^(m - s) = 0.
%
%   Inputs:
%       s: cardinality of the support sub-sample
%       K: number of scenarios
%       beta: confidence parameter
%
%   Outputs:
%       epsilon: probability epsilon
%
% This function is based on the paper
%    S. Garatti and M. C. Campi, "Risk and complexity in scenario optimization," Mathematical Programming, vol. 191, no. 1, pp. 243–279, 2022.
% and the code provided in the appendix.

alphaU = 1 - betaincinv(beta, K-s+1, s);
m1 = s:1:K;
aux1 = sum(triu(log(ones(K-s+1, 1)*m1), 1), 2);
aux2 = sum(triu(log(ones(K-s+1, 1)*(m1 - s)), 1), 2);
coeffs1 = aux2 - aux1;
m2 = K + 1:1:4 * K;
aux3 = sum(tril(log(ones(3*K, 1)*m2)), 2);
aux4 = sum(tril(log(ones(3*K, 1)*(m2 - s))), 2);
coeffs2 = aux3 - aux4;

t1 = 0;
t2 = 1 - alphaU;
poly1 = 1 + beta / (2 * K) - beta / (2 * K) * sum(exp(coeffs1-(K - m1')*log(t1))) - beta / (6 * K) * sum(exp(coeffs2+(m2' - K)*log(t1)));
poly2 = 1 + beta / (2 * K) - beta / (2 * K) * sum(exp(coeffs1-(K - m1')*log(t2))) - beta / (6 * K) * sum(exp(coeffs2+(m2' - K)*log(t2)));
if ~((poly1 * poly2) > 0)
    while t2 - t1 > 1e-10
        t = (t1 + t2) / 2;
        polyt = 1 + beta / (2 * K) - beta / (2 * K) * sum(exp(coeffs1-(K - m1')*log(t))) - beta / (6 * K) * sum(exp(coeffs2+(m2' - K)*log(t)));
        if polyt > 0
            t2 = t;
        else
            t1 = t;
        end
    end
    epsilon = t1;
end
end
