function [y, excitation] = comb_pluck(T, L, K, a, p, excitation_length)
% COMB_PLUCK  Karplus-Strong string driven by the comb pluck -K (delta[n] - delta[n-p]).
%   [y, excitation] = comb_pluck(T, L, K, a, p, excitation_length) returns T + 1 samples
%   (one leading zero, then T causal samples) and the excitation, unscaled.
%   The loop is w = x + K y[n-L], a two-tap average j = (w[n] + w[n-1]) / 2 and the
%   all-pass y = a j[n] + j[n-1] - a y[n-1].
    excitation = zeros(1, excitation_length);
    excitation(1) = -K;
    excitation(p + 1) = K;

    x = zeros(1, T);
    x(1:min(T, excitation_length)) = excitation(1:min(T, excitation_length));

    w = zeros(1, T + L);
    j = zeros(1, T + L);
    y = zeros(1, T + L);
    for n = 1:T
        m = n + L;
        w(m) = x(n) + K * y(n);
        j(m) = 0.5 * w(m) + 0.5 * w(m - 1);
        y(m) = a * j(m) + j(m - 1) - a * y(m - 1);
    end
    y = y(L:L + T)';
end
