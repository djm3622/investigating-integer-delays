function [y, excitation] = triangle_pluck(T, L, K, a, p)
% TRIANGLE_PLUCK  Karplus-Strong string driven by -K times a triangle: 0 at sample 0,
%   1 at p, 0 at L. Returns T + 1 samples (one leading zero, then T causal samples)
%   and the L + 1 excitation samples, unscaled. Same loop as COMB_PLUCK, as one filter.
    validateattributes(T, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(L, {'numeric'}, {'scalar', 'integer', '>=', 2});
    validateattributes(p, {'numeric'}, {'scalar', 'integer', '>=', 1, '<', L});
    if T <= L
        error('T must exceed L.');
    end

    n = 0:L;
    excitation = -K * min(n / p, (L - n) / (L - p));
    x = zeros(1, T);
    x(1:L + 1) = excitation;

    b = [a / 2, (a + 1) / 2, 1 / 2];
    denominator = zeros(1, L + 3);
    denominator(1) = 1;
    denominator(2) = a;
    denominator(L + 1:L + 3) = -K * b;

    y = [0; filter(b, denominator, x)'];
end
