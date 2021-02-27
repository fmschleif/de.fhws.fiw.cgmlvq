function D = euclid(X, W, omat)
    % D = (X' - P')' * (omat' * omat) * (X' - P');
    %Note that (B'A') = (AB)', therefore the formula can be written more intuitively in the 
    %simpler form, which is also cheaper to compute:
    D = norm( omat*(X-W).' )^2;
    D = real(D);
end