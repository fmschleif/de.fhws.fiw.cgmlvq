% Wrapper around "ifft" to retrieve the original time domain signal "y", given 
% Fourier coefficients "X" and the length "L" of the original time domain signal.

function [ y ] = iFourier( X, L )
    y = zeros(size(X,1), L);    %the size of the original data matrix 
    enabled = zeros(1, L);
    r = size(X,2) - 1;
    enabled(1:1+r) = 1; 
    enabled(end-r+1:end) = 1;
    
    if size(y(:,enabled==1),2) == size([X fliplr(conj(X(:,2:end)))],2)
        y(:,enabled==1) = [X fliplr(conj(X(:,2:end)))];
    end
    
    y = ifft(y, [], 2);
end

