%Wrapper around "fft" to obtain Fourier series of "x" truncated at 
%"r" coefficients. Ignores the symmetric part.
%of the spectrum.

function [ Y ] = Fourier( x, r )
    Y = fft(x, [], 2);
    enabled = zeros(1,size(Y,2));
    enabled(1:r+1) = 1; %preserve DC and r positive frequencies from low to high
    Y = Y(:, enabled==1); 
% below code is experimental, was used for cutting different frequencies
% etc, and also different smoothing technique.
% rectangle = zeros(1,size(Y,2));
%     enabled = zeros(1,size(Y,2));
%     if isempty(relevs)
%         enabled(1:r+1) = 1; %preserve DC and r positive frequencies from low to high
%     else
%         for i=1:r
%             [~, index] = max(relevs); %get the most relevant frequency
%             enabled(index) = 1;
%             relevs(index) = -1; %to get the next maximum
%         end
%     end
%     
%     if strcmp(method,'gauss')
%        sigma = 8;
%        rectangle(1:r+1) = exp(-(1:r+1).^ 2 / (2 * sigma ^ 2));
%        rectangle(end-r+1:end) = fliplr(rectangle(2:r+1)); 
%     else %method is normal
%         rectangle = enabled;     
%     end
%     
%     for i=1:size(Y,1)
%         Y(i,:) = Y(i,:).*rectangle;
%     end
%     
end

