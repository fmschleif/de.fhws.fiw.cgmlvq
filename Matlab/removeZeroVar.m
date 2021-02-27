function [ out ] = removeZeroVar( in )
    ts = size(in,2);
    out = in;
    
    j=1;
    for i=1:ts
       if var(in(:,i)) ~= 0
            out(:, j) = in(:, i);
            j = j + 1;
       end
    end
   
    out = out(:, 1:j-1);
end

