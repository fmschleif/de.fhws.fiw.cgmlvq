function [ confmat, misses ] = normalConfmat( confmat )
    nrClasses = size(confmat,1);
    misses = 0;
    for j=1:nrClasses
       geheel = sum(confmat(j,:));
       for i=1:nrClasses
           if j ~= i
                misses = misses + confmat(j,i);
           end
           confmat(j,i) = (confmat(j,i)/geheel)*100; 
       end
    end
end

