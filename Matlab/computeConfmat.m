function [ confmat, misses ] = computeConfmat( lblval, crout )
    nrClasses = length(unique(lblval));
    confmat = zeros(nrClasses,nrClasses);
    nrExamples = length(lblval);
    misses=0;
    for i=1:nrExamples
       confmat(lblval(i), crout(i)) = confmat(lblval(i), crout(i)) + 1;
       if lblval(i) ~= crout(i)
           misses = misses+1;
       end
    end
end

