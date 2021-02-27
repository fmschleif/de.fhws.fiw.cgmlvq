%create a dataset
nrExamples = 300;
lbl = zeros(1,nrExamples);

fvec = zeros(nrExamples, 2);
rng(569);
%create randomized feature f1 and f2
for j=1:nrExamples
   fvec(j,1) = rand+rand*i;
   fvec(j,2) = rand+rand*i;
   % in this case only real parts
    if real(fvec(j,1)) > real(fvec(j,2))
      lbl(j) = 1; 
    else
       lbl(j) = 2; 
    end
       
%     if imag(fvec(j,1)) > imag(fvec(j,2))
%         lbl(j) = 1; 
%     else
%         lbl(j) = 2; 
%     end
%     
%     if real(fvec(j,1)) > imag(fvec(j,2))
%         lbl(j) = 1; 
%     else
%         lbl(j) = 2; 
%     end
%     
%     if real(fvec(j,1)) > imag(fvec(j,1))
%         lbl(j) = 1; 
%     else
%         lbl(j) = 2; 
%     end
end

testCorr.sys = run_single(fvec,lbl',150,[1 2]);
abs(testCorr.sys.lambda)
testCorr.sys.lambda

%same data, but then concatenated real and imaginary parts
fvecConcat = [real(fvec) imag(fvec)];

textCorr.sysConcat = run_single(fvecConcat,lbl',50,[1 2]);
textCorr.sysConcat.lambda
