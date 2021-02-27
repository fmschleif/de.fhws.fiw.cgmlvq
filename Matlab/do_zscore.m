
% perform a z-score transformation of fvec and return 
% rescaled vectors and scaling parameters

function [fvec,mf,st] = do_zscore(fvec)

% input
% fvec :   feature vectors

% output
% fvec :  z-score transformed feature vectors
% mf   :  vector of feauter means used in z-score transformation
% st   :  vector of standard deviations in z-score transformation

ndim=size(fvec,2);     % dimension ndim of data
    
mf=zeros(1,ndim); st=mf;  % initialize vectors mf and st

for i=1:ndim
    mf(i)=mean(fvec(:,i));              % mean of feature i
    st(i)=std(fvec(:,i));               % st.dev. of feature i 
    fvec(:,i)=(fvec(:,i)-mf(i))/st(i);  % transformed feature 
end;

