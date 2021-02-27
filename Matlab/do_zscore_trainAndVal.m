function [ fvec, fvecval,mf ,st ] = do_zscore_trainAndVal( fvec, fvecval )
    ndim = size(fvec,2); 
    mf=zeros(1,ndim); st=ones(1,ndim); % initialize feature means and st.dev. 

  [fvec,mf,st] = do_zscore(fvec);  % transform training data
  for i =1:size(fvecval,1)
     fvecval(i,:)= (fvecval(i,:)-mf)./st; % transfrom validation set
                                          % accordingly 
  end

end

