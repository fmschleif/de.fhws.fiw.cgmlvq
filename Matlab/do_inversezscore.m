function[fvec] = do_inversezscore(fvec, mf, st)

    ndim=size(fvec,2);
    
    for i=1:ndim
       fvec(:,i)=fvec(:,i)*st(i)+mf(i);
    end
end

