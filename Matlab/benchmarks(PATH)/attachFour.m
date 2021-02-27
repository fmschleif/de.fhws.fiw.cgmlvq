function [ st ] = attachFour(st, r)
    L = size(st.orig.tra,2);

    st.FourierA.tra = Fourier(st.orig.tra,r,'normal',[]);
    st.FourierA.val = Fourier(st.orig.val,r,'normal',[]);
    
    %back transformation smooted
    st.FourierABack.tra = iFourier(st.FourierA.tra,r,L);
    st.FourierABack.val = iFourier(st.FourierA.val,r,L);
end

