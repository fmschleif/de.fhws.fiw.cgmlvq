%function that takes a struct with a training set and validation set and takes the Fourier transform
%truncated at r frequencies, and saves the following representations of it:

%   1. The complex coeficients (of r+1 coefficients DC included)
%   2. The real and imaginary parts concatenated of 1 
%   3. The complex truncated back transformed to original space

% function can also take a struct with only data examples not specificly
% specified into a training and validation set split, output are the same
% three cases. is determined by boolean if this is a struct of the first
% type of this type.
function [ st ] = takeFourierAt( st,r, type)
    if type == 1
        %this is the case for specified training and val set
        L = size(st.tra,2);
        %complex case
        st.FourierComp.tra = Fourier(st.tra,r); st.FourierComp.lbltra = st.lbltra;
        st.FourierComp.val = Fourier(st.val,r); st.FourierComp.lblval = st.lblval;
        
        %concat case
        st.FourierConc.tra = removeZeroVar([real(st.FourierComp.tra) imag(st.FourierComp.tra)]); st.FourierConc.lbltra = st.lbltra;
        st.FourierConc.val = removeZeroVar([real(st.FourierComp.val) imag(st.FourierComp.val)]); st.FourierConc.lblval = st.lblval;
        
        %backtransform to original space case
        st.back.tra = iFourier(st.FourierComp.tra,L); st.back.lbltra = st.lbltra;
        st.back.val = iFourier(st.FourierComp.val,L); st.back.lblval = st.lblval;
    end
    
    if type == 2
       %this is the fvec case for no prespecified training and val set
       L = size(st.fvec,2);
       %complex case
       st.fvecComp = Fourier(st.fvec,r);
       
       %concat case
       st.fvecConc = removeZeroVar([real(st.fvecComp) imag(st.fvecComp)]);
       
       %backtransform case
       st.fvecBack = iFourier(st.fvecComp,L);
    end
    
    if type ~= 1 && type ~= 2
        display('type can only be 1 or 2');
    end
end

