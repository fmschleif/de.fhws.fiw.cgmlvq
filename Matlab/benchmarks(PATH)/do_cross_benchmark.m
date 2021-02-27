%function [ st ] = do_cross_benchmark( st, nruns, epochs, prctg, plbl, relAnalysis, maxF, name )
function [ st ] = do_cross_benchmark( st, nruns, epochs, prctg, plbl, maxF, name )
    mkdir(strcat(name, '\truncFour'));
    mkdir(strcat(name, '\truncFourConc'));
    mkdir(strcat(name, '\truncBack'));
    %str = strcat('thesis\03_GraphicFiles\', name, '\benchresultsC\');
    L = size(st.fvec, 2); %length of feature vectors
    %determine max frequencies
    nrFreqs = floor(L/2);
    if nrFreqs > maxF %if there are more than maxF freqs in the Fourier space
        rMax = maxF; %set the maximum to maxF
    else %if there are less than maxF freqs in the Fourier space
        rMax = nrFreqs - 5; %limit the max frequencies to 5 lower than the full Fourier representation 
    end
    
    st = takeFourierAt(st,nrFreqs,2);
    
    %perform training in original feature space
    [~,rocV,lcurves,~,~,permus]=run_validation(st.fvec, st.lbl, epochs, nruns, prctg, plbl);
    benchout.orig.confmat=rocV.confmat;
    benchout.orig.aucval = lcurves.maucval(epochs);
    benchout.orig.tetra = lcurves.mtetra; benchout.orig.teval = lcurves.mteval;
    
    %Relevance coefficient cutting: Explorative, used in the experiment on
    %the Wine dataset. Has been taken out in this version.
%     if relAnalysis
%         [gmlvq_sys] = run_single(fvecFreq,lbl,420);
%         relevs = diag(gmlvq_sys.lambda); %get relevances
%     end
    %Learn in comp. Fourier space and view performance on validation set (AUC)
    [~,rocV,lcurves]=run_validation(st.fvecComp, st.lbl, epochs, nruns, prctg, plbl, permus);
    benchout.Fourier.confmat=rocV.confmat;
    benchout.Fourier.aucval = lcurves.maucval(epochs);
    benchout.Fourier.tetra = lcurves.mtetra; benchout.Fourier.teval = lcurves.mteval;
    
    %Learn in conc. Fourier space and view performance on validation set (AUC)
    [~,rocV,lcurves]=run_validation(st.fvecConc, st.lbl, epochs, nruns, prctg, plbl, permus);
    benchout.FourierConc.confmat=rocV.confmat;
    benchout.FourierConc.aucval = lcurves.maucval(epochs);
    benchout.FourierConc.tetra = lcurves.mtetra; benchout.FourierConc.teval = lcurves.mteval;
    %training error curves
    plotTeTraTeVal(benchout,name,epochs);
    
    j=1;
    %Now truncation at n coefficients
    for r=5:5:rMax
        %truncate at r coefficients
        st = takeFourierAt(st,r,2);
        %r comp. Fourier space
        [~,rocV,lcurves]=run_validation(st.fvecComp, st.lbl, epochs, nruns, prctg, plbl, permus);
        benchout.truncFour.confmat(:,:,j)=rocV.confmat;
        benchout.truncFour.aucval(j) = lcurves.maucval(epochs);
        
        %r conc. Fourier space
        [~,rocV,lcurves]=run_validation(st.fvecConc, st.lbl, epochs, nruns, prctg, plbl, permus);
        benchout.truncFourConc.confmat(:,:,j)=rocV.confmat;
        benchout.truncFourConc.aucval(j) = lcurves.maucval(epochs);
  
        %r back original space
       [~,rocV,lcurves]=run_validation(st.fvecBack, st.lbl, epochs, nruns, prctg, plbl,permus);
       benchout.truncBack.confmat(:,:,j)=rocV.confmat;
       benchout.truncBack.aucval(j) = lcurves.maucval(epochs);

       j=j+1;
    end
    
    plotAUCs(benchout,rMax,1,name);
    plotConfmats(benchout,rMax,name,plbl);
    st.benchout = benchout;
end



