% Function for benchmarking a dataset with a specified train/validation set. 
% First learning in the original space is performed. 
% Then learning in complex- and concatenated Fourier space is performed. 
% Last, learning in truncated complex- and concatenated Fourier space is performed, 
% and original space after a transform to truncated Fourier space and back transform to original space.
% The benchmark results, benchout, are attached to the struct st. 
% The graphical display of the results (AUC- and confusion matrices) 
% are plotted and saved in the project root in the folder name.

%function [ st ] = do_benchmark(st, epochs, plbl, relAnalysis, maxF, name)
function [ st ] = do_benchmark(st, epochs, plbl, maxF, name)
    mkdir(strcat(name, '\truncFour'));
    mkdir(strcat(name, '\truncFourConc'));
    mkdir(strcat(name, '\truncBack'));
    %str = strcat('thesis\03_GraphicFiles\', str, '\benchresults\');
    L = size(st.tra, 2); %length of feature vectors
    %determine max coefficients
    nrFreqs = floor(L/2);
    if nrFreqs > maxF %if there are more than 100 freqs in the Fourier space
        rMax = maxF; %set the maximum to 100
    else %if there are less than 100 freqs in the Fourier space
        rMax = nrFreqs - 5; %limit the max frequencies to 5 lower than the full Fourier representation 
    end
    
    st = takeFourierAt(st,nrFreqs,1);
    %Learn in original space and view performance on validation set (AUC)
    [benchout.orig.aucval,benchout.orig.confmat,~,benchout.orig.tetra,benchout.orig.teval] = evaluateStrat(st,epochs,plbl,0);
    
    %Learn in comp. Fourier space and view performance on validation set (AUC)
    [benchout.Fourier.aucval,benchout.Fourier.confmat,~,benchout.Fourier.tetra,benchout.Fourier.teval] = evaluateStrat(st.FourierComp,epochs,plbl,0);

    %Learn in conc. Fourier space and view performance on validation set (AUC)
    [benchout.FourierConc.aucval,benchout.FourierConc.confmat,~,benchout.FourierConc.tetra,benchout.FourierConc.teval] = evaluateStrat(st.FourierConc,epochs,plbl,0);
    %training error curves
    plotTeTraTeVal(benchout,name,epochs);

    %Relevance coefficient cutting: Explorative, used in the experiment on
    %the Wine dataset. Has been taken out in this version.
%     [trainFreq] = Fourier(train, nrFreqs, 'normal', []); %wrapper around Fourier
%     if relAnalysis
%         disp('Performing relevance analysis on frequencies');
%         [gmlvq_sys] = run_single(trainFreq,lbltra,350);
%         relevs = diag(gmlvq_sys.lambda); %get relevances
%     end
  
    j=1;
    %Now truncation at n coefficients
    for r=5:5:rMax
        %truncate at r coefficients
        st = takeFourierAt(st,r,1);
        %r comp. Fourier space
        [benchout.truncFour.aucval(j),benchout.truncFour.confmat(:,:,j),~] = evaluateStrat(st.FourierComp,epochs,plbl,0);
        
        %r conc. Fourier space
        [benchout.truncFourConc.aucval(j),benchout.truncFourConc.confmat(:,:,j),~] = evaluateStrat(st.FourierConc,epochs,plbl,0);
        
        %r back original space
        [benchout.truncBack.aucval(j),benchout.truncBack.confmat(:,:,j),~] = evaluateStrat(st.back,epochs,plbl,0);

        j=j+1;
    end
    
    plotAUCs(benchout,rMax,1, name);
    plotConfmats(benchout, rMax, name, plbl);
    st.benchout = benchout;
end

