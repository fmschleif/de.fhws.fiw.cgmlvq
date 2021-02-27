%Performs a single run of GMLVQ learning on the dataset "fvec" in Fourier
%space truncated at "r" coefficients. 

%Another option is to pass Fourier- or complex data to the run_single function
%directly, the only thing this function adds is a display of the back 
%transformed prototypes and the feature relevance in Fourier space after
%learning.

function [ output_system, training_curves, param_set, backProts ] = CGMLVQ( fvec, lbl, r, epochs, plbl )  
    L = size(fvec, 2); %length of feature vectors
    nrClasses = length(unique(lbl));
    
    [fvecFreq] = Fourier(fvec, r); %wrapper around Fourier

     %train on the coefficients in Fourier space
    [gmlvq_system,training_curves, param_set]=run_single(fvecFreq, lbl, epochs, plbl);
    [backProts] = iFourier(gmlvq_system.protosInv, L); %wrapper around inverse Fourier
    
    plotDataset( real(backProts), unique(lbl), 'northeast', 'Prototypes transformed to original space', 2, nrClasses, 1 )
    
    figure
    bar(diag(gmlvq_system.lambda));
    xlabel('Feature index')
    ylabel('Relevance')
    title('Relevances in Fourier space')
     
    output_system = gmlvq_system;
end

