%In this experiment we want to see on the Wine dataset the difference between:
% 1. Keeping the 20 lowest frequencies performance
% 2. Keeping the 20 most relevant frequencies as resulted from a thorough
% GMLVQ run on the full Fourier space.
%the two are compared using 5 runs of cross validation (90% training, 10%
%testing, the first run determines the 5 splits, and the second run also
%trains on the same 5 splits

load datasets/Wine_TRAIN;
load datasets/Wine_TEST;

lblTra = Wine_TRAIN(:,1);
lblVal = Wine_TEST(:,1);
Wine_TRAIN = Wine_TRAIN(:,2:end);
Wine_TEST = Wine_TEST(:,2:end);
fvec = [Wine_TRAIN; Wine_TEST]; %put train and test in one matrix
lbl = [lblTra; lblVal]; %labels in one column vector

[fvecFreq] = Fourier(fvec, 117, 'normal', []);
%perform a thorough (400+ epochs) GMLVQ training on the entire dataset to learn the
%relevances
gmlvq_system = run_single(fvecFreq, lbl, 420);
%assign the relevances of the features
relevs = diag(gmlvq_system.lambda);

tEpochs = 65;
%perform a cross validation test for 20 lowest frequencies
[fvecFreq] = Fourier(fvec, 20, 'normal', []);
[~,~,lcurves_mean,~,~,permus]=run_validation(fvecFreq, lbl, tEpochs, 5, 25, [1 2]);
lcurves_mean.maucval(tEpochs)

%perform a cross validation test for the 20 most relevant frequencies
[fvecFreq] = Fourier(fvec, 20, 'normal', relevs);
[~,~,lcurves_mean,~,~,~]=run_validation(fvecFreq, lbl, tEpochs, 5, 25, [1 2], permus); %train on same splits
lcurves_mean.maucval(tEpochs)
