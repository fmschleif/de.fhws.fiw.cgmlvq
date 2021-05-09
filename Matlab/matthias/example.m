addpath( '..' );
addpath( '../benchmarks(PATH)' );
addpath( '../display(PATH)' );
addpath( '../fourier(PATH)' );


  A = load( '../iris.mat' );
% A = load( '../twoclass-simple.mat' );
% A = load( '../twoclass-difficult.mat' );


X_train = A.fvec( 1:120, : );
y_train = A.lbl( 1:120, : );

X_test = A.fvec( 121:150, : );
y_test = A.lbl( 121:150, : );



%%% CGMLVQ (fft) %%%

% [output_system, training_curves, param_set, backProts] = CGMLVQ( X_train, y_train, 2, 50, unique(y_train)' );

% display( output_system );
% display( training_curves );
% display( param_set );
% display( backProts );

% [X_test] = Fourier( X_test, 2 );

% [crisp, score] = classify_gmlvq( output_system, X_test, 1, y_test );

% display( crisp );



%%% run_single %%%

[gmlvq_system, training_curves, param_set] = run_single( X_train, y_train, 50, unique(y_train)' );

% [crisp, score, margin, costf] = classify_gmlvq( gmlvq_system, X_test, 1, y_test );

% display( crisp );



%%% run_validation %%%

% [gmlvq_mean, roc_validation, lcurves_mean, lcurves_sdt, param_set ] = run_validation( A.fvec, A.lbl, 40, 10, 10, unique(A.lbl)' );

% averaged confusion matrix (Nearest prototype classifier, given as percentages)
% roc_val.confmat
% 96.2984    0        3.0565   0        0.6451    0         0
%  0       100.0000   0        0        0         0         0
%  4.0174    0       93.5162   0        2.0924    0.3740    0
%  4.1374    0        2.3858  73.5697   8.4987   11.4085    0
% 23.9696    0       22.0283   4.9418  49.0603    0         0
%  0         0        0        0        0       100.0000    0
%  0         0        0        0        0         0       100.0
% row index: true class label of test sample
% column index: class label predicted by GMLVQ system