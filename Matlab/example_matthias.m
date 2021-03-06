addpath( 'benchmarks(PATH)' );
addpath( 'display(PATH)' );
addpath( 'fourier(PATH)' );

  A = load( 'iris.mat' );
% A = load( 'twoclass-simple.mat' );

[output_system, training_curves, param_set, backProts] = CGMLVQ( A.fvec, A.lbl, 2, 50, unique(A.lbl)' );

display( output_system );
display( training_curves );
display( param_set );
display( backProts );