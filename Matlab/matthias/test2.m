% ======
% readme
% ======

% set doztr to 0 in set_parameters


% ====
% init
% ====

addpath( '..' );
addpath( '../benchmarks(PATH)' );
addpath( '../display(PATH)' );
addpath( '../fourier(PATH)' );


iris = load( 'iris_modified.csv' );

X_train = iris(   1:120, 1:4 );
X_test  = iris( 121:150, 1:4 );

y_train = iris(   1:120, 5 );
y_test  = iris( 121:150, 5 );


% ==============
% classify_gmlvq
% ==============

[gmlvq_system, ~, ~] = run_single( X_train, y_train, 50, unique(y_train)' );

[crisp, ~, ~, ~] = classify_gmlvq( gmlvq_system, X_test, 0, y_test );

my_write( 'test_classify_gmlvq_crisp.csv', crisp );


% ==========
% run_single
% ==========

[gmlvq_system, ~, ~] = run_single( X_train, y_train, 50, unique(y_train)' );

my_write( 'test_run_single_protos.csv', gmlvq_system.protos );
my_write( 'test_run_single_protosInv.csv', gmlvq_system.protosInv );
my_write( 'test_run_single_lambda.csv', gmlvq_system.lambda );
my_write( 'test_run_single_mean_features.csv', gmlvq_system.mean_features );
my_write( 'test_run_single_std_features.csv', gmlvq_system.std_features );