% ======
% readme
% ======

% set doztr to 1 (default) in set_parameters


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

[crisp, ~, ~, ~] = classify_gmlvq( gmlvq_system, X_test, 1, y_test );

my_write( 'test_classify_gmlvq_doztr_crisp.csv', crisp );


% =============
% compute_costs
% =============

rng( 'default' ); 
rng( 291024 );

[proti, omi] = set_initial( X_train, y_train, unique(y_train)', 1, false );


[costf, crout, marg, score] = compute_costs( X_train, y_train, proti, unique(y_train)', omi, 0 );

my_write( 'test_compute_costs_costf.csv', costf );
my_write( 'test_compute_costs_crout.csv', crout );
my_write( 'test_compute_costs_marg.csv', marg );
my_write( 'test_compute_costs_score.csv', score );


[costf ,crout, marg, score] = compute_costs( X_train, y_train, proti, unique(y_train)', omi, 1 );

my_write( 'test_compute_costs_mu1_costf.csv', costf );
my_write( 'test_compute_costs_mu1_crout.csv', crout );
my_write( 'test_compute_costs_mu1_marg.csv', marg );
my_write( 'test_compute_costs_mu1_score.csv', score );


% ===========
% compute_roc
% ===========

rng( 'default' ); 
rng( 291024 );

[proti, omi] = set_initial( X_train, y_train, unique(y_train)', 1, false );

[~, ~, ~, score] = compute_costs( X_train, y_train, proti, unique(y_train)', omi, 0 );

[tpr, fpr, auroc, thresh] = compute_roc( y_train>1, score );

my_write( 'test_compute_roc_tpr.csv', tpr );
my_write( 'test_compute_roc_fpr.csv', fpr );
my_write( 'test_compute_roc_auroc.csv', auroc );
my_write( 'test_compute_roc_thresh.csv', thresh );


% ============
% do_batchstep
% ============

rng( 'default' ); 
rng( 291024 );

[proti, omi] = set_initial( X_train, y_train, unique(y_train)', 1, false );

[~, ~, ~, ~, etam, etap, ~, ~, ~, ~] = set_parameters( X_train );


[prot, omat] = do_batchstep( X_train, y_train, proti, unique(y_train)', omi, etap, etam, 1, 0 );

my_write( 'test_do_batchstep_mode0_prot.csv', prot );
my_write( 'test_do_batchstep_mode0_omat.csv', omat );


[prot, omat] = do_batchstep( X_train, y_train, proti, unique(y_train)', omi, etap, etam, 0, 1 );

my_write( 'test_do_batchstep_mode1_prot.csv', prot );
my_write( 'test_do_batchstep_mode1_omat.csv', omat );


[prot, omat] = do_batchstep( X_train, y_train, proti, unique(y_train)', omi, etap, etam, 0, 2 );

my_write( 'test_do_batchstep_mode2_prot.csv', prot );
my_write( 'test_do_batchstep_mode2_omat.csv', omat );


[prot, omat] = do_batchstep( X_train, y_train, proti, unique(y_train)', omi, etap, etam, 0, 3 );

my_write( 'test_do_batchstep_mode3_prot.csv', prot );
my_write( 'test_do_batchstep_mode3_omat.csv', omat );


% ================
% do_inversezscore
% ================

[fvec, mf, st] = do_zscore( X_train );

[fvec] = do_inversezscore( fvec, mf, st );

my_write( 'test_do_inversezscore_fvec.csv', fvec );


% =========
% do_zscore
% =========

[fvec, mf, st] = do_zscore( X_train );

my_write( 'test_do_zscore_fvec.csv', fvec );
my_write( 'test_do_zscore_mf.csv', mf );
my_write( 'test_do_zscore_st.csv', st );


% ======
% euclid
% ======

rng( 'default' ); 
rng( 291024 );

[proti, omi] = set_initial( X_train, y_train, unique(y_train)', 1, false );

omat = omi / sqrt(sum(sum(omi*omi)));

D = euclid( X_train(1,:), proti(1,:), omat );

my_write( 'test_euclid_D.csv', D );


% =======
% Fourier
% =======

Y = Fourier( X_train, 0 );

my_write( 'test_fourier_Y.csv', Y );


% ===========
% set_initial
% ===========

rng( 'default' ); 
rng( 291024 );

[proti, omi] = set_initial( X_train, y_train, unique(y_train)', 2, true );

my_write( 'test_set_initial_proti.csv', proti );
my_write( 'test_set_initial_omi.csv', omi );


rng( 'default' ); 
rng( 291024 );

[proti, omi] = set_initial( X_train, y_train, unique(y_train)', 3, false );

my_write( 'test_set_initial_mode3_proti.csv', proti );
my_write( 'test_set_initial_mode3_omi.csv', omi );


% ==========
% run_single
% ==========

[gmlvq_system, training_curves, ~] = run_single( X_train, y_train, 50, unique(y_train)' );

my_write( 'test_run_single_doztr_protos.csv', gmlvq_system.protos );
my_write( 'test_run_single_doztr_protosInv.csv', gmlvq_system.protosInv );
my_write( 'test_run_single_doztr_lambda.csv', gmlvq_system.lambda );
my_write( 'test_run_single_doztr_plbl.csv', gmlvq_system.plbl );
my_write( 'test_run_single_doztr_mean_features.csv', gmlvq_system.mean_features );
my_write( 'test_run_single_doztr_std_features.csv', gmlvq_system.std_features );
my_write( 'test_run_single_doztr_costs.csv', training_curves.costs );