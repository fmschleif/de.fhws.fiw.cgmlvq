import csv
import numpy as np
import os
import unittest

from ..cgmlvq import CGMLVQ


class Test_CGMLVQ( unittest.TestCase ):


    def setUp( self ):

        super().__init__()

        iris = self.load_data( "iris_modified.csv" )

        self.X_train = np.array( iris[0:120,0:4] )
        self.y_train = np.array( iris[0:120,4], dtype=int )

        self.X_test = np.array( iris[120:150,0:4] )
        self.y_test = np.array( iris[120:150,4], dtype=int )


    def load_data( self, file_name ):

        data = []

        csv_file = open( os.path.join(os.getcwd(), "Python\\tests", file_name) )

        csv_reader = csv.reader( csv_file, delimiter="," )

        for row in csv_reader:

            data.append( row )

        csv_file.close()

        return np.array( data, dtype=np.cfloat )


    def test_classify_gmlvq( self ):

        cgmlvq = CGMLVQ()
        cgmlvq._CGMLVQ__run_single( self.X_train, self.y_train )

        crisp, _, _, _ = cgmlvq._CGMLVQ__classify_gmlvq( self.X_test )

        np.testing.assert_array_almost_equal( crisp, self.load_data('test_classify_gmlvq_doztr_crisp.csv') )


        cgmlvq.set_params( doztr=False )
        cgmlvq._CGMLVQ__run_single( self.X_train, self.y_train )

        crisp, _, _, _ = cgmlvq._CGMLVQ__classify_gmlvq( self.X_test )

        np.testing.assert_array_almost_equal( crisp, self.load_data('test_classify_gmlvq_crisp.csv') )


    def test_compute_costs( self ):

        cgmlvq = CGMLVQ()

        proti, omi = cgmlvq._CGMLVQ__set_initial( self.X_train, self.y_train, np.unique(self.y_train) )

        costf, crout, marg, score = cgmlvq._CGMLVQ__compute_costs( self.X_train, self.y_train, proti, np.unique(self.y_train), omi, 0 )

        np.testing.assert_array_almost_equal( costf, self.load_data('test_compute_costs_costf.csv')[0] )
        np.testing.assert_array_almost_equal( crout, self.load_data('test_compute_costs_crout.csv')    )
        np.testing.assert_array_almost_equal( marg,  self.load_data('test_compute_costs_marg.csv')     )
        np.testing.assert_array_almost_equal( score, self.load_data('test_compute_costs_score.csv')    )


        cgmlvq.set_params( mu=1 )

        proti, omi = cgmlvq._CGMLVQ__set_initial( self.X_train, self.y_train, np.unique(self.y_train) )

        costf, crout, marg, score = cgmlvq._CGMLVQ__compute_costs( self.X_train, self.y_train, proti, np.unique(self.y_train), omi, 1 )

        np.testing.assert_array_almost_equal( costf, self.load_data('test_compute_costs_mu1_costf.csv')[0] )
        np.testing.assert_array_almost_equal( crout, self.load_data('test_compute_costs_mu1_crout.csv')    )
        np.testing.assert_array_almost_equal( marg,  self.load_data('test_compute_costs_mu1_marg.csv')     )
        np.testing.assert_array_almost_equal( score, self.load_data('test_compute_costs_mu1_score.csv')    )


    def test_compute_euclid( self ):

        cgmlvq = CGMLVQ()

        proti, omi = cgmlvq._CGMLVQ__set_initial( self.X_train, self.y_train, np.unique(self.y_train) )

        omat = omi / np.sqrt(sum(sum(omi*omi)))

        D = cgmlvq._CGMLVQ__compute_euclid( self.X_train[0,:], proti[0,:], omat )

        np.testing.assert_array_almost_equal( D, self.load_data('test_euclid_D.csv') )


    def test_do_batchstep( self ):

        cgmlvq = CGMLVQ()

        proti, omi = cgmlvq._CGMLVQ__set_initial( self.X_train, self.y_train, np.unique(self.y_train) )

        etam, etap, _, _, _ = cgmlvq._CGMLVQ__set_parameters( self.X_train )


        cgmlvq.set_params( mode=0, mu=1 )

        prot, omat = cgmlvq._CGMLVQ__do_batchstep( self.X_train, self.y_train, proti, np.unique(self.y_train), omi, etap, etam )

        np.testing.assert_array_almost_equal( prot, self.load_data('test_do_batchstep_mode0_prot.csv') )
        np.testing.assert_array_almost_equal( omat, self.load_data('test_do_batchstep_mode0_omat.csv') )


        cgmlvq.set_params( mode=1, mu=0 )

        prot, omat = cgmlvq._CGMLVQ__do_batchstep( self.X_train, self.y_train, proti, np.unique(self.y_train), omi, etap, etam )

        np.testing.assert_array_almost_equal( prot, self.load_data('test_do_batchstep_mode1_prot.csv') )
        np.testing.assert_array_almost_equal( omat, self.load_data('test_do_batchstep_mode1_omat.csv') )


        cgmlvq.set_params( mode=2 )

        prot, omat = cgmlvq._CGMLVQ__do_batchstep( self.X_train, self.y_train, proti, np.unique(self.y_train), omi, etap, etam )

        np.testing.assert_array_almost_equal( prot, self.load_data('test_do_batchstep_mode2_prot.csv') )
        np.testing.assert_array_almost_equal( omat, self.load_data('test_do_batchstep_mode2_omat.csv') )


        cgmlvq.set_params( mode=3 )

        prot, omat = cgmlvq._CGMLVQ__do_batchstep( self.X_train, self.y_train, proti, np.unique(self.y_train), omi, etap, etam )

        np.testing.assert_array_almost_equal( prot, self.load_data('test_do_batchstep_mode3_prot.csv') )
        np.testing.assert_array_almost_equal( omat, self.load_data('test_do_batchstep_mode3_omat.csv') )


    def test_do_fourier( self ):

        cgmlvq = CGMLVQ()

        Y = cgmlvq._CGMLVQ__do_fourier( self.X_train )

        np.testing.assert_array_almost_equal( Y, self.load_data('test_fourier_Y.csv') )


    def test_do_inversezscore( self ):

        cgmlvq = CGMLVQ()

        fvec, mf, st = cgmlvq._CGMLVQ__do_zscore( self.X_train )

        fvec = cgmlvq._CGMLVQ__do_inversezscore( fvec, mf, st )

        np.testing.assert_array_almost_equal( fvec, self.load_data('test_do_inversezscore_fvec.csv') )


    def test_do_zscore( self ):

        cgmlvq = CGMLVQ()

        fvec, mf, st = cgmlvq._CGMLVQ__do_zscore( self.X_train )

        np.testing.assert_array_almost_equal( fvec, self.load_data('test_do_zscore_fvec.csv') )
        np.testing.assert_array_almost_equal( mf, self.load_data('test_do_zscore_mf.csv') )
        np.testing.assert_array_almost_equal( st, self.load_data('test_do_zscore_st.csv') )


    def test_run_single( self ):

        cgmlvq = CGMLVQ()

        cgmlvq._CGMLVQ__run_single( self.X_train, self.y_train )

        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['protos'], self.load_data('test_run_single_doztr_protos.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['protosInv'], self.load_data('test_run_single_doztr_protosInv.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['lambda'], self.load_data('test_run_single_doztr_lambda.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['mean_features'], self.load_data('test_run_single_doztr_mean_features.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['std_features'], self.load_data('test_run_single_doztr_std_features.csv') )


        cgmlvq.set_params( doztr=False )
        cgmlvq._CGMLVQ__run_single( self.X_train, self.y_train )

        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['protos'], self.load_data('test_run_single_protos.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['protosInv'], self.load_data('test_run_single_protosInv.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['lambda'], self.load_data('test_run_single_lambda.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['mean_features'], self.load_data('test_run_single_mean_features.csv') )
        np.testing.assert_array_almost_equal( cgmlvq.gmlvq_system['std_features'], self.load_data('test_run_single_std_features.csv') )


    def test_set_initial( self ):

        cgmlvq = CGMLVQ()
        cgmlvq.set_params( mode=2, rndinit=True )

        proti, omi = cgmlvq._CGMLVQ__set_initial( self.X_train, self.y_train, np.unique(self.y_train) )

        np.testing.assert_array_almost_equal( proti, self.load_data('test_set_initial_proti.csv') )
        np.testing.assert_array_almost_equal( omi, self.load_data('test_set_initial_omi.csv') )


        cgmlvq.set_params( mode=3, rndinit=False )

        proti, omi = cgmlvq._CGMLVQ__set_initial( self.X_train, self.y_train, np.unique(self.y_train) )

        np.testing.assert_array_almost_equal( proti, self.load_data('test_set_initial_mode3_proti.csv') )
        np.testing.assert_array_almost_equal( omi, self.load_data('test_set_initial_mode3_omi.csv') )


if "__main__" == __name__:

    unittest.main()