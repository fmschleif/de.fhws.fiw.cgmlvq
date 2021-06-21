""" Complex Generalized Matrix Learning Vector Quantization """

# Author: Matthias Nunn

from scipy.fft import fft
from scipy.linalg import sqrtm

import numpy as np


class CGMLVQ:

    """ A classifier for complex valued data based on gmlvq.

    Parameters
    ----------
    coefficients : int, default=0
        Number of signal values in the frequency domain. If coefficients is 0, no fft is executed.

    totalsteps : int, default=50
        Number of batch gradient steps to be performed in each training run.

    doztr : bool, default=True
        If true, do z transformation, otherwise you may have to adjust step sizes.

    mode : int, default=1
        Control LVQ version

        - 0 = GMLVQ: matrix without null space correction
        - 1 = GMLVQ: matrix with null-space correction
        - 2 = GRLVQ: diagonal matrix only, sensitive to step sizes
        - 3 = GLVQ: relevance matrix proportional to identity (with Euclidean distance), "normalized identity matrix"

    mu : int, default=0
        Controls penalty of singular relevance matrix

        - 0 = unmodified GMLVQ algorithm (recommended for initial experiments)
        - > 0 = non-singular relevance matrix is enforced, mu controls dominance of leading eigenvectors continuously, prevents singular Lambda

    rndinit : bool, default=False
        If true, initialize the relevance matrix randomly (if applicable), otherwise it is proportional to the identity matrix.

    Example
    -------
    >>> X = [[0], [1], [2], [3]]
    >>> Y = [1, 1, 2, 2]
    >>> from cgmlvq import CGMLVQ
    >>> cgmlvq = CGMLVQ()
    >>> cgmlvq.fit( X, y )
    >>> print( cgmlvq.predict([[0], [1]]) )

    Notes
    -----
    Based on the Matlab implementation from Michiel Straat.
    """

    def __init__( self, coefficients=0, totalsteps=50, doztr=True, mode=1, mu=0, rndinit=False ):

        self.coefficients = coefficients
        self.totalsteps = totalsteps
        self.doztr = doztr
        self.mode = mode
        self.mu = mu
        self.rndinit = rndinit


    def fit( self, X, y ):

        """ Fit the classifier from the training dataset.

        Parameters
        ----------
        X : Training data
        y : Target values
        """

        X = np.array( X, dtype=np.cdouble )
        y = np.array( y, dtype=int )

        if self.coefficients > 0:
            X = self.__do_fourier( X )

        self.__run_single( X, y )


    def get_params( self ):

        """ Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """

        return { 'coefficients': self.coefficients, 'totalsteps': self.totalsteps, 'doztr': self.doztr, 'mode': self.mode, 'mu': self.mu, 'rndinit': self.rndinit }


    def predict( self, X ):

        """ Predict the class labels for the provided data.

        Parameters
        ----------
        X : Test data

        Returns
        -------
        y : Class labels for each data sample
        """

        if self.gmlvq_system is None:
            raise ValueError( 'Changed parameter coefficients or doztr. Please call method fit again!' )

        X = np.array( X, dtype=np.cdouble )

        if self.coefficients > 0:
            X = self.__do_fourier( X )

        crisp = self.__classify_gmlvq( X )

        return crisp[0]


    def set_params( self, **params ):

        """ Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters
        """

        if 'coefficients' in params:
            if params['coefficients'] >= 0:
                self.coefficients = params['coefficients']
                self.gmlvq_system = None
            else:
                raise ValueError( 'Invalid parameter coefficients. Check the list of available parameters!' )

        if 'totalsteps' in params:
            if params['totalsteps'] >= 0:
                self.totalsteps = params['totalsteps']
            else:
                raise ValueError( 'Invalid parameter totalsteps. Check the list of available parameters!' )

        if 'doztr' in params:
            if type( params['doztr'] ) == bool:
                self.doztr = params['doztr']
                self.gmlvq_system = None
            else:
                raise ValueError( 'Invalid parameter doztr. Check the list of available parameters!' )

        if 'mode' in params:
            if params['mode'] >= 0 and params['mode'] <= 3:
                self.mode = params['mode']
            else:
                raise ValueError( 'Invalid parameter mode. Check the list of available parameters!' )

        if 'mu' in params:
            if params['mu'] >= 0:
                self.mu = params['mu']
            else:
                raise ValueError( 'Invalid parameter mu. Check the list of available parameters!' )

        if 'rndinit' in params:
            if type( params['rndinit'] ) == bool:
                self.rndinit = params['rndinit']
            else:
                raise ValueError( 'Invalid parameter rndinit. Check the list of available parameters!' )


    def __check_arguments( self, X, y, ncop ):

        """ Check consistency of some arguments and input parameters.

        Parameters
        ----------
        X : feature vectors in data set
        y : data set labels
        ncop : number of copies in step size control procedure

        Returns
        -------
        y : data set labels, protentially transposed for consistency
        wlbl : prototype labels
        """

        m = X.shape[0]
        n = X.shape[1]

        y = np.array( [y], dtype=int )
        if y.shape[1] > 1:  # if lbl is row vector
            y = y.T         # transpose to column vector

        wlbl = np.unique( y )

        if X.shape[0] != len(y):
            raise ValueError('number of training labels differs from number of samples')

        if min(y) != 1 or max(y) != len(np.unique(y)):
            raise ValueError('data labels should be: 1,2,3,...,nclasses')

        st = np.zeros( X.shape[1] )
        for i in range( 0, X.shape[1] ):
            st[i] = np.std( X[:,i], ddof=1 )  # standard deviation of feature i

        if min(st) < 1.e-10:
            raise ValueError('at least one feature displays (close to) zero variance')

        if ncop >= self.totalsteps:
            raise ValueError('number of gradient steps must be larger than ncop (5)')

        if m <= n and self.mode == 0:
            print('dim. > # of examples, null-space correction recommended')

        if not self.doztr and self.mode < 3:
            print('rescale relevances for proper interpretation')

        return y, wlbl


    def __classify_gmlvq( self, X ):

        """ Apply a gmlvq classifier to a given data set with unknown class labels for predication or known class labels for testing/validation

        Parameters
        ----------
        X : set of feature vectors to be classified

        Returns
        -------
        crisp : crisp labels of Nearest-Prototype-Classifier
        """

        m = X.shape[0]

        X = np.copy( X )
        y = np.ones( (1, m) ).T  # fake labels for __compute_costs

        w       = self.gmlvq_system['w']
        lambdaa = self.gmlvq_system['lambda']
        wlbl    = self.gmlvq_system['wlbl']
        mf      = self.gmlvq_system['mean_features']
        st      = self.gmlvq_system['std_features']

        omat = sqrtm( lambdaa )  # symmetric matrix square root as one representation of the distance measure

        # if z-transformation was applied in training, apply the same here:
        if self.doztr:
            for i in range( 0, m ):
                X[i, :] = (X[i, :] - mf) / st

        # cost function can be computed without penalty term for margins/score
        _, crisp, _, _ = self.__compute_costs( X, y, w, wlbl, omat, 0 )

        return crisp


    def __compute_costs( self, X, y, w, wlbl, omega, mu ):

        """ Calculates gmlvq cost function, labels and margins for a set of labelled feature vectors, given a particular lvq system.

        Parameters
        ----------
        X : nvf feature vectors of dim. ndim
        y : data labels
        w : prototypes
        wlbl : prototype labels
        omega : global matrix omega
        mu : mu>0 controls penalty for singular relevance matrix

        Returns
        -------
        costf : glvq-costs per example
        crout : crisp classifier outputs
        marg : margins of classifying the examples
        score : distance based "scores"
        """

        m = X.shape[0]
        c = len( wlbl )

        costf = 0
        marg  = np.zeros( (1, m) )
        score = np.zeros( (1, m) )
        crout = np.zeros( (1, m) )

        omega = omega / np.sqrt(sum(sum(omega * omega)))  # normalized omat

        for i in range( 0, m ):  # loop through examples

            Xi = X[i, :]  # actual example
            yi = y[i]     # actual example

            # calculate squared distances to all prototypes
            d = np.empty( (c, 1) )  # define squared distances
            d[:] = np.nan

            for j in range( 0, c ):  # distances from all prototypes
                d[j] = self.__compute_euclid( Xi, w[j, :], omega )

            # find the two winning prototypes
            correct   = np.where( np.array([wlbl]) == yi )[1]  # all correct prototype indices
            incorrect = np.where( np.array([wlbl]) != yi )[1]  # all wrong   prototype indices

            dJ, JJ = d[correct].min(0), d[correct].argmin(0)      # correct winner
            dK, KK = d[incorrect].min(0), d[incorrect].argmin(0)  # wrong winner

            # winner indices
            jwin = correct[JJ][0]
            kwin = incorrect[KK][0]

            costf = costf + (dJ-dK) / (dJ+dK) / m

            marg[0, i] = (dJ-dK) / (dJ+dK)  # gmlvq margin of example i

            # un-normalized difference of distances
            if yi == 1:
                score[0, i] = dK - dJ
            else:
                score[0, i] = dJ - dK

            # the class label according to nearest prototype
            crout[0, i] = wlbl[jwin] * (dJ <= dK) + wlbl[kwin] * (dJ > dK)

        # add penalty term
        if mu > 0:
            costf = costf - mu / 2 * np.log(np.linalg.det(omega @ omega.conj().T)) / m

        return costf, crout, marg, score


    def __compute_euclid( self, X, w, omega ):

        # d = (X - w).conj().T @ omega.conj().T @ omega @ (X - w)
        # d = d.real

        # simpler form, which is also cheaper to compute
        d = np.linalg.norm(omega @ np.array([X - w]).T)**2
        d = d.real

        return d


    def __do_batchstep( self, X, y, w, wlbl, omega, etap, etam ):

        """ Perform a single step of batch gradient descent GMLVQ with given step size for matrix and prototype updates (input parameter) only for one global quadratic omega matrix, potentially diagonal (mode=2)
            optional: null-space correction for full matrix only (mode=1)

        Parameters
        ----------
        X : feature vectors
        y : training labels
        w : prototypes before the step
        wlbl : prototype labels
        omega : global matrix before the step
        etap : prototype learning rate
        etam : matrix learning rate

        Returns
        -------
        w : prototypes after update
        omega : omega matrix after update
        """

        m = X.shape[0]
        n = X.shape[1]
        c = len( wlbl )

        lambdaa = omega.conj().T @ omega

        # initialize change of w and omega
        chp = 0 * w
        chm = 0 * omega

        for i in range( 0, m ):  # loop through (sum over) all training examples

            Xi = X[i,:]  # actual example
            yi = y[i]    # actual example

            # calculate squared distances to all prototypes
            d = np.empty( (c, 1) )  # define squared distances
            d[:] = np.nan

            for j in range( 0, c ):  # distances from all prototypes
                d[j] = self.__compute_euclid( Xi, w[j,:], omega )

            # find the two winning prototypes
            correct   = np.where( np.array([wlbl]) == yi )[1]  # all correct prototype indices
            incorrect = np.where( np.array([wlbl]) != yi )[1]  # all wrong   prototype indices

            dJ, JJ = d[correct].min(0), d[correct].argmin(0)      # correct winner
            dK, KK = d[incorrect].min(0), d[incorrect].argmin(0)  # wrong winner

            # winner indices
            jwin = correct[JJ][0]
            kwin = incorrect[KK][0]

            # winning prototypes
            wJ = w[jwin,:]
            wK = w[kwin,:]

            # GMLVQ prototype update for one example Xi
            DJ = np.array([ Xi - wJ ]).T  # displacement vectors
            DK = np.array([ Xi - wK ]).T  # displacement vectors

            norm_factor = (dJ + dK)**2  # denominator of prefactor

            dwJ = -(dK/norm_factor) * lambdaa @ DJ  # change of correct winner
            dwK =  (dJ/norm_factor) * lambdaa @ DK  # change of incorrect winner

            # matrix update, single (global) matrix omega for one example
            f1 = ( dK/norm_factor) * (omega@DJ) @ DJ.conj().T
            f2 = (-dJ/norm_factor) * (omega@DK) @ DK.conj().T

            # negative gradient update added up over examples
            chp[jwin,:] = chp[jwin,:] - dwJ.conj().T  # correct   winner summed update
            chp[kwin,:] = chp[kwin,:] - dwK.conj().T  # incorrect winner summed update
            chm = chm - (f1 + f2)                     # matrix summed update

        # singularity control: add derivative of penalty term times mu
        if self.mu > 0:
            chm = chm + self.mu * np.linalg.pinv( omega.conj().T )

        # compute normalized gradient updates (length 1)
        # separate nomralization for prototypes and the matrix
        # computation of actual changes, diagonal matrix imposed here if nec.
        n2chw = np.sum( chp.conj() * chp ).real

        if self.mode == 2:               # if diagonal matrix used only
            chm = np.diag(np.diag(chm))  # reduce to diagonal changes

        n2chm = np.sum(np.sum(np.absolute(chm)**2))  # total 'length' of matrix update

        # final, normalized gradient updates after 1 loop through examples
        w = w + etap * chp / np.sqrt(n2chw)
        omega = omega + etam * chm / np.sqrt(n2chm)

        # if diagonal matrix only
        if self.mode == 2:                     # probably obsolete as chm diagonal
            omega = np.diag( np.diag(omega) )  # reduce to diagonal matrix

        #  nullspace correction using Moore Penrose pseudo-inverse
        if self.mode == 1:
            xvec = np.concatenate((X, w))                                      # concat. protos and fvecs
            omega = (omega @ xvec.conj().T) @ np.linalg.pinv( xvec.conj().T )  # corrected omega matrix

        if self.mode == 3:
            omega = np.identity( n )  # reset to identity regardless of gradients

        # normalization of omega, corresponds to Trace(lambda) = 1
        omega = omega / np.sqrt(np.sum(np.sum(np.absolute(omega)**2)))

        # one full, normalized gradient step performed, return omega and w
        return w, omega


    def __do_fourier( self, X ):

        """ Wrapper around "fft" to obtain Fourier series of "x" truncated at "r" coefficients. Ignores the symmetric part of the spectrum.
        """

        Y = fft( X )

        enabled = np.zeros( Y.shape[1] )

        enabled[ 0 : self.coefficients+1 ] = 1

        Y = Y[:, enabled==1]

        return Y


    def __do_inversezscore( self, X, mf, st ):

        n = X.shape[1]

        for i in range( 0, n ):
            X[:,i] = X[:,i] * st[0,i] + mf[0,i]

        return X


    def __do_zscore( self, X ):

        """ Perform a z-score transformation of fvec

        Parameters
        ----------
        X : feature vectors

        Returns
        -------
        X : z-score transformed feature vectors
        mean : vector of means used in z-score transformation
        std : vector of standard deviations in z-score transformation
        """

        n = X.shape[1]

        mean = np.zeros( (1, n), dtype=np.cdouble )
        std = np.zeros( (1, n) )

        for i in range( 0, n ):
            mean[0,i] = np.mean(X[:,i])
            std[0,i] = np.std(X[:,i], ddof=1)
            X[:, i] = (X[:,i] - mean[0,i]) / std[0,i]

        return X, mean, std


    def __get_initial( self, X, y, wlbl ):

        """ Initialization of prototypes close to class conditional means small random displacements to break ties

        Parameters
        ----------
        X : feature vectors
        y : data labels
        wlbl : prototype labels

        Returns
        -------
        w : prototypes matrix
        omega : omega matrix
        """

        n = X.shape[1]
        c = len( wlbl )

        w = np.zeros( (c, n), dtype=np.cdouble )

        for i in range( 0, c ):  # compute class-conditional means
            w[i,:] = np.mean( X[np.where(y == wlbl[i]), :][0], axis=0 )

        # reproducible random numbers
        np.random.seed( 291024 )

        # displace randomly from class-conditional means
        w = w * (0.99 + 0.02 * np.random.rand(w.shape[1], w.shape[0]).T)

        # (global) matrix initialization, identity or random
        omega = np.identity( n )  # works for all values of mode if rndinit == 0

        if self.mode != 3 and self.rndinit:  # does not apply for mode==3 (GLVQ)
            omega = np.random.rand( n, n ).T - 0.5
            omega = omega.conj().T @ omega  # square symmetric
            # matrix of uniform random numbers

        if self.mode == 2:
            omega = np.diag(np.diag(omega))  # restrict to diagonal matrix

        omega = omega / np.sqrt(sum(sum(abs(omega)**2)))

        return w, omega


    def __get_parameters( self, X ):

        """ Set general parameters
            Set initial step sizes and control parameters of modified procedure based on [Papari, Bunte, Biehl]

        Parameters
        ----------
        X : feature vectors

        Returns
        -------
        etam : initital step size for diagonal matrix updates
        etap : initital step size for prototype update
        decfac : step size factor (decrease) for Papari steps
        incfac : step size factor (increase) for all steps
        ncop : number of waypoints stored and averaged
        """

        # parameters of stepsize adaptation

        if self.mode < 2:  # full matrix updates with (0) or w/o (1) null space correction
            etam = 2
            etap = 1

        elif self.mode == 2:  # diagonal relevances only, DISCOURAGED
            etam = 0.2
            etap = 0.1

        elif self.mode == 3:  # GLVQ, equivalent to Euclidean distance
            etam = 0
            etap = 1

        decfac = 1.5
        incfac = 1.1
        ncop = 5

        return etam, etap, decfac, incfac, ncop


    def __run_single( self, X, y ):

        etam, etap, decfac, incfac, ncop = self.__get_parameters( X )

        y, wlbl = self.__check_arguments( X, y, ncop )

        m = X.shape[0]
        c = len( np.unique(wlbl) )

        te = np.zeros( (self.totalsteps+1, 1) )  # define total error
        cf = np.zeros( (self.totalsteps+1, 1) )  # define cost function
        cw = np.zeros( (self.totalsteps+1, c) )  # define class-wise errors

        if self.doztr:
            X, mf, st = self.__do_zscore( X.copy() )  # perform z-score transformation
        else:
            _, mf, st = self.__do_zscore( X.copy() )  # evaluate but don't apply

        w, omega = self.__get_initial( X, y, wlbl )

        # copies of prototypes and omegas stored in w_copy and omega_copy for the adaptive step size procedure
        w_copy = np.zeros( (w.shape[1], ncop, w.shape[0], ), dtype=np.cdouble )
        omega_copy = np.zeros( (omega.shape[1], ncop, omega.shape[0]), dtype=np.cdouble )

        # calculate initial values for learning curves
        costf, _, marg, _ = self.__compute_costs( X, y, w, wlbl, omega, self.mu )

        te[0] = np.sum(marg>0) / m
        cf[0] = costf

        # perform the first ncop init steps of gradient descent
        for i in range( 0, ncop ):

            # actual batch gradient step
            w, omega = self.__do_batchstep( X, y, w, wlbl, omega, etap, etam )
            w_copy[:,i,:] = w.T
            omega_copy[:,i,:] = omega.T

            # determine and save training set performances
            costf, _, marg, _ = self.__compute_costs( X, y, w, wlbl, omega, self.mu )

            te[i+1] = np.sum(marg>0) / m
            cf[i+1] = costf

            # compute training set errors and cost function values
            for j in range( 1, c+1 ):  # starting with 1 because of the labels
                # compute class-wise errors (positive margin = error)
                cw[i+1, j-1] = np.sum(marg[0, np.where(y == j)[0]] > 0) / np.sum(y == j)

        # perform totalsteps training steps
        for i in range( ncop, self.totalsteps ):

            # calculate mean positions over latest steps
            # note: normalization does not change cost function value but is done here for consistency
            w_mean = np.mean( w_copy, 1 ).T
            omega_mean = np.mean( omega_copy, 1 ).T
            omega_mean = omega_mean / np.sqrt(np.sum(np.sum(np.abs(omega_mean)**2)))

            # compute cost functions for mean prototypes and mean matrix
            costmp, _, _, _ = self.__compute_costs( X, y, w_mean, wlbl, omega,      0       )
            costmm, _, _, _ = self.__compute_costs( X, y, w,      wlbl, omega_mean, self.mu )

            # remember old positions for Papari procedure
            ombefore = omega.copy()
            protbefore = w.copy()

            # perform next step and compute costs etc.
            w, omega = self.__do_batchstep( X, y, w, wlbl, omega, etap, etam )

            # by default, step sizes are increased in every step
            etam = etam * incfac  # (small) increase of step sizes
            etap = etap * incfac  # at each learning step to enforce oscillatory behavior

            # costfunction values to compare with for Papari procedure
            # evaluated w.r.t. changing only matrix or prototype
            costfp, _, _, _ = self.__compute_costs( X, y, w,          wlbl, ombefore, 0       )
            costfm, _, _, _ = self.__compute_costs( X, y, protbefore, wlbl, omega,    self.mu )

            # heuristic extension of Papari procedure
            # treats matrix and prototype step sizes separately
            if costmp <= costfp:  # decrease prototype step size and jump
                # to mean prototypes
                etap = etap / decfac
                w = w_mean

            if costmm <= costfm:  # decrease matrix step size and jump
                # to mean matrix
                etam = etam / decfac
                omega = omega_mean

            # update the copies of the latest steps, shift stack of stored configs.
            # plenty of room for improvement, I guess ...
            for iicop in range( 0, ncop-1 ):
                w_copy[:,iicop,:] = w_copy[:,iicop+1,:]
                omega_copy[:,iicop,:] = omega_copy[:,iicop+1,:]

            w_copy[:,ncop-1,:] = w.T
            omega_copy[:,ncop-1,:] = omega.T

            # determine training and test set performances
            # here: costfunction without penalty term!
            costf0, _, marg, _ = self.__compute_costs( X, y, w, wlbl, omega, 0 )

            # compute total and class-wise training set errors
            te[i+1] = np.sum(marg>0) / m
            cf[i+1] = costf0

            for j in range( 1, c+1 ):
                cw[i+1, j-1] = np.sum(marg[0, np.where(y == j)[0]] > 0) / np.sum(y == j)

        lambdaa = omega.conj().T @ omega  # actual relevance matrix

        # if the data was z transformed: the inverse prototypes
        # if self.doztr:
        #     wInverse = self.__do_inversezscore( w.copy(), mf, st )

        self.gmlvq_system = { 'w': w, 'lambda': lambdaa, 'wlbl': wlbl, 'mean_features': mf, 'std_features': st }
        self.training_curves = { 'costs': cf, 'train_error': te, 'class_wise': cw }