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
            X = self.__fourier( X )

        self.__run_single( X, y )


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
            X = self.__fourier( X )

        crisp, _, _, _ = self.__classify_gmlvq( X )

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


    def get_params( self ):

        """ Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """

        return { 'coefficients': self.coefficients, 'totalsteps': self.totalsteps, 'doztr': self.doztr, 'mode': self.mode, 'mu': self.mu, 'rndinit': self.rndinit }


    def __check_arguments( self, X, y, ncop ):

        """ Check consistency of some arguments and input parameters.

        Parameters
        ----------
        X : feature vectors in data set
        y : data set labels
        ncop : number of copies in step size control procedure

        Returns
        -------
        lbl : data set labels, protentially transposed for consistency
        plbl : prototype labels
        """

        y = np.array( [y], dtype=int )
        if y.shape[1] > 1:  # if lbl is row vector
            y = y.T         # transpose to column vector

        plbl = np.unique( y )

        if X.shape[0] != len(y):
            raise ValueError('number of training labels differs from number of samples')

        if min(y) != 1 or max(y) != len(np.unique(y)):
            raise ValueError('data labels should be: 1,2,3,...,nclasses')

        if len(np.unique(plbl)) != len(np.unique(y)):
            raise ValueError('number of prototype labels must equal number of classes')

        if sum(np.unique(plbl.T) != np.unique(y)) > 0:
            raise ValueError('prototype labels inconsistent with data, please rename/reorder')

        st = np.zeros( X.shape[1] )
        for i in range( 0, X.shape[1] ):
            st[i] = np.std( X[:,i], ddof=1 )  # standard deviation of feature i

        if min(st) < 1.e-10:
            raise ValueError('at least one feature displays (close to) zero variance')

        if ncop >= self.totalsteps:
            raise ValueError('number of gradient steps must be larger than ncop')

        return y, plbl


    def __classify_gmlvq( self, X, y=0 ):

        """ Apply a gmlvq classifier to a given data set with unknown class labels for predication or known class labels for testing/validation

        Parameters
        ----------
        X : set of feature vectors to be classified

        Returns
        -------
        crisp : crisp labels of Nearest-Prototype-Classifier
        score : distance based score with respect to class 1
        margin : GLVQ-type margin with respect to class 1 evaluated in glvqcosts
        costf : GLVQ costfunction (if ground truth is known)
        """

        # for classification with unknown ground truth labels, use:
        # crisp, score = classify_gmlvq( fvec )

        # if used for testing with known test labels (lbl) you can use:
        # crisp, score, margin, costf = classify_gmlvq( fvec, lbl )

        X = np.copy( X )

        prot    = self.gmlvq_system['protos']         # prototypes
        lambdaa = self.gmlvq_system['lambda']         # relevance matrix lambda
        plbl    = self.gmlvq_system['plbl']           # prototype labels
        mf      = self.gmlvq_system['mean_features']  # mean features from potential z-score
        st      = self.gmlvq_system['std_features']   # st.dev. from potential z-score transf.

        omat = sqrtm( lambdaa )  # symmetric matrix square root as one representation of the distance measure

        nfv = X.shape[0]  # number of feature vectors in training set

        if y == 0:                   # ground truth unknown
            y = np.ones((1, nfv)).T  # fake labels, meaningless

        # if z-transformation was applied in training, apply the same here:
        if self.doztr:
            for i in range( 0, nfv ):
                X[i, :] = (X[i, :] - mf) / st

        # call glvqcosts, crout=crisp labels
        # score between 0= "class 1 certainly" and 1= "any other class"
        # margin and costf are meaningful only if lbl is ground truth

        # cost function can be computed without penalty term for margins/score
        costf, crisp, margin, score = self.__compute_costs( X, y, prot, plbl, omat, 0 )

        return crisp, score, margin, costf


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

        nfv = X.shape[0]
        npp = len( wlbl )

        costf = 0
        marg  = np.zeros( (1,nfv) )
        score = np.zeros( (1,nfv) )
        crout = np.zeros( (1,nfv) )

        omega = omega / np.sqrt(sum(sum(omega * omega)))  # normalized omat

        for i in range( 0, nfv ):  # loop through examples

            Xi = X[i, :]  # actual example
            yi = y[i]     # actual example

            # calculate squared distances to all prototypes
            d = np.empty( (npp, 1) )  # define squared distances
            d[:] = np.nan

            for j in range( 0, npp ):  # distances from all prototypes
                d[j] = self.__euclid( Xi, w[j, :], omega )

            # find the two winning prototypes
            correct   = np.where( np.array([wlbl]) == yi )[1]  # all correct prototype indices
            incorrect = np.where( np.array([wlbl]) != yi )[1]  # all wrong   prototype indices

            dJ, JJ = d[correct].min(0), d[correct].argmin(0)      # correct winner
            dK, KK = d[incorrect].min(0), d[incorrect].argmin(0)  # wrong winner

            # winner indices
            jwin = correct[JJ][0]
            kwin = incorrect[KK][0]

            costf = costf + (dJ-dK) / (dJ+dK) / nfv

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
            costf = costf - mu / 2 * np.log(np.linalg.det(omega @ omega.conj().T)) / nfv

        return costf, crout, marg, score


    def __do_batchstep( self, X, y, proti, plbl, omegai, etap, etam ):

        """ Perform a single step of batch gradient descent GMLVQ with given step size for matrix and prototype updates (input parameter) only for one global quadratic omega matrix, potentially diagonal (mode=2)
            optional: null-space correction for full matrix only (mode=1)

        Parameters
        ----------
        X : nvf feature vectors of dim. ndim
        y : training labels
        proti : prototypes before the step
        plbl : prototype labels
        omegai : global matrix before the step
        etap,etam : prototype/matrix learning rate

        Returns
        -------
        prot : prototypes after update
        omat : omega matrix after update
        """

        ndim = X.shape[1]  # dimension of feature vectors
        nfv = len(y)        # number of feature vectors (training samples)
        npt = proti.shape[0]  # number of prototypes

        # omega and lambdaa before step
        omat = omegai
        lambdaa = omat.conj().T @ omat

        prot = proti  # prototypes before step

        chp = 0 * prot
        chm = 0 * omat  # initialize change of prot,omega

        for i in range( 0, nfv ):  # loop through (sum over) all training examples

            fvi = X[i, :]  # actual example
            lbi = y[i]     # actual example

            # calculate squared distances to all prototypes
            dist = np.empty( (npt, 1) )  # define squared distances
            dist[:] = np.nan

            for j in range( 0, npt ):  # distances from all prototypes
                dist[j] = self.__euclid( fvi, prot[j,:], omat )

            # find the two winning prototypes
            correct   = np.where( np.array([plbl]) == lbi )[1]  # all correct prototype indices
            incorrect = np.where( np.array([plbl]) != lbi )[1]  # all wrong   prototype indices

            dJ, JJ = dist[correct].min(0), dist[correct].argmin(0)      # correct winner
            dK, KK = dist[incorrect].min(0), dist[incorrect].argmin(0)  # wrong winner

            # winner indices
            jwin = correct[JJ][0]
            kwin = incorrect[KK][0]

            # winning prototypes
            wJ = prot[jwin,:]
            wK = prot[kwin,:]

            # GMLVQ prototype update for one example fvi
            DJ = np.array([ fvi - wJ ]).T  # displacement vectors
            DK = np.array([ fvi - wK ]).T  # displacement vectors
            # for complex valued data, "." has been added for normal transpose

            norm_factor = (dJ + dK)**2  # denominator of prefactor

            dwJ = -(dK/norm_factor) * lambdaa @ DJ  # change of correct winner
            dwK =  (dJ/norm_factor) * lambdaa @ DK  # change of incorrect winner

            # matrix update, single (global) matrix omat for one example
            f1 = ( dK/norm_factor) * (omat@DJ) @ DJ.conj().T
            f2 = (-dJ/norm_factor) * (omat@DK) @ DK.conj().T

            # negative gradient update added up over examples
            chp[jwin,:] = chp[jwin,:] - dwJ.conj().T  # correct   winner summed update
            chp[kwin,:] = chp[kwin,:] - dwK.conj().T  # incorrect winner summed update
            chm = chm - (f1 + f2)                     # matrix summed update

        # singularity control: add derivative of penalty term times mu
        if self.mu > 0:
            chm = chm + self.mu * np.linalg.pinv( omat.conj().T )

        # compute normalized gradient updates (length 1)
        # separate nomralization for prototypes and the matrix
        # computation of actual changes, diagonal matrix imposed here if nec.
        n2chw = np.sum( chp.conj() * chp ).real

        if self.mode == 2:                 # if diagonal matrix used only
            chm = np.diag(np.diag(chm))  # reduce to diagonal changes

        n2chm = np.sum(np.sum(np.absolute(chm)**2))  # total 'length' of matrix update

        # n2chm = sum(sum(abs(chm).^2));% total 'length' of matrix update
        # for complex valued data abs has been added

        # final, normalized gradient updates after 1 loop through examples
        prot = prot + etap * chp / np.sqrt(n2chw)
        omat = omat + etam * chm / np.sqrt(n2chm)

        # if diagonal matrix only
        if self.mode == 2:                   # probably obsolete as chm diagonal
            omat = np.diag( np.diag(omat) )  # reduce to diagonal matrix

        #  nullspace correction using Moore Penrose pseudo-inverse
        if self.mode == 1:
            xvec = np.concatenate((X, prot))                                 # concat. protos and fvecs
            omat = (omat @ xvec.conj().T) @ np.linalg.pinv( xvec.conj().T )  # corrected omega matrix

        if self.mode == 3:
            omat = np.identity( ndim )  # reset to identity regardless of gradients

        # normalization of omega, corresponds to Trace(lambda) = 1
        omat = omat / np.sqrt(np.sum(np.sum(np.absolute(omat)**2)))

        # one full, normalized gradient step performed, return omat and prot
        return prot, omat


    def __do_inversezscore( self, X, mf, st ):

        ndim = X.shape[1]

        for i in range( 0, ndim ):
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
        mf : vector of feauter means used in z-score transformation
        st : vector of standard deviations in z-score transformation
        """

        ndim = X.shape[1]  # dimension ndim of data

        mf = np.zeros( (1, ndim), dtype=np.cdouble )  # initialize vectors mf and st
        st = np.zeros( (1, ndim) )

        for i in range( 0, ndim ):
            mf[0,i] = np.mean(X[:,i])               # mean of feature i
            st[0,i] = np.std(X[:,i], ddof=1)        # st.dev. of feature i
            X[:, i] = (X[:,i] - mf[0,i]) / st[0,i]  # transformed feature

        return X, mf, st


    def __euclid( self, X, w, omega ):

        # d = (X - w).conj().T @ omega.conj().T @ omega @ (X - w)
        # d = d.real

        # simpler form, which is also cheaper to compute
        d = np.linalg.norm(omega @ np.array([X - w]).T)**2
        d = d.real

        return d


    def __fourier( self, X ):

        """ Wrapper around "fft" to obtain Fourier series of "x" truncated at "r" coefficients. Ignores the symmetric part of the spectrum.
        """

        Y = fft( X )

        enabled = np.zeros( Y.shape[1] )

        enabled[ 0 : self.coefficients+1 ] = 1

        Y = Y[:, enabled==1]

        return Y


    def __set_initial( self, X, y, wlbl ):

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

        ndim = X.shape[1]     # dimension of feature vectors
        nprots = len( wlbl )  # total number of prototypes

        w = np.zeros( (nprots, ndim), dtype=np.cdouble )

        for i in range( 0, nprots ):  # compute class-conditional means
            w[i, :] = np.mean( X[np.where(y == wlbl[i]), :][0], axis=0 )

        # reproducible random numbers
        np.random.seed( 291024 )

        # displace randomly from class-conditional means
        w = w * (0.99 + 0.02 * np.random.rand(w.shape[1], w.shape[0]).T)

        # (global) matrix initialization, identity or random
        omega = np.identity( ndim )  # works for all values of mode if rndinit == 0

        if self.mode != 3 and self.rndinit:  # does not apply for mode==3 (GLVQ)
            omega = np.random.rand( ndim, ndim ).T - 0.5
            omega = omega.conj().T @ omega  # square symmetric
            # matrix of uniform random numbers

        if self.mode == 2:
            omega = np.diag(np.diag(omega))  # restrict to diagonal matrix

        omega = omega / np.sqrt(sum(sum(abs(omega)**2)))

        return w, omega


    def __set_parameters( self, X ):

        """ Set general parameters
            Set initial step sizes and control parameters of modified procedure based on [Papari, Bunte, Biehl]

        Parameters
        ----------
        X : feature vectors
        """

        nfv = X.shape[0]
        ndim = X.shape[1]

        # parameters of stepsize adaptation
        if self.mode < 2:  # full matrix updates with (0) or w/o (1) null space correction
            etam = 2  # suggestion: 2
            etap = 1  # suggestion: 1

        elif self.mode == 2:  # diagonal relevances only, DISCOURAGED
            etam = 0.2  # initital step size for diagonal matrix updates
            etap = 0.1  # initital step size for prototype update

        elif self.mode == 3:  # GLVQ, equivalent to Euclidean distance
            etam = 0
            etap = 1

        decfac = 1.5  # step size factor (decrease) for Papari steps
        incfac = 1.1  # step size factor (increase) for all steps
        ncop = 5      # number of waypoints stored and averaged

        if nfv <= ndim and self.mode == 0:
            print('dim. > # of examples, null-space correction recommended')

        if not self.doztr and self.mode < 3:
            print('rescale relevances for proper interpretation')

        return etam, etap, decfac, incfac, ncop


    def __run_single( self, X, y ):

        etam0, etap0, decfac, incfac, ncop = self.__set_parameters( X )

        etam = etam0  # initial step size matrix
        etap = etap0  # intitial step size prototypes

        y, plbl = self.__check_arguments( X, y, ncop )

        nfv = X.shape[0]               # number of feature vectors in training set
        ncls = len( np.unique(plbl) )  # number of classes

        te = np.zeros( (self.totalsteps+1, 1) )   # define total error
        cf = np.zeros( (self.totalsteps+1, 1) )   # define cost function
        cw = np.zeros( (self.totalsteps+1, ncls) )  # define class-wise errors

        stepsizem = np.zeros( (self.totalsteps+1, 1) )  # define stepsize matrix in the course of training
        stepsizep = np.zeros( (self.totalsteps+1, 1) )  # define stepsize prototypes in the course ...

        if self.doztr:
            X, mf, st = self.__do_zscore( X.copy() )  # perform z-score transformation
        else:
            _, mf, st = self.__do_zscore( X.copy() )  # evaluate but don't apply

        # initialize prototypes and omega
        prot, om = self.__set_initial( X, y, plbl )

        # copies of prototypes and omegas stored in protcop and omcop
        # for the adaptive step size procedure
        protcop = np.zeros( (prot.shape[1], ncop, prot.shape[0], ), dtype=np.cdouble )
        omcop   = np.zeros( (om.shape[1], ncop, om.shape[0]), dtype=np.cdouble )

        # calculate initial values for learning curves
        costf, _, marg, _ = self.__compute_costs( X, y, prot, plbl, om, self.mu )
        te[0] = np.sum(marg>0) / nfv
        cf[0] = costf
        stepsizem[0] = etam
        stepsizep[0] = etap

        # perform the first ncop init steps of gradient descent
        for i in range( 0, ncop ):

            # actual batch gradient step
            prot, om = self.__do_batchstep( X, y, prot, plbl, om, etap, etam )
            protcop[:,i,:] = prot.T
            omcop[:,i,:] = om.T

            # determine and save training set performances
            costf, _, marg, _ = self.__compute_costs( X, y, prot, plbl, om, self.mu )
            te[i+1] = np.sum(marg>0) / nfv
            cf[i+1] = costf
            stepsizem[i+1] = etam
            stepsizep[i+1] = etap

            # compute training set errors and cost function values
            for j in range( 1, ncls+1 ):  # starting with 1 because of the labels
                # compute class-wise errors (positive margin = error)
                cw[i+1, j-1] = np.sum(marg[0, np.where(y == j)[0]] > 0) / np.sum(y == j)

        # perform totalsteps training steps
        for i in range( ncop, self.totalsteps ):

            # calculate mean positions over latest steps
            # note: normalization does not change cost function value but is done here for consistency
            protmean = np.mean( protcop, 1 ).T
            ommean = np.mean( omcop, 1 ).T
            ommean = ommean / np.sqrt(np.sum(np.sum(np.abs(ommean)**2)))

            # compute cost functions for mean prototypes, mean matrix and both
            costmp, _, _, _ = self.__compute_costs( X, y, protmean, plbl, om,     0       )
            costmm, _, _, _ = self.__compute_costs( X, y, prot,     plbl, ommean, self.mu )

            # remember old positions for Papari procedure
            ombefore = om.copy()
            protbefore = prot.copy()

            # perform next step and compute costs etc.
            prot, om = self.__do_batchstep( X, y, prot, plbl, om, etap, etam )

            # by default, step sizes are increased in every step
            etam = etam * incfac  # (small) increase of step sizes
            etap = etap * incfac  # at each learning step to enforce oscillatory behavior

            # costfunction values to compare with for Papari procedure
            # evaluated w.r.t. changing only matrix or prototype
            costfp, _, _, _ = self.__compute_costs( X, y, prot,       plbl, ombefore, 0       )
            costfm, _, _, _ = self.__compute_costs( X, y, protbefore, plbl, om,       self.mu )

            # heuristic extension of Papari procedure
            # treats matrix and prototype step sizes separately
            if costmp <= costfp:  # decrease prototype step size and jump
                # to mean prototypes
                etap = etap / decfac
                prot = protmean

            if costmm <= costfm:  # decrease matrix step size and jump
                # to mean matrix
                etam = etam / decfac
                om = ommean

            # update the copies of the latest steps, shift stack of stored configs.
            # plenty of room for improvement, I guess ...
            for iicop in range( 0, ncop-1 ):
                protcop[:,iicop,:] = protcop[:,iicop+1,:]
                omcop[:,iicop,:] = omcop[:,iicop+1,:]

            protcop[:,ncop-1,:] = prot.T
            omcop[:,ncop-1,:] = om.T

            # determine training and test set performances
            # here: costfunction without penalty term!
            costf0, _, marg, _ = self.__compute_costs( X, y, prot, plbl, om, 0 )

            # compute total and class-wise training set errors
            te[i+1] = np.sum(marg>0) / nfv
            cf[i+1] = costf0

            for j in range( 1, ncls+1):
                cw[i+1, j-1] = np.sum(marg[0, np.where(y == j)[0]] > 0) / np.sum(y == j)

            stepsizem[i+1] = etam
            stepsizep[i+1] = etap

        # if the data was z transformed then also save the inverse prototypes,
        # actually it is not necessary since the mf and st are returned.
        if self.doztr:
            protsInv = self.__do_inversezscore( prot.copy(), mf, st )
        else:
            protsInv = prot

        lambdaa = om.conj().T @ om  # actual relevance matrix

        self.gmlvq_system = { 'protos': prot, 'protosInv': protsInv, 'lambda': lambdaa, 'plbl': plbl, 'mean_features': mf, 'std_features': st }
        self.training_curves = { 'costs': cf, 'train_error': te, 'class_wise': cw }
        self.param_set = { 'etam0': etam0, 'etap0': etap0, 'etamfin': etam, 'etapfin': etap, 'decfac': decfac, 'infac': incfac, 'ncop': ncop }