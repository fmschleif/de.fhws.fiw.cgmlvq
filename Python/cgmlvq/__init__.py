from scipy.fft import fft, ifft
from scipy.linalg import sqrtm

import math
import numpy as np


class CGMLVQ:

    """ A classifier for complex valued data based on gmlvq.

    Parameters
    ----------
    coefficients : int, default=0
        Number of signal values in the frequency domain. If coefficients is 0, no fft is executed.

    totalsteps : int, default=50
        Number of batch gradient steps to be performed in each training run.

    Attributes
    ----------
    doztr : bool, default=(True/False)
        If true, do z transformation, otherwise you may have to adjust step sizes.

    rndinit : bool, default=(True/False)
        If true, initialize the relevance matrix randomly (if applicable), otherwise it is proportional to the identity matrix.

    mu : int, default=0
        Controls penalty of singular relevance matrix

        - 0 = unmodified GMLVQ algorithm (recommended for initial experiments)
        - > 0 = non-singular relevance matrix is enforced, mu controls dominance of leading eigenvectors continuously, prevents singular Lambda

    mode : int, default=1
        Control LVQ version

        - 0 = GMLVQ: matrix without null space correction
        - 1 = GMLVQ: matrix with null-space correction
        - 2 = GRLVQ: diagonal matrix only, sensitive to step sizes
        - 3 = GLVQ: relevance matrix proportional to identity (with Euclidean distance), "normalized identity matrix"

    Example
    -------
    >>> X = [[0], [1], [2], [3]]
    >>> Y = [0, 0, 1, 1]
    >>> from cgmlvq import CGMLVQ
    >>> cgmlvq = CGMLVQ()
    >>> cgmlvq.set_params( mode=0 )
    >>> cgmlvq.fit( X, y )
    >>> print( cgmlvq.predict([[0], [1]]) )

    Notes
    -----
    Based on the Matlab implementation from Michiel Straat.
    """

    doztr = True
    mode = 1
    mu = 0
    rndinit = False


    def __init__( self, coefficients=0, totalsteps=50 ):

        self.coefficients = coefficients
        self.totalsteps = totalsteps


    def fit( self, X, y ):

        """ Fit the classifier from the training dataset.

        Parameters
        ----------
        X : Training data
        y : Target values
        """

        X = np.array( X )
        y = np.array( y )

        if self.coefficients > 0:
            X = self.__fourier__( X )

        self.__run_single__( X, y, np.unique(y).T )


    def predict( self, X ):

        """ Predict the class labels for the provided data.

        Parameters
        ----------
        X : Test data

        Returns
        -------
        y : Class labels for each data sample
        """

        if self.coefficients > 0:
            X = self.__fourier__( X )

        crisp, _, _, _ = self.__classify_gmlvq__( X )

        return crisp[0]


    def set_params( self, **params ):

        """ Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters
        """

        if 'doztr' in params:
            if type( params['doztr'] ) == bool:
                self.doztr = params['doztr']
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


    def __check_arguments__( self, plbl, lbl, fvec, ncop ):

        """ Check consistency of some arguments and input parameters.

        Parameters
        ----------
        plbl : prototype labels
        lbl : data set labels
        fvec : feature vectors in data set
        ncop : number of copies in step size control procedure

        Returns
        -------
        lbl : data set labels, protentially transposed for consistency
        """

        lbl = np.array([ lbl ], dtype=int )
        if lbl.shape[1] > 1:  # if lbl is row vector
            lbl = lbl.T       # transpose to column vector

        if fvec.shape[0] != len(lbl):
            raise ValueError('number of training labels differs from number of samples')

        if min(lbl) != 1 or max(lbl) != len(np.unique(lbl)):
            raise ValueError('data labels should be: 1,2,3,...,nclasses')

        if len(np.unique(plbl)) != len(np.unique(lbl)):
            raise ValueError('number of prototype labels must equal number of classes')

        if sum(np.unique(plbl.T) != np.unique(lbl)) > 0:
            raise ValueError('prototype labels inconsistent with data, please rename/reorder')

        st = np.zeros( fvec.shape[1] )
        for i in range(0, fvec.shape[1]):
            st[i] = np.std( fvec[:,i], ddof=1 )  # standard deviation of feature i

        if min(st) < 1.e-10:
            raise ValueError('at least one feature displays (close to) zero variance')

        if ncop >= self.totalsteps:
            raise ValueError('number of gradient steps must be larger than ncop')

        return lbl


    def __classify_gmlvq__( self, fvec, lbl=0 ):

        """ Apply a gmlvq classifier to a given data set with unknown class labels for predication or known class labels for testing/validation

        Parameters
        ----------
        fvec : set of feature vectors to be classified

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

        prot    = self.gmlvq_system['protos']         # prototypes
        lambdaa = self.gmlvq_system['lambda']         # relevance matrix lambda
        plbl    = self.gmlvq_system['plbl']           # prototype labels
        mf      = self.gmlvq_system['mean_features']  # mean features from potential z-score
        st      = self.gmlvq_system['std_features']   # st.dev. from potential z-score transf.

        omat = sqrtm( lambdaa )  # symmetric matrix square root as one representation of the distance measure

        nfv = fvec.shape[0]           # number of feature vectors in training set

        if lbl == 0:                     # ground truth unknown
            lbl = np.ones( (1, nfv) ).T  # fake labels, meaningless

        # if z-transformation was applied in training, apply the same here:
        if self.doztr:
            for i in range( 0, nfv ):
                fvec[i,:] = (fvec[i,:] - mf) / st

        # call glvqcosts, crout=crisp labels
        # score between 0= "class 1 certainly" and 1= "any other class"
        # margin and costf are meaningful only if lbl is ground truth

        # cost function can be computed without penalty term for margins/score
        costf, crisp, margin, score = self.__compute_costs__( fvec, lbl, prot, plbl, omat, 0 )

        return crisp, score, margin, costf


    def __compute_costs__( self, fvec, lbl, prot, plbl, omat, mu ):

        """ Calculates gmlvq cost function, labels and margins for a set of labelled feature vectors, given a particular lvq system.

        Parameters
        ----------
        fvec : nvf feature vectors of dim. ndim fvec(1:nfv,1:ndim);
        lbl : data labels  lbl(1:nfv);
        prot : prototypes
        plbl : prototype labels
        omat : global matrix omega
        mu : mu>0 controls penalty for singular relevance matrix

        Returns
        -------
        costf : glvq-costs per example
        marg : margins of classifying the examples
        crout : crisp classifier outputs
        score : distance based "scores"
        """

        nfv = fvec.shape[0]
        npp = len( plbl )

        costf = 0
        marg  = np.zeros( (1,nfv) )
        score = np.zeros( (1,nfv) )
        crout = np.zeros( (1,nfv) )

        omat = omat / np.sqrt(sum(sum(omat*omat)))  # normalized omat

        for i in range( 0, nfv ):  # loop through examples

            fvi = fvec[i,:]  # actual example
            lbi = lbl[i]     # actual example

            # calculate squared distances to all prototypes
            dist = np.empty( (npp, 1) )  # define squared distances
            dist[:] = np.nan

            for j in range( 0, npp ):  # distances from all prototypes
                dist[j] = self.__euclid__( fvi, prot[j,:], omat )

            # find the two winning prototypes
            correct   = np.where( np.array([plbl]) == lbi )[1]  # all correct prototype indices
            incorrect = np.where( np.array([plbl]) != lbi )[1]  # all wrong   prototype indices

            dJ, JJ = dist[correct].min(0), dist[correct].argmin(0)      # correct winner
            dK, KK = dist[incorrect].min(0), dist[incorrect].argmin(0)  # wrong winner

            # winner indices
            jwin = correct[JJ][0]
            kwin = incorrect[KK][0]

            costf = costf + (dJ-dK) / (dJ+dK) / nfv

            marg[0, i] = (dJ-dK) / (dJ+dK)  # gmlvq margin of example iii

            # un-normalized difference of distances
            if lbi == 1:
                score[0, i] = dK - dJ
            else:
                score[0, i] = dJ - dK

            crout[0, i] = plbl[jwin] * (dJ <= dK) + plbl[kwin] * (dJ > dK)
            # the class label according to nearest prototype

        # add penalty term
        if mu > 0:
            costf = costf - mu / 2 * np.log(np.linalg.det(omat @ omat.conj().T)) / nfv

        return costf, crout, marg, score


    def __compute_roc__( self, binlbl, score, nthresh=5000 ):

        """ Threshold-based computation of the ROC scores are rescaled to fall into the range 0...1 and then compared with nthresh equi-distant thresholds note that nthresh must be large enough to guarantee correct ROC computation.

        Parameters
        ----------
        binlbl : binarized labels 0,1 for two classes (user-defined selection in multi-class problems)
        score : continuous valued score, e.g. glvq-score 0....1
        nthresh : number of threshold values, default: 2000
        """

        # Remark:
        # an alternative could be the more sophisticated built-in  roc
        # provided in the  Neural Network and/or Statistics toolboxe
        # which does not work with a fixed list of thresholds
        # but requires interpolation techniques for threshold-average
        # or the determination of the NPC performance. Syntax:
        # [tpr,fpr,thresholds] = roc(target,output);
        # only available with appropriate toolbox

        # heuristic rescaling of the scores to range 0....1  to be done: re-scale according to observed range of values
        score = 1 / (1 + np.exp(score/2))

        binlbl = binlbl.astype(int)  # True/False to 0,1
        target = binlbl.T            # should be 0,1

        tu = np.unique(target, axis=1)  # define binary target values
        t1 = tu[0][0]  # target value t1 representing "negative" class
        t2 = tu[0][1]  # target value t2 representing "positive" class

        # for proper "threshold-averaged" ROC (see paper by Fawcett)
        # we use "nthresh" equi-distant thresholds between 0 and 1

        if len(binlbl) > 1250:
            nthresh = 4 * len(binlbl)

        nthresh = 2 * math.floor(nthresh/2)  # make sure nthresh is even
        thresh = np.linspace(0,1,nthresh+1)  # odd number is actually used
        # so value 0.5 is in the list

        fpr = np.zeros( (1, nthresh+1) )  # false positive rates
        tpr = np.zeros( (1, nthresh+1) )  # true positive rates
        tpr[0][0] = 1  # only positives, so tpr=fpr=1
        fpr[0][0] = 1  # only positives, so tpr=fpr=1

        for i in range( 0, nthresh-1 ):
            # count true positves, false positives etc.
            tp = np.sum( target[score >  thresh[i+1]] == t2 )
            fp = np.sum( target[score >  thresh[i+1]] == t1 )
            fn = np.sum( target[score <= thresh[i+1]] == t2 )
            tn = np.sum( target[score <= thresh[i+1]] == t1 )
            # compute corresponding rates
            tpr[0][i+1] = tp / (tp + fn)
            fpr[0][i+1] = fp / (tn + fp)

        # simple numerical integration (trapezoidal rule)
        auroc = -np.trapz(tpr, fpr)[0]  # minus sign due to order of values

        return tpr, fpr, auroc, thresh


    def __do_batchstep__( self, fvec, lbl, proti, plbl, omegai, etap, etam ):

        """ Perform a single step of batch gradient descent GMLVQ with given step size for matrix and prototype updates (input parameter) only for one global quadratic omega matrix, potentially diagonal (mode=2)
            optional: null-space correction for full matrix only (mode=1)

        Parameters
        ----------
        fvec : nvf feature vectors of dim. ndim  fvec(1:nfv,1:ndim);
        lbl : training labels  lbl(1:nfv);
        proti : prototypes before the step
        plbl : prototype labels
        omegai : global matrix before the step
        etap,etam : prototype/matrix learning rate

        Returns
        -------
        prot : prototypes after update
        omat : omega matrix after update
        """

        ndim = fvec.shape[1]  # dimension of feature vectors
        nfv = len(lbl)        # number of feature vectors (training samples)
        npt = proti.shape[0]  # number of prototypes

        # omega and lambdaa before step
        omat = omegai
        lambdaa = omat.conj().T @ omat

        prot = proti  # prototypes before step

        chp = 0 * prot
        chm = 0 * omat  # initialize change of prot,omega

        for i in range( 0, nfv ):  # loop through (sum over) all training examples

            fvi = fvec[i,:]  # actual example
            lbi = lbl[i]     # actual example

            # calculate squared distances to all prototypes
            dist = np.empty( (npt, 1) )  # define squared distances
            dist[:] = np.nan

            for j in range( 0, npt ):  # distances from all prototypes
                dist[j] = self.__euclid__( fvi, prot[j,:], omat )

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

        # singularity control: add  derivative of penalty term times mu
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
        if self.mode == 2:                     # probably obsolete as chm diagonal
            omat = np.diag( np.diag(omat) )  # reduce to diagonal matrix

        #  nullspace correction using Moore Penrose pseudo-inverse
        if self.mode == 1:
            xvec = np.concatenate( (fvec, prot) )                            # concat. protos and fvecs
            omat = (omat @ xvec.conj().T) @ np.linalg.pinv( xvec.conj().T )  # corrected omega matrix

        if self.mode == 3:
            omat = np.identity( ndim )  # reset to identity regardless of gradients

        # normalization of omega, corresponds to Trace(lambda) = 1
        omat = omat / np.sqrt(np.sum(np.sum(np.absolute(omat)**2)))

        # one full, normalized gradient step performed, return omat and prot
        return prot, omat


    def __do_inversezscore__( self, fvec, mf, st ):

        ndim = fvec.shape[1]

        for i in range( 0, ndim ):
            fvec[:,i] = fvec[:,i] * st[0,i] + mf[0,i]

        return fvec


    def __do_zscore__( self, fvec ):

        """ Perform a z-score transformation of fvec

        Parameters
        ----------
        fvec : feature vectors

        Returns
        -------
        fvec : z-score transformed feature vectors
        mf : vector of feauter means used in z-score transformation
        st : vector of standard deviations in z-score transformation
        """

        ndim = fvec.shape[1]  # dimension ndim of data

        mf = np.zeros( (1, ndim), dtype=np.cdouble )  # initialize vectors mf and st
        st = np.zeros( (1, ndim) )

        for i in range( 0, ndim ):
            mf[0,i] = np.mean( fvec[:,i] )               # mean of feature i
            st[0,i] = np.std( fvec[:,i], ddof=1 )        # st.dev. of feature i
            fvec[:,i] = (fvec[:,i] - mf[0,i]) / st[0,i]  # transformed feature

        return fvec, mf, st


    def __euclid__( self, X, W, omat ):

        # D = (X' - P')' * (omat' * omat) * (X' - P');
        # Note that (B'A') = (AB)', therefore the formula can be written more intuitively in the simpler form, which is also cheaper to compute:

        D = np.linalg.norm( omat @ np.array([X-W]).T )**2
        D = D.real

        return D


    def __fourier__( self, X ):

        """ Wrapper around "fft" to obtain Fourier series of "x" truncated at "r" coefficients. Ignores the symmetric part of the spectrum.
        """

        Y = fft( X )

        enabled = np.zeros( Y.shape[1] )

        enabled[ 0 : self.coefficients+1 ] = 1

        Y = Y[:, enabled==1]

        return Y


    def __set_initial__( self, fvec, lbl, plbl ):

        """ Initialization of prototypes close to class conditional means small random displacements to break ties

        Parameters
        ----------
        fvec : feature vectors
        lbl : data labels
        plbl : prototype labels
        """

        ndim = fvec.shape[1]  # dimension of feature vectors
        nprots = len( plbl )  # total number of prototypes

        proti = np.zeros( (nprots, ndim), dtype=np.cdouble )

        for ic in range(0, nprots):  # compute class-conditional means
            proti[ic, :] = np.mean( fvec[np.where(lbl == plbl[ic]), :][0], axis=0 )

        # reproducible random numbers
        np.random.seed( 291024 )

        # displace randomly from class-conditional means
        proti = proti * (0.99 + 0.02 * np.random.rand(proti.shape[1], proti.shape[0]).T)
        # to do: run k-means per class

        # (global) matrix initialization, identity or random
        omi = np.identity( ndim )  # works for all values of mode if rndinit == 0

        if self.mode != 3 and self.rndinit:  # does not apply for mode==3 (GLVQ)
            omi = np.random.rand( ndim, ndim ).T - 0.5
            omi = omi.conj().T @ omi  # square symmetric
            # matrix of uniform random numbers

        if self.mode == 2:
            omi = np.diag(np.diag(omi))  # restrict to diagonal matrix

        omi = omi / np.sqrt(sum(sum(abs(omi)**2)))

        return proti, omi


    def __set_parameters__( self, fvec ):

        """ Set general parameters
            Set initial step sizes and control parameters of modified procedure based on [Papari, Bunte, Biehl]
        """

        nfv = fvec.shape[0]
        ndim = fvec.shape[1]

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

        if self.doztr and self.mode < 3:
            print('rescale relevances for proper interpretation')

        return etam, etap, decfac, incfac, ncop


    def __run_single__( self, fvec, lbl, plbl ):

        plbl = np.array( plbl, dtype=int )  # TODO: evtl in check_arguments

        etam0, etap0, decfac, incfac, ncop = self.__set_parameters__( fvec )

        etam = etam0  # initial step size matrix
        etap = etap0  # intitial step size prototypes

        lbl = self.__check_arguments__( plbl, lbl, fvec, ncop )

        nfv = fvec.shape[0]            # number of feature vectors in training set
        ncls = len( np.unique(plbl) )  # number of classes

        te = np.zeros( (self.totalsteps+1, 1) )   # define total error
        cf = np.zeros( (self.totalsteps+1, 1) )   # define cost function
        auc = np.zeros( (self.totalsteps+1, 1) )  # define AUC(ROC)

        cw = np.zeros( (self.totalsteps+1, ncls) )  # define class-wise errors

        stepsizem = np.zeros( (self.totalsteps+1, 1) )  # define stepsize matrix in the course of training
        stepsizep = np.zeros( (self.totalsteps+1, 1) )  # define stepsize prototypes in the course ...

        if self.doztr:
            fvec, mf, st = self.__do_zscore__( fvec.copy() )  # perform z-score transformation
        else:
            _, mf, st = self.__do_zscore__( fvec.copy() )     # evaluate but don't apply

        # initialize prototypes and omega
        proti, omi = self.__set_initial__( fvec, lbl, plbl )

        # initial values
        prot = proti
        om = omi

        # copies of prototypes and omegas stored in protcop and omcop
        # for the adaptive step size procedure
        protcop = np.zeros( (prot.shape[1], ncop, prot.shape[0], ), dtype=np.cdouble )
        omcop   = np.zeros( (om.shape[1], ncop, om.shape[0]), dtype=np.cdouble )

        # calculate initial values for learning curves
        costf, _, marg, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, self.mu )
        te[0] = np.sum(marg>0) / nfv
        cf[0] = costf
        stepsizem[0] = etam
        stepsizep[0] = etap

        _, _, auroc, _ = self.__compute_roc__( lbl>1, score )

        auc[0] = auroc

        # perform the first ncop steps of gradient descent
        for inistep in range( 0, ncop ):

            # actual batch gradient step
            prot, om = self.__do_batchstep__( fvec, lbl, prot, plbl, om, etap, etam )
            protcop[:,inistep,:] = prot.T
            omcop[:,inistep,:] = om.T

            # determine and save training set performances
            costf, _, marg, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, self.mu )
            te[inistep+1] = np.sum(marg>0) / nfv
            cf[inistep+1] = costf
            stepsizem[inistep+1] = etam
            stepsizep[inistep+1] = etap

            # compute training set errors and cost function values
            for icls in range( 1, ncls+1 ):  # starting with 1 because of the labels
                # compute class-wise errors (positive margin = error)
                cw[inistep+1, icls-1] = np.sum(marg[0, np.where(lbl==icls)[0]] > 0) / np.sum(lbl==icls)

            # training set roc with respect to class 1 versus all others only
            _, _, auroc, _ = self.__compute_roc__( lbl>1, score )
            auc[inistep+1] = auroc

        # perform totalsteps training steps
        for jstep in range( ncop, self.totalsteps ):

            # calculate mean positions over latest steps
            protmean = np.mean( protcop, 1 ).T
            ommean = np.mean( omcop, 1 ).T
            ommean = ommean / np.sqrt(np.sum(np.sum(np.abs(ommean)**2)))
            # note: normalization does not change cost function value
            #       but is done here for consistency

            # compute cost functions for mean prototypes, mean matrix and both
            costmp, _, _, _ = self.__compute_costs__( fvec, lbl, protmean, plbl, om,     0       )
            costmm, _, _, _ = self.__compute_costs__( fvec, lbl, prot,     plbl, ommean, self.mu )

            # remember old positions for Papari procedure
            ombefore = om.copy()
            protbefore = prot.copy()

            # perform next step and compute costs etc.
            prot, om = self.__do_batchstep__( fvec, lbl, prot, plbl, om, etap, etam )

            costf, _, _, _ = self.__compute_costs__( fvec, lbl, prot, plbl, om, self.mu )

            # by default, step sizes are increased in every step
            etam = etam * incfac  # (small) increase of step sizes
            etap = etap * incfac  # at each learning step to enforce oscillatory behavior

            # costfunction values to compare with for Papari procedure
            # evaluated w.r.t. changing only matrix or prototype
            costfp, _, _, _ = self.__compute_costs__( fvec, lbl, prot,       plbl, ombefore, 0       )
            costfm, _, _, _ = self.__compute_costs__( fvec, lbl, protbefore, plbl, om,       self.mu )

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
            costf0, _, marg, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, 0 )

            # compute total and class-wise training set errors
            te[jstep+1] = np.sum(marg>0) / nfv
            cf[jstep+1] = costf0

            for icls in range( 1, ncls+1 ):
                cw[jstep+1,icls-1] = np.sum(marg[0,np.where(lbl==icls)[0]] > 0) / np.sum(lbl==icls)

            stepsizem[jstep+1] = etam
            stepsizep[jstep+1] = etap

            # ROC with respect to class 1 (negative) vs. all others (positive)
            binlbl = lbl > 1
            _, _, auroc, _ = self.__compute_roc__( binlbl, score )
            auc[jstep+1] = auroc

        # if the data was z transformed then also save the inverse prototypes,
        # actually it is not necessary since the mf and st are returned.
        if self.doztr:
            protsInv = self.__do_inversezscore__( prot.copy(), mf, st )
        else:
            protsInv = prot

        lambdaa = om.conj().T @ om  # actual relevance matrix

        self.gmlvq_system = { 'protos': prot, 'protosInv': protsInv, 'lambda': lambdaa, 'plbl': plbl, 'mean_features': mf, 'std_features': st }
        self.training_curves = { 'costs': cf, 'train_error': te, 'class_wise': cw, 'auroc': auc }
        self.param_set = { 'etam0': etam0, 'etap0': etap0, 'etamfin': etam, 'etapfin': etap, 'decfac': decfac, 'infac': incfac, 'ncop': ncop }