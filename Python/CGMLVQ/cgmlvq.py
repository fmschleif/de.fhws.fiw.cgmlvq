from scipy.fft import fft

import math
import numpy as np


class CGMLVQ:


    def __init__( self, coefficients, epochs ):

        self.coefficients = coefficients
        self.epochs = epochs


    def fit( self, X, y ):

        X = np.array( X )
        y = np.array( y )

        row_length = X.shape[1]

        number_of_classes = len( np.unique(y) )

        X_frequency = self.__fourier__( X, self.coefficients )

        gmlvq_system, training_curves, param_set = self.__single__( X_frequency, y, self.coefficients, y.T )


    def predict( self, X ):

        pass


    def __fourier__( self, X, r ):  # r = coefficients

        """ Wrapper around "fft" to obtain Fourier series of "x" truncated at "r" coefficients. Ignores the symmetric part of the spectrum.
        """

        Y = fft( X )  # TODO: correct: fft( X, axis=2 )

        enabled = np.zeros( Y.shape[1] )

        enabled[0:r+1] = 1

        Y = Y[:, enabled==1]

        return Y


    def __single__( self, fvec, lbl, totalsteps, plbl ):

        # TODO: implement

        self.__do_zscore__( fvec )

        return 1, 2, 3


    def __do_zscore__( self, fvec ):

        # TODO: test

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

        mf = st = np.zeros( (1, ndim) )  # initialize vectors mf and st

        for i in range(0, ndim):
            mf[i] = np.mean( fvec[:,i] )           # mean of feature i
            st[i] = np.std( fvec[:,i] )            # st.dev. of feature i
            fvec[:,i] = (fvec[:,i]-mf[i]) / st[i]  # transformed feature

        return fvec, mf, st


    def __compute_costs__( self, fvec, lbl, prot, plbl, omat, mu ):

        # TODO: test

        """ Calculates gmlvq cost function, labels and margins for a set of labelled feature vectors, given a particular lvq system

        Parameters
        ----------
        fvec : nvf feature vectors of dim. ndim fvec(1:nfv,1:ndim);
        lbl : data labels  lbl(1:nfv);
        prot : prototypes
        plbl : prototype labels
        omat : global matrix omega
        mu : TODO

        Returns
        -------
        costf : glvq-costs per example
        marg : margins of classifying the examples
        crout : crisp classifier outputs
        score : distance based "scores"
        """

        nfv= fvec.shape[0]
        ndim = fvec.shape[1]  # and dim. of feature vectors
        np = len( plbl )

        costf = 0
        marg = np.zeros( (1,nfv) )
        score = marg
        crout = np.zeros( (1,nfv) )

        omat = omat / np.sqrt(sum(sum(omat*omat)))  # normalized omat

        for iii in range(0, nfv):  # loop through examples
            fvc = fvec[iii, :]
            lbc = lbl[iii]
            distl = np.empty( np, 1 )
            distl[:] = np.nan
            for jk in range(0, np):
                distl[jk] = self.__euclid__( fvc, prot[jk,:], omat )
            # find the two winning prototypes for example iii
            correct   = np.where( plbl == lbc )
            incorrect = np.where( plbl != lbc )
            [dJJ, JJJ] = min( distl(correct) )
            [dKK, KKK] = min( distl(incorrect) )
            JJ = correct[JJJ]
            KK = incorrect[KKK]  # winner indices

            costf = costf + (dJJ-dKK) / (dJJ+dKK) / nfv

            marg[iii] = (dJJ-dKK) / (dJJ+dKK)  # gmlvq margin of example iii

            # un-normalized difference of distances
            if( lbc == 1 ):
                # score(iii)= 1./(1+exp((dKK-dJJ)/2))  # "the larger the better"
                score[iii] = dKK-dJJ
                # distdiff(iii)=dKK-dJJ
                # score (iii) = 0.5* (1+marg(iii))
            else:
                # score(iii)= 1./(1+exp((dJJ-dKK)/2))  # "the larger the worse"
                score[iii] = dJJ-dKK
                # distdiff(iii)=dJJ-dKK
                # score (iii) = 0.5* (1-marg(iii))

            crout[iii] = plbl[JJ] * (dJJ<=dKK) + plbl(KK) * (dJJ>dKK)
            # the class label according to nearest prototype

        # add penalty term
        if( mu > 0 ):
            costf = costf-mu / 2*np.log(np.det(omat*omat.T)) / nfv

        return costf, marg, crout, score


    def __euclid__( self, X, W, omat ):

        # TODO: test

        # D = (X' - P')' * (omat' * omat) * (X' - P');
        # Note that (B'A') = (AB)', therefore the formula can be written more intuitively in the simpler form, which is also cheaper to compute:

        D = np.linalg.norm( omat*(X-W).T )^2
        D = D.real

        return D


    def __compute_roc__( self, binlbl, score, nthresh=5000 ):

        # TODO: test

        """ Threshold-based computation of the ROC scores are rescaled to fall into the range 0...1 and then compared with nthresh equi-distant thresholds note that nthresh must be large enough to guarantee correct ROC computation.

        Parameters
        ----------
        binllbl : binarized labels 0,1 for two classes (user-defined selection in multi-class problems)
        score : continuous valued score, e.g. glvq-socre 0....1
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
        score = 1./(1+np.exp(score/2))

        target = binlbl.T  # should be 0,1
        output = score
        len( binlbl )

        tu = np.unique(target)  # define binary target values
        t1 = tu(1)           # target value t1 representing "negative" class
        t2 = tu(2)           # target value t2 representing "positive" class

        npos = sum(target==t2)  # number of positive samples
        nneg = sum(target==t1)  # number of negative samples

        # for proper "threshold-averaged" ROC (see paper by Fawcett)
        # we use "nthresh" equi-distant thresholds between 0 and 1

        if( len(binlbl) > 1250 ):
            nthresh = 4 * len(binlbl)

        nthresh = 2 * math.floor(nthresh/2)  # make sure nthresh is even
        thresh = np.linspace(0,1,nthresh+1)  # odd number is actually used
        # so value 0.5 is in the list

        fpr = np.zeros( 1, nthresh+1 )  # false positive rates
        tpr = fpr                       # true positive rates
        tpr[0] = 1  # only positives, so tpr=fpr=1
        fpr[0] = 1     # only positives, so tpr=fpr=1

        for i in range(0, nthresh-1):
            # count true positves, false positives etc.
            tp = sum( target(score>thresh[i+1]) == t2 )
            fp = sum( target(score>thresh[i+1]) == t1 )
            fn = sum( target(score<=thresh[i+1]) == t2 )
            tn = sum( target(score<=thresh[i+1]) == t1 )
            # compute corresponding rates
            tpr[i+1] = tp/(tp+fn)
            fpr[i+1] = fp/(tn+fp)

        # simple numerical integration (trapezoidal rule)
        auroc = -np.trapz(fpr,tpr)  # minus sign due to order of values

        return tpr, fpr, auroc, thresh


    def __do_batchstep__( self, fvec, lbl, proti, plbl, omegai, etap, etam, mu, mode ):

        # TODO: implement

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
        mu : mu>0 controls penalty for singular relevance matrix
        mode : 0 for full, 1 for matrix with nullspace correction, 2 for diagonal matrix only (GRLVQ), 3 for normalized identity matrix (GLVQ)

        Returns
        -------
        prot : prototypes after update
        omat : omega matrix after update
        """

        ndim = fvec.sahpe[1]    # dimension of feature vectors
        nfv = len(lbl)      # number of feature vectors (training samples)
        cls = np.unique(lbl)      # set of class labels
        np = proti.shape[0]     # number of prototypes

        omat = omegai
        lambdaa = omat.T * omat   # omega and lambdaa before step

        # omat=sqrtm(lambdaa);
        prot = proti              # prototypes before step

        chp = 0*prot
        chm = 0*omat  # initialize change of prot,omega

        for i in range(0, nfv):  # loop through (sum over) all training examples
            fvi = fvec[i,:]  # actual example
            lbi = lbl(i)  # actual example
            # calculate squared distances to all prototypes
            dist = nan(np,1)                 # define squared distances
            for j in range(0, np):                       # distances from all prototypes
                dist[j] = self.__euclid__( fvi, prot[j,:], omat )

            # find the two winning prototypes
            correct = np.where( plbl == lbi )     # all correct prototype indices
            incorrect = np.where( plbl != lbi )     # all wrong   prototype indices

            [dJ, JJ] = min( dist(correct) )    # correct winner
            [dK, KK] = min( dist(incorrect) )  # wrong winner

            # winner indices
            jwin = correct(JJ)
            kwin = incorrect (KK)

            # winning prototypes
            wJ = prot[jwin,:]
            wK = prot[kwin,:]

            # GMLVQ prototype update for one example fvi
            DJ = (fvi-wJ).T;  DK = (fvi-wK).T        # displacement vectors
            #for complex valued data, "." has been added for normal transpose

            norm_factor = (dJ+dK)^2                # denominator of prefactor
            dwJ = -(dK/norm_factor)*lambdaa*DJ      # change of correct winner
            dwK =  (dJ/norm_factor)*lambdaa*DK      # change of incorrect winner
            # matrix update, single (global) matrix omat for one example
            f1 = ( dK/norm_factor)*(omat*DJ)*DJ.T   # term 1 of matrix change
            f2 = (-dJ/norm_factor)*(omat*DK)*DK.T   # term 2 of matrix change

            # negative gradient update added up over examples
            chp[jwin,:] = chp[jwin,:] - dwJ.T  # correct    winner summed update
            chp[kwin,:] = chp[kwin,:] - dwK.T  # incorrect winner summed update
            chm = chm - (f1 + f2)             # matrix summed update

        # singularity control: add  derivative of penalty term times mu
        if( mu > 0 ):
            chm = chm + mu * np.linalg.pinv(omat.T)

        # compute normalized gradient updates (length 1)
        # separate nomralization for prototypes and the matrix
        # computation of actual changes, diagonal matrix imposed here if nec.
        n2chw = 0               # zero initial value of sum
        for ni in range(0, np):             # loop through (sum over) set of prototypes  # total length of concatenated prototype vec.
            n2chw = n2chw + np.dot(chp[ni,:],chp[ni,:])

        if (mode==2):             # if diagonal matrix used only
            chm = np.diag(np.diag(chm))  # reduce to diagonal changes

        n2chm = sum(sum(abs(chm).^2))  # total 'length' of matrix update

        #     n2chm = sum(sum(abs(chm).^2));% total 'length' of matrix update
        #     for complex valued data abs has been added

        # final, normalized gradient updates after 1 loop through examples
        prot = prot  + etap * chp/np.sqrt(n2chw)
        omat = omat  + etam * chm/np.sqrt(n2chm)

        # if diagonal matrix only
        if( mode == 2 ):                  # probably obsolete as chm diagonal
            omat = np.diag(np.diag(omat))  # reduce to diagonal matrix

        #  nullspace correction using Moore Penrose pseudo-inverse
        if( mode == 1 ):
            xvec = [fvec;prot]               # concat. protos and fvecs
            omat = ((omat*xvec.T)*np.linalg.pinv(xvec.T))  # corrected omega matrix

        if( mode == 3 ):
            omat = np.identity(ndim)  # reset to identity regardless of gradients

        # normalization of omega, corresponds to Trace(lambda) = 1
        omat = omat / np.sqrt( sum(sum(abs(omat).^2)) )

        # one full, normalized gradient step performed, return omat and prot
        return prot, omat