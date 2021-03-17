from .helper import vdot

from scipy.fft import fft, ifft

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

        gmlvq_system, training_curves, param_set = self.__single__( X_frequency, y, self.epochs, np.unique(y).T )

        backProts = self.__iFourier__( gmlvq_system["protosInv"], row_length )  # wrapper around inverse Fourier


    def predict( self, X ):

        pass


    def __check_arguments__( self, plbl, lbl, fvec, ncop, totalsteps ):

        # check consistency of some arguments and input parameters

        # plbl:  prototype labels
        # lbl :  data set labels
        # fvec:  feature vectors in data set
        # ncop:  number of copies in step size control procedure
        # totalsteps: total number of batch gradient update steps

        # output:
        # lbl :  data set labels, protentially transposed for consistency

        # TODO: matthias
        lbl = np.array([ lbl ], dtype=int )
        if( lbl.shape[1] > 1 ):   # lbl may be column or row vector
            lbl = lbl.T
            print('vector lbl has been transposed')

        if fvec.shape[0] != len(lbl):
            raise ValueError('number of training labels differs from number of samples')

        if( min(lbl)!=1 or max(lbl)!=len(np.unique(lbl)) ):
            print('unique(lbl)=  ' + str(np.unique(lbl)))
            raise ValueError('data labels should be: 1,2,3,...,nclasses')

        if( len(np.unique(plbl))>2 ):
            print( ['multi-class problem, ROC analysis is for class 1 (neg.)', ' vs. all others (pos.)'] )

        if( len(np.unique(plbl)) != len(np.unique(lbl)) ):
            print('unique(plbl)=   ' + str(np.unique(plbl)))
            raise ValueError('number of prototype labels must equal number of classes')

        if( sum(np.unique(plbl.T) != np.unique(lbl)) > 0 ):
           print('unique(plbl)=   ' + str(np.unique(plbl)))
           raise ValueError('prototype labels inconsistent with data, please rename/reorder')

        st = np.zeros( fvec.shape[1] )
        for i in range(0, fvec.shape[1]):
            st[i] = np.std( fvec[:,i], ddof=1 )  # standard deviation of feature i

        print(' ')
        print('minimum standard deviation of features: ' + str(min(st)))
        print(' ')

        if( min(st) < 1.e-10 ):
            raise ValueError('at least one feature displays (close to) zero variance')

        if( ncop >= totalsteps ):
            raise ValueError('number of gradient steps must be larger than ncop')

        return lbl


    def __compute_costs__( self, fvec, lbl, prot, plbl, omat, mu ):

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

        nfv = fvec.shape[0]
        ndim = fvec.shape[1]  # and dim. of feature vectors
        npp = len( plbl )

        costf = 0
        marg = np.zeros( (1,nfv) )
        score = np.zeros( (1,nfv) )
        crout = np.zeros( (1,nfv) )

        omat = omat / np.sqrt(sum(sum(omat*omat)))  # normalized omat

        for iii in range(0, nfv):  # loop through examples

            fvc = fvec[iii,:]
            lbc = lbl[iii]

            distl = np.empty( (npp, 1) )
            distl[:] = np.nan

            for jk in range(0, npp):
                distl[jk] = self.__euclid__( fvc, prot[jk,:], omat )

            # find the two winning prototypes for example iii
            correct   = np.where( np.array([plbl]) == lbc )[1]
            incorrect = np.where( np.array([plbl]) != lbc )[1]

            dJJ = min( distl[correct] )
            dKK = min( distl[incorrect] )

            JJJ = np.where( distl[correct]   == dJJ )[0]
            KKK = np.where( distl[incorrect] == dKK )[0]

            JJ = correct[JJJ][0]
            KK = incorrect[KKK][0]  # winner indices

            costf = costf + (dJJ-dKK) / (dJJ+dKK) / nfv

            marg[0,iii] = (dJJ-dKK) / (dJJ+dKK)  # gmlvq margin of example iii

            # un-normalized difference of distances
            if( lbc == 1 ):
                # score(iii)= 1./(1+exp((dKK-dJJ)/2))  # "the larger the better"
                score[0,iii] = dKK - dJJ
                # distdiff(iii)=dKK-dJJ
                # score (iii) = 0.5* (1+marg(iii))
            else:
                # score(iii)= 1./(1+exp((dJJ-dKK)/2))  # "the larger the worse"
                score[0,iii] = dJJ - dKK
                # distdiff(iii)=dJJ-dKK
                # score (iii) = 0.5* (1-marg(iii))

            crout[0,iii] = plbl[JJ] * (dJJ<=dKK) + plbl[KK] * (dJJ>dKK)
            # the class label according to nearest prototype

        # add penalty term
        if( mu > 0 ):
            costf = costf - mu / 2 * np.log(np.linalg.det(omat*omat.T)) / nfv

        return costf, crout, marg, score


    def __compute_roc__( self, binlbl, score, nthresh=5000 ):

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
        score = 1 / (1 + np.exp(score/2))

        binlbl = binlbl.astype(int)  # True/False to 0,1
        target = binlbl.T  # should be 0,1
        output = score
        len( binlbl )

        tu = np.unique(target, axis=1)  # define binary target values
        t1 = tu[0][0]           # target value t1 representing "negative" class
        t2 = tu[0][1]           # target value t2 representing "positive" class

        npos = np.sum(target == t2)  # number of positive samples
        nneg = np.sum(target == t1)  # number of negative samples

        # for proper "threshold-averaged" ROC (see paper by Fawcett)
        # we use "nthresh" equi-distant thresholds between 0 and 1

        if( len(binlbl) > 1250 ):
            nthresh = 4 * len(binlbl)

        nthresh = 2 * math.floor(nthresh/2)  # make sure nthresh is even
        thresh = np.linspace(0,1,nthresh+1)  # odd number is actually used
        # so value 0.5 is in the list

        fpr = np.zeros( (1, nthresh+1) )  # false positive rates
        tpr = np.zeros( (1, nthresh+1) )                       # true positive rates
        tpr[0][0] = 1  # only positives, so tpr=fpr=1
        fpr[0][0] = 1  # only positives, so tpr=fpr=1

        for i in range(0, nthresh-1):
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


    def __do_batchstep__( self, fvec, lbl, proti, plbl, omegai, etap, etam, mu, mode ):

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

        ndim = fvec.shape[1]          # dimension of feature vectors
        nfv = len(lbl)                # number of feature vectors (training samples)
        cls = np.unique(lbl, axis=0)  # set of class labels
        npt = proti.shape[0]          # number of prototypes

        # omega and lambdaa before step
        omat = omegai
        lambdaa = vdot( omat.T, omat )

        # omat=sqrtm(lambdaa);
        prot = proti              # prototypes before step

        chp = 0 * prot
        chm = 0 * omat  # initialize change of prot,omega

        for i in range( 0, nfv ):  # loop through (sum over) all training examples

            # TODO: doppelter Code

            fvi = fvec[i,:]  # actual example
            lbi = lbl[i]     # actual example

            # calculate squared distances to all prototypes
            dist = np.empty( (npt, 1) )  # define squared distances
            dist[:] = np.nan

            for j in range(0, npt):  # distances from all prototypes
                dist[j] = self.__euclid__( fvi, prot[j,:], omat )

            # find the two winning prototypes
            correct   = np.where( np.array([plbl]) == lbi )[1]  # all correct prototype indices
            incorrect = np.where( np.array([plbl]) != lbi )[1]  # all wrong   prototype indices

            dJ = min( dist[correct] )    # correct winner
            dK = min( dist[incorrect] )  # wrong winner

            JJ = np.where( dist[correct]   == dJ )[0]
            KK = np.where( dist[incorrect] == dK )[0]

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

            dwJ = vdot( -(dK/norm_factor) * lambdaa, DJ )  # change of correct winner
            dwK = vdot(  (dJ/norm_factor) * lambdaa, DK )  # change of incorrect winner

            # matrix update, single (global) matrix omat for one example
            f1 = vdot( (( dK/norm_factor) * vdot(omat, DJ)), DJ.T )
            f2 = vdot( ((-dJ/norm_factor) * vdot(omat, DK)), DK.T )

            f1 = np.conj( f1 )  # vorzeichen bei imaginärteil umkehren
            f2 = np.conj( f2 )

            # negative gradient update added up over examples
            chp[jwin,:] = chp[jwin,:] - dwJ.T  # correct   winner summed update
            chp[kwin,:] = chp[kwin,:] - dwK.T  # incorrect winner summed update
            chm = chm - (f1 + f2)              # matrix summed update

        # singularity control: add  derivative of penalty term times mu
        if( mu > 0 ):
            chm = chm + mu * np.linalg.pinv( omat.T )

        # compute normalized gradient updates (length 1)
        # separate nomralization for prototypes and the matrix
        # computation of actual changes, diagonal matrix imposed here if nec.
        n2chw = np.sum( chp.conj() * chp ).real

        if( mode == 2 ):                 # if diagonal matrix used only
            chm = np.diag(np.diag(chm))  # reduce to diagonal changes

        n2chm = np.sum(np.sum(np.absolute(chm)**2))  # total 'length' of matrix update

        # n2chm = sum(sum(abs(chm).^2));% total 'length' of matrix update
        # for complex valued data abs has been added

        # final, normalized gradient updates after 1 loop through examples
        prot = prot + np.conj( etap * chp / np.sqrt(n2chw) )
        omat = omat +          etam * chm / np.sqrt(n2chm)

        # if diagonal matrix only
        if( mode == 2 ):                     # probably obsolete as chm diagonal
            omat = np.diag( np.diag(omat) )  # reduce to diagonal matrix

        #  nullspace correction using Moore Penrose pseudo-inverse
        if( mode == 1 ):
            xvec = np.concatenate( (fvec, prot) )                       # concat. protos and fvecs
            omat = np.dot(np.dot(omat,xvec.T), np.linalg.pinv(xvec.T))  # corrected omega matrix

        if( mode == 3 ):
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
        st = np.zeros( (1, ndim), dtype=np.cdouble )  # TODO: Matthias: reicht normal?

        for i in range( 0, ndim ):
            mf[0,i] = np.mean( fvec[:,i] )             # mean of feature i
            st[0,i] = np.std( fvec[:,i], ddof=1 )      # st.dev. of feature i
            fvec[:,i] = (fvec[:,i]-mf[0,i]) / st[0,i]  # transformed feature

        return fvec, mf, st


    def __euclid__( self, X, W, omat ):

        # D = (X' - P')' * (omat' * omat) * (X' - P');
        # Note that (B'A') = (AB)', therefore the formula can be written more intuitively in the simpler form, which is also cheaper to compute:

        D = np.linalg.norm( np.dot(omat, np.array([X-W]).T) )**2
        D = D.real

        return D


    def __euclid_cc__( self, X, W, omat ):

        # complex conjugating

        D = np.linalg.norm( vdot(omat, np.array([X-W]).T) )**2
        D = D.real

        return D


    def __fourier__( self, X, r ):  # r = coefficients

        """ Wrapper around "fft" to obtain Fourier series of "x" truncated at "r" coefficients. Ignores the symmetric part of the spectrum.
        """

        Y = fft( X )  # TODO: correct: fft( X, axis=2 )

        enabled = np.zeros( Y.shape[1] )

        enabled[0:r+1] = 1

        Y = Y[:, enabled==1]

        return Y


    def __iFourier__( self, X, L ):

        """ Wrapper around "ifft" to retrieve the original time domain signal "y", given Fourier coefficients "X" and the length "L" of the original time domain signal.
        """

        y = np.zeros( (X.shape[0], L) )  # the size of the original data matrix
        enabled = np.zeros( (1, L) )

        r = X.shape[1] - 1

        last_index = len( enabled[0,:] )
        enabled[0, 0:1+r] = 1
        enabled[0, last_index-r+1:last_index] = 1

        one = ( y[:, np.where(enabled==1)[1]] ).shape[1]
        two = np.concatenate( (X, np.fliplr(np.conj(X[:,1:None]))), axis=1 ).shape[1]
        if( one == two ):
            y[:, np.where(enabled==1)[1]] = np.concatenate( (X, np.fliplr(np.conj(X[:,1:None]))), axis=1 )

        y = ifft(y)

        return y


    def __set_initial__( self, fvec, lbl, plbl, mode, rndinit ):

        """ Initialization of prototypes close to class conditional means small random displacements to break ties

        Parameters
        ----------
        fvec : feature vectors
        lbl : data labels
        plbl : prototype labels
        mode : 0,1 for full matrix (GMLVQ)
               2 for diagonal matrix (GRLVQ)
               3 for prop. to identity (GLVQ)
        """

        ndim = fvec.shape[1]  # dimension of feature vectors
        nprots = len( plbl )  # total number of prototypes

        proti = np.zeros( (nprots, nprots), dtype=np.cdouble )

        for ic in range(0, nprots):  # compute class-conditional means
            proti[ic, :] = np.mean( fvec[np.where(lbl == plbl[ic]), :][0], axis=0 )

        mat_rand = np.array([ [0.0710, 0.0537, 0.7551],
                              [0.2888, 0.5008, 0.4310],
                              [0.9612, 0.3751, 0.9873] ])

        # displace randomly from class-conditional means
        proti = proti * (0.99 + 0.02 * mat_rand)  # TODO: Matlab erzeugt imemr die selbe random-Matrix in jedem Durchlauf, daher für Testzwecke die genommen. Originalcode: np.random.rand(proti.shape[0], proti.shape[1])
        # to do: run k-means per class

        # (global) matrix initialization, identity or random
        omi = np.identity( ndim )          # works for all values of mode if rndinit == 0

        if( mode != 3 and rndinit == 1 ):  # does not apply for mode==3 (GLVQ)
            omi = np.random.rand(ndim) - 0.5
            omi = omi.T*omi  # square symmetric
            #  matrix of uniform random numbers

        if( mode == 2 ):
            omi = np.diag(np.diag(omi))  # restrict to diagonal matrix

        omi = omi / np.sqrt(sum(sum(abs(omi)**2)))

        return proti, omi


    def __set_parameters__( self, fvec ):

        """ Set general parameters
            Set initial step sizes and control parameters of modified procedure based on [Papari, Bunte, Biehl]
        """

        nfv = fvec.shape[0]
        ndim = fvec.shape[1]

        # GMLVQ parameters, explained below
        showplots = 1
        doztr     = 1
        mode      = 1
        rndinit   = 0
        mu        = 0

        # showplots (0 or 1): plot learning curves etc? recommended: 1
        # doztr (0 or 1): perform z-score transformation based on training set
        # mode
        # 0 for matrix without null-space correction
        # 1 for matrix with null-space correction
        # 2 for diagonal matrix (GRLVQ)                         discouraged
        # 3 for GLVQ with Euclidean distance (equivalent)
        # rndinit
        # 0 for initialization of relevances as identity matrix
        # 1 for randomized initialization of relevance matrix
        # mu
        # control parameter of penalty term for singularity of Lambda
        # mu=0: unmodified GMLVQ
        # mu>0: prevents singular Lambda
        # mu very large: Lambda proportional to Identity (Euclidean)


        # parameters of stepsize adaptation
        if( mode < 2 ):  # full matrix updates with (0) or w/o (1) null space correction
            etam = 2  # suggestion: 2
            etap = 1  # suggestion: 1
            if( mode == 0 ):
                print('matrix relevances without null-space correction')
            if (mode==1):
                print('matrix relevances with null-space correction')

        elif( mode == 2 ): # diagonal relevances only, DISCOURAGED
            print('diagonal relevances, not encouraged, sensitive to step sizes')
            etam   = 0.2  # initital step size for diagonal matrix updates
            etap   = 0.1  # initital step size for prototype update

        elif( mode == 3 ): # GLVQ, equivalent to Euclidean distance
            print('GLVQ without relevances')
            etam=0
            etap = 1

        decfac = 1.5       # step size factor (decrease) for Papari steps
        incfac = 1.1       # step size factor (increase) for all steps
        ncop = 5           # number of waypoints stored and averaged

        if( nfv <= ndim and mode == 0 ):
            print('dim. > # of examples, null-space correction recommended')

        if( doztr == 0 ):
            print('no z-score transformation, you may have to adjust step sizes')
            if( mode < 3 ):
                print('rescale relevances for proper interpretation')

        return showplots, doztr, mode, rndinit, etam, etap, mu, decfac, incfac, ncop


    def __single__( self, fvec, lbl, totalsteps, plbl ):

        # TODO: Matthias
        plbl = np.array( plbl, dtype=int )

        showplots, doztr, mode, rndinit, etam0, etap0, mu, decfac, incfac, ncop = self.__set_parameters__( fvec )

        etam = etam0  # initial step size matrix
        etap = etap0  # intitial step size prototypes

        lbl = self.__check_arguments__( plbl, lbl, fvec, ncop, totalsteps )

        # reproducible random numbers
        #rng('default')
        rngseed=291024
        #rng(rngseed)

        nfv = fvec.shape[0]           # number of feature vectors in training set
        ndim = fvec.shape[1]          # dimension of feature vectors
        ncls = len( np.unique(plbl) )  # number of classes
        nprots = len( plbl )          # total number of prototypes

        te = np.zeros( (totalsteps+1, 1) )  # define total error
        cf = np.zeros( (totalsteps+1, 1) )   # define cost function
        auc = np.zeros( (totalsteps+1, 1) )  # define AUC(ROC)

        cw = np.zeros( (totalsteps+1, ncls) )  # define class-wise errors

        stepsizem = np.zeros( (totalsteps+1, 1) )  # define stepsize matrix in the course of training
        stepsizep = np.zeros( (totalsteps+1, 1) )  # define stepsize prototypes in the course ...

        mf = np.zeros( (1, ndim) )  # initialize feature means
        st = np.ones( (1, ndim) )  # and standard deviations

        if( doztr == 1 ):
            fvec, mf, st = self.__do_zscore__( fvec )  # perform z-score transformation
        else:
            _, mf, st = self.__do_zscore__( fvec )      # evaluate but don't apply

        # initialize prototypes and omega
        proti, omi = self.__set_initial__( fvec, lbl, plbl, mode, rndinit )

        # initial values
        prot = proti
        om = omi

        # copies of prototypes and omegas stored in protcop and omcop
        # for the adaptive step size procedure
        protcop = np.zeros( (prot.shape[0], ncop, prot.shape[1]), dtype=np.cdouble )  # !! spalte, zeile, dim
        omcop   = np.zeros( (om.shape[0], ncop, om.shape[1]), dtype=np.cdouble )

        # calculate initial values for learning curves
        costf, _, marg, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, mu )
        te[0] = np.sum(marg>0) / nfv
        cf[0] = costf
        stepsizem[0] = etam
        stepsizep[0] = etap

        tpr, fpr, auroc, thresholds = self.__compute_roc__( lbl>1, score )

        auc[0] = auroc

        # perform the first ncop steps of gradient descent
        for inistep in range( 0, ncop ):

            # actual batch gradient step
            prot, om = self.__do_batchstep__( fvec, lbl, prot, plbl, om, etap, etam, mu, mode )
            protcop[:,inistep,:] = prot.T
            omcop[:,inistep,:] = om.T

            # determine and save training set performances
            costf, _, marg, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, mu )
            te[inistep+1] = np.sum(marg>0) / nfv
            cf[inistep+1] = costf
            stepsizem[inistep+1] = etam
            stepsizep[inistep+1] = etap

            # compute training set errors and cost function values
            for icls in range( 1, ncls+1 ):  # TODO: Matthias: bleibt bei 1 wegen labels ?! und +1 bei ncls hinzu und -1 bei icls zwei zeilen drunter dazu
                # compute class-wise errors (positive margin = error)
                cw[inistep+1, icls-1] = np.sum(marg[0, np.where(lbl==icls)[0]] > 0) / np.sum(lbl==icls)

            # training set roc with respect to class 1 versus all others only
            tpr, fpr, auroc, thresholds = self.__compute_roc__( lbl>1, score )
            auc[inistep+1] = auroc

        # compute cost functions, crisp labels, margins and scores
        # scores with respect to class 1 (negative) or all others (positive)
        _, _, _, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, mu )  # TODO: Matthias: om conj!!

        # perform totalsteps training steps
        for jstep in range( ncop, totalsteps ):

            # calculate mean positions over latest steps
            protmean = np.squeeze( np.mean(protcop,1) )
            ommean = np.squeeze( np.mean(omcop,1) )
            ommean = ommean / np.sqrt(np.sum(np.sum(np.abs(ommean)**2)))
            # note: normalization does not change cost function value
            #       but is done here for consistency

            # compute cost functions for mean prototypes, mean matrix and both
            costmp, _, _, score = self.__compute_costs__( fvec, lbl, protmean, plbl, om,     0  )
            costmm, _, _, score = self.__compute_costs__( fvec, lbl, prot,     plbl, ommean, mu )
            # [costm, ~,~,score ] = compute_costs(fvec,lbl,protmean,plbl,ommean,mu);

            # remember old positions for Papari procedure
            ombefore = om.copy()
            protbefore = prot.copy()

            # perform next step and compute costs etc.
            prot, om = self.__do_batchstep__( fvec, lbl, prot, plbl, om, etap, etam, mu, mode )

            costf, _, _, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, mu )

            # by default, step sizes are increased in every step
            etam = etam * incfac  # (small) increase of step sizes  # TODO: Matthias: etam wird oben evtl. referenziert!
            etap = etap * incfac  # at each learning step to enforce oscillatory behavior

            # costfunction values to compare with for Papari procedure
            # evaluated w.r.t. changing only matrix or prototype
            costfp, _, _, score = self.__compute_costs__( fvec, lbl, prot,       plbl, ombefore, 0  )
            costfm, _, _, score = self.__compute_costs__( fvec, lbl, protbefore, plbl, om,       mu )

            # heuristic extension of Papari procedure
            # treats matrix and prototype step sizes separately
            if( costmp <= costfp ):  # decrease prototype step size and jump
                # to mean prototypes
                etap = etap / decfac
                prot = protmean

            if( costmm <= costfm ):  # decrease matrix step size and jump
                # to mean matrix
                etam = etam / decfac
                om = ommean

            # update the copies of the latest steps, shift stack of stored configs.
            # plenty of room for improvement, I guess ...
            for iicop in range( 0, ncop-1 ):
                protcop[:,iicop,:] = protcop[:,iicop+1,:]
                omcop[:,iicop,:] = omcop[:,iicop+1,:]

            protcop[:,ncop-1,:] = prot
            omcop[:,ncop-1,:] = om

            # determine training and test set performances
            # here: costfunction without penalty term!
            costf0, _, marg, score = self.__compute_costs__( fvec, lbl, prot, plbl, om, 0 )

            # compute total and class-wise training set errors
            te[jstep+1] = np.sum(marg>0) / nfv
            cf[jstep+1] = costf0

            for icls in range( 1, ncls+1 ):  # TODO: Matthias range von 1 an und ncls erhöht
                cw[jstep,icls-1] = np.sum(marg[np.where(lbl==icls)[1]] > 0) / np.sum(lbl==icls)

            stepsizem[jstep+1] = etam
            stepsizep[jstep+1] = etap

            # ROC with respect to class 1 (negative) vs. all others (positive)
            binlbl = lbl > 1
            tpr, fpr, auroc, thresholds = self.__compute_roc__( binlbl, score )
            auc[jstep+1] = auroc

        # if the data was z transformed then also save the inverse prototypes,
        # actually it is not necessary since the mf and st are returned.
        if( doztr == 1 ):
            protsInv = self.__do_inversezscore__( prot, mf, st )
        else:
            protsInv = prot

        lambdaa = om.T * om   # actual relevance matrix

        # define structures corresponding to the trained system and training curves
        gmlvq_system = {'protos':prot, 'protosInv':protsInv, 'lambda':lambdaa, 'plbl':plbl, 'mean_features':mf, 'std_features':st}
        training_curves = {'costs':cf, 'train_error':te, 'class_wise':cw, 'auroc':auc}
        param_set = {'totalsteps':totalsteps, 'doztr':doztr, 'mode':mode, 'rndinit':rndinit, 'etam0':etam0, 'etap0':etap0, 'etamfin':etam, 'etapfin':etap, 'mu':mu, 'decfac':decfac, 'infac':incfac, 'ncop':ncop, 'rngseed':rngseed}

        return gmlvq_system, training_curves, param_set