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
    if one == two:
        y[:, np.where(enabled==1)[1]] = np.concatenate( (X, np.fliplr(np.conj(X[:,1:None]))), axis=1 )

    y = ifft(y)

    return y