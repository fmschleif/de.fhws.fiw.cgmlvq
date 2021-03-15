import numpy as np


def vdot( a, b ):

    """ NumPy implements complex-conjugating dot product only for vectors with numpy.vdot.
        This function wraps the numpy.vdot function, so that matrices can be handled.
    """

    # mxm|mxm
    if( a.shape == b.shape ):
        return vdot_matrix_matrix( a, b )

    # mx1|1xm
    if( a.shape[1] == 1 and b.shape[0] == 1 and a.shape[0] == b.shape[1] ):
        return vdot_vector_vector( a, b )

    # mxm|mx1
    if( is_squared_matrix(a) and a.shape[0] == b.shape[0] and b.shape[1] == 1 ):
        return vdot_matrix_vector( a, b )


def vdot_matrix_matrix( A, B ):

    """ Calculating the complex-conjugating dot product.

    Parameters
    ----------
    A : mxm matrix
    B : mxm matrix

    Returns
    -------
    AB : mxm matrix
    """

    length = len(A)

    AB = np.zeros( (length,length), dtype=np.cdouble )

    for i in range(0, length):
        for ii in range(0, length):
            AB[i,ii] = np.vdot( A[i], B[:,ii] )

    return AB


def vdot_vector_vector( a, b ):

    """ Calculating the complex-conjugating dot product.

    Parameters
    ----------
    a : mx1 matrix
    b : 1xm matrix

    Returns
    -------
    AB : mxm matrix
    """

    length = len(a)

    AB = np.zeros( (length,length), dtype=np.cdouble )

    for i in range(0, length):
        for ii in range(0, length):
            AB[i,ii] = np.vdot( a[i,0], b[0,ii] )

    return AB


def vdot_matrix_vector( A, b ):

    """ Calculating the complex-conjugating dot product.

    Parameters
    ----------
    A : mxm matrix
    b : mx1 matrix

    Returns
    -------
    Ab : mx1 matrix
    """

    length = len(A)

    Ab = np.zeros( (length,1), dtype=np.cdouble )

    for i in range(0, length):
        Ab[i,0] = np.vdot( A[i], b )

    return Ab


def is_squared_matrix( M ):

    return M.shape[0] == M.shape[1]