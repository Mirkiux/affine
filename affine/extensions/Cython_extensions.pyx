#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
from __future__ import division

#Cython specific imports
cimport numpy as np
cimport cython

#import numpy as np
import numpy as np

np.import_array()


cdef extern from "capsule.h":
    void *Capsule_AsVoidPtr(object ptr)

# DOC: Types in use
# NOTE: This is taken from statsmodels statespace code

# `blas_lapack.pxd` contains typedef statements for BLAS and LAPACK functions
# NOTE: Temporary fix
#from statsmodels.src.blas_lapack cimport *
from blas_lapack cimport *

try:
    # Scipy >= 0.12.0 exposes Fortran BLAS functions directly
    from scipy.linalg.blas import cgerc
except:
    # Scipy < 0.12.0 exposes Fortran BLAS functions in the `fblas` submodule
    from scipy.linalg.blas import fblas as blas
else:
    from scipy.linalg import blas

# DOC: These ints are used for BLAS function calls
cdef int dim_one = 1
cdef int dim_zero = 0
cdef int dim_mone = -1
cdef int inc = 1
cdef int FORTRAN = 1

# dimension holders
cdef np.npy_intp dim1[1]
cdef np.npy_intp dim2[2]

# DOC: We are creating a function for each of the four numpy types

# DOC: Set up appropriate pointers to blas functions
cdef sgemm_t *sgemm = <sgemm_t*>Capsule_AsVoidPtr(blas.sgemm._cpointer)
cdef scopy_t *scopy = <scopy_t*>Capsule_AsVoidPtr(blas.scopy._cpointer)

# DOC: This is the numpy type for this function definition
ctypedef np.float32_t sDTYPE_t

# DOC: Scalars used for BLAS calls
cdef sDTYPE_t sscalar_one = 1
cdef sDTYPE_t sscalar_zero = 0
cdef sDTYPE_t sscalar_mone = -1
cdef sDTYPE_t sscalar_half = 1 / 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def sgen_pred_coef(
    unsigned int max_mat,
    np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] lam_0,
    np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] lam_1,
    np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] delta_0,
    np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] delta_1,
    np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] mu,
    np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] phi,
    np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] sigma):
    """
    Returns tuple of arrays
    Generates prediction coefficient vectors A and B

    Parameters
    ----------
    max_mat : the maximum maturity to be calculated
    lam_0 : numpy array
    lam_1 : numpy array
    delta_0 : numpy array
    delta_1 : numpy array
    mu : numpy array
    phi : numpy array
    sigma : numpy array

    Returns
    -------
    a_solve : numpy array
        Array of constants relating factors to yields
    b_solve : numpy array
        Array of coeffiencts relating factors to yields
    """
    cdef unsigned int mat
    cdef unsigned int max_mat_m1 = max_mat - 1

    #sizes needed
    cdef int factors = mu.shape[dim_zero]
    cdef int mu_size = mu.size
    cdef int factors_sqr = phi.size

    # generate predictions arrays
    dim2[:] = [max_mat, 1]
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] a_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)
    dim2[:] = [max_mat, factors]
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] b_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)

    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] n_inv = \
        np.asfortranarray(1 / np.add(range(max_mat), 1).reshape((max_mat, 1)),
                          np.float32)

    a_pre[dim_zero] = -delta_0
    b_pre[dim_zero] = -delta_1[:,dim_zero]

    dim1[:] = [factors]
    cdef np.ndarray[sDTYPE_t, ndim=1, mode="fortran"] b_el_holder = \
        np.PyArray_EMPTY(1, dim1, np.NPY_FLOAT32, FORTRAN)
    dim2[:] = [factors, dim_one]
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] mu_sigma_lam0 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] b_pre_prep = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)
    dim1[:] = [dim_one]
    cdef np.ndarray[sDTYPE_t, ndim=1, mode="fortran"] \
        a_b_mu_sig_lam = np.PyArray_EMPTY(1, dim1, np.NPY_FLOAT32, FORTRAN)
    dim2[:] = [dim_one, factors]
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] b_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] b_sig_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)
    dim2[:] = [dim_one, dim_one]
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] half_b_sig_sig_b_delta = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)
    dim2[:] = [factors, factors]
    cdef np.ndarray[sDTYPE_t, ndim=2, mode="fortran"] phi_sigma_lam_1 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT32, FORTRAN)

    for mat in range(max_mat_m1):
        # Set next value of a
        #NOTE: set these to arrays to be referenced uniquely
        np.PyArray_CopyInto(b_el_holder, b_pre[mat])
        np.PyArray_CopyInto(a_b_mu_sig_lam, a_pre[mat])

        # This creates a filler array that initially has values of mu
        scopy(&mu_size, &mu[dim_zero, dim_zero], &inc,
                       &mu_sigma_lam0[dim_zero, dim_zero], &inc)
        # This creates a filler array that initially has values of delta_0
        scopy(&dim_one, &delta_0[dim_zero, dim_zero], &inc,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &inc)


        # DOC: Calcualte a_pre
        sgemm("N", "N", &factors, &dim_one, &factors,
                       &sscalar_mone, &sigma[dim_zero, dim_zero],
                       &factors, &lam_0[dim_zero, dim_zero], &factors,
                       &sscalar_one,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors)
        sgemm("T", "N", &dim_one, &dim_one, &factors,
                       &sscalar_one, &b_el_holder[dim_zero], &factors,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors,
                       &sscalar_one, &a_b_mu_sig_lam[dim_zero],
                       &dim_one)

        sgemm("T", "N", &dim_one, &factors, &factors,
                       &sscalar_one, &b_el_holder[dim_zero], &factors,
                       &sigma[dim_zero, dim_zero], &factors,
                       &sscalar_zero, &b_sig[dim_zero, dim_zero],
                       &dim_one)
        sgemm("N", "T", &dim_one, &factors, &factors,
                       &sscalar_one, &b_sig[dim_zero, dim_zero],
                       &dim_one, &sigma[dim_zero, dim_zero], &factors,
                       &sscalar_zero, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one)
        sgemm("N", "N", &dim_one, &dim_one, &factors,
                       &sscalar_half, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one, &b_el_holder[dim_zero], &factors,
                       &sscalar_mone,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &dim_one)
        a_pre[mat + 1] = a_b_mu_sig_lam[dim_zero] + \
                         half_b_sig_sig_b_delta[dim_zero, dim_zero]

        # DOC: Calcualte b_pre
        # Filler array that has initial values of phi
        scopy(&factors_sqr, &phi[dim_zero, dim_zero], &inc, &phi_sigma_lam_1[dim_zero, dim_zero],
                       &inc)
        # Filler array that has initial value of delta_1
        scopy(&factors, &delta_1[dim_zero, dim_zero], &inc, &b_pre_prep[dim_zero, dim_zero], &inc)
        # set next value of b
        sgemm("N", "N", &factors, &factors, &factors,
                       &sscalar_mone, &sigma[dim_zero, dim_zero], &factors,
                       &lam_1[dim_zero, dim_zero], &factors, &sscalar_one,
                       &phi_sigma_lam_1[dim_zero, dim_zero], &factors)
        sgemm("T", "N", &factors, &dim_one, &factors,
                       &sscalar_one, &phi_sigma_lam_1[dim_zero, dim_zero], &factors,
                       &b_el_holder[dim_zero], &factors, &sscalar_mone,
                       &b_pre_prep[dim_zero, dim_zero], &factors)
        b_pre[mat + 1] = b_pre_prep[:, dim_zero]

    return np.multiply(-a_pre, n_inv), np.multiply(-b_pre, n_inv)

# DOC: Set up appropriate pointers to blas functions
cdef dgemm_t *dgemm = <dgemm_t*>Capsule_AsVoidPtr(blas.dgemm._cpointer)
cdef dcopy_t *dcopy = <dcopy_t*>Capsule_AsVoidPtr(blas.dcopy._cpointer)

# DOC: This is the numpy type for this function definition
ctypedef np.float64_t dDTYPE_t

# DOC: Scalars used for BLAS calls
cdef dDTYPE_t dscalar_one = 1
cdef dDTYPE_t dscalar_zero = 0
cdef dDTYPE_t dscalar_mone = -1
cdef dDTYPE_t dscalar_half = 1 / 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def dgen_pred_coef(
    unsigned int max_mat,
    np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] lam_0,
    np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] lam_1,
    np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] delta_0,
    np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] delta_1,
    np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] mu,
    np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] phi,
    np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] sigma):
    """
    Returns tuple of arrays
    Generates prediction coefficient vectors A and B

    Parameters
    ----------
    max_mat : the maximum maturity to be calculated
    lam_0 : numpy array
    lam_1 : numpy array
    delta_0 : numpy array
    delta_1 : numpy array
    mu : numpy array
    phi : numpy array
    sigma : numpy array

    Returns
    -------
    a_solve : numpy array
        Array of constants relating factors to yields
    b_solve : numpy array
        Array of coeffiencts relating factors to yields
    """
    cdef unsigned int mat
    cdef unsigned int max_mat_m1 = max_mat - 1

    #sizes needed
    cdef int factors = mu.shape[dim_zero]
    cdef int mu_size = mu.size
    cdef int factors_sqr = phi.size

    # generate predictions arrays
    dim2[:] = [max_mat, 1]
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] a_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)
    dim2[:] = [max_mat, factors]
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] b_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)

    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] n_inv = \
        np.asfortranarray(1 / np.add(range(max_mat), 1).reshape((max_mat, 1)),
                          np.float64)

    a_pre[dim_zero] = -delta_0
    b_pre[dim_zero] = -delta_1[:,dim_zero]

    dim1[:] = [factors]
    cdef np.ndarray[dDTYPE_t, ndim=1, mode="fortran"] b_el_holder = \
        np.PyArray_EMPTY(1, dim1, np.NPY_FLOAT64, FORTRAN)
    dim2[:] = [factors, dim_one]
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] mu_sigma_lam0 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] b_pre_prep = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)
    dim1[:] = [dim_one]
    cdef np.ndarray[dDTYPE_t, ndim=1, mode="fortran"] \
        a_b_mu_sig_lam = np.PyArray_EMPTY(1, dim1, np.NPY_FLOAT64, FORTRAN)
    dim2[:] = [dim_one, factors]
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] b_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] b_sig_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)
    dim2[:] = [dim_one, dim_one]
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] half_b_sig_sig_b_delta = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)
    dim2[:] = [factors, factors]
    cdef np.ndarray[dDTYPE_t, ndim=2, mode="fortran"] phi_sigma_lam_1 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_FLOAT64, FORTRAN)

    for mat in range(max_mat_m1):
        # Set next value of a
        #NOTE: set these to arrays to be referenced uniquely
        np.PyArray_CopyInto(b_el_holder, b_pre[mat])
        np.PyArray_CopyInto(a_b_mu_sig_lam, a_pre[mat])

        # This creates a filler array that initially has values of mu
        dcopy(&mu_size, &mu[dim_zero, dim_zero], &inc,
                       &mu_sigma_lam0[dim_zero, dim_zero], &inc)
        # This creates a filler array that initially has values of delta_0
        dcopy(&dim_one, &delta_0[dim_zero, dim_zero], &inc,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &inc)


        # DOC: Calcualte a_pre
        dgemm("N", "N", &factors, &dim_one, &factors,
                       &dscalar_mone, &sigma[dim_zero, dim_zero],
                       &factors, &lam_0[dim_zero, dim_zero], &factors,
                       &dscalar_one,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors)
        dgemm("T", "N", &dim_one, &dim_one, &factors,
                       &dscalar_one, &b_el_holder[dim_zero], &factors,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors,
                       &dscalar_one, &a_b_mu_sig_lam[dim_zero],
                       &dim_one)

        dgemm("T", "N", &dim_one, &factors, &factors,
                       &dscalar_one, &b_el_holder[dim_zero], &factors,
                       &sigma[dim_zero, dim_zero], &factors,
                       &dscalar_zero, &b_sig[dim_zero, dim_zero],
                       &dim_one)
        dgemm("N", "T", &dim_one, &factors, &factors,
                       &dscalar_one, &b_sig[dim_zero, dim_zero],
                       &dim_one, &sigma[dim_zero, dim_zero], &factors,
                       &dscalar_zero, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one)
        dgemm("N", "N", &dim_one, &dim_one, &factors,
                       &dscalar_half, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one, &b_el_holder[dim_zero], &factors,
                       &dscalar_mone,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &dim_one)
        a_pre[mat + 1] = a_b_mu_sig_lam[dim_zero] + \
                         half_b_sig_sig_b_delta[dim_zero, dim_zero]

        # DOC: Calcualte b_pre
        # Filler array that has initial values of phi
        dcopy(&factors_sqr, &phi[dim_zero, dim_zero], &inc, &phi_sigma_lam_1[dim_zero, dim_zero],
                       &inc)
        # Filler array that has initial value of delta_1
        dcopy(&factors, &delta_1[dim_zero, dim_zero], &inc, &b_pre_prep[dim_zero, dim_zero], &inc)
        # set next value of b
        dgemm("N", "N", &factors, &factors, &factors,
                       &dscalar_mone, &sigma[dim_zero, dim_zero], &factors,
                       &lam_1[dim_zero, dim_zero], &factors, &dscalar_one,
                       &phi_sigma_lam_1[dim_zero, dim_zero], &factors)
        dgemm("T", "N", &factors, &dim_one, &factors,
                       &dscalar_one, &phi_sigma_lam_1[dim_zero, dim_zero], &factors,
                       &b_el_holder[dim_zero], &factors, &dscalar_mone,
                       &b_pre_prep[dim_zero, dim_zero], &factors)
        b_pre[mat + 1] = b_pre_prep[:, dim_zero]

    return np.multiply(-a_pre, n_inv), np.multiply(-b_pre, n_inv)

# DOC: Set up appropriate pointers to blas functions
cdef cgemm_t *cgemm = <cgemm_t*>Capsule_AsVoidPtr(blas.cgemm._cpointer)
cdef ccopy_t *ccopy = <ccopy_t*>Capsule_AsVoidPtr(blas.ccopy._cpointer)

# DOC: This is the numpy type for this function definition
ctypedef np.complex64_t cDTYPE_t

# DOC: Scalars used for BLAS calls
cdef cDTYPE_t cscalar_one = 1
cdef cDTYPE_t cscalar_zero = 0
cdef cDTYPE_t cscalar_mone = -1
cdef cDTYPE_t cscalar_half = 1 / 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def cgen_pred_coef(
    unsigned int max_mat,
    np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] lam_0,
    np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] lam_1,
    np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] delta_0,
    np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] delta_1,
    np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] mu,
    np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] phi,
    np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] sigma):
    """
    Returns tuple of arrays
    Generates prediction coefficient vectors A and B

    Parameters
    ----------
    max_mat : the maximum maturity to be calculated
    lam_0 : numpy array
    lam_1 : numpy array
    delta_0 : numpy array
    delta_1 : numpy array
    mu : numpy array
    phi : numpy array
    sigma : numpy array

    Returns
    -------
    a_solve : numpy array
        Array of constants relating factors to yields
    b_solve : numpy array
        Array of coeffiencts relating factors to yields
    """
    cdef unsigned int mat
    cdef unsigned int max_mat_m1 = max_mat - 1

    #sizes needed
    cdef int factors = mu.shape[dim_zero]
    cdef int mu_size = mu.size
    cdef int factors_sqr = phi.size

    # generate predictions arrays
    dim2[:] = [max_mat, 1]
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] a_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)
    dim2[:] = [max_mat, factors]
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] b_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)

    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] n_inv = \
        np.asfortranarray(1 / np.add(range(max_mat), 1).reshape((max_mat, 1)),
                          np.complex64)

    a_pre[dim_zero] = -delta_0
    b_pre[dim_zero] = -delta_1[:,dim_zero]

    dim1[:] = [factors]
    cdef np.ndarray[cDTYPE_t, ndim=1, mode="fortran"] b_el_holder = \
        np.PyArray_EMPTY(1, dim1, np.NPY_COMPLEX64, FORTRAN)
    dim2[:] = [factors, dim_one]
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] mu_sigma_lam0 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] b_pre_prep = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)
    dim1[:] = [dim_one]
    cdef np.ndarray[cDTYPE_t, ndim=1, mode="fortran"] \
        a_b_mu_sig_lam = np.PyArray_EMPTY(1, dim1, np.NPY_COMPLEX64, FORTRAN)
    dim2[:] = [dim_one, factors]
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] b_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] b_sig_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)
    dim2[:] = [dim_one, dim_one]
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] half_b_sig_sig_b_delta = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)
    dim2[:] = [factors, factors]
    cdef np.ndarray[cDTYPE_t, ndim=2, mode="fortran"] phi_sigma_lam_1 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX64, FORTRAN)

    for mat in range(max_mat_m1):
        # Set next value of a
        #NOTE: set these to arrays to be referenced uniquely
        np.PyArray_CopyInto(b_el_holder, b_pre[mat])
        np.PyArray_CopyInto(a_b_mu_sig_lam, a_pre[mat])

        # This creates a filler array that initially has values of mu
        ccopy(&mu_size, &mu[dim_zero, dim_zero], &inc,
                       &mu_sigma_lam0[dim_zero, dim_zero], &inc)
        # This creates a filler array that initially has values of delta_0
        ccopy(&dim_one, &delta_0[dim_zero, dim_zero], &inc,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &inc)


        # DOC: Calcualte a_pre
        cgemm("N", "N", &factors, &dim_one, &factors,
                       &cscalar_mone, &sigma[dim_zero, dim_zero],
                       &factors, &lam_0[dim_zero, dim_zero], &factors,
                       &cscalar_one,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors)
        cgemm("T", "N", &dim_one, &dim_one, &factors,
                       &cscalar_one, &b_el_holder[dim_zero], &factors,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors,
                       &cscalar_one, &a_b_mu_sig_lam[dim_zero],
                       &dim_one)

        cgemm("T", "N", &dim_one, &factors, &factors,
                       &cscalar_one, &b_el_holder[dim_zero], &factors,
                       &sigma[dim_zero, dim_zero], &factors,
                       &cscalar_zero, &b_sig[dim_zero, dim_zero],
                       &dim_one)
        cgemm("N", "T", &dim_one, &factors, &factors,
                       &cscalar_one, &b_sig[dim_zero, dim_zero],
                       &dim_one, &sigma[dim_zero, dim_zero], &factors,
                       &cscalar_zero, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one)
        cgemm("N", "N", &dim_one, &dim_one, &factors,
                       &cscalar_half, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one, &b_el_holder[dim_zero], &factors,
                       &cscalar_mone,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &dim_one)
        a_pre[mat + 1] = a_b_mu_sig_lam[dim_zero] + \
                         half_b_sig_sig_b_delta[dim_zero, dim_zero]

        # DOC: Calcualte b_pre
        # Filler array that has initial values of phi
        ccopy(&factors_sqr, &phi[dim_zero, dim_zero], &inc, &phi_sigma_lam_1[dim_zero, dim_zero],
                       &inc)
        # Filler array that has initial value of delta_1
        ccopy(&factors, &delta_1[dim_zero, dim_zero], &inc, &b_pre_prep[dim_zero, dim_zero], &inc)
        # set next value of b
        cgemm("N", "N", &factors, &factors, &factors,
                       &cscalar_mone, &sigma[dim_zero, dim_zero], &factors,
                       &lam_1[dim_zero, dim_zero], &factors, &cscalar_one,
                       &phi_sigma_lam_1[dim_zero, dim_zero], &factors)
        cgemm("T", "N", &factors, &dim_one, &factors,
                       &cscalar_one, &phi_sigma_lam_1[dim_zero, dim_zero], &factors,
                       &b_el_holder[dim_zero], &factors, &cscalar_mone,
                       &b_pre_prep[dim_zero, dim_zero], &factors)
        b_pre[mat + 1] = b_pre_prep[:, dim_zero]

    return np.multiply(-a_pre, n_inv), np.multiply(-b_pre, n_inv)

# DOC: Set up appropriate pointers to blas functions
cdef zgemm_t *zgemm = <zgemm_t*>Capsule_AsVoidPtr(blas.zgemm._cpointer)
cdef zcopy_t *zcopy = <zcopy_t*>Capsule_AsVoidPtr(blas.zcopy._cpointer)

# DOC: This is the numpy type for this function definition
ctypedef np.complex128_t zDTYPE_t

# DOC: Scalars used for BLAS calls
cdef zDTYPE_t zscalar_one = 1
cdef zDTYPE_t zscalar_zero = 0
cdef zDTYPE_t zscalar_mone = -1
cdef zDTYPE_t zscalar_half = 1 / 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
def zgen_pred_coef(
    unsigned int max_mat,
    np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] lam_0,
    np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] lam_1,
    np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] delta_0,
    np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] delta_1,
    np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] mu,
    np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] phi,
    np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] sigma):
    """
    Returns tuple of arrays
    Generates prediction coefficient vectors A and B

    Parameters
    ----------
    max_mat : the maximum maturity to be calculated
    lam_0 : numpy array
    lam_1 : numpy array
    delta_0 : numpy array
    delta_1 : numpy array
    mu : numpy array
    phi : numpy array
    sigma : numpy array

    Returns
    -------
    a_solve : numpy array
        Array of constants relating factors to yields
    b_solve : numpy array
        Array of coeffiencts relating factors to yields
    """
    cdef unsigned int mat
    cdef unsigned int max_mat_m1 = max_mat - 1

    #sizes needed
    cdef int factors = mu.shape[dim_zero]
    cdef int mu_size = mu.size
    cdef int factors_sqr = phi.size

    # generate predictions arrays
    dim2[:] = [max_mat, 1]
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] a_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)
    dim2[:] = [max_mat, factors]
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] b_pre = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)

    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] n_inv = \
        np.asfortranarray(1 / np.add(range(max_mat), 1).reshape((max_mat, 1)),
                          np.complex128)

    a_pre[dim_zero] = -delta_0
    b_pre[dim_zero] = -delta_1[:,dim_zero]

    dim1[:] = [factors]
    cdef np.ndarray[zDTYPE_t, ndim=1, mode="fortran"] b_el_holder = \
        np.PyArray_EMPTY(1, dim1, np.NPY_COMPLEX128, FORTRAN)
    dim2[:] = [factors, dim_one]
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] mu_sigma_lam0 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] b_pre_prep = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)
    dim1[:] = [dim_one]
    cdef np.ndarray[zDTYPE_t, ndim=1, mode="fortran"] \
        a_b_mu_sig_lam = np.PyArray_EMPTY(1, dim1, np.NPY_COMPLEX128, FORTRAN)
    dim2[:] = [dim_one, factors]
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] b_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] b_sig_sig = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)
    dim2[:] = [dim_one, dim_one]
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] half_b_sig_sig_b_delta = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)
    dim2[:] = [factors, factors]
    cdef np.ndarray[zDTYPE_t, ndim=2, mode="fortran"] phi_sigma_lam_1 = \
        np.PyArray_EMPTY(2, dim2, np.NPY_COMPLEX128, FORTRAN)

    for mat in range(max_mat_m1):
        # Set next value of a
        #NOTE: set these to arrays to be referenced uniquely
        np.PyArray_CopyInto(b_el_holder, b_pre[mat])
        np.PyArray_CopyInto(a_b_mu_sig_lam, a_pre[mat])

        # This creates a filler array that initially has values of mu
        zcopy(&mu_size, &mu[dim_zero, dim_zero], &inc,
                       &mu_sigma_lam0[dim_zero, dim_zero], &inc)
        # This creates a filler array that initially has values of delta_0
        zcopy(&dim_one, &delta_0[dim_zero, dim_zero], &inc,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &inc)


        # DOC: Calcualte a_pre
        zgemm("N", "N", &factors, &dim_one, &factors,
                       &zscalar_mone, &sigma[dim_zero, dim_zero],
                       &factors, &lam_0[dim_zero, dim_zero], &factors,
                       &zscalar_one,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors)
        zgemm("T", "N", &dim_one, &dim_one, &factors,
                       &zscalar_one, &b_el_holder[dim_zero], &factors,
                       &mu_sigma_lam0[dim_zero, dim_zero], &factors,
                       &zscalar_one, &a_b_mu_sig_lam[dim_zero],
                       &dim_one)

        zgemm("T", "N", &dim_one, &factors, &factors,
                       &zscalar_one, &b_el_holder[dim_zero], &factors,
                       &sigma[dim_zero, dim_zero], &factors,
                       &zscalar_zero, &b_sig[dim_zero, dim_zero],
                       &dim_one)
        zgemm("N", "T", &dim_one, &factors, &factors,
                       &zscalar_one, &b_sig[dim_zero, dim_zero],
                       &dim_one, &sigma[dim_zero, dim_zero], &factors,
                       &zscalar_zero, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one)
        zgemm("N", "N", &dim_one, &dim_one, &factors,
                       &zscalar_half, &b_sig_sig[dim_zero, dim_zero],
                       &dim_one, &b_el_holder[dim_zero], &factors,
                       &zscalar_mone,
                       &half_b_sig_sig_b_delta[dim_zero, dim_zero], &dim_one)
        a_pre[mat + 1] = a_b_mu_sig_lam[dim_zero] + \
                         half_b_sig_sig_b_delta[dim_zero, dim_zero]

        # DOC: Calcualte b_pre
        # Filler array that has initial values of phi
        zcopy(&factors_sqr, &phi[dim_zero, dim_zero], &inc, &phi_sigma_lam_1[dim_zero, dim_zero],
                       &inc)
        # Filler array that has initial value of delta_1
        zcopy(&factors, &delta_1[dim_zero, dim_zero], &inc, &b_pre_prep[dim_zero, dim_zero], &inc)
        # set next value of b
        zgemm("N", "N", &factors, &factors, &factors,
                       &zscalar_mone, &sigma[dim_zero, dim_zero], &factors,
                       &lam_1[dim_zero, dim_zero], &factors, &zscalar_one,
                       &phi_sigma_lam_1[dim_zero, dim_zero], &factors)
        zgemm("T", "N", &factors, &dim_one, &factors,
                       &zscalar_one, &phi_sigma_lam_1[dim_zero, dim_zero], &factors,
                       &b_el_holder[dim_zero], &factors, &zscalar_mone,
                       &b_pre_prep[dim_zero, dim_zero], &factors)
        b_pre[mat + 1] = b_pre_prep[:, dim_zero]

    return np.multiply(-a_pre, n_inv), np.multiply(-b_pre, n_inv)
