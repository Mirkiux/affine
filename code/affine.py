"""
This defines the class objection Affine, intended to solve affine models of the
term structure
This class inherits from statsmodels LikelihoodModel class
"""

import numpy as np
import statsmodels.api as sm
import pandas as px
import scipy.linalg as la
import re

from numpy import linalg as nla
from numpy import ma
from statsmodels.tsa.api import VAR
from statsmodels.base.model import LikelihoodModel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.numdiff import (approx_hess, approx_fprime)
from operator import itemgetter
from scipy import optimize
from util import flatten, select_rows, retry

#debugging
import pdb

#############################################
# Create affine class system                   #
#############################################

class Affine(LikelihoodModel):
    """
    This class defines an affine model of the term structure
    """
    def __init__(self, yc_data, var_data, rf_rate=None, maxlags=4, freq='M',
                 latent=False, no_err=None):
        """
        Attempts to solve affine model
        yc_data : DataFrame 
            yield curve data
        var_data : DataFrame
            data for var model
        rf_rate : DataFrame
            rf_rate for short_rate, used in latent factor case
        max_lags: int
            number of lags for VAR system
        freq : string
            frequency of data
        no_err : list of ints
            list of the column indexes of yields to be measured without error
            ex: [0, 3, 4] 
            (1st, 4th, and 5th columns in yc_data to be estimatd without error)
        """
        self.yc_data = yc_data
        self.var_data = var_data
        self.rf_rate = rf_rate
        self.yc_names = yc_data.columns
        self.num_yields = len(yc_data.columns)
        self.names = names = var_data.columns
        k_ar = self.k_ar = maxlags
        neqs = self.neqs = len(names)
        self.freq = freq
        self.latent = latent
        self.no_err = no_err

        #generates mths: list of mths in yield curve data
        mths = self._mths_list()
        self.mths = mths

        assert len(yc_data.dropna(axis=0)) == len(var_data.dropna(axis=0)) \
                                                - k_ar + 1, \
            "Number of non-null values unequal in VAR and yield curve data"

        if latent:

            lat = self.lat = len(no_err)

            self.err = list(set(range(len(mths))).difference(no_err))

            self.pos_list = self._gen_pos_list()

            self.no_err_mth, self.err_mth = self._gen_mth_list()
            #gen position list for processing list input to solver
            self.noerr_cols, self.err_cols = self._gen_col_names()
            #set to unconditional mean of short_rate
            self.delta_0 = np.mean(rf_rate)

        #with all observed factors, mu, phi, and sigma are directly generated
        #from OLS VAR one step estimation
        else:
            self.delta_0 = 0
            self.lat = 0
            delta_1 = np.zeros([neqs*k_ar, 1])
            #delta_1 is vector of zeros, with one grabbing fed_funds rate
            delta_1[np.argmax(var_data.columns == 'fed_funds')] = 1
            self.delta_1_nolat = delta_1

        self.mu_ols, self.phi_ols, self.sigma_ols = self._gen_OLS_res()

        #maybe this should be done in setup script...
        #get VAR input data ready
        x_t_na = var_data.copy()
        for lag in range(k_ar-1):
            for var in var_data.columns:
                x_t_na[var + '_m' + str(lag + 1)] = px.Series(var_data[var].
                        values[:-(lag+1)], index=var_data.index[lag + 1:])

        var_data_vert = self.var_data_vert = x_t_na.dropna(axis=0)
        self.periods = len(self.var_data)

        super(Affine, self).__init__(var_data_vert)

    def solve(self, guess_params, lam_0_e=None, lam_1_e=None, delta_1_e=None,
            mu_e=None, phi_e=None, sigma_e=None, method="ls", alg="newton",
            attempts=5, maxfev=10000, maxiter=10000, ftol=1e-100, xtol=1e-100,
            full_output=False):
        """
        Attempt to solve affine model

        guess_params : list
            List of starting values for parameters to be estimated
            In row-order and ordered as masked arrays

        For all estimate paramters:
        elements marked with 'E' or 'e' are estimated
        n = number of variables in fully-specified VAR(1) at t
        lam_0_e : array-like, n x 1
            shape of constant vector of risk pricing equation
        lam_1_e : array-like, n x n
            shape of parameter array of risk pricing equation
        delta_1_e : array-like, n x 1
            shape of initial parameter vector, corresponding to short-rate
            equation
        mu_e : array-like, n x 1
            shape of constant vector for VAR process
        phi_e : array-like, n x n
            shape of parameter array for VAR process
        sigma_e : array-like, n x n
            shape of variance, covariance array for VAR process

        method : string
            solution method
            ls = linear least squares
            nls = nonlinear least squares
            ml = maximum likelihood
            angpiazml = ang and piazzesi multi-step ML
        alg : str {'newton','nm','bfgs','powell','cg', or 'ncg'}
            algorithm used for numerical approximation
            Method can be 'newton' for Newton-Raphson, 'nm' for Nelder-Mead,
            'bfgs' for Broyden-Fletcher-Goldfarb-Shanno, 'powell' for modified
            Powell's method, 'cg' for conjugate gradient, or 'ncg' for Newton-
            conjugate gradient. `method` determines which solver from
            scipy.optimize is used.  The explicit arguments in `fit` are passed
            to the solver.  Each solver has several optional arguments that are
            not the same across solvers.  See the notes section below (or
            scipy.optimize) for the available arguments.
        attempts : int
            Number of attempts to retry solving if singular matrix Exception
            raised by Numpy

        scipy.optimize params
        maxfev : int
            maximum number of calls to the function for solution alg
        maxiter : int
            maximum number of iterations to perform
        ftol : float
            relative error desired in sum of squares
        xtol : float
            relative error desired in the approximate solution
        full_output : bool
            non_zero to return all optional outputs
        """
        #Notes for you:
        #remember that Ang and Piazzesi treat observed and unobserved factors
        #as orthogonal
        #observed factor parameters can thus be estimated using OLS
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert

        dim = neqs * k_ar + lat
        if lam_0_g is not None:
            assert np.shape(lam_0_g) == (dim, 1), "Shape of lam_0_g incorrect"
        if lam_1_g is not None:
            assert np.shape(lam_1_g) == (dim, dim), \
                "Shape of lam_1_g incorrect"

        #creates single input vector for params to solve
        if lat:
            #assertions for correction passed in parameters
            assert np.shape(delta_1_g) == (dim, 1), "Shape of delta_1_g" \
                "incorrect"
            assert np.shape(mu_g) == (dim, 1), "Shape of mu incorrect"
            assert np.shape(phi_g) == (dim, dim), "Shape of phi_g incorrect"
            assert np.shape(sigma_g) == (dim, dim), "Shape of sig_g incorrect"
            #This might need to be removed towrads purer solver class
            delta_1_g, mu_g, phi_g, sigma_g = \
                    self._pass_ols(delta_1=delta_1_g, mu=mu_g, phi=phi_g,
                                   sigma=sigma_g)
            params = self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g,
                                          delta_1=delta_1_g, mu=mu_g,
                                          phi=phi_g, sigma=sigma_g)

        else:
            params = self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g)

        if method == "ls":
            func = self._affine_nsum_errs
            solver = retry(optimize.leastsq, attempts)
            reslt = solver(func, params, maxfev=maxfev, xtol=xtol,
                           full_output=full_output)
            solv_params = reslt[0]
            output = reslt[1:]

        elif method == "nls":
            func = self._affine_pred
            #need to stack
            yield_stack = self._stack_yields(yc_data)
            #run optmization
            solver = retry(optimize.curve_fit, attempts)
            reslt = solver(func, var_data_vert, yield_stack, p0=params,
                           maxfev=maxfev, xtol=xtol, full_output=True)
            solv_params = reslt[0]
            solv_cov = reslt[1]

        elif method == "ml":
            solver = retry(self.fit, attempts)
            solve = solver(start_params=params, method=alg, maxiter=maxiter,
                    maxfun=maxfev, xtol=xtol, fargs=(lam_0_e, lam_1_e,
                        delta_1_e, mu_e, phi_e, sigma_e))
            solv_params = solve.params
            tvalues = solve.tvalues
        elif method == "angpiazml":
            solve = solver(start_params=params, method=alg, maxiter=maxiter,
                    maxfun=maxfev, xtol=xtol, fargs=(lam_0_g, lam_1_g,
                        delta_1_g, mu_g, phi_g, sigma_g))

        lam_0, lam_1, delta_1, mu, phi, sigma = \
                self._param_to_array(params=solv_params, delta_1=delta_1_g,
                                      mu=mu_g, phi=phi_g, sigma=sigma_g)

        a_solve, b_solve = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1,
                                              delta_1=delta_1, mu=mu, phi=phi,
                                              sigma=sigma)

        #This will need to be refactored
        #if full_output:
            #return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, output 
        if method == "nls":
            return lam_0, lam_1, delta_1, mu, phi, sigma, a_solve, b_solve, \
                    solv_cov
        elif method == "ls":
            return lam_0, lam_1, delta_1, mu, phi, sigma, a_solve, b_solve, \
                    output
        elif method == "ml":
            return lam_0, lam_1, delta_1, mu, phi, sigma, a_solve, b_solve, \
                    tvalues

    def score(self, params, *args):
        """
        Return the gradient of the loglike at params

        Parameters
        ----------
        params : list

        Notes
        -----
        Return numerical gradient
        """
        #would be nice to have additional arguments here
        loglike = self.loglike
        return approx_fprime(params, loglike, epsilon=1e-8, args=(args))

    def hessian(self, params, *args):
        """
        Returns numerical hessian.
        """
        #would be nice to have additional arguments here
        loglike = self.loglike
        my_stuff = args
        return approx_hess(params, loglike, args=(args))

    def loglike(self, params, lam_0, lam_1, delta_1, mu, phi, sigma):
        """
        Loglikelihood used in latent factor models
        """
        lat = self.lat
        per = self.periods

        #all of the params don't seem to be moving
        #only seems to be for certain solution methods

        lam_0, lam_1, delta_1, mu, phi, sigma \
            = self._param_to_array(params=params, delta_1=delta_1, mu=mu, \
                                   phi=phi, sigma=sigma)

        solve_a, solve_b = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1, \
                                delta_1=delta_1, mu=mu, phi=phi, sigma=sigma)

        #first solve for unknown part of information vector
        var_data_c, jacob, yield_errs  = self._solve_unobs(a_in=solve_a,
                                                           b_in=solve_b)

        # here is the likelihood that needs to be used
        # sigma is implied VAR sigma
        # use two matrices to take the difference

        errors = var_data_c.values.T[:, 1:] - mu - np.dot(phi,
                var_data_c.values.T[:, :-1])

        sign, j_logdt = nla.slogdet(jacob)
        j_slogdt = sign * j_logdt

        sign, sigma_logdt = nla.slogdet(np.dot(sigma, sigma.T))
        sigma_slogdt = sign * sigma_logdt

        like = -(per - 1) * j_slogdt - (per - 1) * 1.0 / 2 * sigma_slogdt - \
               1.0 / 2 * np.sum(np.dot(np.dot(errors.T, \
               la.inv(np.dot(sigma, sigma.T))), errors)) - (per - 1) / 2.0 * \
               np.log(np.sum(np.var(yield_errs, axis=1))) - 1.0 / 2 * \
               np.sum(yield_errs**2/np.var(yield_errs, axis=1)[None].T)

        return like

    # def angpiazml(start_params, method=, maxiter=, maxfun=, xtol=, delta_1_g=,
    #               mu_g=, phi_g=, sigma_g=):
    #     """
    #     Performs three step ML ala Ang and Piazzesi (2003)

    #     multistep : step in ang and piazzesi method
    #         0 : NA
    #         1 : set both lam_0 and lam_1 equal to zero
    #         2 : set lam_0 eqaul to 0 
    #     """
    #     #1) keep lam_0 and lam_1 equal to zero and estimate to get starting
    #     #parameters for \theta

    #     #2)Hold lam_0 constant while estimating to get starting values for
    #     #lam_1

    #     #3) Set insignificant parameters in lam_1 equal to zero and estimate
    #     #lamda_0

    #     #4) Set insignficant parameters in lam_0 equal to 0. Re-estimate whole
    #     # system abiding by all parameter guesses and parameters equalt to 0

    #     #estimate with lam_0_g 
    #     lam_0_g

    #     params = 

    def gen_pred_coef(self, lam_0, lam_1, delta_1, mu, phi, sigma):
        """
        Generates prediction coefficient vectors A and B
        lam_0 : array
        lam_1 : array
        delta_1 : array
        phi : array
        sigma : array
        """
        #This should be passed to a C function, it is slow right now
        mths = self.mths
        delta_0 = self.delta_0
        max_mth = max(mths)
        #generate predictions
        a_pre = np.zeros((max_mth, 1))
        a_pre[0] = -delta_0
        b_pre = []
        b_pre.append(-delta_1)

        for mth in range(max_mth-1):
            a_pre[mth+1] = (a_pre[mth] + np.dot(b_pre[mth].T, \
                            (mu - np.dot(sigma, lam_0))) + \
                            (1.0/2)*np.dot(np.dot(np.dot(b_pre[mth].T, sigma), \
                            sigma.T), b_pre[mth]) - delta_0)[0][0]
            b_pre.append(np.dot((phi - np.dot(sigma, lam_1)).T, \
                                b_pre[mth]) - delta_1)
        n_inv = 1.0/np.add(range(max_mth), 1).reshape((max_mth, 1))
        a_solve = -(a_pre*n_inv)
        b_solve = np.zeros_like(b_pre)
        for mth in range(max_mth):
            b_solve[mth] = np.multiply(-b_pre[mth], n_inv[mth])
        return a_solve, b_solve

    def _affine_nsum_errs(self, params):
        """
        This function generates the sum of the prediction errors
        """
        lat = self.lat
        mths = self.mths
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert

        lam_0, lam_1, delta_1, mu, phi, sigma = self._param_to_array(params=params)

        a_solve, b_solve = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1,
                                              delta_1=delta_1, mu=mu,phi=phi,
                                              sigma=sigma)

        errs = []

        yc_data_val = yc_data.values
        
        for ix, mth in enumerate(mths):
            act = np.flipud(yc_data_val[:, ix])
            pred = a_solve[mth - 1] + np.dot(b_solve[mth - 1].T, 
                                        np.fliplr(var_data_vert.T))[0]
            errs = errs + (act - pred).tolist()
        return errs

    def _solve_unobs(self, a_in, b_in):
        """
        Solves for unknown factors

        Parameters
        ----------
        a_in : list of floats (periods)
            List of elements for A constant in factors -> yields relationship
        b_in : array (periods, neqs * k_ar + lat)
            Array of elements for B coefficients in factors -> yields
            relationship

        Returns
        -------
        var_data_c : DataFrame 
            VAR data including unobserved factors
        jacob : array (neqs * k_ar + num_yields)**2
            Jacobian used in likelihood
        yield_errs : array (num_yields - lat, periods)
            The errors for the yields estimated with error
        """
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert
        yc_names = self.yc_names
        num_yields = self.num_yields
        names = self.names
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        no_err = self.no_err
        err = self.err
        no_err_mth = self.no_err_mth
        err_mth = self.err_mth
        noerr_cols = self.noerr_cols
        err_cols = self.err_cols

        yc_data_names = yc_names.tolist()
        no_err_num = len(noerr_cols)
        err_num = len(err_cols)

        #need to combine the two matrices
        #these matrices will collect the final values
        a_all = np.zeros([num_yields, 1])
        b_all_obs = np.zeros([num_yields, neqs * k_ar])
        b_all_unobs = np.zeros([num_yields, lat])

        a_sel = np.zeros([no_err_num, 1])
        b_sel_obs = np.zeros([no_err_num, neqs * k_ar])
        b_sel_unobs = np.zeros([no_err_num, lat])
        for ix, y_pos in enumerate(no_err):
            a_sel[ix, 0] = a_in[no_err_mth[ix] - 1]
            b_sel_obs[ix, :, None] = b_in[no_err_mth[ix] - 1][:neqs * k_ar]
            b_sel_unobs[ix, :, None] = b_in[no_err_mth[ix] - 1][neqs * k_ar:]

            a_all[y_pos, 0] = a_in[no_err_mth[ix] - 1]
            b_all_obs[y_pos, :, None] = b_in[no_err_mth[ix] - 1][:neqs * k_ar]
            b_all_unobs[y_pos, :, None] = \
                    b_in[no_err_mth[ix] - 1][neqs * k_ar:]
        #now solve for unknown factors using long matrices

        unobs = np.dot(la.inv(b_sel_unobs), 
                    yc_data.filter(items=noerr_cols).values.T - a_sel - \
                    np.dot(b_sel_obs, var_data_vert.values.T))

        #re-initialize a_sel, b_sel_obs, and b_sel_obs
        a_sel = np.zeros([err_num, 1])
        b_sel_obs = np.zeros([err_num, neqs * k_ar])
        b_sel_unobs = np.zeros([err_num, lat])
        for ix, y_pos in enumerate(err):
            a_all[y_pos, 0] =  a_sel[ix, 0] = a_in[err_mth[ix] - 1]
            b_all_obs[y_pos, :, None] = b_sel_obs[ix, :, None] = \
                    b_in[err_mth[ix] - 1][:neqs * k_ar]
            b_all_unobs[y_pos, :, None] = b_sel_unobs[ix, :, None] = \
                    b_in[err_mth[ix] - 1][neqs * k_ar:]

        yield_errs = yc_data.filter(items=err_cols).values.T - a_sel - \
                        np.dot(b_sel_obs, var_data_vert.values.T) - \
                        np.dot(b_sel_unobs, unobs)

        var_data_c = var_data_vert.copy()
        for factor in range(lat):
            var_data_c["latent_" + str(factor)] = unobs[factor, :]
        meas_mat = np.zeros((num_yields, err_num))

        for col_index, col in enumerate(err_cols):
            row_index = yc_data_names.index(col)
            meas_mat[row_index, col_index] = 1

        jacob = self._construct_J(b_obs=b_all_obs, 
                                    b_unobs=b_all_unobs, meas_mat=meas_mat)
        
        return var_data_c, jacob, yield_errs 

    def _mths_list(self):
        """
        This function just grabs the mths of yield curve points and return
        a list of them
        """
        mths = []
        columns = self.yc_names
        matcher = re.compile(r"(.*?)([0-9]+)$")
        for column in columns:
            mths.append(int(re.match(matcher, column).group(2)))
        return mths

    def _param_to_array(self, params, lam_0_e, lam_1_e, delta_1_e, mu_e,
                        phi_e):
        """
        Process params input into appropriate arrays

        Parameters
        ----------
        delta_1 : array (neqs * k_ar + lat, 1)
            delta_1 prior to complete model solve
        mu : array (neqs * k_ar + lat, 1)
            mu prior to complete model solve
        phi : array (neqs * k_ar + lat, neqs * k_ar + lat)
            phi prior to complete model solve
        sigma : array (neqs * k_ar + lat, neqs * k_ar + lat)
            sigma prior to complete model solve
        """
        all_arrays = [lam_0_e, lam_1_e, delta_1_e, mu_e, phi_e, sigma_e]

        arg_sep = self._gen_arg_sep([ma.count_masked(struct) for struct in \
                                     all_arrays])

        for pos, struct in enumerate(all_arrays):
            struct[ma.getmask(struct)] = params[arg_sep[pos]:arg_sep[pos + 1]]

        return all_arrays

    def _affine_pred(self, data, *params):
        """
        Function based on lambda and data that generates predicted yields
        data : DataFrame
        params : tuple of floats
            parameter guess
        """
        mths = self.mths
        yc_data = self.yc_data

        lam_0, lam_1, delta_1, mu, phi, sigma = self._param_to_array(params)

        a_test, b_test = self.gen_pred_coef(lam_0, lam_1, delta_1, mu, phi,
                                            sigma)

        pred = px.DataFrame(index=yc_data.index)

        for i in mths:
            pred["l_tr_m" + str(i)] = a_test[i-1] + np.dot(b_test[i-1].T,
                                      data.T).T[:,0]

        pred = self._stack_yields(pred)

        return pred

    def _stack_yields(self, orig):
        """
        Stacks yields into single column ndarray
        """
        mths = self.mths
        obs = len(orig)
        new = np.zeros((len(mths) * obs))
        for col, mth in enumerate(orig.columns):
            new[col*obs:(col+1)*obs] = orig[mth].values
        return new
    
    def _gen_OLS_res(self):
        """
        Runs VAR on macro data and retrieves parameters
        """
        #run VAR to generate parameters for known 
        var_data = self.var_data
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        freq = self.freq

        var_fit = VAR(var_data, freq=freq).fit(maxlags=k_ar)

        coefs = var_fit.params.values
        sigma_u = var_fit.sigma_u

        obs_var = neqs * k_ar

        mu = np.zeros([k_ar*neqs, 1])
        mu[:neqs] = coefs[0, None].T

        phi = np.zeros([k_ar * neqs, k_ar * neqs])
        phi[:neqs] = coefs[1:].T
        phi[neqs:obs_var, :(k_ar - 1) * neqs] = np.identity((k_ar - 1) * neqs)

        sigma = np.zeros([k_ar * neqs, k_ar * neqs])
        sigma[:neqs, :neqs] = sigma_u
        sigma[neqs:obs_var, neqs:obs_var] = np.identity((k_ar - 1) * neqs)
        
        return mu, phi, sigma

    def _ml_meth(self, params, lam_0_g, lam_1_g, delta_1_g, mu_g, phi_g, sigma_g):
        """
        This is a wrapper for the simple maximum likelihodd solution method
        """
        lam_0_g
        #solve_a = self.gen_pred_coef(lam_0=lam_0_g, lam_1_g=lam_1_g

    def _gen_pos_list(self):
        """
        Generates list of positions from draw parameters from list
        Notes: this is only lengths of parameters that we are solving for using
        numerical maximization
        """
        neqs = self.neqs
        k_ar = self.k_ar
        lat = self.lat

        pos_list = []
        pos = 0
        len_lam_0 = neqs + lat
        len_lam_1 = neqs**2 + (neqs * lat) + (lat * neqs) + lat**2
        len_delta_1 = lat
        len_mu = lat
        len_phi = lat * lat
        len_sig = lat * lat
        length_list = [len_lam_0, len_lam_1, len_delta_1, len_mu, len_phi,
                       len_sig]

        for length in length_list:
            pos_list.append(length + pos)
            pos += length

        return pos_list
    
    def _gen_arg_sep(self, arg_lenths):
        """
        Generates list of positions 
        """
        arg_sep = [0]
        for length in arg_lengths:
            pos_list.append(length + pos)
            pos += length
        return arg_sep

    def _gen_col_names(self):
        """
        Generate column names for err and noerr
        """
        yc_names = self.yc_names
        no_err = self.no_err
        err = self.err
        noerr_cols = []
        err_cols = []
        for index in no_err:
            noerr_cols.append(yc_names[index])
        for index in err:
            err_cols.append(yc_names[index])
        return noerr_cols, err_cols

    def _gen_mth_list(self):
        """
        Generate list of mths measured with and wihout error
        """
        yc_names = self.yc_names
        no_err = self.no_err
        mths = self.mths
        err = self.err

        no_err_mth = []
        err_mth = []

        for index in no_err:
            no_err_mth.append(mths[index])
        for index in err:
            err_mth.append(mths[index])

        return no_err_mth, err_mth

    def _construct_J(self, b_obs, b_unobs, meas_mat): 
        """
        Consruct jacobian matrix
        meas_mat : array 
        LEFT OFF here 
        """
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        num_yields = self.num_yields
        num_obsrv = neqs * k_ar

        #now construct Jacobian
        msize = neqs * k_ar + num_yields 
        jacob = np.zeros([msize, msize])
        jacob[:num_obsrv, :num_obsrv] = np.identity(neqs*k_ar)

        jacob[num_obsrv:, :num_obsrv] = b_obs
        jacob[num_obsrv:, num_obsrv:num_obsrv + lat] = b_unobs
        jacob[num_obsrv:, num_obsrv + lat:] = meas_mat

        return jacob
