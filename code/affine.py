"""
This defines the class objection Affine, intended to solve affine models of the
term structure
This class inherits from statsmodels LikelihoodModel class
"""

import numpy as np
import statsmodels.api as sm
import pandas as px
import scipy.linalg as la

from statsmodels.tsa.api import VAR
from statsmodels.base.model import LikelihoodModel
from statsmodels.tools.numdiff import (approx_hess, approx_fprime)
from operator import itemgetter
from scipy import optimize

#debugging
#import pdb

#############################################
# Create affine class system                   #
#############################################

class Affine(LikelihoodModel):
    """
    This class defines an affine model of the term structure
    """
    def __init__(self, yc_data, var_data, rf_rate=None, maxlags=4,
                 freq='M', latent=0, no_err=None):
        """
        Attempts to solve affine model
        yc_data : yield curve data
        var_data : data for var model
        rf_rate : rf_rate for short_rate, used in latent factor case
        max_lags: number of lags for VAR system
        freq : frequency of data
        latent : # number of latent variables
        no_err : list of the yields that are estimated without error
        """
        self.yc_data = yc_data

        #gen VAR instance
        mod = VAR(var_data, freq=freq)
        vreg = mod.fit(maxlags=maxlags)

        #generates mths and mth_only
        self._proc_to_mth()

        #number of latent variables to include
        lat = self.latent = latent
        self.no_err = no_err

        self.k_ar = k_ar = vreg.k_ar
        self.neqs = neqs = vreg.neqs
        self.params = params = vreg.params.values
        sigma_u = vreg.sigma_u

        if lat:
            assert len(no_err) >= lat, "One yield estimated with no err"\
                                        + "for each latent variable"

            #order is lam_0, lam_1, delt_1, phi, sig
            len_lst = [neqs+lat, (neqs + lat)**2, lat, lat**2, lat**2]
            pos_lst = []
            acc = 0
            for lengths in len_lst:
                pos_lst.append(lengths+acc)
                acc += lengths
            self.pos_lst = pos_lst
            yc_data_cols = yc_data.columns.tolist()
            self.noerr_indx = list(set(yc_data_cols).intersection(no_err))
            self.err_indx = list(set(yc_data_cols).difference(no_err))

        mu = np.zeros([k_ar*neqs+lat, 1])
        mu[:neqs] = params[0, None].T
        self.mu = mu

        phi = np.zeros([k_ar*neqs, k_ar*neqs])
        phi[:neqs] = params[1:].T
        phi[neqs:, :(k_ar-1)*neqs] = np.identity((k_ar-1)*neqs)
        self.phi = phi

        sig = np.zeros([k_ar*neqs, k_ar*neqs])
        sig[:neqs, :neqs] = sigma_u
        self.sig = sig

        if lat == 0:
            self.delta_0 = 0
            delta_1 = np.zeros([neqs*k_ar, 1])
            #delta_1 is vector of zeros, with one grabbing fed_funds rate
            delta_1[np.argmax(var_data.columns == 'fed_funds')] = 1
            self.delta_1 = delta_1

        else:
            #this is the method outlined by Ang and Piazzesi (2003)
            reg_data = var_data.copy()
            reg_data['intercept'] = 1
            par = sm.OLS(rf_rate, reg_data).fit().params
            self.delta_0 = par.values[-1]
            delta_1 = np.zeros([neqs*k_ar+lat, 1])
            delta_1[:neqs, 0] = par.values[:neqs]
            self.delta_1 = delta_1

        #get VAR input data ready
        x_t_na = var_data.copy()
        for lag in range(k_ar-1):
            for var in var_data.columns:
                x_t_na[var + '_m' + str(lag+1)] = px.Series(var_data[var].
                        values[:-(lag+1)], index=var_data.index[lag+1:])

        #check this, looks fine
        self.var_data = x_t_na.dropna(axis=0)

        super(Affine, self).__init__(var_data)


    def solve(self, lam_0_g, lam_1_g, method="ls", delt_1_g=None, phi_g=None,
            sig_g=None, maxfev=10000, ftol=1e-100, xtol=1e-100,
            full_output=False):
        """
        Attempt to solve affine model

        method : string
            ls = linear least squares
            cf = nonlinear least squares
        lam_0_g : array (n x 1),
            guess for elements of lam_0
        lam_1_g : array (n x n),
            guess for elements of lam_1
        delt_1_g : array 
            guess for elements of delt_1
        phi_g : array
            guess for elements of phi
        sig_g : array
            guess for elements of sigma
        scipy.optimize.leastsq params
        maxfev : int
            maximum number of calls to the function for solution alg
        ftol : float
            relative error desired in sum of squares
        xtol : float
            relative error desired in the approximate solution
        full_output : bool
            non_zero to return all optional outputs
        """
        lat = self.latent
        neqs = self.neqs
        lam = []
        x_t = self.var_data
        mth_only = self.mth_only

        #assert np.shape(lam_0_g) == neqs + lat, "Length of lam_0_g not correct"
        #assert len(lam_1_g) == (neqs + lat)**2, "Length of lam_1_g not correct"
        #creates single input vector for params to solve
        if lat:
            assert len(delt_1_g) == lat, "Length of delt_1_g not correct"
            assert len(phi_g) == lat**2, "Length of phi_g not correct"
            assert len(sig_g) == lat**2, "Length of sig_g not correct"
            lam = np.asarray(lam_0_g + lam_1_g + delt_1_g + phi_g + sig_g)
        else:
            lam_0_list = flatten(lam_0_g[:neqs])
            lam_1_list = flatten(lam_1_g[:neqs, :neqs])
            for param in range(len(lam_0_list)):
                lam.append(lam_0_list[param])
            for param in range(len(lam_1_list)):
                lam.append(lam_1_list[param])

        #this should be specified in function call
        if method == "ls":
            func = self._affine_nsum_errs
            reslt = optimize.leastsq(func, lam, maxfev=maxfev,
                                xtol=xtol, full_output=full_output)
            lam_solv = reslt[0]
            output = reslt[1:]
            func = self._affine_pred
        elif method == "cf":
            func = self._affine_pred
            #need to stack
            yield_stack = self._stack_yields(mth_only)
            #run optmization
            reslt = optimize.curve_fit(func, x_t, yield_stack, p0=lam,
                                       maxfev=maxfev, xtol=xtol,
                                       full_output=full_output)

            lam_solv = reslt[0]
            lam_cov = reslt[1]

        lam_0, lam_1, delta_1, phi, sig = self._proc_lam(*lam_solv)

        a_solve, b_solve = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sig)

        #if full_output:
            #return lam_0, lam_1, delta_1, phi, sig, a_solve, b_solve, output 
        if method == "cf":
            return lam_0, lam_1, delta_1, phi, sig, a_solve, b_solve, lam_cov
        elif method == "ls":
            return lam_0, lam_1, delta_1, phi, sig, a_solve, b_solve, output

    def score(self, lam):
        """
        Return the gradient of the loglike at AB_mask.

        Parameters
        ----------
        AB_mask : unknown values of A and B matrix concatenated

        Notes
        -----
        Return numerical gradient
        """
        loglike = self._affine_nsum_errs
        return approx_fprime(lam, loglike, epsilon=1e-8)

    def hessian(self, lam):
        """
        Returns numerical hessian.
        """
        loglike = self._affine_nsum_errs
        return approx_hess(lam, loglike)[0]

    #def loglike(self, params):
    #    """
    #    Loglikelihood used in latent factor models
    #    """
    #    # here is the likelihood that needs to be used
    #    # sig is implied VAR sig
    #    # use two matrices to take the difference
    #    like = -(T - 1) * np.logdet(J) - (T - 1) * 1.0 / 2 * \
    #            np.logdet(np.dot(sig, sig.T)) - 1.0 / 2 * \
    #            np.sum(np.dot(np.dot(errors.T, np.inv(np.dot(sig, sig.T))),\
    #                          err)) - (T - 1) / 2.0 * \
    #            np.log(np.sum(np.var(meas_err, axis=1))) - 1.0 / 2 * \
    #            np.sum(meas_err/np.var(meas_err, axis=1))

    def gen_pred_coef(self, lam_0_ab, lam_1_ab, delta_1, phi, sig):
        """
        Generates prediction coefficient vectors A and B
        lam_0_ab : array
        lam_1_ab : array
        delta_1 : array
        phi : array
        sig : array
        """
        mths = self.mths
        delta_0 = self.delta_0
        mu = self.mu
        max_mth = max(mths)
        #generate predictions
        a_pre = np.zeros((max_mth, 1))
        a_pre[0] = -delta_0
        b_pre = []
        b_pre.append(-delta_1)
        for mth in range(max_mth-1):
            a_pre[mth+1] = (a_pre[mth] + np.dot(b_pre[mth].T, \
                            (mu - np.dot(sig, lam_0_ab))) + \
                            (1.0/2)*np.dot(np.dot(np.dot(b_pre[mth].T, sig), \
                            sig.T), b_pre[mth]) - delta_0)[0][0]
            b_pre.append(np.dot((phi - np.dot(sig, lam_1_ab)).T, \
                                b_pre[mth]) - delta_1)
        n_inv = 1.0/np.add(range(max_mth), 1).reshape((max_mth, 1))
        a_solve = -(a_pre*n_inv)
        b_solve = np.zeros_like(b_pre)
        for mths in range(max_mth):
            b_solve[mths] = np.multiply(-b_pre[mths], n_inv[mths])
        return a_solve, b_solve

    def _affine_nsum_errs(self, lam):
        """
        This function generates the sum of the prediction errors
        """
        lat = self.latent
        mths = self.mths
        mth_only = self.mth_only
        x_t = self.var_data

        lam_0, lam_1, delta_1, phi, sig = self._proc_lam(*lam)

        a_solve, b_solve = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sig)

        #this is explosive

        if lat:
            x_t = self._solve_x_t_unkn(a_solve, b_solve)

        errs = []
        
        for i in mths:
            act = np.flipud(mth_only['l_tr_m' + str(i)].values)
            pred = a_solve[i-1] + np.dot(b_solve[i-1].T, np.fliplr(x_t.T))[0]
            errs = errs + (act-pred).tolist()
        return errs

    def _solve_x_t_unkn(self, a_in, b_in, x_t = None):
        """
        This is still under development
        It should solve for the unobserved factors in the x_t VAR data
        """
        lat = self.latent
        no_err = self.no_err
        mth_only = self.mth_only
        yc_data = self.yc_data
        x_t_new = np.append(x_t, np.zeros((x_t.shape[0], lat)), axis=1)
        errors = x_t[1:] - mu - np.dot(phi, x_t[:-1])
        if x_t is None:
            x_t = self.var_data
        T = x_t.shape[0]

        # solve for unknown factors
        noerr_indx = self.noerr_indx
        a_noerr = select_rows(noerr_indx, a_in)
        b_0_noerr = select_rows(noerr_indx, b_in)
        # this is the right hand for solving for the unobserved latent 
        # factors
        r_hs = yc_data[no_err] - a_noerr[None].T - np.dot(b_0_noerr, x_t)
        lat = la.solve(b_u, r_hs)

        #solve for pricing error on other yields
        err_indx = self.err_indx
        a_err = select_rows(err_indx, a_in)
        b_0_err = select_rows(err_indx, b_in)
        r_hs = yc_data[no_err] - a_noerr[None].T - np.dot(b_0_noerr, x_t)
        meas_err = la.solve(b_m, r_hs)

        #create Jacobian (J) here
        
        #this taken out for test run, need to be added back in
        #J = 

    def _proc_to_mth(self):
        """
        This function transforms the yield curve data so that the names are all
        in months
        (not sure if this is necessary)
        """
        frame = self.yc_data
        mths = []
        fnd = 0
        n_cols = len(frame.columns)
        for col in frame.columns:
            if 'm' in col:
                mths.append(int(col[6]))
                if fnd == 0:
                    mth_only = px.DataFrame(frame[col],
                            columns = [col],
                            index=frame.index)
                    fnd = 1
                else:
                    mth_only[col] = frame[col]
            elif 'y' in col:
                mth = int(col[6:])*12
                mths.append(mth)
                mth_only[('l_tr_m' + str(mth))] = frame[col]
        col_dict = dict([( mth_only.columns[x], mths[x]) for x in
                    range(n_cols)])
        cols = np.asarray(sorted(col_dict.iteritems(),
                        key=itemgetter(1)))[:,0].tolist()
        mth_only = mth_only.reindex(columns = cols)
        mths.sort()
        self.mths = mths
        self.mth_only = mth_only

    #def _unk_likl(self):
    #    likl = -(T-1)*np.logdet(J) - (T-1)*1.0/2*np.logdet(np.dot(sig,\
    #            sig.T)) - 1.0/2*

    def _proc_lam(self, *lam):
        """
        Process lam input into appropriate parameters
        """
        lat = self.latent
        neqs = self.neqs
        k_ar = self.k_ar

        if lat:

            pos_lst = self.pos_lst

            lam_0_est = lam[:pos_lst[0]]
            lam_1_est = lam[pos_lst[0]:pos_lst[1]]
            delt_1_g = lam[pos_lst[1]:pos_lst[2]]
            phi_g = lam[pos_lst[2]:pos_lst[3]]
            sig_g = lam[pos_lst[3]:]

            lam_0 = np.zeros([k_ar*neqs+lat, 1])
            lam_0[:neqs, 0] = np.asarray(lam_0_est[:neqs]).T
            lam_0[-lat:, 0] = np.asarray(lam_0_est[-lat:]).T

            lam_1 = np.zeros([k_ar*neqs+lat, k_ar*neqs+lat])
            lam_1[:neqs, :neqs] = np.reshape(lam_1_est[:neqs**2], (neqs, neqs))
            nxt = neqs*lat
            lam_1[:neqs, -lat:] = np.reshape(lam_1_est[neqs**2:\
                                            neqs**2 + nxt],(neqs,lat))
            nxt = nxt + neqs**2
            lam_1[-lat:, :neqs] = np.reshape(lam_1_est[nxt: \
                                            nxt+lat*neqs], (lat, neqs))
            nxt = nxt + lat*neqs
            lam_1[-lat:, -lat:] = np.reshape(lam_1_est[nxt: \
                                            nxt + lat**2], (lat, lat))
            delta_1 = self.delta_1.copy()
            delta_1[-lat:, 0] = np.asarray(delt_1_g)

            #add rows/columns for unk params
            phi_n = self.phi.copy()
            add = np.zeros([lat, np.shape(phi_n)[1]])
            phi_n = np.append(phi_n, add, axis=0)
            add = np.zeros([np.shape(phi_n)[0], lat])
            phi = np.append(phi_n, add, axis=1)
            #fill in parm guesses
            phi[-lat:, -lat:] = np.reshape(phi_g, (lat, lat))

            #add rows/columns for unk params
            sig_n = self.sig.copy()
            add = np.zeros([lat, np.shape(sig_n)[1]])
            sig_n = np.append(sig_n, add, axis=0)
            add = np.zeros([np.shape(sig_n)[0], lat])
            sig = np.append(sig_n, add, axis=1)
            sig[-lat:, -lat:] = np.reshape(sig_g, (lat, lat))

        else:
            lam_0_est = lam[:neqs]
            lam_1_est = lam[neqs:]

            lam_0 = np.zeros([k_ar*neqs, 1])
            lam_0[:neqs] = np.asarray([lam_0_est]).T

            lam_1 = np.zeros([k_ar*neqs, k_ar*neqs])
            lam_1[:neqs, :neqs] = np.reshape(lam_1_est, (neqs, neqs))

            delta_1 = self.delta_1
            phi = self.phi
            sig = self.sig

        return lam_0, lam_1, delta_1, phi, sig

    def _affine_pred(self, x_t, *lam):
        """
        Function based on lambda and x_t that generates predicted yields
        x_t : X_inforionat
        """
        mths = self.mths
        mth_only = self.mth_only

        lam_0, lam_1, delta_1, phi, sig = self._proc_lam(*lam)

        a_test, b_test = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sig)

        pred = px.DataFrame(index=mth_only.index)

        for i in mths:
            pred["l_tr_m" + str(i)] = a_test[i-1] + np.dot(b_test[i-1].T,
                                      x_t.T).T[:,0]

        pred = self._stack_yields(pred)

        return pred

    def _stack_yields(self, orig):
        """
        Stacks yields into single column ndarray
        """
        mths = self.mths
        obs = len(orig)
        new = np.zeros((len(mths)*obs))
        for col, mth in enumerate(orig.columns):
            new[col*obs:(col+1)*obs] = orig[mth].values
        return new

def flatten(array):
    """
    Flattens array to list values
    """
    a_list = []
    if array.ndim == 1:
        for index in range(np.shape(array)[0]):
            a_list.append(array[index])
        return a_list
    elif array.ndim == 2:
        rshape = np.reshape(array, np.size(array))
        for index in range(np.shape(rshape)[0]):
            a_list.append(rshape[index])
        return a_list
    
def select_rows(rows, array):
    """
    Creates 2-dim submatrix only of rows from list rows
    array must be 2-dim
    """
    if array.ndim == 1:
        new_array = array[rows[0]]
        if len(rows) > 1:
            for row in rows[1:]:
                new_array = np.append(new_array, array[row])
    elif array.ndim == 2:
        new_array = array[rows[0], :]
        if len(rows) > 1:
            for row in enumerate(rows[1:]):
                new_array = np.append(new_array, array[row, :], axis=0)
    return new_array
