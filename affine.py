import numpy as np
import socket
import pickle
import os
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.base.model import LikelihoodModel
from statsmodels.sandbox.regression.numdiff import (approx_hess,
                                                        approx_fprime)

import pandas as px
import pandas.core.datetools as dt
from operator import itemgetter
from scipy import optimize
import scipy.linalg as la

import matplotlib.pyplot as plt

#############################################
# Create affine class system                   #
#############################################

class BSR(LikelihoodModel):
    def __init__(self, yc_data, var_data, rf_rate=None, maxlags=4,
                 freq='M', latent=0, no_err=None):
        """
        Attemps to solve BSR model
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
        data = var_data.values
        names = var_data.columns
        dates = var_data.index

        #mod = VAR(data, names=names, dates=dates, freq='monthly')
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
            for x in len_lst:
                pos_lst.append(x+acc)
                acc += x
            self.pos_lst = pos_lst

        mu = np.zeros([k_ar*neqs+lat,1])
        mu[:neqs] = params[0,None].T
        self.mu = mu

        phi = np.zeros([k_ar*neqs, k_ar*neqs])
        phi[:neqs] = params[1:].T
        phi[neqs:,:(k_ar-1)*neqs] = np.identity((k_ar-1)*neqs)
        self.phi = phi

        sig = np.zeros([k_ar*neqs, k_ar*neqs])
        sig[:neqs, :neqs] = sigma_u
        self.sig = sig

        if lat == 0:
            self.delta_0 = 0
            delta_1 = np.zeros([neqs*k_ar,1])
            #delta_1 is vector of zeros, with one grabbing fed_funds rate
            delta_1[np.argmax(var_data.columns == 'fed_funds')] = 1
            self.delta_1 = delta_1

        else:
            #this is the method outlined by Ang and Piazzesi (2003)
            reg_data = var_data.copy()
            reg_data['intercept'] = 1
            par = sm.OLS(rf_rate, reg_data).fit().params
            self.delta_0 = par.values[-1]
            delta_1 = np.zeros([neqs*k_ar+lat,1])
            delta_1[:neqs,0] = par.values[:neqs]
            self.delta_1 = delta_1

        #get VAR input data ready
        x_t_na = var_data.copy()
        for t in range(k_ar-1):
            for var in var_data.columns:
                x_t_na[var + '_m' + str(t+1)] = px.Series(var_data[var].
                        values[:-(t+1)], index=var_data.index[t+1:])
        self.var_data = x_t_na.dropna(axis=0)

        super(BSR, self).__init__(var_data)


    def solve(self, lam_0_g, lam_1_g, delt_1_g=None, phi_g=None,
            sig_g=None, maxfev=10000, xtol=1e-100,
              full_output=False):
        lat = self.latent
        neqs = self.neqs
        k_ar = self.k_ar

        assert len(lam_0_g) == neqs + lat, "Length of lam_0_g not correct"
        assert len(lam_1_g) == (neqs + lat)**2, "Length of lam_1_g not correct"
        if lat:
            assert len(delt_1_g) == lat, "Length of delt_1_g not correct"
            assert len(phi_g) == lat**2, "Length of phi_g not correct"
            assert len(sig_g) == lat**2, "Length of sig_g not correct"
            lam = np.asarray(lam_0_g + lam_1_g + delt_1_g + phi_g + sig_g)
        else:
            lam = np.asarray(lam_0_g + lam_1_g)

        func = self._BSR_nsum_errs
        reslt = optimize.leastsq(func, lam, maxfev=maxfev,
                                xtol=xtol, full_output=full_output)
        lam_solv = reslt[0]
        output = reslt[1:]

        lam_0, lam_1, delta_1, phi, sig = self._proc_lam(lam_solv)

        a, b = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sig)

        if full_output:
            return lam_0, lam_1, delta_1, phi, sig, a, b, output
        else:
            return lam_0, lam_1, delta_1, phi, sig, a, b

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
        loglike = self._BSR_nsum_errs
        return approx_fprime(lam, loglike, epsilon=1e-8)

    def hessian(self, lam):
        """
        Returns numerical hessian.
        """
        loglike = self._BSR_nsum_errs
        return approx_hess(lam, loglike)[0]

    def gen_pred_coef(self, lam_0_ab, lam_1_ab, delta_1, phi, sig):
        lat = self.latent
        mths = self.mths
        delta_0 = self.delta_0
        mu = self.mu
        max_mth = max(mths)
        #generate predictions
        A = np.zeros((max_mth, 1))
        A[0] = -delta_0
        B = []
        B.append(-delta_1)
        for x in range(max_mth-1):
            A[x+1] = (A[x] + np.dot(B[x].T, (mu - np.dot(sig,  \
            lam_0_ab))) + (1.0/2)*np.dot(np.dot(np.dot(B[x].T, \
            sig), sig.T), B[x]) - delta_0)[0][0]
            B.append(np.dot((phi - np.dot(sig, lam_1_ab)).T, B[x]) - delta_1)
        n_inv = 1.0/np.add(range(max_mth), 1).reshape((max_mth,1))
        a = -(A*n_inv)
        b = np.zeros_like(B)
        for x in range(max_mth):
            b[x] = np.multiply(-B[x], n_inv[x])
        return a, b

    def _BSR_nsum_errs(self, lam):
        """
        This function generates the sum of the prediction errors
        """
        lat = self.latent
        no_err = self.no_err
        neqs = self.neqs
        k_ar = self.k_ar
        mths = self.mths
        mth_only = self.mth_only
        X_t = self.var_data

        lam_0, lam_1, delta_1, phi, sig = self._proc_lam(lam)

        a, b = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sig)

        #this is explosive

        if lat:
            X_t = self._solve_X_t_unkn(a, b, X_t)

        errs = []

        for i in mths:
            act = np.flipud(mth_only['l_tr_m' + str(i)].values)
            pred = a[i-1] + np.dot(b[i-1].T, np.fliplr(X_t.T))[0]
            errs = errs + (act-pred).tolist()
        return errs

    def _solve_X_t_unkn(self, a, b, X_t):
        lat = self.latent
        no_err = self.no_err
        mth_only = self.mth_only
        X_t_new = np.append(X_t, np.zeros((X_t.shape[0],lat)), axis=1)
        T = X_t.shape[0]
        #here is the likelihood that needs to be used
        # sigma is implied VAR sigma
        #like = -(T - 1)*np.logdet(J) - (T - 1)*1.0 / 2 * \
            #np.logdet(np.dot(sigma,sigma.T)) - 1.0 / 2 * \
            # use two matrices to take the difference
            #np.sum(

        for t in range(X_t.shape[0]):
            #mx = c
            m = np.zeros((lat,lat))
            c = np.zeros(lat)
            for x, i in enumerate(no_err):
                act = mth_only['l_tr_m' + str(i)].values[t]
                #print "i"
                #print i
                #print "act"
                #print act
                #print "a[i-1]"
                #print a[i-1]
                #print "b[i-1].T[:,:-lat]"
                #print b[i-1].T[:,:-lat]
                #print "X_t.values.T[:,t,None]"
                #print X_t.values.T[:,t,None]
                c[x] = act - a[i-1] - np.dot(b[i-1].T[:,:-lat],\
                        X_t.values.T[:,t,None])
                m[x] = b[i-1].T[:,-lat:]
            X_t_new[t,-lat:] = la.solve(m,c)
        return X_t_new

    def _proc_to_mth(self):
        frame = self.yc_data
        mths = []
        fnd = 0
        n_cols = len(frame.columns)
        for x in frame.columns:
            srt_ord = []
            if 'm' in x:
                mths.append(int(x[6]))
                if fnd == 0:
                    mth_only = px.DataFrame(frame[x],
                            columns = [x],
                            index=frame.index)
                    fnd = 1
                else:
                    mth_only[x] = frame[x]
            elif 'y' in x:
                mth = int(x[6:])*12
                mths.append(mth)
                mth_only[('l_tr_m' + str(mth))] = frame[x]
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

    def _proc_lam(self, lam):
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

            lam_0 = np.zeros([k_ar*neqs+lat,1])
            lam_0[:neqs,0] = np.asarray(lam_0_est[:neqs]).T
            lam_0[-lat:,0] = np.asarray(lam_0_est[-lat:]).T

            lam_1 = np.zeros([k_ar*neqs+lat, k_ar*neqs+lat])
            lam_1[:neqs,:neqs] = np.reshape(lam_1_est[:neqs**2], (neqs,neqs))
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
            delta_1[-lat:,0] = np.asarray(delt_1_g)

            #add rows/columns for unk params
            phi_n = self.phi.copy()
            add = np.zeros([lat, np.shape(phi_n)[1]])
            phi_n = np.append(phi_n, add, axis=0)
            add = np.zeros([np.shape(phi_n)[0], lat])
            phi = np.append(phi_n, add, axis=1)
            #fill in parm guesses
            phi[-lat:, -lat:] = np.reshape(phi_g, (lat,lat))

            #add rows/columns for unk params
            sig_n = self.sig.copy()
            add = np.zeros([lat, np.shape(sig_n)[1]])
            sig_n = np.append(sig_n, add, axis=0)
            add = np.zeros([np.shape(sig_n)[0], lat])
            sig = np.append(sig_n, add, axis=1)
            sig[-lat:, -lat:] = np.reshape(sig_g, (lat,lat))

        else:
            lam_0_est = lam[:neqs]
            lam_1_est = lam[neqs:]

            lam_0 = np.zeros([k_ar*neqs,1])
            lam_0[:neqs] = np.asarray([lam_0_est]).T

            lam_1 = np.zeros([k_ar*neqs, k_ar*neqs])
            lam_1[:neqs,:neqs] = np.reshape(lam_1_est, (neqs,neqs))

            delta_1 = self.delta_1
            phi = self.phi
            sig = self.sig

        return lam_0, lam_1, delta_1, phi, sig
