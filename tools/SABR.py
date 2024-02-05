#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:34:25 2024

@author: ted
"""
import numpy as np
from scipy import optimize


def SABR_FUNC(strike, forward, expiryTime, alpha, beta, nu, rho):
    """
    
    Description
    -----------
    
    SABR function as described in "Foreign Exchange Option Pricing: A Practitioner's Guide," by Iain J. Clark

    Parameters
    ----------
    strike : float
        strike price of the option.
    forward : float
        forward price.
    expiryTime : float
        maturity of option.
    alpha : float
        SABR param.
    beta : float
        SABR param.
    nu : float
        SABR param.
    rho : float
        SABR param.

    Returns
    -------
    sigma : float
        new implied volatility.

    """
    
    eps=1e-3

    A = (forward*strike)**(1.0-beta)

    
    if abs(forward-strike)<eps:
        logM = np.log(forward/strike)
    else:
        epsilon = (forward-strike)/strike
        logM = epsilon - 0.5 * epsilon * epsilon
        
    z = (nu / alpha) * np.sqrt(A) * logM
    C = (1.0-beta) * (1.0-beta) * (logM **2)


    X = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1-rho))
    numerator = (1.0 + logM**2 / 24.0 + logM**4  / 1920.0)
    denominator = (1.0 + C / 24.0 + C * C / 1920.0)
    
    Term1 = numerator / denominator
    
    Term2 = 1.0 + expiryTime * (-beta * (2 - beta) * alpha * alpha / (24.0 * A) +
                                0.25 * rho * beta * nu * alpha / np.sqrt(A) +
                                (2.0 - 3.0 * rho * rho) * (nu * nu / 24.0))

    if z**2 > eps*10:
        Term3 = z / X
    else:
        Term3 = 1.0 - 0.5 * rho * z - (3.0 * rho * rho - 2.0) * z * z / 12.0
    F = alpha * (forward * strike)**( beta / 2.0)
    sigma = F * Term1 * Term3 * Term2
    
    return sigma


def calibrate_SABR(market_vol,K,S,T,r,q,params):
    """
    
    Description
    ----------
    This function optimizes the SABR function parameters using the scipy.optimize.minimize package. 
    
    
    Parameters
    ----------
    market_vol : NumPy array
        implied volatility of market options.
    K : NumPy array
        Array of strikes.
    S : float
        Current stock price.
    T : NumPy array
        Array of maturities.
    r : NumPy array
        Domestic rate.
    q : NumPy array
        foreign rate or dividend rate.
    params : list
        Initial guesses of SABR parameters. 

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    F = S * np.exp((r-q)*T)

    RMSE = np.empty((np.size(T)))
    calibrated_params = np.empty((np.size(T),4))
    for i in range(np.size(T)):
        def func(params):
                params[0] = max(params[0], 1e-8)      # Prevent alpha going negative
                params[1] = max(params[1], 1e-8)      # Prevent beta going negative
                params[2] = max(params[2], 1e-8)      # Prevent nu going negative
                params[3] = max(params[3], -0.999)    # Prevent nu going negative
                params[3] = min(params[3], 0.999)     # Prevent nu going negative
                try:
                    calc_vols = np.array([SABR_FUNC(strike, F[i], T[i], *params) for strike in K[i,:]])
    
                    error = ((calc_vols - np.array(market_vol[i,:]))**2 ).mean() **.5
                    
                except:

                    calc_vols = np.array([SABR_FUNC(strike, F, T, *params) for strike in K])
      
                    error = ((calc_vols - np.array(market_vol[i]))**2 ).mean() **.5
                    
                return error
        
        constraints_dict = (
            {'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: 0.99 - x[1]},
            {'type': 'ineq', 'fun': lambda x: x[1]},
            {'type': 'ineq', 'fun': lambda x: x[2]},
            {'type': 'ineq', 'fun': lambda x: 1. - x[3]**2}
        )
        
        result = optimize.minimize(func, params, constraints=constraints_dict, options={'eps': 1e-5})
        new_params = result['x']
        
        # Have try condition based on the dimensionality of our strikes, K. 
        try:
            newVols = np.array([SABR_FUNC(strike, F[i], T[i], *new_params) for strike in K[i,:]])
            RMSE[i] = np.sqrt( (1/np.size(K))* np.sum(newVols - market_vol[i,:]**2))
            
            calibrated_params[i,:] = new_params
            
            params = new_params
        except:
            newVols = np.array([SABR_FUNC(strike, F, T, *new_params) for strike in K])
            RMSE[i] = np.sqrt( (1/np.size(K))* np.sum(newVols - market_vol[i]**2))
            
            calibrated_params[i,:] = new_params
            
            params = new_params
 
        
    return calibrated_params,RMSE
