#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:59:38 2024

@author: ted
"""
import numpy as np
from scipy import optimize 

"""

This module is based on the work "Quasi-Explicit Calibration of Gatheral’s SVI model," by Zeliade White.

Parameters m and sigma are calibrated using Nelder-Mead and a,d,c are pre and post-calibrated using least-squares optimization.

Available from: https://www.semanticscholar.org/paper/Quasi-Explicit-Calibration-of-Gatheral’s-SVI-model-White/7a12d2d0b5f2dd208ba7842ff8ac6507372797c9

"""

def svi_parameterization(x, a, d, c, m, sigma):
    """
    
    Parameters
    ----------
    x : float
        log-money.
    a : float
        SVI param.
    d : float
        SVI param.
    c : float
        SVI param.
    m : float
        SVI param.
    sigma : float
        SVI param.

    Returns
    -------
    float
        Total variance.

    """
    
    y = (x - m) / sigma
    
    return a + d*y + c*np.sqrt(y**2 + 1)

class svi_model:
    """
    Creating an SVI object for SVI parameterization.
    """
    
    def __init__(self, a, d, c, m, sigma):
        """

        Parameters
        ----------
        a : float
            SVI param.
        d : float
            SVI param.
        c : float
            SVI param.
        m : float
            SVI param.
        sigma : float
            SVI param.

        Returns
        -------
        None.

        """
        self.a = a           
        self.d = d
        self.c = c
        self.m = m
        self.sigma = sigma
        
    def __call__(self, x):
        """
        

        Parameters
        ----------
        x : float
            log-moneyness.

        Returns
        -------
        float
            SVI parameterization object.

        """
        return svi_parameterization(x,self.a,self.d,self.c,self.m,self.sigma)
    
def svi_optimise(iv,x,init_m_sigma,maxiter=10,tolerance=1e-12):
    """
    

    Parameters
    ----------
    iv : NumPy array
        implied volatility.
    x : NumPy array
        log-moneyness.
    init_m_sigma : list
        2 element list containing initial guess for m and sigma.
    maxiter : int, optional
        max iterations. The default is 10.
    tolerance : float, optional
        tolerance until optimization is broken. The default is 1e-12.

    Returns
    -------
    list
        list of the SVI_model parameters and the RMSE.

    """
    
    optimise_rmse=1

    def svi_rmse(iv,y,a,d,c):
        return np.sqrt( 1/np.size(iv)*np.sum(a + d*y + c*np.sqrt(y**2 + 1) - iv)**2)
    
    def calculate_ADC(iv,x,m,sigma):
        """
        
        Calculating SVI params: a, d, c

        """
        y = (x - m) / sigma
        s = max(sigma,1e-6)
        bnd = ( (0,0,0), (max(iv.max(),1e-6), 2*np.sqrt(2)*s, 2*s*np.sqrt(2) ) ) 
        z = np.sqrt(np.square(y)+1)
        
        
        A = np.column_stack([np.ones(len(iv)),np.sqrt(2)/2*(y+z),np.sqrt(2)/2*(-y+z)])
       
        a, d, c = optimize.lsq_linear(A,iv,bnd,tol=1e-12,verbose=False).x
        
        return a, np.sqrt(2)/2 * (d - c), np.sqrt(2)/2 * (d + c)
    

    def optimise_m_sigma(m_sigma):
        """
        
        Parameters
        ----------
        m_sigma : list
            m and sigma SVI params.

        Returns
        -------
        float
            The RMSE functon.

        """
        
        m, sigma = m_sigma
        y = (x - m) / sigma 
        
        a, d, c = calculate_ADC(iv,x,m,sigma)
        
        return np.sum( (a + d*y + c*np.sqrt(y**2 + 1) - iv)**2 )

    for i in range(1,maxiter+1):

        m_star, sigma_star = optimize.minimize(optimise_m_sigma,
                                               init_m_sigma,
                                               method='Nelder-Mead',
                                               bounds=( (2*min(x.min(),0), 2*max(x.max(),0)),(1e-6,1) ),
                                               tol=1e-12).x
        
        a_star, d_star, c_star = calculate_ADC(iv, x, m_star, sigma_star)
        
        optimise_rmse = svi_rmse(iv, (x-m_star)/sigma_star, a_star, d_star, c_star)
   
        if i>1 and optimise_rmse<1e-6:
            break
        init_m_sigma = [m_star, sigma_star]
        
    result = np.array([a_star,d_star,c_star,m_star,sigma_star,optimise_rmse])
    
    return result



