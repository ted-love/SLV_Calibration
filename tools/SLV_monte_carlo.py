#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:05:34 2024

@author: ted
"""

import numpy as np

def SLV_MC(V0, kappa, sigma, theta, rho, r, q, Lev_func, S0, T, Nsteps, Nsims):
   
    
    dt = T/Nsteps
    
    ####### FULL TRUNCATION ########
    f1 = lambda x : x
    f2 = lambda x : np.maximum(x,0)
    f3 = lambda x : np.maximum(x,0)
    
    Z_v = np.random.normal(0,1,[Nsims,Nsteps])
    Z_2 = np.random.normal(0,1,[Nsims,Nsteps])
    Z_s = rho*Z_v + np.sqrt(1-(rho**2))*Z_2
    
    V = np.full(shape=(Nsims,Nsteps+1),fill_value=V0)
    V_tilde = V

    S = np.full(shape=(Nsims,Nsteps+1),fill_value=S0)

    t_vec = np.ones(Nsims)
    for i in range(1,Nsteps+1):
        
        S[:,i] = S[:,i-1]*np.exp((r((i-1)*dt)-q((i-1)*dt) -(0.5*V[:,i-1]))*dt + Lev_func(t_vec*(i-1)*dt,S[:,i-1]) * np.sqrt(V[:,i-1])*np.sqrt(dt)*Z_s[:,i-1])
        
        V_tilde[:,i] = f1(V_tilde[:,i-1]) + kappa*dt*(theta - f2(V_tilde[:,i-1])) \
            + sigma*np.sqrt(f3(V_tilde[:,i-1]))*Z_v[:,i-1]*np.sqrt(dt)
            
        V[:,i] = f3(V_tilde[:,i])
    
    return S
 
