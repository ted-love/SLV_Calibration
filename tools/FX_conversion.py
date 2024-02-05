#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:30:08 2024

@author: ted
"""

from scipy import interpolate
import numpy as np
from scipy.stats import norm


def get_strikes(data,S_0,r_d ,r_f ,D10_BF ,D25_BF,D10_RR ,D25_RR ,ATM_vol,T):
    
    
    forward = S_0 * np.exp((r_d - r_f)*T)
    K_atm = forward * np.exp(0.5*T*ATM_vol**2)
        
    Treasury_Curve = interpolate.CubicSpline(T, r_d)
    Implied_Dividend_Curve = interpolate.CubicSpline(T, r_f)
    
    def convert_RR_BF_no_drft(S,T,atm_vol,RR_BF):
        return S*np.exp(0.5 * T * atm_vol**2 + (RR_BF*atm_vol*np.sqrt(T)))
    
    
    
    def calculate_imp_vol(S,T,K,atm_vol,RR_BF):
        return 1/np.sqrt(T) * ( np.log(K/S) - 0.5 * atm_vol**2 *T)/(RR_BF * atm_vol)
    
    
    
    phi = 1
    
    
    def calc_K(S_0,T,vol,r_d,r_f,delta,flag,drift):
        
        if flag=='c':
            phi=1
        elif flag=='p':
            phi=-1
        
        if drift=='y':
            drift = np.exp(r_f*T)
        elif drift=='n':
            drift = 1
        forward = S_0 * np.exp((r_d-r_f)*T)
    
        
        
        return forward * np.exp(-phi*norm.ppf(phi*delta*drift) * vol * np.sqrt(T)+
                                0.5 *vol**2 * T)
        
    
    
    
    sigma_p_25_no_drift = D25_BF[:6] + ATM_vol[:6] - 0.5*D25_RR[:6]
    
    sigma_c_25_no_drift = sigma_p_25_no_drift + D25_RR[0]
    
    
    K_call_25_no_drift = calc_K(S_0,T[:6],sigma_c_25_no_drift,r_d[:6],r_f[:6],0.25,'c','y')
    K_put_25_no_drift = calc_K(S_0,T[:6],sigma_p_25_no_drift,r_d[:6],r_f[:6],-0.25,'p','y')
    
    sigma_p_10_no_drift = D10_BF[:6] + ATM_vol[:6] - 0.5*D10_RR[:6]
    
    sigma_c_10_no_drift = sigma_p_10_no_drift + D10_RR[0]
    
    
    K_call_10_no_drift = calc_K(S_0,T[:6],sigma_c_10_no_drift,r_d[:6],r_f[:6],0.10,'c','y')
    K_put_10_no_drift = calc_K(S_0,T[:6],sigma_p_10_no_drift,r_d[:6],r_f[:6],-0.10,'p','y')
    
    
    
    
    sigma_p_25_drift = D25_BF[6:] + ATM_vol[6:] - 0.5*D25_RR[6:]
    
    sigma_c_25_drift = sigma_p_25_drift + D25_RR[0]
    
    
    K_call_25_drift = calc_K(S_0,T[6:],sigma_c_25_drift,r_d[6:],r_f[6:],0.25,'c','y')
    K_put_25_drift = calc_K(S_0,T[6:],sigma_p_25_drift,r_d[6:],r_f[6:],-0.25,'p','y')
    
    sigma_p_10_drift = D10_BF[6:] + ATM_vol[6:] - 0.5*D10_RR[6:]
    
    sigma_c_10_drift = sigma_p_10_drift + D10_RR[0]
    
    
    K_call_10_drift = calc_K(S_0,T[6:],sigma_c_10_drift,r_d[6:],r_f[6:],0.10,'c','y')
    K_put_10_drift = calc_K(S_0,T[6:],sigma_p_10_drift,r_d[6:],r_f[6:],-0.10,'p','y')
    
    
    
    K = []
    sigma = []
    T_new = []
    for i in range(np.size(T)):
        tau = T[i]
        if tau <=1:
            sigma.append(sigma_p_10_no_drift[i])
            sigma.append(sigma_p_25_no_drift[i])
            sigma.append(ATM_vol[i])
            sigma.append(sigma_c_25_no_drift[i])
            sigma.append(sigma_c_10_no_drift[i])
    
    
            K.append(K_put_10_no_drift[i])
            K.append(K_put_25_no_drift[i])
            K.append(K_atm[i])
            K.append(K_call_25_no_drift[i])
            K.append(K_call_10_no_drift[i])
    
            T_new.append(tau)
            T_new.append(tau)
            T_new.append(tau)
            T_new.append(tau)
            T_new.append(tau)
            
        else:
            sigma.append(sigma_p_10_drift[i-6])
            sigma.append(sigma_p_25_drift[i-6])
            sigma.append(ATM_vol[i])
            sigma.append(sigma_c_25_drift[i-6])
            sigma.append(sigma_c_10_drift[i-6])
    
            K.append(K_put_10_drift[i-6])
            K.append(K_put_25_drift[i-6])       
            K.append(K_atm[i])
            K.append(K_call_25_drift[i-6])
            K.append(K_call_10_drift[i-6])
            
            T_new.append(tau)
            T_new.append(tau)
            T_new.append(tau)
            T_new.append(tau)
            T_new.append(tau)
    
    
    T = np.array(T_new).reshape(10,5)
    K = np.array(K).reshape(10,5)
    sigma = np.array(sigma).reshape(10,5)
    
    return sigma,T,K