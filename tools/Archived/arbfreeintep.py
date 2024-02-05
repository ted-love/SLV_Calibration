#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:32:44 2024

@author: ted
"""
import numpy as np
import py_vollib_vectorized
from tools.thin_plate_spline import ThinPlateSpline
from qpsolvers import solve_qp
from tools.ismember import ismember
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def fitSpline(v, u, g, gamma):
    """
    

    Parameters
    ----------
    v : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    g : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    """
    
    
    v = v.flatten()
    u = u.flatten()
    g = g.flatten()
    gamma = gamma.flatten()
    
    m = np.size(v)
    n = np.size(u)    
    y = np.zeros((m,1))
   
    
    for s in range(m):
        for i in range(n-1):
            h = u[i+1] - u[i]
            if u[i]<=v[s] and v[s] <= u[i+1]:
                y[s] = ((v[s]-u[i])*g[i+1] + (u[i+1]-v[s])*g[i])/h - 1/6*(v[s]-u[i])*(u[i+1]-v[s]) \
                    * ( (1+(v[s]-u[i])/h)*gamma[i+1] + (1+(u[i+1]-v[s])/h)*gamma[i])
                    
        if v[s] < u.min():
            dg = (g[2]-g[1]) / (u[2] - u[1]) - (u[2]-u[1]) * gamma[2] /6
            y[s] = g[1] - (u[1]-v[s])*dg
            
        if v[s] > u.max():
            dg =(g[n-1]-g[n-2]) / (u[n-1]-u[n-2]) + (u[n-1]-u[n-2]) * gamma[n-2] / 6
            y[s] = g[n-1] + (v[s] - u[n-1])*dg
            
    return y


def quad_program(U,y,A,b,lowerbound,upperbound):
    lambd=1e-2
    #lambd=5000
    
    n = np.size(U)
    LB = lowerbound.copy()
    UB = upperbound.copy()
   
    for i in range(np.size(LB)):
        if LB[0,i] > UB[0,i]:
          
            LB[0,i] = UB[0,i]
    h = np.diff(U)
    
    Q = np.zeros((n,n-2))
    R = np.zeros((n-2,n-2))
    for j in range(1,n-1):
        Q[j-1,j-1] = 1 / h[j-1]
        Q[j,j-1] = - 1/h[j-1] - 1/h[j]
        Q[j+1,j-1] = 1/h[j-1]
    
    for i in range(1,n-1):
        R[i-1,i-1] = (h[i-1] + h[i]) / 3
        if n-2 > i:
            R[i-1,i] = h[i] / 6
            R[i,i-1] = h[i] / 6
    
    y = np.append(y,np.ones(n-2))
    y = np.expand_dims(y, 1)
    b_1 = np.hstack(( np.eye(n) , np.zeros((n,np.shape(R)[1])) ))
    b_2 = np.hstack(( np.zeros((np.shape(R)[0],n)) , lambd * R))
    P = np.vstack((b_1,b_2))
    
    x_0 = y.copy()
    x_0[n:] = 1e-3
    
    Aeq = np.vstack((Q,-R.T)).T
    beq = np.zeros((np.shape(Aeq)[0],1))
    """
    print('P:',P)
    print('y:',-y)
    print('A:',A)
    print('b:',b)
    print('Aeq:',Aeq)
    print('beq:',beq)
    print('Lb:',LB)
    print('UB:',UB)
    print('x_0:',x_0)
    """
    x = solve_qp(P,-y,A,b,Aeq,b=beq,lb=LB,ub=UB,initvals=x_0,solver='osqp')
   
   # print(x)
   # print('\n')
    g_temp = x[:n]
    
    gamma_1 = np.append(np.array(0),x[n:2*n-1])
    gamma_temp = np.append(gamma_1,np.array(0))
    return g_temp, gamma_temp


def arbitrage_free_interpolation(Data,vol_df,N):
    """
    

    Parameters
    ----------
    Data : Data Class Object
        Data class object containing all the data.

    vol_df : Pandas DataFrame
        contains all the implied vols
    Returns
    -------
    aaaaaa : TYPE
        DESCRIPTION.

    """
   
    Maturities = np.unique(Data.T)
    Strikes = np.unique(Data.K)
    S_0 = Data.S
    interest_rates = np.unique(Data.r)
    dividend_rates = np.unique(Data.q)
    IVs = vol_df.values.T / 100
    flag = Data.flag
    
    """
    print(np.shape(Maturities))
    print(np.shape(Strikes))
    print(np.shape(S_0))
    print(np.shape(interest_rates))
    print(np.shape(dividend_rates))
    print(np.shape(IVs))
    print(np.shape(flag))
    """
    T,K = np.meshgrid(Maturities,Strikes)
    r,_ = np.meshgrid(interest_rates,Strikes)
    q,_ = np.meshgrid(dividend_rates,Strikes)



    t = T.flatten()
    k = K.flatten()
    rr = r.flatten()
    qq = q.flatten()
    flag = []
    for i in k:
        if i < S_0:
            flag.append('p')
        else:
            flag.append('c')
    flag = np.array(flag)
    """
    Maturities = (1/365.25) * np.loadtxt('data/T.csv',delimiter=',')
    
    IVs = np.loadtxt('data/ivs.csv',delimiter=',')
    Strikes = np.loadtxt('data/K.csv',delimiter=',') 
    interest_rates = np.array([0.0436,0.0447,0.0453,0.0457,0.0471,0.0485,0.0493,0.0504])
    dividend_rates = np.zeros(8)
    
    S_0 = 7.2689e3
    """
 
    
    #option_prices =  py_vollib_vectorized.vectorized_black_scholes_merton(flag, S_0, k, t, rr, IVs, qq, return_as='numpy')
    
   # option_prices = option_prices.reshape(np.shape(IVs)[0],np.shape(IVs)[1])
    
    
   
    
    tps = ThinPlateSpline(alpha=0.)
    
    imp_vol = []
    ti=[]
    k=[]
    rr=[]
    qq=[]
    f = []
   
    
    for j in range(np.shape(IVs)[1]):
        for i in range(np.shape(IVs)[0]):
            if not np.isnan(IVs[i,j]):
                imp_vol.append(IVs[i,j])
                ti.append(T[i,j])
                k.append(K[i,j])
                rr.append(r[i,j])
                qq.append(q[i,j])
                f.append(S_0 * np.exp((r[i,j]-q[i,j])*T[i,j]))
    
    
    T = np.expand_dims(np.array(ti),1)
    K = np.expand_dims(np.array(k),1)
    q = np.expand_dims(np.array(qq),1)
    r = np.expand_dims(np.array(rr),1)

    imp_vol = np.expand_dims(np.array(imp_vol),1)

    
    x = np.concatenate((K,T),axis=1)
    y = imp_vol 
    
    
    #%%
    tps.fit(x,y)
    #print('moneyness: ',moneyness)
    
   # kappa_ini = np.linspace(np.ceil((10*moneyness).min())/10,np.floor((10*moneyness).max())/10,N).reshape(N,1)
    #print(np.shape(kappa_ini))
   # print('kappa 1' ,kappa_ini)
    kappa_ini = np.linspace(K.min(),K.max(),N).reshape(N,1)
    print('kappa 2',kappa_ini)
    tau_ini = np.unique(T)
    #print(np.shape(tau_ini))
    
    tau,kappa = np.meshgrid(tau_ini,kappa_ini)

    tau = np.expand_dims(tau.flatten(),1)
    kappa = np.expand_dims(kappa.flatten(),1)
    tau_kappa = np.concatenate((kappa,tau),1)
    
    new_total_var = tps.transform(tau_kappa)
    #print('new_total_var')
    
    imp_vol_interpolated = np.sqrt(new_total_var / tau)
    imp_vol_interpolated = new_total_var
    
    
    print('imp_vol_interp: ',imp_vol_interpolated)
    #flag = (np.array(['c']*np.size(tau))).reshape(np.shape(tau)[0],np.shape(tau)[1])
    flag = []
    
    for i in kappa:
        if i < S_0:
            flag.append('p')
        else:
            flag.append('c')
        
    
    flag=np.array(flag)
    


    np.where(tau.flatten()==np.unique(tau))
    _,idx = ismember(tau.flatten(),np.unique(tau))
    r = interest_rates[idx]
    q = dividend_rates[idx]
    #new_K = kappa_ini * S_0 * np.exp((interest_rates-dividend_rates)*tau_ini) 
    
    call_prices = py_vollib_vectorized.vectorized_black_scholes_merton(flag.squeeze(), S_0, kappa.flatten(), tau.flatten(), r, imp_vol_interpolated.flatten(), q, return_as='numpy')
    call_prices = call_prices.reshape(np.size(kappa_ini),np.size(tau_ini))
   # print(kappa.flatten())
   # print(tau.flatten())
  #  print(imp_vol_interpolated)
  #  print('t',tau)
    print(py_vollib_vectorized.vectorized_black_scholes_merton(flag.squeeze(), S_0, kappa.flatten(), tau.flatten(), 0, imp_vol_interpolated.flatten(), 0, return_as='numpy'))
    tt = np.size(np.unique(tau))
    kk = np.size(np.unique(kappa))
    #print(tt)
    #print(kk)
    g = np.zeros((kk,tt))
    #print(g)
    #print('kk:',kk)
    #print('kappa: ',kappa)
    gamma = g.copy()
    u = g.copy()
    tau = np.unique(tau)
    kappa = np.unique(kappa)
  #  print(forward)
    
    
    
    #%%
    for t in range(tt-1,-1,-1):
        
       # print('t: ', t)
        
        u[:,t] = kappa 
        y = call_prices[:,t]
       # print(y)
        
        n = np.size(u[:,t])
        h = np.diff(u[:,t])
     #   print(h)
        a_temp = np.hstack((np.array([h[0]/6]) , np.zeros(n-3)))
        b_temp = np.hstack((np.zeros(n-2), a_temp))
        A_1 = np.hstack((np.array([1/h[0],-1/h[0]]) , b_temp))
        a_temp = np.hstack((np.zeros((n-2)) , -1/h[n-3]))
        b_temp = np.hstack((1/h[n-2],np.zeros(n-3)))
        c_temp = np.hstack((a_temp,b_temp))
        A_2 = np.hstack((c_temp,h[n-2]/6))
        A = np.vstack((A_1,A_2))
        b_small = np.array([[np.exp(-tau[t] * interest_rates[t])],[0]])
        
        
        lb_1 = S_0 * np.exp( -dividend_rates[t] * tau[t]) \
            - u[:,t].reshape(1,np.size(u[:,t])) * np.exp(-interest_rates[t]*tau[t])
        lb_2 = np.maximum(lb_1, 0)
        lowerbound = np.concatenate((lb_2, np.zeros((1,n-2))),axis=1)
    
        
        if t==tt-1:
            a_temp = np.array([S_0 * np.exp(-dividend_rates[t] *tau[t])])
            b_temp = np.inf * np.ones((2*n-3))
            upperbound = np.concatenate((a_temp,b_temp)).reshape(1,1 + 2*n-3)
        else:
            
            a_temp = np.exp( 0.5 * (dividend_rates[t]+dividend_rates[t+1])*(tau[t+1] - tau[t]) )*g[:,t+1]
            
            b_temp =  np.inf * np.ones(n-2)
            upperbound = np.append(a_temp,b_temp).reshape(1,np.size(a_temp) + np.size(b_temp))
        
        """
        print('\nu:\n',u[:,t])

        print('\nA:\n',A)

        print('\ny:\n',y)
        print('\nb_small:\n',b_small)

        """
        g[:,t],gamma[:,t] = quad_program(u[:,t],y,A,b_small,lowerbound,upperbound)
    
    
    
    price = np.zeros(np.size(K))
    
    for t in range(np.size(tau)):
        pos_maturity = T == tau[t]
        
        price[pos_maturity.squeeze()] = fitSpline(K[pos_maturity], u, g, gamma).squeeze()
        
        
    #%%
    
    
    flag = np.array(['c']*np.size(qq))
    
    K=K.squeeze()
    rr = np.array(rr)
    ti=np.array(ti)
    qq=np.array(qq)
    iv = py_vollib_vectorized.vectorized_implied_volatility(price, S_0, K, ti, rr, flag, q=qq, model='black_scholes_merton',return_as='numpy') 
    
    """
    non_nan = np.where(~np.isnan(iv))
    
    iv=iv[non_nan]
    K = K[non_nan]
    rr = rr[non_nan]
    T = ti[non_nan]
    flag=flag[non_nan]
    qq=qq[non_nan]
    """
    
    return iv,K,T,rr,qq,flag
    
    
    



