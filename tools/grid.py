#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:42:27 2023

@author: ted
"""
import numpy as np
from scipy import sparse

def mapping_S(x, K_mid, c):
    """
    
    Parameters
    ----------
    x : int
        value from 0 to m1.
    K_mid : float
        mid strike price of the options.
    c : float
        A reasonable size scaler for the mapping, c = K_mid/5.

    Returns
    -------
    float
        New spatial coordinate for the non-uniform log-returns grid, X_i = log(S_i/S_0).

    """
    return K_mid + c * np.sinh(x)


def mapping_V(x, d):
    """
    

    Parameters
    ----------
    x : int
        value from 0 to m2.
    d : float
        A reasonable size scaler for the mapping, d = V_max / 500.

    Returns
    -------
   float
       New spatial coordinate for the non-uniform variance grid.
       
    """
    
    return d * np.sinh(x)


def make_grid(m1, m2, Smax, S_0, K_mid, Vmax, V_0):
    """
    
    Creating non-uniform spatial grids for the log-returns, X = log(S_t/S_0) and the variance, V.
    
    Parameters
    ----------
    m1 : int
        Number of spatial grid points for the log-returns. X=log(S_t / S_0).
    m2 : int
        Number of spatial grid points for the variance, V. 
    Smax : float
        Maximum value for the stock price in the spatial grid.
    S_0 : float
        Current stock price.
    K_mid : float
        mid strike price of the options.
    Vmax : float
        Maximum value for the variance in the spatial grid.
    V_0 : float
        Current Variance.

    Returns
    -------
    X_vec : NumPy array
        Spatial grid of the log-returns as a vector, X=log(S_t / S_0), with shape (m1,1).
    Delta_X : NumPy array
        The difference of the X_vec array with shape (m1,1). The 1st and last element has value = 0.
    V_vec : NumPy array
        Spatial grid of the variance as a vector, with shape (m2,1).
    Delta_V : NumPy array
        The difference of the V_vec array with shape (m2,1). The 1st and last element has value = 0.
    X_vec_mesh : NumPy array
        X-axis return of a meshgrid with X_vec and V_vec, has shape (m2,m1).
    V_vec_mesh : NumPy array
        Y-axis return of a meshgrid with X_vec and V_vec, has shape (m2,m1).

    """
    

    c = K_mid / 5        # Scaling parameter for new non-uniform log-returns coordinates 
    d = Vmax / 500       # Scaling parameter for new non-uniform variance coordinates
    
    # Creating non-uniform grid for stock price
    Delta_xi = (1.0 / m1) * (np.arcsinh((Smax - K_mid) / c) - np.arcsinh(-K_mid / c))
    Uniform_s = [np.arcsinh(-K_mid / c) + i * Delta_xi for i in range(m1)]
    S_vec = [mapping_S(Uniform_s[i], K_mid, c) for i in range(m1)]
    
    
    # Making sure S_0 is inside the non-uniform grid
    if len(np.where(np.array(S_vec)==S_0)[0])==0:
        S_vec.append(S_0)
        S_vec.sort()
        S_vec.pop(-1)
    S_vec = np.array(S_vec)
    
    
    # Converting to log-returns
    X_vec = np.log(S_vec/S_0)
    X_vec[0] = X_vec[1] - 1.5*(X_vec[2] - X_vec[1])
    
    Delta_X = [X_vec[i] - X_vec[i-1] for i in range(1,m1)]
    Delta_X = np.insert(Delta_X,0,0)
    Delta_X = np.append(Delta_X,0)
    
    # Creating non-uniform grid for variance
    Delta_eta = (1.0 / m2) * np.arcsinh(Vmax / d)
    Uniform_v = [i * Delta_eta for i in range(m2)]
    V_vec = [mapping_V(Uniform_v[i], d) for i in range(m2)]
    
    # Making sure V_0 is inside the non-uniform grid
    if len(np.where(np.array(V_vec)==V_0)[0])==0:
        V_vec.append(V_0)
        V_vec.sort()
        V_vec.pop(-1)
    
    V_vec.pop(0)
    V_vec.insert(0,0.000001)
    Delta_V = [V_vec[i] - V_vec[i-1] for i in range(1,m2)]
    Delta_V.insert(0,0)
    Delta_V = np.append(Delta_V,0)
    
    X_vec_mesh, V_vec_mesh = np.meshgrid(X_vec, V_vec)

    return X_vec, Delta_X, V_vec, Delta_V, X_vec_mesh, V_vec_mesh


def make_derivatives(m1, m2, Delta_X, Delta_V):
    """
    

    Parameters
    ----------
    m1 : int
        Number of spatial grid points for the log returns. X=log(S_t / S_0).
    m2 : int
        Number of spatial grid points for the variance, V. 
    Delta_X : NumPy array
        Change in the spatial grid of the log-returns as a vector with shape (m1+1,1).
    Delta_V : NumPy array
        Change in the spatial grid of the variance as a vector with shape (m2+1,1).

    Returns
    -------
    D_x : NumPy array
        1st order derivative array of log-returns with shape (m1,m1).
    D_xx : NumPy array
        2nd order derivative array of log-returns with shape (m1,m1).
    D_v : NumPy array
        1st order derivative array of variance with shape (m2,m2).
    D_vv : NumPy array
        2nd order derivative array of variance with shape (m2,m2).

    """


    D_x = np.zeros((m1,m1))
    for i in range(1,m1-1):
        D_x[i,i-1] = -Delta_X[i + 1] / (Delta_X[i] * (Delta_X[i] + Delta_X[i + 1]))
        D_x[i,i] = (Delta_X[i + 1] - Delta_X[i]) / (Delta_X[i] * Delta_X[i + 1])
        D_x[i,i+1] = Delta_X[i] / (Delta_X[i + 1] * (Delta_X[i] + Delta_X[i + 1]))
    
    """
    Adjusting derivative boundaries
    """
    D_x[0,0] = (-2 * Delta_X[1] - Delta_X[2]) / (Delta_X[1] * (Delta_X[1] + Delta_X[2]))
    D_x[0,1] = (Delta_X[1] + Delta_X[2]) / (Delta_X[1] * Delta_X[2])
    D_x[0,2] = -Delta_X[1] / (Delta_X[2] * (Delta_X[1] + Delta_X[2]))
    
    D_x[m1-1,m1-2] = -1 / Delta_X[m1-1]
    D_x[m1-1,m1-1] = 1 / Delta_X[m1-1]
    
    
    D_xx = np.zeros((m1,m1))
    for i in range(1,m1-1):
        D_xx[i,i-1] = 2 / (Delta_X[i] * (Delta_X[i] + Delta_X[i+1]))
        D_xx[i,i] = -2 / (Delta_X[i] * Delta_X[i+1])
        D_xx[i,i+1] = 2 / (Delta_X[i+1] * (Delta_X[i] + Delta_X[i+1]))
    
    """
    Adjusting derivative boundaries
    """
    D_xx[0,0] = D_x[0,0]
    D_xx[0,1] = D_x[0,1]
    D_xx[0,2] = D_x[0,2]
    D_xx[m1-1,m1-2] = D_x[m1-1,m1-2] 
    D_xx[m1-1,m1-1] = D_x[m1-1,m1-1] 
    
    
    D_v = np.zeros((m2,m2))
    for i in range(1,m2-1):
        D_v[i,i-1] = -Delta_V[i + 1] / (Delta_V[i] * (Delta_V[i] + Delta_V[i + 1]))
        D_v[i,i] = (Delta_V[i + 1] - Delta_V[i]) / (Delta_V[i] * Delta_V[i + 1])
        D_v[i,i+1] = Delta_V[i] / (Delta_V[i + 1] * (Delta_V[i] + Delta_V[i + 1]))
    
    """
    Adjusting derivative boundaries
    """
    D_v[0,0] = (-2 * Delta_V[1] - Delta_V[2]) / (Delta_V[1] * (Delta_V[1] + Delta_V[2]))
    D_v[0,1] = (Delta_V[1] + Delta_V[2]) / (Delta_V[1] * Delta_V[2])
    D_v[0,2] = -Delta_V[1] / (Delta_V[2] * (Delta_V[1] + Delta_V[2]))
    D_v[m2-1,m2-2] = -1 / Delta_V[m2-1]
    D_v[m2-1,m2-1] = 1 / Delta_V[m2-1]
    
    
    D_vv = np.zeros((m2,m2))
    for i in range(1,m2-1):
        D_vv[i,i-1] = 2 / (Delta_V[i] * (Delta_V[i] + Delta_V[i+1]))
        D_vv[i,i] = -2 / (Delta_V[i] * Delta_V[i+1])
        D_vv[i,i+1] = 2 / (Delta_V[i+1] * (Delta_V[i] + Delta_V[i+1]))
    
    """
    Adjusting derivative boundaries
    """
    D_vv[m2-1,m2-2] = D_v[m2-1,m2-2]
    D_vv[m2-1,m2-1] = D_v[m2-1,m2-1]
    
    return D_x,D_xx,D_v,D_vv


def make_fokker_matrices(n, m1, m2, m, rho, sigma, r_d, r_f, kappa, theta, V_vec, Lev, D_x, D_xx, D_v, D_vv, matrix_helpers):
    """
    
    Spatial Discretisation of the Fokker-Planck PDE (Kolmogorov Forward PDE) matrices. 


    Parameters
    ----------
    n : int
        Time iteration of the PDE solver.
    m1 : int
        Number of spatial grid points for the log returns. X=log(S_t / S_0).
    m2 : int
        Number of spatial grid points for the variance, V. 
    m : int
        Number of spatial points for X and V, m = m1 * m2.
    rho : Float
        Correlation between Stock and Variance.
    sigma : Float
        Volatility of Volatility.
    r_d : Float
        Domestic interest rate.
    r_f : Float
        Foreign interest rate / Dividend Yield.
    kappa : Float
        Rate of mean-reversion for the variance.
    theta : Float
        Long-term vol, also known as v_bar.
    V_vec : NumPy array
        Spatial vector of the variance with shape (m2,1).
    Lev : NumPy array
        Leverage function with shape (m1, N), where N is time grid points.
    D_x : NumPy array
        1st order derivative array of log-returns with shape (m1,m1).
    D_xx : NumPy array
        2nd order derivative array of log-returns with shape (m1,m1).
    D_v : NumPy array
        1st order derivative array of variance with shape (m2,m2).
    D_vv : NumPy array
        2nd order derivative array of variance with shape (m2,m2).
    matrix_helpers : List
        List containing the ODEs from the 1st iteration that do not change - allows for fast computation.

    Returns
    -------
    A : scipy.sparse.csc_matrix object
        Sum of the A_i system of ODes with shape (m1*m2,m1*m2).
        
    A0 : scipy.sparse.csc_matrix object
        Mixed derivatives array with shape (m1*m2,m1*m2).
        
    A1 : scipy.sparse.csc_matrix object
        Log-return derivatives (1st and 2nd order) array with shape (m1*m2,m1*m2).
        
    A2 : scipy.sparse.csc_matrix object
        Variance derivatives (1st and 2nd order) array with shape (m1*m2,m1*m2).
        
    matrix_helpers : List
        List containing the ODEs from the 1st iteration that do not change - allows for fast computation.

    """
    
    if n==1:
        

        I_x = np.eye(m1)
        I_v = np.eye(m2)
        V_vec_I = V_vec * I_v
        
        
        L_kron = sparse.csc_matrix(np.kron(I_v, Lev))
        L2_kron = L_kron.power(2)

        a0 = rho * sigma * np.kron( D_v.T @ np.sqrt(V_vec_I) @ np.sqrt(V_vec_I)  , D_x ) 
        a0 = sparse.csc_matrix(a0)
        A0 = a0 @ L_kron
        A0 = sparse.csc_matrix(A0)
        
        a1_1 = 0.5 * np.kron(V_vec_I , D_xx.T)
        a1_1 = sparse.csc_matrix(a1_1)
        a1_2 =  (r_d - r_f)*np.kron(I_v , D_x.T)
        a1_2 = sparse.csc_matrix(a1_2)
        a1_3 = 0.5 * np.kron(V_vec_I , D_x.T)
        a1_3 = sparse.csc_matrix(a1_3)
        
        A1 =  a1_1 @ L2_kron \
             + a1_2 \
            - a1_3 @ L2_kron
    
        A2 = np.kron( 0.5 * (sigma**2) * D_vv.T @ V_vec_I + 
              kappa*D_v.T @ (theta*I_v - V_vec_I) , I_x)
        A2 = sparse.csc_matrix(A2)
        
        A = A0 + A1 + A2
        
        matrix_helpers = [a0,a1_1,a1_2,a1_3,A2,I_v]
        
        return A,A0,A1,A2,matrix_helpers 
    
    else:
        
        [a0,a1_1,a1_2,a1_3,A2,I_v] = matrix_helpers 
     
        L_kron = sparse.kron(I_v, Lev)
        L2_kron = L_kron.power(2)
         
        A0 = a0 @ L_kron
        A0 = sparse.csc_matrix(A0)
        
        A1 =  a1_1 @ L2_kron \
             + a1_2 \
            - a1_3 @ L2_kron
            
        A = A0 + A1 + A2
            
        return A,A0,A1,A2,matrix_helpers 

