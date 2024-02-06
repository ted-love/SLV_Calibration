#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:47:32 2023

@author: ted
"""

import numpy as np
import pandas as pd
from tools.Heston_Calibration_Class import Data_Class
from scipy import interpolate
from scipy import sparse
import yfinance as yf
from tools.Levenberg_Marquardt import levenberg_Marquardt
from tools.Heston_COS_METHOD import heston_cosine_method
from tools import grid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default='browser'
from tools import FX_conversion
from tools.SABR import calibrate_SABR,SABR_FUNC
import datetime
from tools.SVI import svi_optimise, svi_model
from tools import FX_correlation
import py_vollib_vectorized
from tools import plotting
from tools.SLV_monte_carlo import SLV_MC


#%%

"""
Using pre-loaded data
"""

data = pd.read_csv('FX_data.csv')

S_0 = data['S_0'][0]

r_d = np.array(data['r_d'])/100
r_f = np.array(data['r_f'])/100
T = np.array(data['T'])/12

D10_BF =np.array( data['10_D_BF'])/100
D25_BF = np.array(data['25_D_BF'])/100
D10_RR = np.array(data['10_D_RR'])/100
D25_RR = np.array(data['25_D_RR'])/100
ATM_vol = np.array( data['ATM'])/100


forward = S_0 * np.exp((r_d - r_f)*T)

# Converting Risk-Reversal and Butterfly quotes into Strikes from FX_conversion.py file
market_vol, T, K = FX_conversion.get_strikes(data,S_0,r_d ,r_f ,D10_BF ,D25_BF,D10_RR ,D25_RR ,ATM_vol,T)


Domestic_Curve = interpolate.CubicSpline(np.unique(T), r_d)
Foreign_Curve = interpolate.CubicSpline(np.unique(T), r_f)

domestic_rate = Domestic_Curve(T.flatten())
foreign_rate = Foreign_Curve(T.flatten())

V_0 = ATM_vol[0]      # Using ATM implied volatility as V_0 for Heston parameter.


"""
Downloading Historical EUR/USD price data using Yahoo Finance. 
"""
start_date = datetime.datetime.today()- datetime.timedelta(40000) 
end_date = datetime.datetime.today()
stock ="EURUSD=X"
FX_daily = yf.download(stock, start_date, end_date)



"""
Creating a Data_Class object. This stores all the options data information, i.e. strike, maturity, price, IV etc.
This allows us to delete irregular options from the calibration procedure and keep the information consistent
throughout the whole procedure without requiring global variables. 
"""
#%%
Data = Data_Class()

Data.S = S_0 
Data.K = K.flatten()
Data.T = T.flatten()
Data.market_vol = market_vol.flatten()
Data.r = domestic_rate
Data.q = foreign_rate


"""
Creating arbitrage option type to get the Black-Scholes price since quotes came without price.
We use an option type s.t. the option is OTM. This makes no difference in the calibration procedure
because we use the Black-scholes price has a 1:1 correspondence with implied volatility. 
"""
flag = []
for i in Data.K:
    if i>=S_0:
        flag.append('c')
    else:
        flag.append('p')

Data.flag = np.array(flag)

market_prices = py_vollib_vectorized.vectorized_black_scholes_merton(Data.flag, S_0, Data.K, Data.T, Data.r, Data.market_vol, Data.q, return_as='numpy')

Data.market_prices = market_prices

rho = FX_correlation.calculate_yearly_correlation(FX_daily)
var = np.var(FX_daily['Close'])


"""
Performing an initial calibration to retrieve the Heston dynamics of the option data set.
Then afterwards we can calibrate the Leverage function to the data set. 

"""

N = 240                  # Number of terms of summation during COS-expansion
L = 20                   # Length of truncation
I = 600                  # Max numbr of accepted iterations of calibration
w = 1e3                  # Weight of initial damping factor   
F = 10                   # Factor to reduce pre-calibration by
precision = 0.01         # Precision of numerical differentiation

"""
Initial Guesses
"""
v_bar_guess = var                       # v_bar : long-term vol
sigma_guess = 0.9                       # sigma : vol of vol
rho_guess = rho                         # rho   : correlation between S and V
kappa_guess = 1.000                     # Kappa : rate of mean-reversion
v0_guess = ATM_vol[0]**2                # v0    : initial vol

initial_guesses = np.array([ v_bar_guess, 
                             sigma_guess,      
                             rho_guess,        
                             kappa_guess,         
                             v0_guess
                             ]).reshape(5,1)

"""
Choose params you want to calibrated. Params not in params_2b_calibrated will be fixed. 
put params to be calibrated: params_2b_calibrated = ['v0','vbar','sigma','rho','kappa']
"""

params_2b_calibrated = ['vbar','sigma','kappa','rho','v0']


# Start calibrating with my own LM algorithm
calibrated_params, counts_accepted, counts_rejected, RMSE = levenberg_Marquardt(Data,initial_guesses,I,w,N,L,precision,params_2b_calibrated,1,1)

# Calibrated params
v_bar, sigma, rho, kappa, v0 = calibrated_params.squeeze()

#%%

# 
calibrated_prices = heston_cosine_method(S_0, Data.K, Data.T, N, L, Data.r, Data.q, v_bar, v0, sigma, rho, kappa, Data.flag)

# The calibrated 
calibrated_vol = py_vollib_vectorized.vectorized_implied_volatility(calibrated_prices, S_0, Data.K, Data.T, Data.r, Data.flag, Data.q, return_as='numpy')


"""
FX data has incomplete set of strikes. Using SABR to interpolate and have a variety of different strikes. 
"""

maturities = np.unique(T)
strikes = K.copy()

params = [0.1]*4

SABR_params,RMSE = calibrate_SABR(market_vol,strikes,S_0,maturities,r_d,r_f,params)

N=10
new_strikes = np.linspace(np.exp(-1)*S_0,np.exp(1)*S_0,N)
new_strikes = np.linspace(K.min(),K.max(),N)


SABR_FUNC_vect = np.vectorize(SABR_FUNC)

new_vol = np.empty((np.size(maturities),N) )
for i in range(np.size(maturities)):
    new_vol[i,:] = SABR_FUNC_vect(new_strikes, forward[i], maturities[i], SABR_params[i,0],SABR_params[i,1],SABR_params[i,2],SABR_params[i,3])


new_strikes,maturities_mesh = np.meshgrid(new_strikes,maturities)


#%%

"""
Using SVI parametisation to remove arbitrage on this surface. 

"""
total_volatility = new_vol**2 * maturities_mesh

S = S_0 
r = r_d
q = r_d

"""
Calibrating the SVI parameters
"""
results = []
for i in range(np.size(maturities)):
    
    tot_vol = total_volatility[i,:]
    
    log_money = np.log(new_strikes[i,:]/(S*np.exp((r[i]-q[i])*maturities[i])))
    
    a, d, c, m, sig, rmse = svi_optimise(tot_vol,log_money,[0.05,0.1],10)
    
    results.append((log_money,tot_vol,(a,d,c,m,sig),rmse))


"""
Applying the SVI parameters to a new set of strikes and expiries
"""
surf = np.zeros((10,25))
K_new = np.linspace(np.exp(-1),np.exp(1),25)
for i in range(10):
   
    model = svi_model(*results[i][2])
    surf[i,:] = model(np.log(K_new))
    

vol_arb_free = np.zeros((16,25))
T_new = np.linspace(maturities.min(),maturities.max(),16)
for j in range(25):
    vol_arb_free[:,j] = interpolate.UnivariateSpline(maturities,surf[:,j],k=1)(T_new)


vol_arb_free = np.sqrt(vol_arb_free / np.expand_dims(T_new, 1))   # 


"""
Removing any nan values
"""
nan_rows = []
for i in range(np.size(T_new)):
    if np.isnan(vol_arb_free[i,:]).any():
        nan_rows.append(i)

for i in nan_rows:
    params = [0.1]*4

    nan_loc = np.where(np.isnan(vol_arb_free[i,:]))[0]
    strikes = np.delete(K_new,nan_loc)
    vol = vol_arb_free[i,:]
    vol = np.delete(vol,nan_loc)
    vol_arb_free[i,:] = interpolate.UnivariateSpline(strikes,vol,k=5)(K_new)
  
K_new_mesh, T_new_mesh = np.meshgrid(K_new,T_new)


#%%

def implied_to_local_vol(sigma, eps, T, K, S, r, q, i, j):
    """
    
    Description
    -----------
    Converting the implied volatilities to the local volatility using Dupire's. 
    We use Wilmott's and Gatheral's parameteristion. For the derivative of sigma (iv) wrt T,
    interpolate the sigma along T at T_i+eps and T_i-eps. Then use numerical differentiation.
    
    Read more from: "Stochastic Volatility Modeling" by Lorenzo Bergomi. 

    Parameters
    ----------
    sigma : NumPy array
        2-D matrix of implied volatilities.
    eps : float
        Numerical differentiation difference.
    T : NumPy array
        Option maturities as a meshgrid.
    K : NumPy array
        Option strikes as a meshgrid.
    S : float
        Current price of Stock.
    r : float
        Domestic interest rate.
    q : float
        Dividend/Foreign rate.
    i : int
        Spatial coordinate of the maturity.
    j : int
        Spatial coordinate of the strike.

    Returns
    -------
    local_vol : float
        local volatility for the i'th maturity and j'th strike.

    """
    
    up = 1+eps
    down = 1-eps
    
    # sigma_down = sigma(T-down,K)
    t_down = T[i,j]*down
    sigma_down = np.sqrt( ((T[i,j]-t_down)/(T[i,j]-T[i-1,j])*T[i-1,j]*sigma[i-1,j]**2 + (t_down-T[i-1,j])/(T[i,j]-T[i-1,j])*T[i,j]*sigma[i,j]**2)/t_down)

    # sigma_up = sigma(T+down,K)
    t_up = T[i,j]*up
    sigma_up = np.sqrt( ((T[i+1,j]-t_up)/(T[i+1,j]-T[i,j])*T[i,j]*sigma[i,j]**2 + (t_up-T[i,j])/(T[i+1,j]-T[i,j])*T[i+1,j]*sigma[i+1,j]**2)/t_up)
    
    # Derivative of implied volatility wrt T 
    iv_T = (sigma_up - sigma_down) / (t_up - t_down)
    
    # Derivative of implied volatility wrt K and double derivative .... 
    iv_K = (sigma[i,j+1] - sigma[i,j-1]) / (K[i,j+1]-K[i,j-1])
    iv_K2 = (sigma[i,j+1] - 2*sigma[i,j] + sigma[i,j-1]) / (K[i,j+1]-K[i,j-1])**2
    
    d1 = (np.log(S/K[i,j])+(r - q +0.5*sigma[i,j]**2)*T[i,j])/(sigma[i,j]*np.sqrt(T[i,j]))

    numerator =  sigma[i,j]**2 + 2*sigma[i,j]*T[i,j] * (iv_T + (r-q)*K[i,j]*iv_K)
    denominator = (1 + d1*K[i,j]*np.sqrt(T[i,j])*iv_K)**2 + sigma[i,j]*T[i,j]*(K[i,j]**2) * ( iv_K2 - d1*np.sqrt(T[i,j])*(iv_K**2))

    local_vol = np.sqrt(numerator/denominator)
    
    return local_vol

eps=1e-3

local_vol_grid=np.empty((np.size(T_new)-2,np.size(K_new)-2))

r_d_new = Domestic_Curve(T_new)
r_f_new = Foreign_Curve(T_new)

for i in range(1,np.size(T_new)-1):
    for j in range(1,np.size(K_new)-1):

        local_vol_grid[i-1,-1+j] =  implied_to_local_vol(vol_arb_free,eps,T_new_mesh,K_new_mesh,S_0,r_d_new[i],r_f_new[i],i,j)
        

TT_new = T_new_mesh[1:-1,1:-1]
XX_new = K_new_mesh[1:-1,1:-1]

local_vol_interpolation = interpolate.Rbf(XX_new.flatten(),TT_new.flatten(),local_vol_grid.flatten(),function='linear')

local_vol = local_vol_interpolation(K_new_mesh,T_new_mesh)

# Plotting arb-free surface, local vol surface and market vol surface. 
plotting.plot_subplots(S_0, market_vol, K, T, vol_arb_free, K_new_mesh, T_new_mesh,local_vol)


#%%

V_0 = v0    # Calibrated initial volatility


"""
Creating spatial grids to solve the Fokker-Planck (Kolmogorov Forward Equation)
"""

# grid [0, S] x [0, V]
m1 = 200               # Spatial steps for log-returns, x=log(S_t/S_0)
m2 = 50                # Spatial steps for variance
m = m1 * m2        
N = 50
T_max = T.max()


K_mid = np.median(strikes)
Smax = S_0 * np.exp(2)
Vmax = 0.5

# Creating non-uniform spatial grid
X_vec, Delta_X, V_vec, Delta_V, X_vec_mesh, V_vec_mesh = grid.make_grid(m1, m2, Smax, S_0, K_mid, Vmax, V_0)

Delta_T = T_max/N/40

T_vec = np.linspace(0,T_max,N)

X_vec = np.array(X_vec).reshape(m1,1)     # Spatial coordinates for the log-returns X=log(S_i,S_0) as a vector
V_vec = np.array(V_vec).reshape(m2,1)     # Spatial coordinates for the variance as a vector

Delta_X =np.expand_dims(Delta_X,1)        # Differences of the X_vec vector 
Delta_V = np.expand_dims(Delta_V,1)       # Differences of the V_vec vector


w_x = np.array([(Delta_X[i][0]+Delta_X[i+1][0]) / 2 for i in range(m1)])  # Midpoint differences 
w_v = np.array([(Delta_V[i][0]+Delta_V[i+1][0]) / 2 for i in range(m2)])  # Midpoint differences


Lev_00 = local_vol_interpolation(0,0)/ V_0  # initial leverage function value at X=0 and t=0, L(0,0)


#%%
def initial_condition(X, V, L_00, v_0, r0, q0, kappa, rho, theta, sigma, dt):
    """
    
    Description
    -----------
    Calculating the initial condition of the transition probability distribution, p(X_i,V_i,0)
    This is given approximated by the bivariate delta function, that is:
    p(X_i,V_i,0) = \delta(X_i) * \delta(V_i - V_0)

    which is this approximated by the bivariate normal distribution. 

    Parameters
    ----------
    X : NumPy array
        Meshrid of the log-returns, X=log(S_t,S_0) with size (m2,m1).
    V : NumPy array
        Meshrid of the variance with size (m2,m1).
    L_00 : float
        Initial value of the leverage function at t=0 and X=0.
    v_0 : float
        current value of volatility.
    r0 : float
        current value of domestic rate.
    q0 : float
        current value of the dividend/foreign rate.
    kappa : float
        rate of mean-reverson.
    rho : float
        correlation between S and V.
    theta : float
        long-term vol.
    sigma : float
        vol of vol.
    dt : float
        time step-sizes.

    Returns
    -------
    biv_density : NumPy array
        The initial transition density with shape (m2,m1)

    """
    
    mu_x = ( r0 - q0 - 0.5 * (L_00**2)*v_0 )*dt    # mean of X
    mu_v = kappa*( theta - v_0 ) * dt              # mean of V

    sigma_x = L_00 * np.sqrt(v_0 * dt)         # standard deviation of X
    sigma_v = sigma * np.sqrt(v_0 * dt)        # standard deviation of v

   
    
    exponential_numerator = (X - mu_x)**2 / (sigma_x**2) + (V - mu_v)**2 / (sigma_v**2) \
                                - (2*rho*(X-mu_x)*(V-mu_v))/(sigma_x*sigma_v)
                                
    biv_density = (1 / (2*np.pi*sigma_x*sigma_v*np.sqrt(1-rho**2))) \
                 *  np.exp(-exponential_numerator/(2*(1-rho**2)))
    
    return biv_density



P_0 = initial_condition(X_vec_mesh,V_vec_mesh-V_0,Lev_00,V_0,r_f[0],r_d[0],kappa,rho,v_bar,sigma,Delta_T)


# Checking density
P_sum = 0
for i in range(m1):
    for j in range(m2):
        P_sum += P_0[j,i] * w_x[i] * w_v[j]
print(P_sum)

# returns ~ 1


alpha = 1/2        # ADI param. If alpha = 0, then scheme is fully explicit. If alpha = 1, then scheme is fully implicit.
theta = v_bar      # Long-term vol. 


r_d = Domestic_Curve(T_vec)     # Adjusting rates with new time-scale
r_f = Foreign_Curve(T_vec)



Leverage_function = np.zeros((m1,N))
Leverage_function[:,0] = local_vol_interpolation(X_vec,np.zeros((m1,1))).squeeze()  # Initial leverage at t=0


D_x, D_xx, D_v, D_vv = grid.make_derivatives(m1, m2, Delta_X, Delta_V)    # Derivative matrices


#%%
def Douglas_Scheme(m, m1, m2, r_d, r_f, N, P0, Leverage_function, Delta_T, alpha, D_x, D_xx, D_v, D_vv, T_vec, X_vec, V_vec, w_x, w_v, local_vol_interp, rho, sigma, kappa, theta):
    """
    
    Description
    -----------
    Douglas ADI scheme. This is solving the Fokker-Planck (Kolmogorov-Forward equation)
    More information can be found here: https://dspace.mit.edu/bitstream/handle/1721.1/56567/18-336Spring-2005/NR/rdonlyres/Mathematics/18-336Spring-2005/1A0AFFF5-36BD-4177-9E36-77D6EDB55E99/0/adi_method.pdf

    """
    P_vec = P0.flatten()
    I = np.identity(m)

    
    eps=1e-8

   
    L = np.diag(Leverage_function[:,0])    
    
    matrix_helpers = []
    for n in range(1,N):    

        A,A_0,A_1,A_2,matrix_helpers =  grid.make_fokker_matrices(n,m1, m2, m, rho, sigma, r_d[n], r_f[n], kappa, theta, V_vec, L, D_x, D_xx, D_v, D_vv, matrix_helpers)
        
        inv_ini_1 = sparse.csc_matrix(I - alpha*Delta_T*A_1)
        inv_1 = sparse.linalg.inv(inv_ini_1)
        
        inv_ini_2 = sparse.csc_matrix(I - alpha*Delta_T*A_2)
        inv_2 = sparse.linalg.inv(inv_ini_2)
  
        Y_0 = P_vec + Delta_T*A*P_vec
        Y_1 = inv_1 * (Y_0 - alpha*Delta_T*A_1*P_vec)
        Y_2 = inv_2 * (Y_1 - alpha*Delta_T*A_2*P_vec)

        P_vec = Y_2   
        P_mat = P_vec.reshape(m2,m1)
        
        """
        If numerator or denominator in the leverage function is negative, make L_n = L_n-1
        
        """
        numerator=0
        denominator = 0
        for j in range(m2):
            numerator += P_mat[j,:]  * w_v[j]
            denominator +=  V_vec[j]*P_mat[j,:] * w_v[j]
        
        numerator += theta * eps
        denominator += eps
        time1 = T_vec[n] * np.ones(m1)
        l_vol = local_vol_interp(time1,np.exp(X_vec.squeeze())*S_0)
         
        for i in range(m1):
            
            if numerator[i] < 0 or denominator[i] < 0:
             
                Leverage_function[i,n] = Leverage_function[i,n-1]

            else:
                Leverage_function[i,n] = l_vol[i] * np.sqrt(numerator[i]/denominator[i])

        L = sparse.csc_matrix(np.diag(Leverage_function[:,n])) 
        print('Iteration ',n," out of ",N-1)

    return P_mat,Leverage_function

P, Leverage_function = Douglas_Scheme(m, m1, m2, r_d, r_f, N, P_0, Leverage_function, Delta_T, alpha, D_x, D_xx, D_v, D_vv, T_vec, X_vec, V_vec, w_x, w_v, local_vol_interpolation, rho, sigma, kappa, theta)

np.savetxt('Leverage_func.csv',Leverage_function,delimiter=',')




T_mesh,X_mesh_T = np.meshgrid(T_vec,X_vec)
plotting.plot_subplots_2(P,X_vec_mesh,V_vec_mesh,Leverage_function,X_mesh_T,T_mesh)

#%% 
"""
Interpolating the leverage function to use for MC simulation
"""

time=np.linspace(0,T_max,N)

time_steps , spatial_steps = np.meshgrid(time,np.exp(X_vec.squeeze())*S_0)

Leverage_function_interpolation = interpolate.Rbf(time_steps.flatten(),spatial_steps.flatten(),Leverage_function.flatten())
#%%
"""
MC simulation
"""
Nsteps = 400
Nsims = 10000

S_mc = SLV_MC(v0, kappa, sigma, theta, rho, Domestic_Curve, Foreign_Curve , Leverage_function_interpolation, S_0, T_max, Nsteps, Nsims)



S_mc = np.genfromtxt('MC_Result.csv',delimiter=',')


#%%

"""
Backing out implied volatility and then calculating RMSE..
"""

num_options = np.size(Data.T)
C=np.empty(num_options)
for i in range(num_options):
    t = Data.T[i]
    flag = Data.flag[i]
    k = Data.K[i]
    step = np.floor((t/T_max) * Nsteps-1)
    step = int(step)
    r = Domestic_Curve(t)
    if flag=='c':
        C[i] = np.exp(-r*t) * np.mean(np.maximum(S_mc[:,step]-k,0))
    if flag=='p':
        C[i] = np.exp(-r*t) * np.mean(np.maximum(k-S_mc[:,step],0))
        

calibrated_vol = py_vollib_vectorized.vectorized_implied_volatility(C, S_0, Data.K, Data.T, Data.r, Data.flag, Data.q, return_as='numpy')



RMSE = np.sqrt( 1/np.size(calibrated_vol) * np.sum((calibrated_vol - Data.market_vol)**2))
print(RMSE)


