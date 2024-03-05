#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:12:11 2024

@author: ted
"""


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy import stats
pio.renderers.default='browser'
from scipy import interpolate
from scipy import sparse
import zstandard as zstd
import io
import yfinance
import datetime
import pdb
from scipy import interpolate
import databento as db
from scipy.sparse import diags
from scipy import sparse




#%%


import py_vollib_vectorized




T = np.array([1]*13)
K = np.linspace(4970,5030,13)
S = 5000.
r = np.array([0.05]*13)
q = np.array([0.025]*13)
flag = np.array(['c']*13)
sigma = np.linspace(0.09,0.20,6)
rev_sig = sigma[::-1]
sig = np.array([0.08])
sigma = np.concatenate((rev_sig,sig,sigma))

C = py_vollib_vectorized.vectorized_black_scholes_merton(flag, S, K, T, r, sigma, q, return_as='numpy')


delta = py_vollib_vectorized.vectorized_vega(flag, S, K, T, r, sigma, q,model='black_scholes', return_as='numpy')

import matplotlib.pyplot as plt

plt.scatter(K,delta)




#%%



def options_chain(symbol):
    """

    Parameters
    ----------
    symbol : Str
        Stock Ticker.

    Returns
    -------
    options : DataFrame
       Options data i.e. bid-ask spread, strikes, expiries etc.
    S : Float
        Spot price of the Stock.

    """

    tk = yf.Ticker(symbol)
    # Expiration dates
    
    exps = tk.options
    if symbol=='^VVIX' or symbol=='^VIX':
        return tk

    S = (tk.info['bid'] + tk.info['ask'])/2 

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)

    options['expirationDate'] = pd.to_datetime(options['expirationDate'])
    options['dte'] = ((options['expirationDate'] - datetime.datetime.today()).dt.days + 1) / 365
    
    options['CALL'] = options['contractSymbol'].str[4:].apply(lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['midPrice'] = (options['bid'] + options['ask']) / 2 
    
    options = options.drop(columns = ['contractSize', 'currency', 'impliedVolatility', 'inTheMoney', 'change', 'percentChange', 'lastTradeDate'])
    
    return options,S,tk


"""
Option chains and historical daily returns

"""


SPX,S,SPX_info = options_chain("^SPX")





SPX.to_csv('SPX_options.csv')





#%%



trader_df = pd.read_csv('xnas-itch-20240215.mbp-10.csv')



#%%


trader_ = trader_df.iloc[:100]




#%%



df1 = pd.read_csv('xnas-itch-20240215.trades.csv')






df2 = pd.read_csv('xnas-itch-20220610.ohlcv-1m.csv')


#%%

options_df = pd.read_csv('SPX_options.csv')


#%%
option_trades = pd.read_csv('opra-pillar-20240215.trades.csv')






#%%
option_trades['symbol'] = option_trades['symbol'].str.replace(r'\s+', '', regex=True)



#%%

option_SPX240216C05000000 = option_trades.loc[option_trades['symbol']=='SPX240216C05000000']





#%%


API_KEY = 'db-pjQNFKFiU7nKLbSuFTwV5X3vAEJQR'


client = db.Historical(API_KEY)

details = client.batch.submit_job(
    dataset="OPRA.PILLAR",
    symbols=["SPX   240216C02875000"],
    schema="mbp-1",
    encoding="csv",
    start="2024-02-01T00:00:00+00:00",
    end="2024-02-27T00:00:00+00:00",
)
print(details)




#%%

client = db.Historical(API_KEY)

details = client.batch.submit_job(
    dataset="OPRA.PILLAR",
    symbols=["SPX   240216C05000000"],
    schema="mbp-1",
    encoding="csv",
    start="2024-02-01T00:00:00+00:00",
    end="2024-02-27T00:00:00+00:00",
)
print(details)





#%%


import databento as db

client = db.Historical(API_KEY)

jobs = client.batch.list_jobs(
    states=["queued", "processing", "done"],
    since="2022-06-01",
)
print(jobs)

#%%


# Download all files for the batch job
client.batch.download(
    output_dir="my_data/",
    job_id=jobs[-1]['id'],
)





#%%



#%%


trades = pd.read_csv('opra-pillar-20240214.trades.csv')

data_feb_14 = pd.read_csv('my_data/OPRA-20240302-K4KR68M3RV/opra-pillar-20240214.mbp-1.csv.zst')




#%%

for i in range(np.size(data_feb_14)):
    if data_feb_14.iloc[i]['action']=='C':
        break




#%%

trades = trades.loc[trades['symbol']=='SPX   240216C05000000']



#%%
time_interval = [trades.iloc[0]['ts_recv'],trades.iloc[-1]['ts_recv']]


LOB_isolated = data_feb_14.loc[(data_feb_14['ts_recv']>=time_interval[0] ) & (data_feb_14['ts_recv']<=time_interval[1])]



#%%


LOB_isolated_only_trades = LOB_isolated[LOB_isolated['sequence'].isin(trades['sequence'].values)]

#%%

LOB_iso = LOB_isolated.iloc[268000:270000]



#%%

LOB_orders = pd.read_csv('AMZN_2012-06-21_34200000_57600000_message_10.csv',
                              names=['time','type','ID','size','price','direction'],index_col=['time'])


col_names = []
for i in range(1,11):
    col_names.append('ask ' + str(i))
    col_names.append('ask vol ' + str(i))
    col_names.append('bid ' + str(i))
    col_names.append('bid vol ' + str(i))
    
LOB_spread = pd.read_csv('AMZN_2012-06-21_34200000_57600000_orderbook_10.csv',names=col_names)





mid_price = ((LOB_spread['ask 1'] +  LOB_spread['bid 1']) / 2  )


#%%
df_bid_price = pd.DataFrame(mid_price,columns=['midPrice'])

df_ref_price = pd.DataFrame(np.zeros(269748),columns=['RefPrice'])

LOB_orders.index = (LOB_orders.index - LOB_orders.index[0])

new_LOB_orders = pd.concat([LOB_orders.reset_index(drop=True),df_bid_price.reset_index(drop=True),df_ref_price.reset_index(drop=True)], axis=1)

new_LOB_orders.index=LOB_orders.index


execution_LO = new_LOB_orders.loc[new_LOB_orders['type']>=4]




X2 = execution_LO['direction'] * execution_LO['midPrice'] - execution_LO['direction'] * execution_LO['price']
X2 = X2.rename('X2')

execution_LO = pd.concat([execution_LO,X2],axis=1)

X = execution_LO['direction'] *( execution_LO['price']-execution_LO['RefPrice'] )
X = X.rename('X')

execution_LO = pd.concat([execution_LO,X],axis=1)


mu = (execution_LO.iloc[0]['midPrice'] - execution_LO.iloc[-1]['midPrice']) / execution_LO.index[-1]


n = 391                         # number of time intervals
time = np.linspace(0,390,n)*60  # seconds in minute intervals







execution_LO=execution_LO[execution_LO['X']>0]

        


execution_LO_bid = execution_LO[execution_LO['direction']==1]
execution_LO_ask = execution_LO[execution_LO['direction']==-1]



#%%
LOB_orders.index = (LOB_orders.index - LOB_orders.index[0])

df_mid_price = pd.DataFrame(mid_price,columns=['midPrice'])


new_LOB_orders = pd.concat([LOB_orders.reset_index(drop=True),df_mid_price.reset_index(drop=True)], axis=1)

new_LOB_orders.index=LOB_orders.index


execution_LO = new_LOB_orders.loc[new_LOB_orders['type']>=4]
#%%
X = execution_LO['direction'] * execution_LO['midPrice'] - execution_LO['direction'] * execution_LO['price']

X = X.rename('X')

execution_LO = pd.concat([execution_LO,X],axis=1)

#%%

execution_LO=execution_LO[execution_LO['X']>0]


execution_LO_bid = execution_LO[execution_LO['direction']==1]
execution_LO_ask = execution_LO[execution_LO['direction']==-1]

#%%

n = 391                         # number of time intervals
time = np.linspace(0,390,n)*60  # seconds in minute intervals




delta = np.unique(execution_LO['X'].values)


null_data = np.zeros((391,np.size(delta)))

count=1

null_data = np.zeros(np.size(delta))

Lambda_bid = pd.Series(data=null_data,index=delta)

for t in range(1,np.size(time)):


    temd_df = execution_LO_bid.loc[time[t-1]:time[t]]
    X = temd_df['X'].value_counts()
    Lambda_bid = Lambda_bid.add(X,fill_value=0)
    

Lambda_bid = Lambda_bid / 390


Lambda_ask = pd.Series(data=null_data,index=delta)

for t in range(1,np.size(time)):


    temd_df = execution_LO_ask.loc[time[t-1]:time[t]]
    X = temd_df['X'].value_counts()
    Lambda_ask = Lambda_ask.add(X,fill_value=0)
    

Lambda_ask = Lambda_ask / 390




    

#%%
Lambda_bid = Lambda_bid[Lambda_bid!=0]
Lambda_ask = Lambda_ask[Lambda_ask!=0]

log_Lambda_bid = np.log(Lambda_bid)


res = stats.linregress(log_Lambda_bid.index,log_Lambda_bid.values)

k_bid = -res.slope
A_bid = np.exp(res.intercept)


log_Lambda_ask = np.log(Lambda_ask)


res = stats.linregress(log_Lambda_ask.index,log_Lambda_ask.values)

k_ask = -res.slope
A_ask = np.exp(res.intercept)



#%%


lambda_a = Lambda_ask
lambda_b = Lambda_bid
k_a = k_ask
k_b = k_bid


    

def Thomas_Algorithm(a,b,c,d):
    n = len(d)
    c_star = np.zeros(n-1)
    d_star = np.zeros(n)
    x = np.zeros(n)
    
    c_star[0] = c[0]/b[0]
    d_star[0] = d[0]/b[0]

    for i in range(1,n-1):
        c_star[i] = c[i]/(b[i] - a[i-1]*c_star[i-1])
        
    for i in range(1,n):
        d_star[i] = (d[i] - a[i-1]*d_star[i-1])/(b[i] - a[i-1]*c_star[i-1])
        
    x[n-1] = d_star[n-1]
    
    for i in range(n-1,0,-1):
        x[i-1] = d_star[i-1] - c_star[i-1]*x[i]
        
    return x



def Thomas_Algorithm_Shermann(a,b,c,d):
    n = len(d)

    alpha = 0.
    beta = c[n-1]
    gamma = -b[0]
    
    x = np.array([1]+[0]*(n-2)+[alpha])
    
    cmod,u = np.empty(n),np.empty(n)

    cmod[0] = alpha / (b[0] - gamma)
    u[0] = gamma / (b[0] - gamma)
    x[0] = x[0] / (b[0] - gamma)
    for i in range(n-1):
        m = 1. / (b[i] - a[i]*cmod[i-1])
        cmod[i] = m * c[i]
        u[i] = m * (-a[i]*u[i-1])
        x[i] = m * (x[i] - a[i]*u[i-1])
        
    for i in range(n-2,-1,-1):
        u[i] = u[i] - cmod[i]*u[i+1]
        x[i] = x[i] - cmod[i]*x[i+1]
        
    factor =  (x[0] + x[n - 1] * beta / gamma) / (1.0 + u[0] + u[n - 1] * beta / gamma)
    for i in range(0,n):
        x[i] = x[i] - factor * u[i]
        
    return x





#%%













#%%


def solve_optimal_bid_ask(A_ask,A_bid,k_a,k_b,gamma,q_min,q_max,T,M,N,v_bar,kappa,rho,eta):
    
    h = q_max - q_min + 1

    q_grid = np.linspace(q_max,q_min,h,dtype=int).tolist()
    q_map = dict((q, i) for i, q in enumerate(q_grid))
    position = lambda q: q_map[q]


    t_vec = np.linspace(0,T,M+1)
        
    dt = T/M
    v_min = 1e-10
    v_max= 0.5
    v_vec = np.linspace(v_min,v_max,N+1)
    
    dv = v_max/N
    q_vec = np.linspace(q_max,q_min,h,dtype=int)
    t_vec,v_mesh = np.meshgrid(t_vec,v_vec)
    
    
    U = np.zeros((N, M+1))
    d_bid_dict={}
    d_ask_dict={}
 
    U_dict = {}
    for q in range(q_min,q_max+1):
        U[:,-1] = np.exp(-gamma*alpha*np.abs(q))
      #  print(np.exp(-gamma*alpha*np.abs(q)))
        U_dict[q] = U.copy()
        
    
            
    for q in range(q_min+1,q_max+1):  
        #p = (1/gamma) * np.log(U_dict[q-1]/U_dict[q]) 
        p = U_dict[q] - U_dict[q-1]
        d_ask_dict[q] = 1/gamma * np.log(1+gamma/k_a) + p + alpha*(1-2*q)
        
    for q in range(q_min,q_max):
        #p = (1/gamma) * np.log(U_dict[q+1]/U_dict[q])
        p = U_dict[q] - U_dict[q+1]

        d_bid_dict[q] =  1/gamma * np.log(1+gamma/k_b) + p + alpha*(1+2*q)

        
    """
    Use boundary condition
        
    """
    
    a_list =[]
    b_list = []
    c_list = []
    d_list = []
    for t in range(M-1,-1,-1):
    
        for q in range(q_min+1,q_max):
 
            F = A_ask * np.exp( -k_a * d_ask_dict[q][:,t+1]) * ( U_dict[q-1][:,t+1] * np.exp(-k_a * d_ask_dict[q][:,t+1]) - U_dict[q][:,t+1]) \
              + A_bid * np.exp( -k_b * d_bid_dict[q][:,t+1]) * ( U_dict[q+1][:,t+1] * np.exp(-k_b * d_bid_dict[q][:,t+1]) - U_dict[q][:,t+1]) \
                
            
            d_vec = - U_dict[q][:,t+1] / dt - F
            
            a_vec = 0.5 * v_vec * eta**2 / dv**2 - 0.5 * ( kappa * (v_bar - v_vec) - rho*eta*v_vec*gamma*q**2 / dv)
            b_vec = - v_vec * eta**2 / dv**2 - 1/dt + 0.5 * v_vec * (gamma**2) * q**2
            c_vec = 0.5 * v_vec * eta**2 / dv**2 + 0.5 * ( kappa * (v_bar - v_vec) - rho*eta*v_vec*gamma*q**2 / dv)
            
      
            U = Thomas_Algorithm_Shermann(a_vec, b_vec, c_vec, d_vec)
            

            U_dict[q][:,t] = U
            if q==15:
                a_list.append(a_vec)
                b_list.append(b_vec)
                c_list.append(c_vec)
                d_list.append(d_vec)
            
        q_truncated = np.linspace(q_min+1,q_max-1,49,dtype=int)
        for v in range(N):
            y = np.empty(h-2)
            idx=0
            for q in range(q_min+1,q_max):  
                y[idx] = U_dict[q][v][t]
                idx+=1
               
            interp = interpolate.Rbf(q_truncated,y)
            
            U_dict[q_min][v][t] = interp(q_min)
            U_dict[q_max][v][t] = interp(q_max)
            
        
        for q in range(q_min+1,q_max+1):  
            
          #  p = (1/gamma) * np.log(U_dict[q-1][:,t]/U_dict[q][:,t]) 
            """
            if np.isnan(p).any():
                print('p: ',p)
                print('U_q-1: ',U_dict[q-1][:,t])
                print('U_q: ', U_dict[q][:,t])
                print('t:', t)
           
            """
            p = U_dict[q][:,t] - U_dict[q-1][:,t]

            d_ask_dict[q][:,t] = 1/gamma * np.log(1+gamma/k_a) + p #+ alpha*(1-2*q)
            
            
        for q in range(q_min,q_max):
            
          #  p = (1/gamma) * np.log(U_dict[q+1][:,t]/U_dict[q][:,t])
            p = U_dict[q][:,t] - U_dict[q+1][:,t]

            d_bid_dict[q][:,t] = 1/gamma * np.log(1+gamma/k_b) + p #+ alpha*(1+2*q)
            
        

           
        
    return U_dict,d_bid_dict,d_ask_dict,a_list,b_list,c_list,d_list
    





k=n*10
T=time.max()
T=1



q=0
S_0 = execution_LO.iloc[0]['midPrice']
2
sigma = 24
gamma = 0.01

d_ask_0 = S_0 + (1-2*q) * gamma * T * sigma**2 / 2
d_bid_0 = S_0 + (-1-2*q) * gamma * T * sigma**2 / 2
alpha=0.0001



delta_a = - S_0 + d_ask_0 + 1/gamma * np.log(1 + gamma/k_ask)
delta_b =  S_0 - d_bid_0 + 1/gamma * np.log(1 + gamma/k_bid)

#Lambda_bid_interpolate = interpolate.make_interd_spline(Lambda_bid.index,Lambda_bid.values)
#Lambda_ask_interpolate = interpolate.make_interd_spline(Lambda_ask.index,Lambda_ask.values)



#lab = Lambda_bid_interpolate(delta_b).item()
#laa = Lambda_ask_interpolate(delta_a).item()
q_max = 50
q_min = -50

kappa=1.
v_bar = 0.15**2
rho=-0.9
eta=0.2

M=300
N=100

A_ask = 1.
A_bid = 1.

k_a = 1.
k_b = 1.
q_min=-25
q_max=25


K = 5
n = np.shape(execution_LO)[0]
n_bar = (n-K+1) / K


H = int(390 *3 / 2)
TSRV = np.empty(H)

time_max = execution_LO.index.values.max()

time_index = np.linspace(0,n,H+1,dtype=int)
time_index = time_index[1:]
time_vec = np.linspace(0,time_max,1000)
time_vec = time_vec[1:]




U_dict,d_bid_dict,d_ask_dict,a_list,b_list,c_list,d_list = solve_optimal_bid_ask(A_ask,A_bid,k_a,k_b,gamma,q_min,q_max,T,M,N,v_bar,kappa,rho,eta)




#%%
idx=0
for j in time_index:
    n = np.size(execution_LO.iloc[K:j]['midPrice'])
    n_bar = (n-K+1) / K
    TSRV_1 = np.sum((execution_LO.iloc[K:j]['midPrice'] - execution_LO.iloc[:j-K]['midPrice'])**2)

    TSRV_2 = np.sum((execution_LO.iloc[1:j]['midPrice'] - execution_LO.iloc[:j-1]['midPrice'])**2)

    TSRV[idx] = 1/K * TSRV_1 - n_bar / n * TSRV_2
    if TSRV[idx]<0:
        break
    print(idx)
    idx+=1
    
    
    
    
    #%%
    
    
    


H = 800 # size of TSRV array

K=10

n = np.size(S)   # S is an array wih size 10,000


time_index = np.linspace(0,n,H+1,dtype=int)
time_index = time_index[1:]  # new time scale for the TSRV

TSRV = np.empty(H)

idx=0
for j in time_index:
    n = n.size(S[K:j])
    n_bar = (n-K+1) / K
    
    TSRV_1 = np.sum((S[K:j]- S[:j-K])**2)
    TSRV_2 = np.sum((S[1:j] - S[:j-1])**2)

    TSRV[idx] = 1/K * TSRV_1 - n_bar / n * TSRV_2

    idx+=1

t=0
idx=0
T = int(H/2)
variance = np.empty(T)
while t < T:
    variance[idx] = (TSRV[2*(t+1)] - TSRV[2*t]) / (time_max/T)
    t+=1

    idx+=1




#%%

S = execution_LO['midPrice']




time_increments = S.index.max() / 200


#%%


loop = True
variance = []
idx=0
while loop:
    S_partition = S[idx * time_increments <= S.index]
    var = S_partition[ S_partition.index<= (idx+1) * time_increments].var()
    
    variance.append(var)

    
    if idx==199:
        break
    idx+=1
    
    
    
variance = np.array(variance)
    


#%%




variance = variance[~np.isnan(variance)]
variance = np.insert(variance, 0, np.array((variance[0])))
#%%

S = execution_LO['midPrice']
idx=0

S_new = []

for i in range(1,201):

    S_new.append(S[S.index<i * time_increments].iloc[-1])
    
S_new.insert(0,S[0])
S_new = np.array(S_new)




#%%

n = 200

dt = time_max / n

term1 = np.sum(np.sqrt(variance[1:]*variance[:-1]))

term2 = 0
for i in range(1,199):
    
    

    term2 += np.sqrt(variance[i]/variance[i-1]) * np.sum(variance[:-1])

numerator = 1/n**2 *term1 - 1/n**2 * term2

denominator = dt/ 2 - dt /2 /n**2 * np.sum(1/variance[:-1]) * np.sum(variance[:-1])
P = numerator / denominator



kappa = 2/dt * (1 + P*dt / 2/n * np.sum(1/variance[:-1]) - 1/n * np.sum(np.sqrt(variance[1:]/variance[:-1])))

eta = np.sqrt(4 / dt/n * np.sum((np.sqrt(variance[1:]) - np.sqrt(variance[:-1]) \
                  - dt / 2 / np.sqrt(variance[1:]) * (P - kappa*np.sqrt(variance[:-1])))**2))

v_bar = (P * eta**2/4) / kappa

dW = (S_new[1:] - S_new[:-1]) / np.sqrt(variance[:-1])

dB = (variance[1:] - variance[:-1] - kappa * (v_bar - variance[:-1]))  * dt / eta / np.sqrt(variance[:-1])

rho = (1/n/dt) * np.sum(dW * dB)



#%%


#%%

bid = np.empty((N,50))
ask = np.empty((N,50))
idx=0
for q in range(q_min,q_max):  

    bid[:,idx] = d_bid_dict[q][:,286]
    idx+=1
idx=0
for q in range(q_min+1,q_max+1):
    ask[:,idx] = d_ask_dict[q][:,286]
    idx+=1
    
#%%

q_i,v_i = np.meshgrid(np.linspace(-25,24,50),np.linspace(1e-6,0.5,100))

fig = go.Figure(go.Surface(z=bid,x=q_i,y=v_i))


fig.show()




q_i,v_i = np.meshgrid(np.linspace(-24,25,50),np.linspace(1e-6,0.5,100))

fig = go.Figure(go.Surface(z=ask,x=q_i,y=v_i))


fig.show()


#%%



h = q_max - q_min + 1

q_grid = np.linspace(q_max,q_min,h,dtype=int).tolist()
q_map = dict((q, i) for i, q in enumerate(q_grid))
position = lambda q: q_map[q]


t_vec = np.linspace(0,T,M+1)
    
dt = T/M
v_min = 1e-10
v_max= 0.5
v_vec = np.linspace(v_min,v_max,N+1)

dv = v_max/N
q_vec = np.linspace(q_max,q_min,h,dtype=int)
t_vec,v_mesh = np.meshgrid(t_vec,v_vec)
q=10


#%%


#%%
S = S_0
dt = T / k
q=0
X_0 = 1e12
X=1e12
j=1
i=100
d_a_dict = []
d_b_dict =[]
S_dict = [S]
sale=0
buy=0
while i <k:
    
    prob_a = np.random.uniform(0,1,1).item()
    prob_b = np.random.uniform(0,1,1).item()
    
    d_a = S + d_ask[q][i] 
    d_b = S - d_bid[q][i]
    
    d_a_dict.append(d_a)
    d_b_dict.append(d_b)
    if prob_a < Lambda_ask_interpolate(d_ask[q][i]).item():
        q = q - 1
        X = X + d_a
       # print(d_ask[q][i])
        
        sale+=1

    if prob_b < Lambda_ask_interpolate(d_bid[q][i]).item():
        q = q + 1
        X = X - d_b     
        buy+=1
    
    S += np.sqrt(k/100)* sigma * np.random.randn(1)[0]
    S_dict.append(S)
    i+=10

    
Utility = X + q*S - alpha*q**2

print("Final Utility: ",Utility)
print('Difference: ', Utility - X_0)


#%%


x = np.linspace(1,100,100)


y = lambda x : x*(100-x)

Y = y(x)

print(np.sum(Y)/100)





#%%