#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:41:00 2024

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

#%%



LOB_orders = pd.read_csv('AMZN_2012-06-21_34200000_57600000_message_10.csv',
                              names=['time','type','ID','size','price','direction'],index_col=['time'])



time = np.array(LOB_orders.index)    # time

N = 80   # steps
dt = (time.max() - time.min()) / N 

new_time  = np.linspace(time.min(),time.max(),80)

empty_data = np.empty((80,4))

LOB_executed_orders = pd.DataFrame(empty_data,index = new_time, columns = ['bid exec. no.','bid volume','ask exec. no.','ask volume'])
LOB_executed_orders = LOB_executed_orders.rename_axis(index='time')
idx=0

for t in new_time:
    
    

    sum_bid_vol = LOB_orders.loc[(LOB_orders['direction'] == 1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 4)]['size'].sum()    
    
        
    sum_bid_vol += LOB_orders.loc[(LOB_orders['direction'] == 1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 5)]['size'].sum()    
    
    
    sum_bid_counts = LOB_orders.loc[(LOB_orders['direction'] == 1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 4)]['size'].count()    

    sum_bid_counts += LOB_orders.loc[(LOB_orders['direction'] == 1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 5)]['size'].count()    

    sum_ask_vol = LOB_orders.loc[(LOB_orders['direction'] == -1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 4)]['size'].sum()    
    sum_ask_vol += LOB_orders.loc[(LOB_orders['direction'] == -1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 5)]['size'].sum()    
                
    
    sum_ask_counts = LOB_orders.loc[(LOB_orders['direction'] == -1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 4)]['size'].count() 
   
    sum_ask_counts += LOB_orders.loc[(LOB_orders['direction'] == -1) & (LOB_orders.index < t+dt)\
                       & (t <= LOB_orders.index) & (LOB_orders['type'] == 5)]['size'].count() 
    
    
    LOB_executed_orders.iloc[idx]['bid volume'] = sum_bid_vol
    LOB_executed_orders.iloc[idx]['bid exec. no.'] = sum_bid_counts
    
    
    LOB_executed_orders.iloc[idx]['ask volume'] = sum_ask_vol
    LOB_executed_orders.iloc[idx]['ask exec. no.'] = sum_ask_counts

    
    

    idx+=1
    

fig_1 = go.Figure()


fig_1.add_trace(go.Bar(x=new_time,y=LOB_executed_orders['bid volume'].values,
              base=-LOB_executed_orders['bid volume'],
              marker_color='blue',
              name='bids'))

fig_1.add_trace(go.Bar(x=new_time,y=LOB_executed_orders['ask volume'].values,
              base=0,
              marker_color='red',
              name='asks'))


fig_1.update_layout(xaxis_title='time',yaxis_title='volume')
#fig_1.show()

fig_2 = go.Figure()


fig_2.add_trace(go.Bar(x=new_time,y=LOB_executed_orders['bid exec. no.'].values,
              base=-LOB_executed_orders['bid exec. no.'],
              marker_color='blue',
              name='bids'))

fig_2.add_trace(go.Bar(x=new_time,y=LOB_executed_orders['ask exec. no.'].values,
              base=0,
              marker_color='red',
              name='asks'))


fig_2.update_layout(xaxis_title='time',yaxis_title='No. of executions')
#fig_2.show()



col_names = []
for i in range(1,11):
    col_names.append('ask ' + str(i))
    col_names.append('ask vol ' + str(i))
    col_names.append('bid ' + str(i))
    col_names.append('bid vol ' + str(i))
    
LOB_spread = pd.read_csv('AMZN_2012-06-21_34200000_57600000_orderbook_10.csv',names=col_names)
                 





time=0

spread = LOB_spread.iloc[time].values

cleaning_data = np.empty((int((np.size(spread)/2)) , 2))
idx=0
jdx=0
while idx <np.size(spread)/2:
    cleaning_data[idx,0] = spread[jdx] 
    cleaning_data[idx,1] = spread[jdx+1] 
    idx+=1
    jdx+=2
    
cleaning_data[:,0] = cleaning_data[:,0] / 1e4
mid_price = (cleaning_data[0,0] + cleaning_data[1,0])/2

colours=[]
for i in range(int(np.size(cleaning_data)/2)):
    if cleaning_data[i,0] < mid_price:
        colours.append('bid')
    if cleaning_data[i,0] > mid_price:
        colours.append('ask')


data_with_colors = {
    'X': cleaning_data[:, 0],
    'Y': cleaning_data[:, 1],
    'Colors': colours
}

# Create a bar chart with color labels and custom legend labels
fig_3 = px.bar(data_with_colors, x='X', y='Y', color='Colors',
               title='Bid-Ask Spread at time t = ' + str(time))

fig_3.add_shape(
    type="line",
    x0=mid_price,
    y0=0,
    x1=mid_price,
    y1=np.amax(cleaning_data[:, 1]) + 2,
    line=dict(color="green", width=2)
)

fig_3.add_asknnotation(
    x=mid_price,
    y=np.amax(cleaning_data[:, 1]) + 6,
    text="Mid Price = $" + str(mid_price),
    showarrow=True,
    arrowhead=1,
    ax=0,
)

# Update legend labels for colors
fig_3.update_traces(marker=dict(color='red'), selector=dict(name='red'))
fig_3.update_traces(marker=dict(color='blue'), selector=dict(name='blue'))

fig_3.update_layout(xaxis_title='Price', yaxis_title='Volume')
#fig_3.show()







mid_price = ((LOB_spread['ask 1'] +  LOB_spread['bid 1']) / 2  )


#%%
df_bid_price = pd.DataFrame(mid_price,columns=['midPrice'])

df_ref_price = pd.DataFrame(np.zeros(269748),columns=['RefPrice'])
#%%
for i in range(np.shape(df_ref_price)[0]):
    if LOB_orders.iloc[i]['direction']==-1:
        df_ref_price.iloc[i]['RefPrice'] = LOB_spread.iloc[i]['ask 1']
    else:
        df_ref_price.iloc[i]['RefPrice'] = LOB_spread.iloc[i]['bid 1']




#%%

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


    temp_df = execution_LO_bid.loc[time[t-1]:time[t]]
    X = temp_df['X'].value_counts()
    Lambda_bid = Lambda_bid.add(X,fill_value=0)
    

Lambda_bid = Lambda_bid / 390


Lambda_ask = pd.Series(data=null_data,index=delta)

for t in range(1,np.size(time)):


    temp_df = execution_LO_ask.loc[time[t-1]:time[t]]
    X = temp_df['X'].value_counts()
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

#lambda_ask = A * np.exp(-k_ask * )

"""
fig = px.scatter(log_Lambda)
fig.show()
"""

#%%


lambda_a = Lambda_ask
lambda_b = Lambda_bid
kappa_a = k_ask
kappa_b = k_bid

"""
kappa_a = 10
kappa_b = 10
lambda_a = 50
lambda_b = 50

"""
    



def solve_optimal_bid_ask(lambda_b,lambda_a,kappa_a,kappa_b,q_min,q_max,T,k):
    """
    
    kappa_a=10
    kappa_b=10
    lambda_a=50
    lambda_b=50

    q_min = -25
    q_max = 25
    alpha = 0.00001
    k=500
    T=1
    """
    h = q_max - q_min + 1

    q_grid = np.linspace(q_max,q_min,h,dtype=int).tolist()
    q_map = dict((q, i) for i, q in enumerate(q_grid))
    position = lambda q: q_map[q]

    dt = T/k

    u = np.zeros((h, k))
    u[:, -1] = -alpha * np.array(q_grid)**2


    t_vec = np.zeros(k)
    t_vec[-1] = T
    for i in range(k-1,0,-1):
    
        t_vec[i-1] = t_vec[i] - dt
        
        u_current = np.zeros(h)
        
        u_previous = u[:,i]
        for q in range(q_max,q_min-1,-1):
        
            if q == q_min:
                u_current[position(q)] = u_previous[position(q)] \
                                         + ((lambda_b / (kappa_b * np.e)) *  np.exp(kappa_b*(u_previous[position(q+1)]-u_previous[position(q)])))*dt
           
            elif q == q_max:
                u_current[position(q)] = u_previous[position(q)] \
                                         + ((lambda_a / (kappa_a * np.e))  * np.exp(kappa_a*(u_previous[position(q-1)]-u_previous[position(q)])))*dt
            
            else:
                u_current[position(q)] = u_previous[position(q)] \
                                         + ((lambda_a / (kappa_a * np.e)) * np.exp(kappa_a*(u_previous[position(q-1)]-u_previous[position(q)]))
                                         + ( lambda_b / (kappa_b * np.e)) * np.exp(kappa_b*(u_previous[position(q+1)]-u_previous[position(q)])))*dt
        
         
        u[:,i-1] = u_current
    
    d_ask = {}
    d_bid = {}
    for q in range(q_max,q_min,-1):
        d_ask[q] = (u[position(q)] - u[position(q-1)]) + (1 / kappa_a) 
    
    for q in range(q_min,q_max):
        d_bid[q] = (u[position(q)] - u[position(q+1)]) + (1 / kappa_b) 
     
    return d_ask, d_bid







k=n*10
T=time.max()

q=0
S_0 = execution_LO.iloc[0]['midPrice']

sigma = 24
gamma = 0.0001

d_ask_0 = S_0 + (1-2*q) * gamma * T * sigma**2 / 2
d_bid_0 = S_0 + (-1-2*q) * gamma * T * sigma**2 / 2
alpha=0.01



delta_a = - S_0 + d_ask_0 + 1/gamma * np.log(1 + gamma/k_ask)
delta_b =  S_0 - d_bid_0 + 1/gamma * np.log(1 + gamma/k_bid)

Lambda_bid_interpolate = interpolate.make_interp_spline(Lambda_bid.index,Lambda_bid.values)
Lambda_ask_interpolate = interpolate.make_interp_spline(Lambda_ask.index,Lambda_ask.values)



lab = Lambda_bid_interpolate(delta_b).item()
laa = Lambda_ask_interpolate(delta_a).item()
q_max = 50
q_min = -50

d_ask,d_bid = solve_optimal_bid_ask(lab,laa,kappa_a,kappa_b,q_min,q_max,T,k)


import matplotlib.pyplot as plt


plt.plot(d_ask[-20],label='-20')
plt.plot(d_ask[-10],label='-10')
plt.plot(d_ask[10],label='10')
plt.plot(d_ask[20],label='20')
plt.title('asks')
plt.legend()

plt.show()
plt.plot(d_bid[-20],label='-20')
plt.plot(d_bid[-10],label='-10')
plt.plot(d_bid[10],label='10')
plt.plot(d_bid[20],label='20')
plt.title('bids')
plt.legend()
plt.show()

#%%
S = S_0
dt = T / k
q=0
X_0 = 1e12
X=1e12
j=1
i=100
p_a_list = []
p_b_list =[]
S_list = [S]
sale=0
buy=0
while i <k:
    
    prob_a = np.random.uniform(0,1,1).item()
    prob_b = np.random.uniform(0,1,1).item()
    
    p_a = S + d_ask[q][i] 
    p_b = S - d_bid[q][i]
    
    p_a_list.append(p_a)
    p_b_list.append(p_b)
    if prob_a < Lambda_ask_interpolate(d_ask[q][i]).item():
        q = q - 1
        X = X + p_a
       # print(d_ask[q][i])
        
        sale+=1

    if prob_b < Lambda_ask_interpolate(d_bid[q][i]).item():
        q = q + 1
        X = X - p_b     
        buy+=1
    
    S += np.sqrt(k/100)* sigma * np.random.randn(1)[0]
    S_list.append(S)
    print(q)
    i+=10

    
Utility = X + q*S - alpha*q**2

print("Final Utility: ",Utility)
print('Difference: ', Utility - X_0)


#%%




#%%






i=0


delta_a = d_ask[0][0]
delta_b = d_bid[0][0]

S = S_0
X = 1e6
q=0
while dt * i < T:
    delta_a = d_ask[q][i]
    delta_b = d_bid[q][i]
    
    LO_a = S + delta_a
    LO_b = S - delta_b
    
    is_ask_hit = False
    is_bid_hit = False
    
    for j in range(20):
        S += np.sqrt(dt/20)* sigma * np.random.randn(1)[0]
        if S > LO_a and is_ask_hit==False:
            X = X + LO_a
            q = q + 1
            is_ask_hit=True
            
        if S < LO_b and is_bid_hit==False:
            X = X + LO_b
            q = q - 1
            is_bid_hit=True
        
        if is_ask_hit == True and is_bid_hit == True:
            break
        
    i+=1
    


Utility = X + q*S - alpha*q**2

print(Utility)

