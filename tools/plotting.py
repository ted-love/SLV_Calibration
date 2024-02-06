#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 16:38:28 2024

@author: ted
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
pio.renderers.default='browser'

def plot_subplots(S_0, market_vol, market_K, market_T, calibrated_vol, calibrated_K, calibrated_T, local_vol):
        
    fig = make_subplots(rows=1, cols=3,
                        specs=[[{'type': 'Surface'}, {'type': 'Surface'}, {'type': 'Surface'}]],
                        subplot_titles=['Market Surface', 'Arb-Free Surface', 'Local Vol Surface'],
                        horizontal_spacing=0.01,
                        shared_xaxes=True,
                        shared_yaxes=True)
    
    fig.add_trace(go.Surface(
        
                    z=100*market_vol,
                    x=np.log(market_K / S_0),
                    y=market_T,
                    
                    showscale=False), row=1, col=1)
    
    fig.add_trace(go.Surface(
        
                    z=100*calibrated_vol,
                    x=np.log(calibrated_K),
                    y=calibrated_T,
                    
                    showscale=False), row=1, col=2)
    
    fig.add_trace(go.Surface(
        
                    z=100*local_vol,
                    x=np.log(calibrated_K),
                    y=calibrated_T,
                    
                    showscale=False), row=1, col=3)
    
    fig.update_scenes(xaxis_title_text='log(K/S)',  
                      yaxis_title_text='T (years)',  
                      zaxis_title_text='Local Volatility (%)')
        
    fig.layout.annotations[0].update(y=0.8)
    fig.layout.annotations[1].update(y=0.8)
    fig.layout.annotations[2].update(y=0.8)

    fig.show()
    
    
    
def plot_subplots_2(P,X_vec_mesh,V_vec_mesh,Leverage_function,X_mesh_T,T_mesh):
    
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'Surface'}, {'type': 'Surface'}]],subplot_titles=['Density',
                                'Leverage Function'],shared_xaxes=False,shared_yaxes=False,horizontal_spacing=0)
    
    fig.add_trace(go.Surface(z=P,x=V_vec_mesh,y=X_vec_mesh,showscale=False), row=1, col=1)
    fig.add_trace(go.Surface(z=Leverage_function ,x=X_mesh_T,y=T_mesh,showscale=False), row=1, col=2)
    
    fig.update_scenes(xaxis_title_text='Variance, V',  
                      yaxis_title_text='log-moneyness: log(S_t/S_0)',  
                      zaxis_title_text='Density, P',row=1,col=1)
    
    fig.update_scenes(xaxis_title_text='log-moneyness: log(S_t/S_0)',  
                      yaxis_title_text='Time (years)',  
                      zaxis_title_text='Leverage Function',row=1,col=2)
    fig.layout.annotations[0].update(y=0.8)
    fig.layout.annotations[1].update(y=0.8)
    fig.update_scenes(aspectratio=dict(x=1, y=1, z=1))
    fig.show()