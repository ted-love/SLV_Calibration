#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:26:52 2024

@author: ted
"""

import numpy as np

def calculate_yearly_correlation(FX_daily):
    
    years = np.arange(np.amin(FX_daily.index.year),np.amax(FX_daily.index.year)+1)
    years = years[1:-1]
    pears_corr = np.empty([np.size(years)])
    
    idx=0
    
    for year in years:
        
        years_1 = FX_daily.loc[FX_daily.index.year==year].index.month.values
        FX_yearly = FX_daily.loc[FX_daily.index.year==year]
        
        variances = []
        means = []
        for month in np.unique(years_1):
            FX_mean_month = FX_yearly.loc[FX_yearly.index.month==month]
            variance = FX_mean_month['Close'].var()
            variances.append(variance)
            mean =  FX_mean_month['Close'].mean()
            means.append(mean)
       
        pears_corr[idx] = np.corrcoef(means,variances)[0,1]
        idx+=1
        
        
    
    
    correlation_each_year = np.mean(pears_corr)
    return correlation_each_year
