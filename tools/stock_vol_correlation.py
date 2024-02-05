#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:35:42 2023

@author: ted
"""
import numpy as np

def calculate_yearly_correlation(SPX_daily,VIX_daily):
    """
    
    Calculating the correlation between the SPX and the VIX

    Parameters
    ----------
    SPX_daily : Pandas.DataFrame
        DataFrame containing the daily returns.
    VIX_daily : Pandas.DataFrame
        DataFrame containing the daily volatility.

    Returns
    -------
    correlation_each_year : TYPE
        DESCRIPTION.

    """
  
    years = np.arange(np.amin(SPX_daily.index.year),np.amax(SPX_daily.index.year)+1)
    pears_corr = np.empty([np.size(years)])
    
    idx=0
    for year in years:
        VIX_year = VIX_daily.loc[VIX_daily.index.year == year,['Close']]
        SPX_year = SPX_daily.loc[SPX_daily.index.year == year,['Close']]
        pears_corr[idx] = SPX_year.corrwith(VIX_year)
    
        idx+=1
    
    
    correlation_each_year = np.mean(pears_corr)
    return correlation_each_year
 

