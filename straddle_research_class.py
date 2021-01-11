#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:38:04 2020

@author: jirong
"""

#Research into weekend straddle and overnight straddle effect

import pandas as pd
import numpy as np
import time
import random
import os
import datetime as dt
import re
import quandl
import yfinance as yf
import matplotlib.pyplot as plt
import util as ut
from sklearn import datasets, model_selection

class StraddleResearch(object):
    
    def __init__(self, path, ticker, date_start, date_end, shift_days, 
                 buy_day, expiry_day, fix_capital, cap, 
                 otm_put_perdiff=None, include_hedge=1):

        """
        Constructor for FuturesResearch class
    
        :param path: path to data folder file (e.g. "./trend_following/quantopian_data/futures_incl_2016.csv")
        :param ticker: ticker of options chain
        :param date_start: Starting date of strategy
        :param date_end: Ending date of strategy
        :param shift_days: Shifting day from purchase date
        :param buy_day: Buy on which day (0: Mon, 6: Sun)
        :param expiry_day: Expire on which day
        :param fix_capital: Fix notional capital of underlying asset (e.g. 600000)
        :param cap: Parameter used to cap forecast strength (see Jupyter notebook)
        :param otm_perdiff: Out of the money put. % away from ATM strike.
        :param include_hedge: Include OTM put. (1: to include. 0: Not to include)
        :return: returns StraddleResearch class
        """   

        
        self.path = path
        self.ticker = ticker
        self.date_start = date_start
        self.date_end = date_end
        self.stock_price = None
        self.shift_days = shift_days
        self.buy_day = buy_day
        self.expiry_day = expiry_day
        self.profits = None
        self.fix_capital = fix_capital
        self.cap = cap
        self.otm_put_perdiff = otm_put_perdiff
        self.include_hedge = include_hedge    
        
        pass
    
    def get_stock_data(self):
        
        prices_df = yf.download(self.ticker, start = self.date_start, end = self.date_end, adjusted = True)
        prices_df = prices_df.ffill(axis = 'rows')
        prices_df['date'] = prices_df.index
        prices_df['date_liquidate'] = prices_df['date'].shift(-1)  
        prices_df['Open_liquidate'] = prices_df['Open'].shift(-1) 
        prices_df['Close_liquidate'] = prices_df['Close'].shift(-1)     
        prices_df['Open_change'] = (prices_df['Open_liquidate'] - prices_df['Close'])/prices_df['Close']       
        prices_df['Close_change'] = (prices_df['Close_liquidate'] - prices_df['Close'])/prices_df['Close']       
        prices_df['date_forward'] = prices_df['date'].shift(-self.shift_days)  #Shift days for expiry date
        prices_df['date_ymd'] = prices_df.index.strftime("%Y-%m-%d") 
        prices_df['date_forward_ymd'] = prices_df['date_ymd'].shift(-self.shift_days)
        prices_df['date_liquidate_ymd'] = prices_df['date_ymd'].shift(-1)                
        prices_df = prices_df[['date', 'date_liquidate','date_forward', 'date_ymd', 'date_liquidate_ymd','date_forward_ymd', 'Close', 'Adj Close','Open_liquidate','Open_change','Close_liquidate','Close_change']]
        prices_df['day_of_week'] = prices_df.date.dt.dayofweek
        prices_df['forward_day'] = prices_df['day_of_week'].shift(-self.shift_days)  
        prices_df['call_close'] = np.nan
        prices_df['put_close'] = np.nan  
        prices_df['call_open'] = np.nan
        prices_df['put_open'] = np.nan  
        prices_df['otm_put_close'] = np.nan 
        prices_df['otm_put_open'] = np.nan         
        prices_df['call_price_change'] = np.nan  
        prices_df['put_price_change'] = np.nan  
        prices_df['call_price_dollar_change'] = np.nan  
        prices_df['put_price_dollar_change'] = np.nan          
        prices_df['otm_put_price_change'] = np.nan        
        prices_df['avg_price_change'] = np.nan     
        prices_df['max_straddles'] = (self.fix_capital/(prices_df['Close']*100))
        prices_df.index = prices_df['date_ymd']
                
        self.stock_price = prices_df
        
        pass                           
    
    def subset_straddle_data(self, buy_day, expiry_day):
        
        """
        Subset stock data according to purchase date and liquidation date  
        :param buy_day: Buy on which day (0: Mon, 6: Sun)
        :param expiry_day: Expire on which day        
        """        
        
        #self.stock_price = self.stock_price[(self.stock_price.day_of_week == 4) & (self.stock_price.forward_day == weekday)]
        self.stock_price = self.stock_price[(self.stock_price.day_of_week == buy_day) & (self.stock_price.forward_day == expiry_day)]
        
        pass


    def read_options_chain_data(self, strike, date_quote, date_liquidate, date_expire):
        
        """
        Read options chain data from CBOE 
        :param strike: Strike price
        :param date_quote: Quotation/Purchase date
        :param date_liquidate: Liquidation date
        :param date_expire: Expiry date    
        :return: returns options df
        """            
        
        call_close = np.nan
        put_close = np.nan    
        otm_put_close = np.nan            
        call_open = np.nan
        put_open = np.nan
        otm_put_open = np.nan
        
        try:
        
            strike = int(strike)
            otm_strike = strike * (1-self.otm_put_perdiff)
            otm_strike = 5 * round(otm_strike/5)
            
            #print(otm_strike)
            
            data = pd.read_csv(self.path + date_quote + '.csv')
            call_close = data['close'][(data['expiration'] == date_expire) & (data['strike'] == strike) & (data['option_type'] == 'C')]        
            put_close = data['close'][(data['expiration'] == date_expire) & (data['strike'] == strike) & (data['option_type'] == 'P')]        
           
            call_close = float(call_close)
            put_close = float(put_close)
            
            data_open = pd.read_csv(self.path + date_liquidate + '.csv')
            call_open = data_open['close'][(data_open['expiration'] == date_expire) & (data_open['strike'] == strike) & (data_open['option_type'] == 'C')]        
            put_open = data_open['close'][(data_open['expiration'] == date_expire) & (data_open['strike'] == strike) & (data_open['option_type'] == 'P')]        
            
            call_open = float(call_open)
            put_open = float(put_open)
                                
        except:       
            
            call_close = np.nan
            put_close = np.nan
            otm_put_close = np.nan                
            call_open = np.nan
            put_open = np.nan
            otm_put_open = np.nan
            
        return call_close, put_close, call_open, put_open, otm_put_close, otm_put_open
     
    def read_options_chain_otm_data(self, strike, date_quote, date_liquidate, date_expire):
        
        """
        Read options chain data from CBOE 
        :param strike: Strike price
        :param date_quote: Quotation/Purchase date
        :param date_liquidate: Liquidation date
        :param date_expire: Expiry date      
        :return: returns options info        
        """         
           
        otm_put_close = -1            
        otm_put_open = -1
        
        try:
        
            strike = int(strike)
            otm_strike = strike * (1-self.otm_put_perdiff)
            otm_strike = 5 * round(otm_strike/5)
            
            #print(otm_strike)
            
            data = pd.read_csv(self.path + date_quote + '.csv')
            
            try:
                otm_put_close = data['close'][(data['expiration'] == date_expire) & (data['strike'] == otm_strike) & (data['option_type'] == 'P')]                    
                otm_put_close = float(otm_put_close)             
                #print('otm_put_close ' + str(otm_put_close))
            except:
                pass

            data_open = pd.read_csv(self.path + date_liquidate + '.csv')

            try:
                otm_put_open = data_open['close'][(data_open['expiration'] == date_expire) & (data_open['strike'] == otm_strike) & (data_open['option_type'] == 'P')]        
                otm_put_open = float(otm_put_open)    
                #print('otm_put_open ' + str(otm_put_open))
            except:
                pass
                                 
        except:       
            
            otm_put_close = -1              
            otm_put_open = -1
            
        return otm_put_close, otm_put_open    
    
    def populate_strategy_data_frame(self):        
        
        """
        Populate strategy data-frame with options chain data      
        """         
        
        for r in range(self.stock_price.shape[0]):
            
            #print(r)
            
            options_data = self.read_options_chain_data(float(self.stock_price.at[self.stock_price.index[r],'Close']), \
                                                        self.stock_price.index[r], \
                                                        self.stock_price.at[self.stock_price.index[r],'date_liquidate_ymd'],
                                                        self.stock_price.at[self.stock_price.index[r],'date_forward_ymd'])
            self.stock_price.at[self.stock_price.index[r],'call_close'] = options_data[0]
            self.stock_price.at[self.stock_price.index[r],'put_close'] = options_data[1]            
            self.stock_price.at[self.stock_price.index[r],'call_open'] = options_data[2]
            self.stock_price.at[self.stock_price.index[r],'put_open'] = options_data[3]  
            #self.stock_price.at[self.stock_price.index[r],'otm_put_close'] = options_data[4]              
            #self.stock_price.at[self.stock_price.index[r],'otm_put_open'] = options_data[5]   

            options_data = self.read_options_chain_otm_data(float(self.stock_price.at[self.stock_price.index[r],'Close']), \
                                                        self.stock_price.index[r], \
                                                        self.stock_price.at[self.stock_price.index[r],'date_liquidate_ymd'],
                                                        self.stock_price.at[self.stock_price.index[r],'date_forward_ymd'])

            if ((type(options_data[0]) == float) & (type(options_data[1]) == float)):

                self.stock_price.at[self.stock_price.index[r],'otm_put_close'] = float(options_data[0])              
                self.stock_price.at[self.stock_price.index[r],'otm_put_open'] = float(options_data[1]) 
            
            pass            
    
    def compute_price_change(self):
        
        """
        Options price change    
        """            
        
        self.stock_price['call_price_change'] = (self.stock_price['call_open'] - self.stock_price['call_close'])/self.stock_price['call_close']      
        self.stock_price['put_price_change'] = (self.stock_price['put_open'] - self.stock_price['put_close'])/self.stock_price['put_close']      
        self.stock_price['avg_price_change'] = (self.stock_price['call_price_change'] + self.stock_price['put_price_change'])/2

        #self.stock_price['num_straddles'] = round(((self.fix_capital/2)/self.stock_price['call_close'] + \
        #                                    (self.fix_capital/2)/self.stock_price['put_close'])/2 /100)      
            
        self.stock_price['call_price_dollar_change'] = (self.stock_price['call_open'] - self.stock_price['call_close'])  
        self.stock_price['put_price_dollar_change'] = (self.stock_price['put_open'] - self.stock_price['put_close'])
        
        pass    
    
    def obtain_vix_filter(self):
        
        """
        Include risk filter
        """         
        
        vix = pd.read_csv('research/^VIX.csv')
        vix = vix[['Date','Close']]
        vix.columns = ['Date', 'vix']
               
        dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
        vix3m = pd.read_csv('research/VIX3M_History.csv', parse_dates = ['DATE'], date_parser = dateparse)
        vix3m.columns = ['Date','vix3m']
                
        self.stock_price = pd.merge(self.stock_price, vix3m, left_on='date', right_on='Date')
        self.stock_price = pd.merge(self.stock_price, vix, left_on='date_ymd', right_on='Date')        
        self.stock_price['buy_indicator'] = self.stock_price['vix'] < self.stock_price['vix3m']
        self.stock_price['signal_strength'] = 1 - (self.stock_price['vix'] / self.stock_price['vix3m']) #0-25%. Scale to 100%
        self.stock_price['signal_strength_adj'] = self.stock_price['signal_strength']/self.cap 
        self.stock_price['signal_strength_adj'] = np.where(self.stock_price['signal_strength_adj']>1,1,self.stock_price['signal_strength_adj']) 
        self.stock_price['signal_strength_adj'] = np.where(self.stock_price['signal_strength_adj']<0,0,self.stock_price['signal_strength_adj'])         
        
        self.stock_price['num_straddles'] = round(self.stock_price['signal_strength_adj'] *  self.stock_price['max_straddles'])           
        
        self.stock_price['otm_put_price_change'] = (self.stock_price['otm_put_open'] - self.stock_price['otm_put_close'])/self.stock_price['otm_put_close']
        self.stock_price['otm_put_price_change'] = np.where(np.isnan(self.stock_price['otm_put_price_change']), 0, self.stock_price['otm_put_price_change'])
        self.stock_price['otm_put_price_change'] = np.where(np.isinf(self.stock_price['otm_put_price_change']), 0, self.stock_price['otm_put_price_change'])
        
        self.stock_price['hedge_loss'] = \
        self.stock_price['num_straddles'] * \
        self.stock_price['otm_put_close'] * 100 *\
        self.stock_price['otm_put_price_change']

        self.stock_price['hedge_loss'] = np.where(np.isnan(self.stock_price['hedge_loss']), 0, self.stock_price['hedge_loss'])
        
        self.stock_price['notional_capital'] = self.stock_price['num_straddles'] * self.stock_price['Close'] * 100
        
        self.stock_price['prop_notional_capital'] = ((self.stock_price['call_close'] + self.stock_price['put_close']) * 100 * self.stock_price['max_straddles'])/ self.fix_capital
        self.stock_price['prop_notional_capital_cont_sig'] = self.stock_price['prop_notional_capital'] * (self.stock_price['num_straddles']/self.stock_price['max_straddles'])
        
        pass
            
        
    def profits_generation(self):
        
        """
        Generating profits
        """         
        
        #Open - Close
        self.stock_price['profits'] = (self.stock_price['call_price_dollar_change'] + self.stock_price['put_price_dollar_change']) * (-100) * self.stock_price['max_straddles']
        #self.stock_price['profits'] = self.stock_price.avg_price_change * self.stock_price['max_straddles'] * (-100) * (self.stock_price['call_close'] + self.stock_price['put_close'])        
        self.stock_price['cum_profits'] = self.stock_price['profits'].cumsum()
        self.profits = self.stock_price[~pd.isnull(self.stock_price['cum_profits'])]
        self.profits['profits_after_filter'] = np.where(self.profits['buy_indicator'] == True, self.profits['profits'], 0)
        self.profits['profits_after_filter_cum_sum'] = self.profits['profits_after_filter'].cumsum()
        #self.profits['returns_after_filter_cont_signal'] = self.profits.avg_price_change * self.profits['signal_strength_adj']    
        #self.profits['profits_after_filter_cont_signal'] = self.profits.avg_price_change * self.profits['signal_strength_adj'] * round(self.stock_price['max_straddles'] * (-100)) * (self.stock_price['call_close'] + self.stock_price['put_close'])  
        self.profits['profits_after_filter_cont_signal'] = self.profits['signal_strength_adj'] * (self.profits['call_price_dollar_change'] + self.profits['put_price_dollar_change']) * (-100) * self.profits['max_straddles']
        self.profits['returns_after_filter_cont_signal'] = self.profits['profits_after_filter_cont_signal']/(self.fix_capital/10)          
        
        if self.include_hedge == 1:
            self.profits['profits_after_filter_cont_signal'] = self.profits['profits_after_filter_cont_signal'] + self.profits['hedge_loss']
            
        self.profits['profits_after_filter_cont_signal_cumsum'] = self.profits['profits_after_filter_cont_signal'].cumsum()
        self.profits['id'] = np.arange(self.profits.shape[0])+1
        self.profits.index = self.profits['id']
           
        pass
    
    def trade_statistic_simulation(self, profits_df, forecast_cap):
        
        """
        Profits dataframe for each bootstrap sample.
        
        :param profits_df: Profits data-frame
        :param forecast_cap: forecast_cap parameter for forecast strength (refer to Jupyter notebook)
        :return: returns trade statistic        
        """         

        profits_df['signal_strength_adj'] = profits_df['signal_strength']/forecast_cap 
        profits_df['signal_strength_adj'] = np.where(profits_df['signal_strength_adj']>1,1,profits_df['signal_strength_adj'])     
        profits_df['signal_strength_adj'] = np.where(profits_df['signal_strength_adj']<0,0,profits_df['signal_strength_adj'])                 
        #profits_df['returns_after_filter_cont_signal'] = profits_df.avg_price_change * profits_df['signal_strength_adj'] 
        #profits_df['profits_after_filter_cont_signal'] = profits_df.avg_price_change * profits_df['signal_strength_adj'] * round(profits_df['max_straddles'] * (-100)) * (profits_df['call_close'] + profits_df['put_close'])  
        profits_df['profits_after_filter_cont_signal'] = profits_df['signal_strength_adj'] * (profits_df['call_price_dollar_change'] + profits_df['put_price_dollar_change']) * (-100) * profits_df['max_straddles']        
        profits_df['returns_after_filter_cont_signal'] = profits_df['profits_after_filter_cont_signal']/(self.fix_capital/10)        
        
        if self.include_hedge == 1:
            profits_df['profits_after_filter_cont_signal'] = profits_df['profits_after_filter_cont_signal'] + profits_df['hedge_loss']        
        
        profits_df['profits_after_filter_cont_signal_cumsum'] = profits_df['profits_after_filter_cont_signal'].cumsum()        
        trade_statistic = (profits_df['returns_after_filter_cont_signal']).describe().to_frame().transpose()
        #trade_statistic = (profits_df['profits_after_filter_cont_signal']).describe().to_frame().transpose()        
        trade_statistic['mean_over_sd'] = trade_statistic['mean']/trade_statistic['std'] 
        trade_statistic['mean_over_min'] = trade_statistic['mean']/trade_statistic['min']
        trade_statistic['mean_over_med'] = trade_statistic['mean']/trade_statistic['50%']         
        trade_statistic['kurtosis'] = ut.get_kurtosis(profits_df['returns_after_filter_cont_signal'])
        trade_statistic['skewness'] = ut.get_skewness(profits_df['returns_after_filter_cont_signal'])
        trade_statistic['final_profits'] = float(profits_df['profits_after_filter_cont_signal_cumsum'].iloc[-1:])
        trade_statistic['profits_per_trade'] = trade_statistic['final_profits']/trade_statistic['count'] 
        trade_statistic['forecast_param'] = forecast_cap
        
        return trade_statistic
    
    def block_boostrap_simulations(self, num_simulations = 10, test_size = 2/3, param_space=[0.05,0.1,0.15,0.2,0.25]):
           
        """
        Block bootstrapping simulation (not used in study)
        """     
             
        #data_sel, _ = model_selection.train_test_split(self.profits, test_size = test_size)          
        data_list = [model_selection.train_test_split(self.profits, test_size = test_size) for i in range(num_simulations)]                
        boostrap_stats = [self.trade_statistic_simulation(data_list[i][0], j) for i in range(num_simulations) for j in param_space]
        boostrap_stats = pd.concat(boostrap_stats, axis=0) 
        
        return boostrap_stats
    
    def boostrap_simulations(self,num_samples=100,num_simulations = 10, param_space=[0.05,0.1,0.15,0.2,0.25]):

        """
        Bootstrapping simulation (used in study)
        :param num_sample: Number of datapoints per bootstrap sample
        :param num_simulations: Number of simulations carried out per parameter space
        :param param_space: Parameter space for forecast_cap
        :return: returns bootstrap statistics dataframe        
        """             

        def rand_int():   
            #random.seed(seed_num)        
            return random.randrange(1, self.profits.shape[0])

        def resample_df_1():                
            index_list = tuple(rand_int() for n in range(num_samples))      
            #print(index_list)
            df = self.profits.set_index('id').reindex(index_list).reset_index().reindex(self.profits.columns, axis=1)  
            return df
        
        data_list = [resample_df_1() for i in range(num_simulations)]                  
        boostrap_stats = [self.trade_statistic_simulation(data_list[i], j) for j in param_space for i in range(num_simulations)]                                
        boostrap_stats = pd.concat(boostrap_stats, axis=0)                 
                                       
        return boostrap_stats    
    
    def execute_flow(self):

        """
        Execute data analysis flow
        """           
        
        self.get_stock_data()    
        self.subset_straddle_data(self.buy_day, self.expiry_day)  #2    
        self.populate_strategy_data_frame()
        self.compute_price_change()   
        self.obtain_vix_filter()
        self.profits_generation()
    
if __name__ == "__main__":       

    #Weekend trade 
    strategy = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                ticker='SPY',
                                date_start = '2005-01-01', 
                                date_end = '2020-12-04', 
                                shift_days = 1,  #1: Mon, 3: Wed, 5: Fri  #Shift day from purchase date
                                buy_day = 4,
                                expiry_day = 0,  #0: Mon, 2: Wed, 4: Fri
                                fix_capital = 600000, #4000
                                cap = 0.25,
                                otm_put_perdiff=0.1,
                                include_hedge = 0
                                )    
    
#        profits_after_filter_cont_signal
# count                        110.000000
# mean                         102.055243
# std                         1273.890769
# min                        -5259.952605
# 25%                         -271.994853
# 50%                          141.297676
# 75%                          754.474869
# max                         3125.337886    
       
#          profits_after_filter_cont_signal                  
#                                      mean      median count
# positive                                                   
# -1                           -1141.512731 -625.309936    36
#  0                               0.000000   -0.000000    15
#  1                             886.788729  725.091101    59    
    
    #Weekend trade 
    strategy = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                ticker='SPY',
                                date_start = '2005-01-01', 
                                date_end = '2020-12-04', 
                                shift_days = 3,  #1: Mon, 3: Wed, 5: Fri  #Shift day from purchase date
                                buy_day = 4,
                                expiry_day = 2,  #0: Mon, 2: Wed, 4: Fri
                                fix_capital = 600000, #4000
                                cap = 0.25,
                                otm_put_perdiff=0.1,
                                include_hedge = 0
                                )    
    
#        profits_after_filter_cont_signal
# count                        155.000000
# mean                         225.308643
# std                          740.055399
# min                        -2728.268839
# 25%                          -43.414088
# 50%                          362.333930
# 75%                          648.277168
# max                         2033.238744    
    
    #Wed expiry  
    strategy = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                ticker='SPY',
                                date_start = '2005-01-01', 
                                date_end = '2020-12-04', 
                                shift_days = 1,  #1 
                                buy_day = 1,     #1
                                expiry_day = 2,  #2
                                fix_capital = 600000, #4000
                                cap = 0.25,
                                otm_put_perdiff=0.1,
                                include_hedge = 0                                
                                )  
    
#          profits_after_filter_cont_signal                  
#                                      mean      median count
# positive                                                   
# -1                           -1043.362822 -570.034193    53
#  0                               0.000000    0.000000    24
#  1                             860.388769  826.857581   108

#        profits_after_filter_cont_signal
# count                        185.000000
# mean                         203.371662
# std                         1173.158340
# min                        -5357.539921
# 25%                          -99.307237
# 50%                          213.287817
# 75%                          939.193838
# max                         3164.944709    
    
    #Fri trade -->doesn't work
    strategy = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                ticker='SPY',
                                date_start = '2005-01-01', 
                                date_end = '2020-12-04', 
                                shift_days = 1,  #1: Mon, 3: Wed, 5: Fri 
                                buy_day = 3,
                                expiry_day = 4,  #0: Mon, 2: Wed, 4: Fri
                                fix_capital = 600000, #4000
                                cap = 0.25,
                                otm_put_perdiff=0.1,
                                include_hedge = 0                                
                                )  

    strategy = StraddleResearch(path = './options_chain_data/qqq_options_unzipped/UnderlyingOptionsEODQuotes_',
                                ticker='QQQ',
                                date_start = '2005-01-01', 
                                date_end = '2020-12-04', 
                                shift_days = 1,  #1: Mon, 3: Wed, 5: Fri 
                                buy_day = 3,
                                expiry_day = 4,  #0: Mon, 2: Wed, 4: Fri
                                fix_capital = 600000, #4000
                                cap = 0.25,
                                otm_put_perdiff=0.1,
                                include_hedge = 0                                
                                )        
    
#          profits_after_filter_cont_signal                  
#                                      mean      median count
# positive                                                   
# -1                           -1540.935555 -745.788818   111
#  0                               0.000000   -0.000000    32
#  1                            1073.799679  901.123759   189    
    
#        profits_after_filter_cont_signal
# count                        332.000000
# mean                          96.097267
# std                         2010.041935
# min                       -19120.841006
# 25%                         -276.014127
# 50%                          306.323837
# 75%                          989.772729
# max                         4082.354643    
    
    
    
    strategy.execute_flow()
    profits = strategy.profits           
    #profits = strategy.profits[5:]
    #stats = strategy.block_boostrap_simulations(10,2/3)
    stats = strategy.boostrap_simulations(num_samples=160,num_simulations = 1000, param_space=[0.05,0.1,0.15,0.2,0.25])    
    
    avg_stats = stats.groupby(['forecast_param']).mean()
    
    profits['positive'] = np.where(profits['profits_after_filter_cont_signal'] > 0, 1, 0)
    profits['positive'] = np.where(profits['profits_after_filter_cont_signal'] < 0, -1, profits['positive'])
    profits[['profits_after_filter_cont_signal']].groupby(profits['positive']).agg(['mean','median','count'])
    profits[['profits_after_filter_cont_signal']].describe()
    
    profits.index = profits['date']
    year_profits = profits.groupby(pd.Grouper(freq="Y")).sum()['profits_after_filter_cont_signal']   
    
    plt.plot(profits['date'], profits[['profits_after_filter_cum_sum']])
    plt.plot(profits['date'], profits[['cum_profits']])
    plt.plot(profits['date'], profits[['profits_after_filter_cont_signal_cumsum']])
    plt.xticks(rotation = 90)    
    plt.show()
    
    #Plot average price change with option price change
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    #ax.scatter(abs(profits.Open_change), profits.avg_price_change, color='r')
    ax.scatter(profits.Open_change, profits.profits_after_filter_cont_signal, color='r')    
    ax.set_ylabel('Profits')
    ax.set_xlabel('SPY abs price change')
    ax.set_title('scatter plot')
    plt.show()    
    
    #Feature engineering. Price change, vix, vix change
    
        
###########################Combining profits dataframe across 2 days and sort#########################
    strategy_fri = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                    ticker='SPY',
                                    date_start = '2005-01-01', 
                                    date_end = '2020-12-04', 
                                    shift_days = 3,  #1: Mon, 3: Wed, 5: Fri  #Shift day from purchase date
                                    buy_day = 4,
                                    expiry_day = 2,  #0: Mon, 2: Wed, 4: Fri
                                    fix_capital = 600000, #4000
                                    cap = 0.25,
                                    otm_put_perdiff=0.1,
                                    include_hedge = 0                                    
                                    )        

    strategy_fri.execute_flow()
    profits_fri = strategy_fri.profits       
    
    strategy_wed = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                    ticker='SPY',
                                    date_start = '2005-01-01', 
                                    date_end = '2020-12-04', 
                                    shift_days = 1,  #1 
                                    buy_day = 1,     #1
                                    expiry_day = 2,  #2
                                    fix_capital = 600000, #4000
                                    cap = 0.25,
                                    otm_put_perdiff=0.1,
                                    include_hedge = 0                                        
                                    )  
    
    strategy_wed.cap = 0.25
    strategy_wed.execute_flow()
    profits_wed = strategy_wed.profits     
    
    strategy_thu = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                    ticker='SPY',
                                    date_start = '2005-01-01', 
                                    date_end = '2020-12-04', 
                                    shift_days = 1,  #1: Mon, 3: Wed, 5: Fri 
                                    buy_day = 3,
                                    expiry_day = 4,  #0: Mon, 2: Wed, 4: Fri
                                    fix_capital = 600000, #4000
                                    cap = 0.25,
                                    otm_put_perdiff=0.1,
                                    include_hedge = 0                                        
                                    )      
    strategy_thu.execute_flow()
    profits_thu = strategy_thu.profits       
       
    #profits = pd.concat([profits_fri, profits_wed, profits_thu], axis=0)
    profits = pd.concat([profits_fri, profits_wed], axis=0)    
    #profits = pd.concat([profits_wed], axis=0)  
    #profits = pd.concat([profits_fri], axis=0)  
    profits = profits.sort_values(by='date')    
    profits['id'] = np.arange(profits.shape[0])+1
    
    profits['positive'] = np.where(profits['profits_after_filter_cont_signal'] > 0, 1, 0)
    profits['positive'] = np.where(profits['profits_after_filter_cont_signal'] < 0, -1, profits['positive'])
    profits[['profits_after_filter_cont_signal']].groupby(profits['positive']).agg(['mean','median','count'])
    profits[['profits_after_filter_cont_signal']].describe()    
    profits['count'] = 1

    profits['profits_after_filter_cont_signal_cumsum'] = profits['profits_after_filter_cont_signal'].cumsum()     
    
    plt.plot(profits['date'], profits[['profits_after_filter_cont_signal_cumsum']])
    plt.xticks(rotation = 90)    
    plt.show()    
    
    plt.plot(profits['id'], profits[['profits_after_filter_cont_signal_cumsum']])   
    
    profits.index = profits['date']
    month_profits = profits.groupby(pd.Grouper(freq="M")).sum()['profits_after_filter_cont_signal']      
    #month_profits['2016']
    
    a = month_profits/60000
    
    year_profits = profits.groupby(pd.Grouper(freq="Y")).sum()['profits_after_filter_cont_signal']       
    
#        profits_after_filter_cont_signal
# count                        345.000000
# mean                         216.382656
# std                          994.255911
# min                        -5357.539921
# 25%                          -63.231190
# 50%                          312.032360
# 75%                          772.372332
# max                         3164.944709    
    
#          profits_after_filter_cont_signal                  
#                                      mean      median count
# positive                                                   
# -1                            -880.268237 -532.927322    95
#  0                               0.000000    0.000000    39
#  1                             750.130327  670.199049   211

    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    
    vix = pd.read_csv('research/^VIX.csv', parse_dates = ['Date'], date_parser = dateparse)
    vix = vix[['Date','Close']]
    vix.columns = ['Date', 'vix']

    dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')           
    vix3m = pd.read_csv('research/VIX3M_History.csv', parse_dates = ['DATE'], date_parser = dateparse)
    vix3m.columns = ['Date','vix3m']

    vix_comb = pd.merge(vix3m, vix, left_on='Date', right_on='Date')
    
    prices_df = yf.download('SPY', start = '2005-01-01', end = '2020-12-08', adjusted = True)
    prices_df = prices_df.ffill(axis = 'rows')
    prices_df['date'] = prices_df.index
    prices_df['date_liquidate'] = prices_df['date'].shift(-1)  
    prices_df['Open_liquidate'] = prices_df['Open'].shift(-1) 
    prices_df['Close_liquidate'] = prices_df['Close'].shift(-1)     
    prices_df['Open_change'] = (prices_df['Open_liquidate'] - prices_df['Close'])/prices_df['Close']       
    prices_df['Close_change'] = (prices_df['Close_liquidate'] - prices_df['Close'])/prices_df['Close']       
    prices_df['date_forward'] = prices_df['date'].shift(-1)  #Shift days for expiry date
    prices_df['date_ymd'] = prices_df.index.strftime("%Y-%m-%d") 
    prices_df['date_forward_ymd'] = prices_df['date_ymd'].shift(-1)
    prices_df['date_liquidate_ymd'] = prices_df['date_ymd'].shift(-1)                
    prices_df = prices_df[['date', 'date_liquidate','date_forward', 'date_ymd', 'date_liquidate_ymd','date_forward_ymd', 'Close', 'Adj Close','Open_liquidate','Open_change','Close_liquidate','Close_change']]
    
    
    vix_equity = pd.merge(vix_comb, prices_df, left_on='Date', right_on='date')
    vix_equity['buy_indicator'] = vix_equity['vix'] < vix_equity['vix3m']
    vix_equity['signal_strength'] = 1 - (vix_equity['vix'] / vix_equity['vix3m']) #0-25%. Scale to 100%
    vix_equity['signal_strength_adj'] = vix_equity['signal_strength']/0.25 
    vix_equity['signal_strength_adj'] = np.where(vix_equity['signal_strength_adj']>1,1,vix_equity['signal_strength_adj']) 
    vix_equity['signal_strength_adj'] = np.where(vix_equity['signal_strength_adj']<0,0,vix_equity['signal_strength_adj'])     

    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    #ax.scatter(abs(profits.Open_change), profits.avg_price_change, color='r')
    ax.scatter(abs(vix_equity.Open_change), vix_equity.signal_strength_adj, color='r')    
    ax.set_ylabel('Forecast')
    ax.set_xlabel('SPY absolute price change')
    ax.set_title('scatter plot')
    plt.show()       
    
    distribution = (vix_equity['signal_strength_adj'] * abs(vix_equity.Open_change) * 600000).describe()
    
    (vix_equity['signal_strength_adj'] * abs(vix_equity.Open_change) * 600000).plot.hist(bins = 500, ylim=(0,50))
    
