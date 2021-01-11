# straddle
Straddle strategy over non trading periods

## Disclaimer
None of the contents constitute an offer to sell, a solicitation to buy, or a recommendation or endorsement for any security or strategy, nor does it constitute an offer to provide investment advisory services Past performance is no indicator of future performance Provided for informational purposes only All investments involve risk, including loss of principal

Note:

I have a vested interest in this strategy. Variant of the strategy has been deployed since start of Dec-2020
The fund is managed systematically through an algorithmic trading infrastructure developed by the fund manager that places order through Interactive Broker’s API.
Margins (from excess liquidity) from my account is used for this strategy.

## Summary
This strategy could be a complementary strategy in a core portfolio to utilize margins more effectively.

Some key performance statistics include,

- 0.27 per cent average gains per trade
- 0.42 per cent median gains per trade
- 1.24 per cent standard deviation per trade
- 3.48 per cent maximum loss per trade
- 67 per cent win rate
- Worst drawdown of this strategy is < 8 per cent.

That being said, this strategy with inherent risk/ huge negative skew. If you were considering this strategy, I would recommend a non-compounding fix capital approach.

## Introduction
Based on literature,

i. Short options returns are higher during non-trading period, the vast majority of which are weekends (Jones, Christopher S., and Joshua Shemesh, 2018)

ii. Implied volatility is usually higher than the actual historical volatility (Coval, Joshua D., and Tyler Shumway, 2001)

iii. Positional Option Trading (Sinclair, Euan. 2009)

iv. Understand the volatility risk premium (Ang, I., Roni Israelov, R. N. Sullivan, and Harsha Tummala., 2018)

This, therefore suggests the possibility to earn a systematic risk premium by selling at-the-money options.

Despite positive edge (expected value) exhibited by the strategy based on historical backtests, there remains an unbounded theoretical downside risk in the strategy. To mitigate such risk, I employed a risk filter and position sizing based on term structure in the volatility market.

## Approach
Place a trade at end of Friday trading session with SPY options expiring on Wednesday.

Liquidate on start of Monday trading session.

I backtested the following approach

Continuous risk filter: Do not trade when VIX term structure is in backwardation i.e. VIX < VIX3M. Position sizing is relative to gap between VIX and VIX3M.

To smooth out returns stream, I convert VIX/VIX3M into a continuous signal,

self.stock_price['signal_strength'] = 1 - (self.stock_price['vix'] / self.stock_price['vix3m']) 
A normalizing denominator, j (0.1, 0.15, 0.2, 0.25 parameters are tested for optimization) is further applied to convert continuous signal to -100% to 100% signal self.stock_price['signal_strengthadj' + str(i)] = self.stock_price['signal_strength']/j

The ensuing signal is the proportion of available capital (or fix capital) used in position sizing.

## Simulations

Boostrapping is carried out to simulate the profit possibility based on historical distribution.
Given that this is a fat-tail high kurtosis strategy with strategy, I also included a section to understand strategy returns distribution based on SPY, VIX and VIX3M data from 2009 onwards (options data utilized in this strategy is only available from Sep-2016 onwards).

## Data
For this study, I purchased the options chain data from CBOE. Based on the terms and conditions imposed by CBOE, I'm unable to share the data here.

## How to use this repository
I developed a StraddleResearch class for this strategy.

You may initialize the strategy class as follow,

strategy_fri = StraddleResearch(path = './options_chain_data/options_unzipped/UnderlyingOptionsEODQuotes_',
                                ticker='SPY',
                                date_start = '2005-01-01', 
                                date_end = '2020-12-04', 
                                shift_days = 3,  #1: Mon, 3: Wed, 5: Fri  #Shift day from purchase date
                                buy_day = 4,
                                expiry_day = 2,  #0: Mon, 2: Wed, 4: Fri
                                fix_capital = 600000,
                                cap = 0.25,
                                otm_put_perdiff=0.1,
                                include_hedge = 0                                    
                                )        

strategy_fri.cap = 0.25
strategy_fri.execute_flow()
profits_fri = strategy_fri.profits

"""
Constructor for StraddleResearch class

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

Pls look at the jupyter notebook (straddles_research.ipynb) to understand how to use the class.
Pls look at the documentation on the StraddleResearch class (straddle_research_class_documentation.pdf).
Note that this is not an engineering project but more of a research project. Error handling and engineering features are not included. Should any of the repository for deployment purpose, these should be included.

