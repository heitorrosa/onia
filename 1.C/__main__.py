import os
import sys
import time
from datetime import *

import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns

if __name__ == '__main__':
    dataset = pd.read_csv('1.C/stock_market_dataset.csv', index_col='Stock')
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset.drop(columns={'Target', 'Inflation_Rate', 'Interest_Rate', 'Sentiment_Score', 'Next_Close', 'GDP_Growth', 'RSI', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower'})

    dataset['Volume_MA20'] = dataset['Volume'].rolling(window=20, min_periods=20).mean()
    dataset['Volatility Spike'] = np.where(dataset['Volume'] > dataset['Volume_MA20'] * 2, 1, 0)

    dataset = dataset[dataset['Date'].dt.year <= 2026]

    stocksDict = {}
    for ticker, group in dataset.groupby('Stock'):
        stocksDict[ticker] = group.set_index('Date').sort_index()

    targetStock = 'TSLA'
    targetData = stocksDict[targetStock].tail(30)
    print(targetData)

    spikeDates = targetData[targetData['Volatility Spike'] == 1].index.tolist()
    smaLine = mpf.make_addplot(targetData['SMA_10'], color='blue', width=2.0)

    mpf.plot(
        targetData, 
        type='candle',
        addplot=smaLine,
        volume=True,
        vlines=dict(vlines=spikeDates, colors='red', alpha=0.2, linewidths=5),
        figratio=(16, 9), 
        figscale=1.5,
        style='yahoo',
        savefig=dict(fname=f'1C {targetStock}.png', dpi=300),
        title=f"{targetStock} - Volatility Analysis"
    )