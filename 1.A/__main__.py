import os
import sys
import time
from datetime import *

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def loadDataset():
    dataset = pd.read_csv('1.A/all_stocks_5yr.csv', index_col="Name")

    dataset.drop(columns=['open', 'high', 'low', 'volume'], inplace=True)
    dataset['date'] = pd.to_datetime(dataset['date'])

    dataset.dropna(subset=['date', 'close'], inplace=True)

    return dataset

def evaluateMA(TICKER):
    df = stocks.loc[f'{TICKER}'].copy()

    df['MA 50'] = df['close'].rolling(window=50, min_periods=50).mean()
    df['MA 200'] = df['close'].rolling(window=200, min_periods=200).mean()

    df.to_csv(f'{TICKER}.csv')

    return df

if __name__ == '__main__':
    stocks = loadDataset()

    stocksList = ['INTC']
    for stock in stocksList:
        try:
            maData = evaluateMA(stock)
        except Exception as e:
            print(f'error: {e}')

        plt.figure(figsize=(12, 6))
        plt.plot(maData['date'], maData['MA 50'], label='MA 50')
        plt.plot(maData['date'], maData['MA 200'], label='MA 200')
        plt.plot(maData['date'], maData['close'], label='Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.title(f'Moving Averages ({maData['date'].min().strftime('%Y-%m-%d')} - {maData['date'].max().strftime('%Y-%m-%d')}) {maData.iloc[0].name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{maData.iloc[0].name}.png')