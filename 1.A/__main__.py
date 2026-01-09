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

    df['MA 50'] = df['close'].rolling(window=50, min_periods=1).mean()
    df['MA 200'] = df['close'].rolling(window=200, min_periods=1).mean()

    return df

if __name__ == '__main__':
    stocks = loadDataset()

    maNVDA = evaluateMA("NVDA")