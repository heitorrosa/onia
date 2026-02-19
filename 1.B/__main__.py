import os
import sys
import time
from datetime import *

import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def datasetCleanup(stocksList):
    for i in range(len(stocksList)):
        fileName = stocksList[i].lower() + ".us.txt"

        baseDir = os.path.dirname(os.path.abspath(__file__))
        datasetFolder = os.path.join(baseDir, "dataset")

        if fileName in os.listdir(datasetFolder):
            source = os.path.join(datasetFolder, fileName)
            dest = os.path.join(datasetFolder, stocksList[i].upper() + '.csv')
            os.rename(source, dest)

    for filename in os.listdir(datasetFolder):
        if filename.endswith(".txt"):
            os.remove(os.path.join(datasetFolder, filename))

if __name__ == '__main__':
    stocksList = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'AVGO', 'CSCO', 'ORCL', 'ADBE', 'CRM', # Technology
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'BLK', 'SCHW', 'PYPL',             # Finance
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'HES', 'OXY',           # Energy
        'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'AMGN', 'DHR',         # Healthcare
        'AMZN', 'WMT', 'KO', 'PEP', 'PG', 'COST', 'MCD', 'NKE', 'PM', 'HD'              # Consumer Goods
    ]

    datasetCleanup(stocksList)

    baseDir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(baseDir, "dataset")
    
    tempCSV = glob.glob(os.path.join(path, "*.csv"))     

    if not tempCSV:
        print(f"No CSV files found in: {path}")
        sys.exit(1)

    dfList = []
    for f in tempCSV:
        df = pd.read_csv(f)
        ticker = os.path.basename(f).split('.')[0].upper()
        df['Ticker'] = ticker
        df['Return'] = df['Close'].pct_change(1)

        dfList.append(df)

    allStocks = pd.concat(dfList, ignore_index=True)
    allStocks = allStocks.drop(columns={'Open', 'High', 'Low', 'Volume', 'OpenInt'})

    allStocks['Date'] = pd.to_datetime(allStocks['Date'])
    allStocks = allStocks.dropna(subset=['Date', 'Close'])

    returnsPivot = allStocks.pivot(index='Date', columns='Ticker', values='Return')
    returnsPivot = returnsPivot.dropna()

    corrMatrix = returnsPivot.corr()

    print(corrMatrix)

    diffPairs = corrMatrix.unstack()
    diffPairs = diffPairs[diffPairs.index.get_level_values(0) != diffPairs.index.get_level_values(1)]

    diffPairs_minCorr = diffPairs.min()
    diffPairs_minPair = diffPairs.idxmin()

    diffPairs_maxCorr = diffPairs.max()
    diffPairs_maxPair = diffPairs.idxmax()

    print(f"min: {diffPairs_minCorr:.4f} for {diffPairs_minPair[0]} and {diffPairs_minPair[1]}")
    print(f"max: {diffPairs_maxCorr:.4f} for {diffPairs_maxPair[0]} and {diffPairs_maxPair[1]}")

    plt.figure(figsize=(24, 18), dpi=100)
    sns.heatmap(
        corrMatrix, 
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.2,
        annot_kws={"size": 6}
    )
    plt.title("S&P 500 Sector Correlation Matrix", fontsize=20)
    plt.tight_layout()
    plt.savefig('1B.png')