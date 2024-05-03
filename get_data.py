# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:24:54 2024

@author: franc
"""

# Import needed libraries.
import yfinance as yf
from datetime import datetime, timedelta

# Create variables to store the start and end dates.
today = datetime.today()
start = today - timedelta(days=365*3)
end = today - timedelta(days=1)

# Download AAPL daily data from Yahoo Finance.
AAPL_data = yf.download('AAPL', start, end, interval='1d')

# Keep all columns except 'Close'.
AAPL_data = AAPL_data.loc[:, AAPL_data.columns != 'Close']

# Export the dataframe as a '.csv' file.
AAPL_data.to_csv('AAPL_daily_data.csv', sep=',', index=True)