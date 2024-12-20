import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_stock_data(df, date_column='date', stock_value_column='stock_value', title='Stock Value Over Time'):
    """
    Plots stock value over time from a CSV file.

    Args:
    - file_path (str): Path to the CSV file.
    - date_column (str): The name of the date column in the CSV file (default is 'date').
    - stock_value_column (str): The name of the stock value column in the CSV file (default is 'stock_value').
    - title (str): Title for the plot (default is 'Stock Value Over Time').
    """

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce', format='%Y-%m-%d %H:%M:%S')

    df.dropna(subset=[date_column], inplace=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df[date_column], df[stock_value_column], label=stock_value_column, color='b')

    plt.xlabel('Date')
    plt.ylabel('Stock Value')
    plt.title(title)

    plt.xticks(rotation=45)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


import talib
import pandas as pd
import matplotlib.pyplot as plt

def analyze_stock_data(df):
    """
    Analyze stock data using TA-Lib technical indicators and visualize the results.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data with columns 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    """
    # Ensure the 'Date' column is in datetime format and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # ------------------ TA-Lib Technical Indicators ------------------
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # ------------------ Visualization ------------------
    # Create subplots for visualizing the data
    plt.figure(figsize=(14, 12))

    # Subplot for Stock Price and Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df['Close'], label='Close Price', color='blue', alpha=0.6)
    plt.plot(df['SMA_50'], label='50-Day SMA', color='red', linestyle='--')
    plt.plot(df['EMA_50'], label='50-Day EMA', color='green', linestyle='--')
    plt.title('Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()

    # Subplot for MACD and Signal Line
    plt.subplot(3, 1, 2)
    plt.plot(df['MACD'], label='MACD', color='blue')
    plt.plot(df['MACD_signal'], label='MACD Signal', color='red', linestyle='--')
    plt.bar(df.index, df['MACD_hist'], label='MACD Histogram', color='gray', alpha=0.5)
    plt.title('MACD and Signal Line')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()

    # Subplot for RSI
    plt.subplot(3, 1, 3)
    plt.plot(df['RSI'], label='RSI', color='orange')
    plt.axhline(30, color='green', linestyle='--', label="Oversold (30)")
    plt.axhline(70, color='red', linestyle='--', label="Overbought (70)")
    plt.title('RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend()

    plt.tight_layout()
    plt.show()
