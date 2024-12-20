import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib


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


    # visualization.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_time_series(df, date_column='Date', value_column='Close', title='Time Series Plot'):
    """
    Plot a time series of stock values or any other metric against time.

    Args:
    - df (pd.DataFrame): DataFrame containing time series data.
    - date_column (str): Column containing the date values (default 'Date').
    - value_column (str): Column containing the value to be plotted (default 'Close').
    - title (str): Title for the plot (default 'Time Series Plot').
    """
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    plt.figure(figsize=(10, 6))
    plt.plot(df[date_column], df[value_column], label=value_column, color='b')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bar(data, title, xlabel, ylabel, colors=None, custom_xticks=None, horizontal=False, bar_width=0.8, grid=True, xtick_rotation=45):
    """
    General bar plot function for visualizing categorical data with customizable options.

    Args:
    - data (pd.Series or pd.DataFrame): Data to plot.
    - title (str): Title for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - colors (list): List of colors for bars (optional).
    - custom_xticks (list or pd.Index): Custom labels for x-ticks (optional).
    - horizontal (bool): Whether to create a horizontal bar plot (default False).
    - bar_width (float): Width of the bars (default 0.8).
    - grid (bool): Whether to display gridlines (default True).
    - xtick_rotation (int): Rotation angle for x-tick labels (default 45 degrees).
    """
    plt.figure(figsize=(12, 8))
    
    if horizontal:
        ax = data.plot(kind='barh', color=colors, width=bar_width)
    else:
        ax = data.plot(kind='bar', color=colors, width=bar_width)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Check if custom_xticks is provided and handle both list and pandas Index
    if custom_xticks is not None:
        if isinstance(custom_xticks, (list, np.ndarray)):  # If it's a list or numpy array
            ax.set_xticks(range(len(custom_xticks)))  # Ensure the ticks are spaced correctly
            ax.set_xticklabels(custom_xticks, rotation=xtick_rotation, ha='right')  # Rotate and align labels for better readability
        elif isinstance(custom_xticks, pd.Index):  # If it's a pandas Index
            ax.set_xticks(range(len(custom_xticks)))  # Ensure the ticks are spaced correctly
            ax.set_xticklabels(custom_xticks.tolist(), rotation=xtick_rotation, ha='right')  # Convert to list for setting tick labels
    else:
        plt.xticks(rotation=xtick_rotation, ha='right')  # Use default tick labels if custom_xticks are not provided
    
    if grid:
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def plot_sentiment(sentiment_counts, company):
    """
    Visualize sentiment classes in a bar chart.

    Args:
    - sentiment_counts (pd.Series): Sentiment class counts.
    - company (str): Company name for the title.
    """
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'grey'}

    plt.figure(figsize=(6, 4))
    ax = sentiment_counts.plot(kind='bar', color=[colors[sentiment] for sentiment in sentiment_counts.index])
    plt.title(f'Sentiment Distribution for {company} Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)

    for i, count in enumerate(sentiment_counts):
        ax.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

    plt.show()

def plot_pie(data, title='Pie Chart', labels=None, autopct='%1.1f%%', colors=None):
    """
    Plot a pie chart to show the distribution of categorical data.

    Args:
    - data (pd.Series): Data containing categories and their counts.
    - title (str): Title for the plot (default 'Pie Chart').
    - labels (list): List of labels for each slice (optional).
    - autopct (str): Format string for percentage display (default '%1.1f%%').
    - colors (list): List of colors for the pie slices (optional).
    """
    plt.figure(figsize=(10, 8))
    if labels is None:
        labels = data.index
    if colors is None:
        colors = plt.cm.Paired.colors  # Default color scheme
    
    plt.pie(data, labels=labels, autopct=autopct, startangle=90, colors=colors)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
    plt.show()




def analyze_stock_data(df):
    """
    Analyze stock data using TA-Lib technical indicators and visualize the results.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data with columns 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    """
    
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


# Visualize sentiment classes in a graph
def plot_sentiment(sentiment_counts, company):
    # Custom colors for each sentiment
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'grey'}

    # Plot the sentiment distribution
    plt.figure(figsize=(6, 4))
    ax = sentiment_counts.plot(kind='bar', color=[colors[sentiment] for sentiment in sentiment_counts.index])
    plt.title('Sentiment Distribution of ' + company + 'Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)

    # Add the count above each bar
    for i, count in enumerate(sentiment_counts):
        ax.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

    plt.show()


# Plot of all the years and the trend of the month 
def plot_monthly_trends_overtime(text_data, exclude_year=None, ylim_overall=None, ylim_2020=None):
    
    # Group by year and month to count articles
    monthly_counts = text_data.groupby(['year', 'month']).size().reset_index(name='count')

    # Convert the 'month' column to integers
    monthly_counts['month'] = monthly_counts['month'].astype(int)

    # Add month name for better labeling
    monthly_counts['month_name'] = monthly_counts['month'].apply(
        lambda x: pd.to_datetime(f'{x}', format='%m').strftime('%B')
    )

    # Reorder months to appear in calendar order
    monthly_counts['month_name'] = pd.Categorical(
        monthly_counts['month_name'], 
        categories=['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December'],
        ordered=True
    )

    # Sort the values by 'year' and 'month_name'
    monthly_counts = monthly_counts.sort_values(by=['year', 'month_name'])

    # Pivot the data for line plotting
    monthly_counts_pivot = monthly_counts.pivot(index='month_name', columns='year', values='count')

    # Plot the line graph for each year (excluding the specified year)
    plt.figure(figsize=(12, 6))
    for year in monthly_counts_pivot.columns:
        if exclude_year is None or year != exclude_year:
            plt.plot(monthly_counts_pivot.index, monthly_counts_pivot[year], label=f'Year {int(year)}', marker='')

    # Title and labels for the overall trend plot
    plt.title('Monthly Article Publication Trends by Year')
    plt.xlabel('Months')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)

    # Set y-axis range if specified
    if ylim_overall:
        plt.ylim(ylim_overall)

    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the overall trend chart
    plt.tight_layout()
    plt.show()

    # Plot for the excluded year separately
    if exclude_year in monthly_counts_pivot.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_counts_pivot.index, monthly_counts_pivot[exclude_year], label=f'Year {exclude_year}', color='blue', marker='o')

        # Title and labels for the specific year plot
        plt.title(f'Monthly Article Publication Trend for Year {exclude_year}')
        plt.xlabel('Months')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)

        # Set y-axis range if specified
        if ylim_2020:
            plt.ylim(ylim_2020)

        plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Show the chart for the specific year trend
        plt.tight_layout()
        plt.show()

