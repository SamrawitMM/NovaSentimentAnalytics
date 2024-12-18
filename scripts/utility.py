import pandas as pd
import nltk

from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re

from textblob import TextBlob
from joblib import Parallel, delayed

# Ensure you download necessary NLTK resources
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def read_csv_file(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Remove any 'Unnamed:' columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    # Get additional info
    column_names = data.columns.tolist()  
    row_count = data.shape[0]  
    
    return {
        'data': data,
        'column_names': column_names,
        'row_count': row_count
    }


# Precompute static resources
ENGLISH_WORD_SET = set(words.words())  # Set of valid English words
STOP_WORDS = set(stopwords.words('english'))  # Set of stopwords
LEMMATIZER = WordNetLemmatizer()  # Lemmatizer instance


def clean_text(text):
    # # Load necessary sets and tools
    # english_word_set = set(words.words())  # set of valid English words
    # stop_words = set(stopwords.words('english'))  # set of stopwords
    # lemmatizer = WordNetLemmatizer()  # lemmatizer instance

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', '', text.translate(str.maketrans("", "", string.punctuation)))

    # Tokenize the cleaned text
    tokens = word_tokenize(text)

    # Filter and lemmatize tokens
    cleaned_words = [
        LEMMATIZER.lemmatize(word)
        for word in tokens
        if word not in STOP_WORDS and len(word) > 2 and word in ENGLISH_WORD_SET
    ]

    return ' '.join(cleaned_words)

# Function to get sentiment and polarity value
def get_sentiment(text):
    try:
        # Sentiment polarity: -1 (negative) to 1 (positive), categorize into positive, negative, neutral
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            sentiment = 'positive'
        elif polarity < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return sentiment, polarity  # Return both sentiment label and polarity value
    except Exception as e:
        print(f"Error processing text: {text}, error: {e}")
        return 'unknown', None  # If error occurs, return unknown sentiment and None for polarity

# Function to perform sentiment analysis on the DataFrame in parallel and save the result in new columns
def sentiment_analysis_parallel(df, column):
    # Apply sentiment analysis on the specified column in parallel
    sentiments = Parallel(n_jobs=-1)(delayed(get_sentiment)(text) for text in df[column])

    # Debug: print first few sentiments to check structure
    print(sentiments[:5])  # You can adjust this number to see more examples if needed
    
    # Convert the result into two separate lists: one for sentiment labels and one for polarity values
    try:
        sentiment_labels, polarity_values = zip(*sentiments)
    except ValueError as e:
        print(f"Error unpacking sentiments: {e}")
        return df  # Return the original DataFrame if there's an error

    # Save the results in the DataFrame as new columns
    df['sentiment'] = sentiment_labels  # New column for sentiment labels
    df['polarity'] = polarity_values    # New column for polarity values
    
    return df  # Return the updated DataFrame


import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

def plot_articles_trends(articles_per_day_avg, articles_per_year_avg):
    """
    This function will plot the average articles published per day of the week and per year.
    
    Args:
    - articles_per_day_avg (Series): Average articles published per day of the week.
    - articles_per_year_avg (Series): Average articles published per year.
    """
    
    plt.figure(figsize=(14, 6))  # Adjust figure size for side-by-side charts

    # Subplot 1: Average articles per day of the week
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first chart
    articles_per_day_avg.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Average Articles Published Per Day of the Week", fontsize=14)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Subplot 2: Average articles per year
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second chart
    articles_per_year_avg.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Average Articles Published Per Year", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(rotation=45)

    # Adjust layout and show the plots
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

def calculate_and_plot(text_data):
    """
    This function will calculate the average articles per day of the week and year, and call the plotting function.
    Args:
    - text_data (DataFrame): The dataset containing the articles data with datetime information.
    """
    # Ensure the 'date' column is in datetime format
    text_data['Date'] = pd.to_datetime(text_data['Date'])

    # # Extract year, month, and day_of_week directly from the 'date' column
    text_data['year'] = text_data['Date'].dt.year
    text_data['month'] = text_data['Date'].dt.month
    text_data['day_of_week'] = text_data['Date'].dt.day_name()

    # Count articles by day of the week (day_of_week column)
    articles_per_day = text_data['day_of_week'].value_counts()

    # Count articles over time (monthly trends) by grouping by year and month
    articles_over_time = text_data.groupby(['year', 'month']).size().sort_index()

    # Calculate the total number of articles published per day of the week across the entire dataset
    articles_per_day_total = text_data.groupby('day_of_week').size()

    # Calculate the average articles per day of the week across all years
    # total_days = len(text_data)  # Total number of days in the dataset
    articles_per_day_avg = articles_per_day_total  # Average per day of the week

    # Ensure the order of the days of the week (Monday to Sunday)
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Reindex to include all days of the week, filling missing days with 0
    articles_per_day_avg = articles_per_day_avg.reindex(ordered_days, fill_value=0)

    # Calculate the total articles per year
    articles_per_year_total = text_data.groupby('year').size()

    # Calculate the average articles per year (total number of articles per year divided by the number of years)
    total_years = len(text_data['year'].unique())  # Count the number of unique years
    articles_per_year_avg = articles_per_year_total   # Average per year

    # Print results
    print("Average Articles Published Per Day of the Week (Monday to Sunday):")
    print(articles_per_day_avg)

    print("\nAverage Articles Published Per Year:")
    print(articles_per_year_avg)

    return articles_per_day_avg, articles_per_year_avg


# Example usage
# Assuming `text_data` is your DataFrame with a 'date' column
# calculate_and_plot(text_data)

# Example usage
# Assuming `text_data` is your DataFrame with datetime index and articles data
# calculate_and_plot(text_data)

def plot_day_month(text_data):
    # # Ensure 'date' column exists
    # df_copy['date'] = pd.to_datetime(df_copy[['year', 'month']].assign(day=1))

    # # Extract day from 'date' column
    # text_data['day'] = text_data['day'].dt.day
    text_data['date'] = pd.to_datetime(text_data['date'], errors='coerce')

    # Extract the day of the month from 'date' column
    text_data['day'] = text_data['date'].dt.day

    # Compute the average number of articles per month (January to December)
    articles_monthly_avg = text_data.groupby('month').size() / text_data['year'].nunique()

    # Compute the average number of articles per day of the month (1 to 30)
    articles_daily_avg = text_data.groupby('day').size() / text_data['month'].nunique()

    # Plotting the trends
    plt.figure(figsize=(16, 8))  # Adjust figure size for better visibility

    # Subplot 1: Monthly trend (January to December)
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first chart
    articles_monthly_avg.plot(kind='bar', color='dodgerblue', edgecolor='black')
    plt.title("Average Articles Published Per Month", fontsize=14)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

    # Subplot 2: Daily trend of the month (1 to 30)
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second chart
    articles_daily_avg.plot(kind='bar', color='tomato', edgecolor='black')
    plt.title("Average Articles Published Per Day (1-30)", fontsize=14)
    plt.xlabel("Day of the Month", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(ticks=range(30), labels=range(1, 31), rotation=45)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()


def plot_year(text_data):
    # Sample dataframe, assuming df_copy contains 'year' and 'month' columns
    # Ensure that df_copy has 'year' and 'month' columns

    # Group by year and month to count articles
    monthly_counts = text_data.groupby(['year', 'month']).size().reset_index(name='count')

    # Convert the 'month' column to integers (in case it's a float)
    monthly_counts['month'] = monthly_counts['month'].astype(int)

    # Add month name for better labeling based on the 'month' column
    monthly_counts['month_name'] = monthly_counts['month'].apply(lambda x: pd.to_datetime(f'{x}', format='%m').strftime('%B'))

    # Reorder months to appear in calendar order
    monthly_counts['month_name'] = pd.Categorical(monthly_counts['month_name'], 
                                                categories=['January', 'February', 'March', 'April', 'May', 'June', 
                                                            'July', 'August', 'September', 'October', 'November', 'December'],
                                                ordered=True)

    # Sort the values by 'year' and 'month_name'
    monthly_counts = monthly_counts.sort_values(by=['year', 'month_name'])

    # Pivot the data for line plotting
    monthly_counts_pivot = monthly_counts.pivot(index='month_name', columns='year', values='count')

    # Plot the line graph for each year
    plt.figure(figsize=(12, 6))
    for year in monthly_counts_pivot.columns:
        if year != 2020:  # Skip the year 2020 for this plot
            plt.plot(monthly_counts_pivot.index, monthly_counts_pivot[year], label=f'Year {int(year)}', marker='')

    # Title and labels for the overall trend plot
    plt.title('Monthly Article Publication Trends by Year')
    plt.xlabel('Months')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability

    # Set y-axis range from 0 to 1250 (you can adjust this based on your data)
    # plt.ylim(0, 1250)

    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the overall trend chart
    plt.tight_layout()
    plt.show()

    # Plot for the year 2020 separately
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_counts_pivot.index, monthly_counts_pivot[2020], label='Year 2020', color='blue', marker='o')

    # Title and labels for 2020 plot
    plt.title('Monthly Article Publication Trend for Year 2020')
    plt.xlabel('Months')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability

    # Set y-axis range from 0 to 2000 (adjust as necessary)
    # plt.ylim(0, 2000)

    # Legend
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the chart for 2020 trend
    plt.tight_layout()
    plt.show()


def top_publishers(text_data):
    # Step 1: Count the number of articles per publisher
    articles_per_publisher = text_data['publisher'].value_counts()

    # Step 2: Select the top 15 publishers
    top_15_publishers = articles_per_publisher.head(10)

    # Step 3: Plot the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(top_15_publishers, labels=top_15_publishers.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Top 10 Publishers and Proportion of Articles Published')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.

    # Show the pie chart
    plt.tight_layout()
    plt.show()
    

def top_headline(text_data):
    # Step 1: Count the number of occurrences of each headline
    headline_counts = text_data['headline'].value_counts()

    # Step 2: Select the top 15 headlines
    top_15_headlines = headline_counts.head(15)

    # Step 3: Plot the data
    plt.figure(figsize=(12, 8))
    top_15_headlines.plot(kind='bar', color='orange')
    plt.title('Top 15 Headlines and Their Occurrences')
    plt.xlabel('Headline')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
    plt.grid(True)
    plt.show()


# Function to get sentiment and polarity value
def get_sentiment(text):
    try:
        # Sentiment polarity: -1 (negative) to 1 (positive), categorize into positive, negative, neutral
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            sentiment = 'positive'
        elif polarity < 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return sentiment, polarity  # Return both sentiment label and polarity value
    except Exception as e:
        print(f"Error processing text: {text}, error: {e}")
        return 'unknown', None  # If error occurs, return unknown sentiment and None for polarity

# Function to perform sentiment analysis on the DataFrame in parallel and save the result in new columns
def sentiment_analysis_parallel(df, column):
    # Apply sentiment analysis on the specified column in parallel
    sentiments = Parallel(n_jobs=-1)(delayed(get_sentiment)(text) for text in df[column])

    # Debug: print first few sentiments to check structure
    print(sentiments[:5])  # You can adjust this number to see more examples if needed
    
    # Convert the result into two separate lists: one for sentiment labels and one for polarity values
    try:
        sentiment_labels, polarity_values = zip(*sentiments)
    except ValueError as e:
        print(f"Error unpacking sentiments: {e}")
        return df  # Return the original DataFrame if there's an error

    # Save the results in the DataFrame as new columns
    df['sentiment'] = sentiment_labels  # New column for sentiment labels
    df['polarity'] = polarity_values    # New column for polarity values
    
    return df  # Return the updated DataFrame


def plot_sentiment(sentiment_counts):
    # Custom colors for each sentiment
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'grey'}

    # Plot the sentiment distribution
    plt.figure(figsize=(6, 4))
    ax = sentiment_counts.plot(kind='bar', color=[colors[sentiment] for sentiment in sentiment_counts.index])
    plt.title('Sentiment Distribution of AAPL Headlines')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)

    # Add the count above each bar
    for i, count in enumerate(sentiment_counts):
        ax.text(i, count + 5, str(count), ha='center', va='bottom', fontsize=10)

    plt.show()