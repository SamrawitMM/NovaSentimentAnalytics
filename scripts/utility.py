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


    import matplotlib.pyplot as plt

def plot_articles_per_day(articles_per_day_avg):
    """
    This function plots the average articles published per day of the week.
    
    Args:
    - articles_per_day_avg (Series): Average articles published per day of the week.
    """
    plt.figure(figsize=(7, 6))
    articles_per_day_avg.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Average Articles Published Per Day of the Week", fontsize=14)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

def plot_articles_per_year(articles_per_year_avg):
    """
    This function plots the average articles published per year.
    
    Args:
    - articles_per_year_avg (Series): Average articles published per year.
    """
    plt.figure(figsize=(7, 6))
    articles_per_year_avg.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Average Articles Published Per Year", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_articles_per_day(articles_per_day_avg)
# plot_articles_per_year(articles_per_year_avg)


def calculate_and_plot(text_data):
    """
    This function will calculate the average articles per day of the week and year, and call the plotting function.
    Args:
    - text_data (DataFrame): The dataset containing the articles data with datetime information.
    """
    # # Ensure the 'date' column is in datetime format
    # text_data['Date'] = pd.to_datetime(text_data['Date'])

    # # # Extract year, month, and day_of_week directly from the 'date' column
    # text_data['year'] = text_data['Date'].dt.year
    # text_data['month'] = text_data['Date'].dt.month
    # text_data['day_of_week'] = text_data['Date'].dt.day_name()

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


import matplotlib.pyplot as plt

def plot_monthly_trend(text_data):
    """
    This function plots the average number of articles published per month.
    
    Args:
    - text_data (pd.DataFrame): The dataframe containing article data with 'month' and 'year' columns.
    """
    # Compute the average number of articles per month (January to December)
    articles_monthly_avg = text_data.groupby('month').size() / text_data['year'].nunique()

    # Plotting the monthly trend
    plt.figure(figsize=(8, 6))  # Adjust figure size for better visibility
    articles_monthly_avg.plot(kind='bar', color='dodgerblue', edgecolor='black')
    plt.title("Average Articles Published Per Month", fontsize=14)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.tight_layout()
    plt.show()

def plot_daily_trend(text_data):
    """
    This function plots the average number of articles published per day of the month.
    
    Args:
    - text_data (pd.DataFrame): The dataframe containing article data with 'day' and 'month' columns.
    """
    # Compute the average number of articles per day of the month (1 to 30)
    articles_daily_avg = text_data.groupby('day').size() / text_data['month'].nunique()

    # Plotting the daily trend of the month
    plt.figure(figsize=(8, 6))  # Adjust figure size for better visibility
    articles_daily_avg.plot(kind='bar', color='tomato', edgecolor='black')
    plt.title("Average Articles Published Per Day (1-30)", fontsize=14)
    plt.xlabel("Day of the Month", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(ticks=range(30), labels=range(1, 31), rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_monthly_trend(text_data)
# plot_daily_trend(text_data)


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




# Function to extract and format the date
def extract_and_format_date(dataframe, date_column):
    dataframe[date_column] = dataframe[date_column].astype(str)
    dataframe['extracted_date'] = dataframe[date_column].str.split(' ').str[0]
    dataframe['formatted_date'] = pd.to_datetime(dataframe['extracted_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    dataframe['valid_date'] = dataframe['formatted_date'].notna()
    dataframe['formatted_date'] = pd.to_datetime(dataframe['formatted_date'])
    return dataframe

# Function to clean time strings
def clean_time_strings(time_str):
    try:
        time_str = time_str.split('-')[0].split('+')[0]
        parts = time_str.split(':')
        if len(parts) >= 3:
            return ':'.join(parts[:3])
        return time_str
    except Exception:
        return None

# Function to extract and format the time
def extract_and_format_time(dataframe, date_column):
    dataframe[date_column] = dataframe[date_column].astype(str)
    dataframe['extracted_time'] = dataframe[date_column].str.split(' ').str[1]
    dataframe['extracted_time'] = dataframe['extracted_time'].apply(clean_time_strings)
    dataframe['formatted_time'] = pd.to_datetime(dataframe['extracted_time'], format='%H:%M:%S', errors='coerce').dt.strftime('%H:%M:%S')
    dataframe['valid_time'] = dataframe['formatted_time'].notna()
    dataframe['timestamp'] = pd.to_datetime(dataframe['formatted_time'], format='%H:%M:%S', errors='coerce')
    dataframe['time'] = dataframe['timestamp'].dt.time
    dataframe['hour'] = dataframe['timestamp'].dt.hour
    dataframe['minute'] = dataframe['timestamp'].dt.minute
    return dataframe

# Function to extract additional date-related features
def extract_date_features(dataframe, formatted_date_column):
    dataframe['year'] = dataframe[formatted_date_column].dt.year
    dataframe['month'] = dataframe[formatted_date_column].dt.month
    dataframe['day'] = dataframe[formatted_date_column].dt.day
    dataframe['day_of_week'] = dataframe[formatted_date_column].dt.day_name()
    return dataframe

# Main preprocessing function
def preprocess_dataset(dataframe, date_column):
    dataframe = extract_and_format_date(dataframe, date_column)
    dataframe = extract_and_format_time(dataframe, date_column)
    dataframe = extract_date_features(dataframe, 'formatted_date')

    dataframe.rename(columns={'formatted_date': 'Date'}, inplace=True)
    dataframe.drop(columns=['extracted_date', 'valid_date', 'date', 'extracted_time','formatted_time', 'valid_time', 'timestamp'], inplace=True)
    return dataframe


import matplotlib.pyplot as plt

def analyze_articles_per_hour(dataframe):
    """
    Analyze and visualize the distribution of articles published per hour.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame containing a column 'hour'.

    Returns:
    - pd.Series: Distribution of articles per hour.
    """
    # Count the number of articles published per hour
    articles_per_hour = dataframe['hour'].value_counts().sort_index()

    # Display the result
    print("Articles Published Per Hour:")
    print(articles_per_hour)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(articles_per_hour.index, articles_per_hour.values, color='orange')
    plt.title('Articles Published Per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Articles')
    plt.xticks(range(0, 24))  # Ensure all hours are displayed
    plt.tight_layout()
    plt.show()

    return articles_per_hour


import matplotlib.pyplot as plt

def analyze_am_pm_distribution(articles_per_hour):
    """
    Analyze and visualize the distribution of articles published during AM and PM hours.

    Parameters:
    - articles_per_hour (pd.Series): Series containing the number of articles published per hour.

    Returns:
    - dict: A dictionary with counts of AM and PM articles.
    """
    # Calculate the number of articles in AM and PM
    am_articles = articles_per_hour.loc[0:11].sum()  # Sum of articles from 0 to 11 (AM)
    pm_articles = articles_per_hour.loc[12:23].sum()  # Sum of articles from 12 to 23 (PM)

    # Prepare data for the pie chart
    labels = ['AM', 'PM']
    values = [am_articles, pm_articles]

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=['skyblue', 'orange']
    )
    plt.title('Articles Published by AM/PM')
    plt.tight_layout()
    plt.show()

    # Print the counts
    print(f"AM Articles: {am_articles}")
    print(f"PM Articles: {pm_articles}")

    return {'AM': am_articles, 'PM': pm_articles}


from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def analyze_topics_with_bertopic(text_data, column_name, n_samples=1000, model_name='all-MiniLM-L6-v2', min_topic_size=10):
    """
    Perform topic modeling using BERTopic and visualize the results.

    Parameters:
    - text_data (pd.DataFrame): The dataset containing text data.
    - column_name (str): The name of the column with text data to analyze.
    - n_samples (int): Number of samples to randomly select for analysis.
    - model_name (str): Pre-trained model name for sentence embeddings.
    - min_topic_size (int): Minimum topic size for BERTopic.

    Returns:
    - dict: A dictionary containing the topic information and top 10 topics.
    """
    # Load the sample data
    sample_data = text_data[column_name].dropna().sample(n=n_samples, random_state=42).tolist()

    # Initialize embedding model
    embedding_model = SentenceTransformer(model_name)

    # Initialize and fit BERTopic model
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=min_topic_size)
    topics, probs = topic_model.fit_transform(sample_data)

    # Get topic information
    topic_info = topic_model.get_topic_info()

    # Display the top 10 topics
    top_topics = topic_info.head(10)
    print("Top 10 Topics:")
    print(top_topics)

    # Visualization
    if len(topic_info) > 0:
        try:
            # Visualizing topic clusters
            fig_topics = topic_model.visualize_topics()
            fig_topics.show()

            # Visualizing the frequency of topics in a bar chart
            fig_barchart = topic_model.visualize_barchart()
            fig_barchart.show()
        except Exception as e:
            print(f"Error during visualization: {e}")
    else:
        print("No topics were generated. Skipping visualizations.")

    return {
        "topic_info": topic_info,
        "top_topics": top_topics
    }


import matplotlib.pyplot as plt
import pandas as pd

def analyze_email_domains(text_data, email_column, top_n=10):
    """
    Analyze and visualize the top email domains from a specified column in the dataset.

    Parameters:
    - text_data (pd.DataFrame): The dataset containing email addresses.
    - email_column (str): The name of the column containing email addresses.
    - top_n (int): The number of top domains to display in the visualization.

    Returns:
    - pd.Series: A series containing the counts of unique email domains.
    """
    # Extract domains from email addresses
    text_data['publisher_domain'] = text_data[email_column].apply(
        lambda x: x.split('@')[-1] if '@' in str(x) else None
    )

    # Count unique domains
    domain_counts = text_data['publisher_domain'].value_counts()

    print(f"{len(domain_counts)} total unique email domains")

    # Show top domains contributing to the feed
    print("Top domains:")
    print(domain_counts.head(top_n))

    # Visualize top domains
    plt.figure(figsize=(10, 6))
    domain_counts.head(top_n).plot(kind='bar', color='lightgreen')
    plt.title(f"Top {top_n} Email Domains Contributing to the News Feed")
    plt.xlabel("Domain")
    plt.ylabel("Count of Articles")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return domain_counts


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

def perform_topic_modeling(
    text_data, 
    column_name, 
    n_samples=1000, 
    n_topics=5, 
    n_terms=10, 
    stop_words='english', 
    output_file='headline_topics.csv'
):
    """
    Perform topic modeling on text data.

    Parameters:
    - text_data (pd.DataFrame): The input dataset containing text data.
    - column_name (str): The column to perform topic modeling on.
    - n_samples (int): Number of rows to sample for the analysis.
    - n_topics (int): Number of topics to extract.
    - n_terms (int): Number of top terms to display per topic.
    - stop_words (str or list): Stop words to exclude in the vectorization.
    - output_file (str): Optional path to save the results as a CSV file.

    Returns:
    - pd.DataFrame: The dataframe with dominant topics assigned to each row.
    - list: A list of extracted topics with their top terms.
    """
    # Step 1: Randomly sample rows
    text_data_sampled = text_data.sample(n=n_samples, random_state=42)

    # Step 2: Preprocess the column
    text_data_sampled[column_name] = text_data_sampled[column_name].str.lower().str.strip()

    # Step 3: Vectorize the text data
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000)
    text_matrix = vectorizer.fit_transform(text_data_sampled[column_name])

    # Step 4: Apply LDA
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(text_matrix)

    # Step 5: Extract topics
    terms = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_terms = [terms[i] for i in topic.argsort()[:-n_terms - 1:-1]]
        topics.append(top_terms)

    print("Identified Topics:")
    for i, topic in enumerate(topics):
        print(f"Topic {i+1}: {', '.join(topic)}")

    # Step 6: Assign topics to rows
    topic_distribution = lda_model.transform(text_matrix)
    text_data_sampled['dominant_topic'] = topic_distribution.argmax(axis=1)

    # Step 7: Visualize Topic Distribution
    topic_counts = text_data_sampled['dominant_topic'].value_counts()
    plt.figure(figsize=(8, 6))
    topic_counts.plot(kind='bar', color='coral')
    plt.title("Distribution of Topics in Headlines")
    plt.xlabel("Topic")
    plt.ylabel("Number of Headlines")
    plt.xticks(rotation=0)
    plt.show()

    # Step 8: Save results (optional)
    if output_file:
        text_data_sampled[[column_name, 'dominant_topic']].to_csv(output_file, index=False)
        print(f"Results saved to '{output_file}'")

    return text_data_sampled, topics


import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_article_trends(text_data, year_column='year', month_column='month', exclude_year=None):
    """
    Plot monthly article publication trends by year, and separately for a specified year.

    Parameters:
    - text_data (pd.DataFrame): The dataset containing 'year' and 'month' columns.
    - year_column (str): The column name representing the year.
    - month_column (str): The column name representing the month.
    - exclude_year (int or None): The year to exclude from the overall trend plot (optional).

    Returns:
    - None
    """
    # Group by year and month to count articles
    monthly_counts = text_data.groupby([year_column, month_column]).size().reset_index(name='count')

    # Convert the 'month' column to integers
    monthly_counts[month_column] = monthly_counts[month_column].astype(int)

    # Add month name for better labeling
    monthly_counts['month_name'] = monthly_counts[month_column].apply(
        lambda x: pd.to_datetime(f'{x}', format='%m').strftime('%B')
    )

    # Reorder months to appear in calendar order
    monthly_counts['month_name'] = pd.Categorical(
        monthly_counts['month_name'], 
        categories=[
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ],
        ordered=True
    )

    # Sort the values by 'year' and 'month_name'
    monthly_counts = monthly_counts.sort_values(by=[year_column, 'month_name'])

    # Pivot the data for line plotting
    monthly_counts_pivot = monthly_counts.pivot(index='month_name', columns=year_column, values='count')

    # Plot the overall trend for each year
    plt.figure(figsize=(12, 6))
    for year in monthly_counts_pivot.columns:
        if year != exclude_year:
            plt.plot(monthly_counts_pivot.index, monthly_counts_pivot[year], label=f'Year {int(year)}', marker='')

    # Add title, labels, and legend for the overall trend plot
    plt.title('Monthly Article Publication Trends by Year')
    plt.xlabel('Months')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Display the overall trend chart
    plt.tight_layout()
    plt.show()

    # Plot the trend for the excluded year (if specified)
    if exclude_year in monthly_counts_pivot.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(
            monthly_counts_pivot.index, 
            monthly_counts_pivot[exclude_year], 
            label=f'Year {exclude_year}', 
            color='blue', 
            marker='o'
        )

        # Add title, labels, and legend for the excluded year plot
        plt.title(f'Monthly Article Publication Trend for Year {exclude_year}')
        plt.xlabel('Months')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Display the chart for the excluded year's trend
        plt.tight_layout()
        plt.show()



import matplotlib.pyplot as plt

def plot_top_publishers_pie_chart(text_data, top_n=10):
    """
    This function takes a DataFrame `text_data` containing article data and plots a pie chart
    of the top N publishers based on the number of articles they have published.

    Parameters:
    text_data (pd.DataFrame): The dataframe containing article data, must have a 'publisher' column.
    top_n (int): The number of top publishers to display (default is 10).

    Returns:
    None
    """
    # Step 1: Count the number of articles per publisher
    articles_per_publisher = text_data['publisher'].value_counts()

    # Step 2: Select the top N publishers
    top_publishers = articles_per_publisher.head(top_n)

    # Step 3: Plot the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(top_publishers, labels=top_publishers.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title(f'Top {top_n} Publishers and Proportion of Articles Published')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.

    # Show the pie chart
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_top_publishers_pie_chart(text_data, top_n=10)


import matplotlib.pyplot as plt

def plot_top_headlines_bar_chart(text_data, top_n=15):
    """
    This function takes a DataFrame `text_data` containing article data and plots a bar chart
    for the top N headlines based on their occurrences.

    Parameters:
    text_data (pd.DataFrame): The dataframe containing article data, must have a 'headline' column.
    top_n (int): The number of top headlines to display (default is 15).

    Returns:
    None
    """
    # Step 1: Count the number of occurrences of each headline
    headline_counts = text_data['headline'].value_counts()

    # Step 2: Select the top N headlines
    top_headlines = headline_counts.head(top_n)

    # Step 3: Plot the data
    plt.figure(figsize=(12, 8))
    top_headlines.plot(kind='bar', color='orange')
    plt.title(f'Top {top_n} Headlines and Their Occurrences')
    plt.xlabel('Headline')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_top_headlines_bar_chart(text_data, top_n=15)


def plot_monthly_trends_overtime(text_data, exclude_year=None, ylim_overall=None, ylim_2020=None):
    """
    Plots monthly article publication trends by year and a separate plot for a specific year.

    Parameters:
        text_data (pd.DataFrame): DataFrame containing 'year' and 'month' columns.
        exclude_year (int, optional): Year to exclude from the overall trend plot.
        ylim_overall (tuple, optional): Y-axis limits for the overall trend plot (min, max).
        ylim_2020 (tuple, optional): Y-axis limits for the 2020 trend plot (min, max).
    """
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


import matplotlib.pyplot as plt

def analyze_headline_lengths(data, headline_column):
    """
    Analyze and visualize the distribution of headline lengths in a dataset.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the headline data.
        headline_column (str): The name of the column with headline text.
    
    Returns:
        pd.Series: Descriptive statistics of headline lengths.
    """
    # Calculate headline lengths and their statistics
    data['headline_length'] = data[headline_column].apply(len)
    headline_stats = data['headline_length'].describe()
    print("Headline Length Statistics:\n", headline_stats)

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data['headline_length'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.title('Distribution of Headline Lengths', fontsize=14)
    plt.xlabel('Headline Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()
    plt.show()

    return headline_stats
