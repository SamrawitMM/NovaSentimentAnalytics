import pandas as pd
import nltk
from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re
from textblob import TextBlob
from joblib import Parallel, delayed
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import talib


# Ensure you download necessary NLTK resources
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

# Preprocess a text input
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


# Calculates the articles per day of the week and for the year to make the value ready for the plot
def calculate_and_plot(text_data):
    
    # Calculate the total number of articles published per day of the week across the entire dataset
    articles_per_day_total = text_data.groupby('day_of_week').size()

    # Calculate the average articles per day of the week across all years
    # total_days = len(text_data)  # Total number of days in the dataset
    articles_per_day_avg = articles_per_day_total / 84  # Average per day of the week

    # Ensure the order of the days of the week (Monday to Sunday)
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Reindex to include all days of the week, filling missing days with 0
    articles_per_day_avg = articles_per_day_avg.reindex(ordered_days, fill_value=0)

    # Calculate the total articles per year
    articles_per_year_total = text_data.groupby('year').size()

    # Calculate the average articles per year (total number of articles per year divided by the number of years)
    articles_per_year_avg = articles_per_year_total / 12   # Average per year

    # Print results
    print("Average Articles Published Per Day of the Week (Monday to Sunday):")
    print(articles_per_day_avg)

    print("\nAverage Articles Published Per Year:")
    print(articles_per_year_avg)

    return articles_per_day_avg, articles_per_year_avg


# Plot top publishers that published more articles
def top_publishers(text_data):
    # Count the number of articles per publisher
    articles_per_publisher = text_data['publisher'].value_counts()

    # Select the top 15 publishers
    top_15_publishers = articles_per_publisher.head(10)


    return top_15_publishers
    

# Top 15 frequent headlines
def top_headline(text_data):
    # Count the number of occurrences of each headline
    headline_counts = text_data['headline'].value_counts()

    # Select the top 15 headlines
    top_15_headlines = headline_counts.head(15)


    return top_15_headlines


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



def analyze_am_pm_distribution(articles_per_hour):

    # Calculate the number of articles in AM and PM
    am_articles = articles_per_hour.loc[0:11].sum()  # Sum of articles from 0 to 11 (AM)
    pm_articles = articles_per_hour.loc[12:23].sum()  # Sum of articles from 12 to 23 (PM)


    # Print the counts
    print(f"AM Articles: {am_articles}")
    print(f"PM Articles: {pm_articles}")

    return {'AM': am_articles, 'PM': pm_articles}

# Get dominant topics from the news feed/articles published
def analyze_topics_with_bertopic(text_data, column_name, n_samples=1000, model_name='all-MiniLM-L6-v2', min_topic_size=10):

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



# Identify publishers  named with email regular expression
def analyze_email_domains(text_data, email_column, top_n=10):
    
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


    return domain_counts


# Get dominant topics
def perform_topic_modeling(
    text_data, 
    column_name, 
    n_samples=1000, 
    n_topics=5, 
    n_terms=10, 
    stop_words='english', 
    output_file='headline_topics.csv'
):

    # Randomly sample rows
    text_data_sampled = text_data.sample(n=n_samples, random_state=42)

    # Preprocess the column
    text_data_sampled[column_name] = text_data_sampled[column_name].str.lower().str.strip()

    # Vectorize the text data
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000)
    text_matrix = vectorizer.fit_transform(text_data_sampled[column_name])

    # Apply LDA
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(text_matrix)

    # Extract topics
    terms = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_terms = [terms[i] for i in topic.argsort()[:-n_terms - 1:-1]]
        topics.append(top_terms)

    print("Identified Topics:")
    for i, topic in enumerate(topics):
        print(f"Topic {i+1}: {', '.join(topic)}")

    # Assign topics to rows
    topic_distribution = lda_model.transform(text_matrix)
    text_data_sampled['dominant_topic'] = topic_distribution.argmax(axis=1)

    # Visualize Topic Distribution
    topic_counts = text_data_sampled['dominant_topic'].value_counts()
 

    # Save results (optional)
    if output_file:
        text_data_sampled[[column_name, 'dominant_topic']].to_csv(output_file, index=False)
        print(f"Results saved to '{output_file}'")

    return text_data_sampled, topics



# Headline descriptive statistics and histogram plot
def analyze_headline_lengths(data, headline_column):
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



def preprocess_stock_data(df):
    """
    Analyze stock data using TA-Lib technical indicators and return the modified DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing stock data with columns 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.

    Returns:
        pd.DataFrame: DataFrame with added columns for SMA, EMA, RSI, MACD.
    """
    

    # Technical Indicators
    df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
    df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    return df



def average_article_overtime(df):
    articles_per_day_total = df.groupby('day').size()
    articles_per_year_total = df.groupby('year').size()
    
    

    unique_weeks = df[['year', 'day_of_week']].drop_duplicates().shape[0]
    unique_month_days = df[['year', 'month', 'day']].drop_duplicates().shape[0]
    unique_months = df[['year', 'month']].drop_duplicates().shape[0]
    unique_years = df[['year']].drop_duplicates().shape[0]
    unique_hours = df[['hour']].drop_duplicates().shape[0]

    articles_per_day_total = df.groupby('day_of_week').size()
    articles_per_year_total = df.groupby('year').size()
    articles_daily_total = df.groupby('day').size() 
    articles_montly_total = df.groupby('month').size()
    articles_per_hour = df['hour'].value_counts().sort_index()


    # Calculate the average articles per day, month, and year
    avg_article_per_day_week = articles_per_day_total / unique_weeks
    avg_article_per_month = articles_montly_total / unique_months
    avg_article_per_year = articles_per_year_total / unique_years
    articles_daily_avg = articles_daily_total/ unique_month_days
    avg_articles_per_hour = articles_per_hour / unique_hours

    
    ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Reindex to include all days of the week, filling missing days with 0
    articles_per_day_of_week_avg = avg_article_per_day_week.reindex(ordered_days, fill_value=0)

    ordered_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    articles_per_month_avg = avg_article_per_month.reindex(ordered_months, fill_value=0)

    # Print results
    # print("Average Articles Published Per Day:")
    # print(avg_article_per_day_week)
    print("\nAverage Articles Published Per Day of the Week (Monday to Sunday):")
    print(articles_per_day_of_week_avg)

    print("\nAverage Articles Published Per Month:")
    print(avg_article_per_month)

    print("\nAverage Articles Published Per Year:")
    print(avg_article_per_year)

    # print("\nAverage Articles Published Per Day of the Week (Monday to Sunday):")
    # print(articles_per_day_of_week_avg)

    print("\Total Articles Each Year:")
    print(articles_per_year_total)

    print("\Average Articles Each Day in a Month ( 1 to 30):")
    print(articles_daily_avg)

    print("\Average Articles Each hour ( 1 to 24):")
    print(avg_articles_per_hour)

    return articles_per_day_of_week_avg, avg_article_per_month, avg_article_per_year, articles_per_year_total, articles_daily_avg, avg_articles_per_hour
