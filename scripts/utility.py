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


# Plot articles published per day of the week to see the trend
def plot_articles_per_day(articles_per_day_avg):
    
    plt.figure(figsize=(7, 6))
    articles_per_day_avg.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Average Articles Published Per Day of the Week", fontsize=14)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()

# Plot articles published per year
def plot_articles_per_year(articles_per_year_avg):

    plt.figure(figsize=(7, 6))
    articles_per_year_avg.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Average Articles Published Per Year", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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



# Plot number of articles published per month
def plot_monthly_trend(text_data):

    # Compute the average number of articles per month (January to December)
    articles_monthly_avg = text_data.groupby('month').size() / 136

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

# Plot day of the month published articles 
def plot_daily_trend(text_data):
    
    # Compute the average number of articles per day of the month (1 to 30)
    articles_daily_avg = text_data.groupby('day').size() / 3955

    # Plotting the daily trend of the month
    plt.figure(figsize=(8, 6))  # Adjust figure size for better visibility
    articles_daily_avg.plot(kind='bar', color='tomato', edgecolor='black')
    plt.title("Average Articles Published Per Day (1-30)", fontsize=14)
    plt.xlabel("Day of the Month", fontsize=12)
    plt.ylabel("Average Number of Articles", fontsize=12)
    plt.xticks(ticks=range(30), labels=range(1, 31), rotation=45)
    plt.tight_layout()
    plt.show()



# Plot year trend over the months
def plot_year(text_data):
   

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


# Plot top publishers that published more articles
def top_publishers(text_data):
    # Count the number of articles per publisher
    articles_per_publisher = text_data['publisher'].value_counts()

    # Select the top 15 publishers
    top_15_publishers = articles_per_publisher.head(10)

    # Step 3: Plot the pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(top_15_publishers, labels=top_15_publishers.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Top 10 Publishers and Proportion of Articles Published')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.

    # Show the pie chart
    plt.tight_layout()
    plt.show()
    

# Top 15 frequent headlines
def top_headline(text_data):
    # Count the number of occurrences of each headline
    headline_counts = text_data['headline'].value_counts()

    # Select the top 15 headlines
    top_15_headlines = headline_counts.head(15)

    # Plot the data
    plt.figure(figsize=(12, 8))
    top_15_headlines.plot(kind='bar', color='orange')
    plt.title('Top 15 Headlines and Their Occurrences')
    plt.xlabel('Headline')
    plt.ylabel('Number of Occurrences')
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
    plt.grid(True)
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



# Plot articles published per hour
def analyze_articles_per_hour(dataframe):

    # Count the number of articles published per hour
    articles_per_hour = dataframe['hour'].value_counts().sort_index() / 24

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


# PM and AM distirubtion as the majority of the hour consentrates in approximately in a similar hour
def analyze_am_pm_distribution(articles_per_hour):

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
    plt.figure(figsize=(8, 6))
    topic_counts.plot(kind='bar', color='coral')
    plt.title("Distribution of Topics in Headlines")
    plt.xlabel("Topic")
    plt.ylabel("Number of Headlines")
    plt.xticks(rotation=0)
    plt.show()

    # Save results (optional)
    if output_file:
        text_data_sampled[[column_name, 'dominant_topic']].to_csv(output_file, index=False)
        print(f"Results saved to '{output_file}'")

    return text_data_sampled, topics


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
