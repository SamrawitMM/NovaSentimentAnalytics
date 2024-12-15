import pandas as pd
import nltk

from nltk.corpus import words, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import re

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

    