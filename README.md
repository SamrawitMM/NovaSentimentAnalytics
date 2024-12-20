# Stock Market Sentiment Analysis

This repository contains a comprehensive analysis of stock market data with a focus on sentiment analysis, financial metrics, and historical stock data of major companies. The project leverages data from Yahoo Finance and analyst ratings, integrating techniques such as topic modeling and sentiment analysis.

## Project Structure

```bash
├── =4.2.0
├── data
│   ├── raw_analyst_ratings.csv
│   └── yfinance_data
│       ├── AAPL_historical_data.csv
│       ├── AMZN_historical_data.csv
│       ├── GOOG_historical_data.csv
│       ├── META_historical_data.csv
│       ├── MSFT_historical_data.csv
│       ├── NVDA_historical_data.csv
│       └── TSLA_historical_data.csv
├── notebooks
│   ├── =4.2.0
│   ├── bertopic_model
│   ├── headline_topics.csv
│   ├── __init__.py
│   ├── main.ipynb
│   └── publisher_topics_no_emails.csv
├── README.md
└── scripts
    ├── __init__.py
    ├── plot.py
    └── utility.py
```


## Data Overview

- **Raw Analyst Ratings**: Contains data from various analysts regarding stock ratings.
- **Yahoo Finance Data**: Historical stock prices for major companies like Apple, Amazon, Google, Meta, Microsoft, Nvidia, and Tesla.
- **Headlines & Publisher Topics**: Analysis of stock-related headlines and identification of topics using sentiment and keyword extraction.

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
- bertopic

## Installation

1. Clone this repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

- The `notebooks/main.ipynb` file contains the primary analysis workflow.
- `scripts/plot.py` can be used to generate visualizations, and `utility.py` contains helper functions for data processing.

## License

This project is licensed under the MIT License.
