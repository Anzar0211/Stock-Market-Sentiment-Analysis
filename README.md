# Financial Sentiment Analysis and Stock Prediction

A comprehensive platform for collecting, analyzing sentiment, and predicting stock movements using data from multiple sources including Twitter, Reddit, StockTwits, and financial news APIs.

## Project Overview

This project combines natural language processing (NLP), machine learning, and financial data analysis to provide insights into market sentiment and predict stock price movements. It collects real-time data from social media platforms and news sources, analyzes sentiment, and uses this information along with historical market data to generate predictions.

## Features

- **Multi-source Data Collection**:

  - Twitter account monitoring
  - Reddit financial subreddits scraping
  - StockTwits sentiment analysis
  - Financial news aggregation from News API

- **Sentiment Analysis**:

  - Text preprocessing pipelines
  - Multiple sentiment analysis models
  - Comparative analysis of sentiment methods

- **Predictive Modeling**:

  - LSTM neural networks for time series prediction
  - Regression and classification models for price movement prediction
  - Feature importance analysis

- **Visualization and Reporting**:
  - Interactive dashboard for real-time monitoring
  - Automated report generation
  - Sector-based analysis

## Directory Structure

```
.
├── config/               # Configuration files and settings
├── dashboard/            # Interactive visualization dashboard
├── data/                 # Data storage directory
│   ├── historical/       # Historical stock price data
│   ├── news/             # Collected news articles
│   ├── reddit/           # Reddit data
│   ├── stocks/           # Stock market data
│   ├── stocktwits/       # StockTwits data
│   └── tweets/           # Twitter data
├── examples/             # Example scripts and demonstrations
├── models/               # Trained machine learning models
├── results/              # Analysis results and visualization outputs
├── sentiment_env/        # Python virtual environment
└── src/                  # Source code
    ├── data_collection/  # Data collection scripts
    ├── predictive_models/# Prediction model implementations
    ├── preprocessing/    # Data preprocessing utilities
    ├── sector_analysis/  # Sector-specific analysis tools
    └── sentiment/        # Sentiment analysis modules
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/financial-sentiment-analysis.git
cd financial-sentiment-analysis
```

2. Create and activate the virtual environment:

```bash
python -m venv sentiment_env
# On Windows
sentiment_env\Scripts\activate
# On MacOS/Linux
source sentiment_env/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure API keys:
   - Create a `.env` file in the root directory
   - Add your API keys for Twitter, Reddit, News API, etc.

## Usage

### Data Collection

```bash
# Collect Twitter data
python fetch_multiple_twitter_accounts.py

# Collect Reddit data
python reddit_scraper.py

# Collect financial news
python news_api_collector.py

# Collect stock price data
python stock_data_collection.py
```

### Running the Dashboard

```bash
python run_dashboard.py
```

### Running Example Analysis

```bash
python examples/apple_sentiment_analysis.py
python examples/stock_price_prediction_demo.py
```

## Example Results

The `examples/` directory contains demonstrations of various analysis techniques:

- Sentiment analysis comparison across different models
- Text processing pipelines for financial text
- Stock price prediction using sentiment features
- Visualization of sentiment distribution across sources

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- This project uses several open-source libraries and APIs
- Special thanks to contributors and data providers
