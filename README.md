# Financial Sentiment Analysis and Stock Prediction

A comprehensive platform for collecting, analyzing sentiment, and predicting stock movements using data from multiple sources including Twitter, Reddit, StockTwits, and financial news APIs.

## Project Overview

This project combines natural language processing (NLP), machine learning, and financial data analysis to provide insights into market sentiment and predict stock price movements. It collects real-time data from social media platforms and news sources, analyzes sentiment, and uses this information along with historical market data to generate predictions.

## What Makes This Project Unique?

- **Multi-Source Sentiment Analysis**: Integrates sentiment from news, Reddit, Twitter, and StockTwits using advanced and custom analyzers (e.g., FinBERT, VADER with financial tuning).
- **Portfolio & Sector Sentiment Aggregation**: Analyzes and visualizes sentiment at the stock, sector, and portfolio level, including distribution and sector comparison.
- **Custom Fear & Greed Index**: Implements a multi-factor index using market breadth, momentum, junk bond demand, and sentiment indicators.
- **Economic Factor Integration**: Combines sector performance, sentiment, and economic indicators for deeper insights.
- **Interactive Dashboard**: Real-time, interactive dashboard for monitoring, prediction, backtesting, and sector/stock comparison.
- **Terminal-Based Model Evaluation**: Run model accuracy evaluation and backtesting directly from the terminal with detailed tabular output, no dashboard required.
- **Data Simulation**: Simulates realistic sector/stock data for robust demo and fallback when real data is missing.

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

## How to Run and Test Everything (Windows)

### 1. Activate the Virtual Environment

```bat
sentiment_env\Scripts\activate
```

### 2. Install Dependencies

```bat
pip install -r requirements.txt
```

### 3. Data Collection

```bat
python fetch_multiple_twitter_accounts.py         # Collect Twitter data
python reddit_scraper.py                          # Collect Reddit data
python news_api_collector.py                      # Collect financial news
python stock_data_collection.py                   # Collect stock price data
```

### 4. Preprocessing and Feature Engineering (if scripts available)

```bat
python src\preprocessing\run_preprocessing.py
python src\preprocessing\run_feature_engineering.py
```

### 5. Run the Dashboard

```bat
python run_dashboard.py
```

- Open your browser and go to: http://localhost:8050

### 6. Terminal-Based Model Accuracy Evaluation

```bat
python evaluate_model_accuracy.py --ticker AAPL --model lstm --days 5 --periods 3 --save-results
python evaluate_model_accuracy.py --ticker MSFT --model random_forest --days 10 --periods 5 --save-results
python evaluate_model_accuracy.py --ticker AAPL --model xgboost --days 7 --periods 4
```

- `--ticker`: Stock symbol (AAPL, MSFT, etc.)
- `--model`: Model type (`lstm`, `random_forest`, `xgboost`)
- `--days`: Prediction horizon in days
- `--periods`: Number of backtest periods
- `--save-results`: Save results to the `results/` directory

### 7. Run Example Analysis Scripts

```bat
python examples\apple_sentiment_analysis.py
python examples\stock_price_prediction_demo.py
python examples\text_processing_sentiment_demo.py
```

### 8. Testing Data Collection and Sentiment Modules

```bat
python test_data_collection.py
python test_reddit_collector.py
python test_scrape_stocktwits.py
python test_sentiment_visualization.py
```

---

## Command Summary Table

| Task                         | Command Example                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------- |
| Collect Twitter data         | `python fetch_multiple_twitter_accounts.py`                                                        |
| Collect Reddit data          | `python reddit_scraper.py`                                                                         |
| Collect news                 | `python news_api_collector.py`                                                                     |
| Collect stock prices         | `python stock_data_collection.py`                                                                  |
| Preprocessing                | `python src\preprocessing\run_preprocessing.py`                                                    |
| Feature engineering          | `python src\preprocessing\run_feature_engineering.py`                                              |
| Run dashboard                | `python run_dashboard.py`                                                                          |
| Model accuracy (terminal)    | `python evaluate_model_accuracy.py --ticker AAPL --model lstm --days 5 --periods 3 --save-results` |
| Example analysis             | `python examples\apple_sentiment_analysis.py`                                                      |
| Test data collection         | `python test_data_collection.py`                                                                   |
| Test Reddit collector        | `python test_reddit_collector.py`                                                                  |
| Test StockTwits              | `python test_scrape_stocktwits.py`                                                                 |
| Test sentiment visualization | `python test_sentiment_visualization.py`                                                           |

---

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- This project uses several open-source libraries and APIs
- Special thanks to contributors and data providers
