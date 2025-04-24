"""
Sector Economic Analysis
Provides analysis of economic factors affecting different sectors through sentiment trends.
Combines sector classification, sentiment analysis, and economic indicators to provide
insights into sector-specific economic trends and factors.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Any, Optional
import logging
import os
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Local imports
from src.sector_analysis.sector_classifier import SectorClassifier
from src.sentiment.sector_sentiment_analyzer import SectorSentimentAnalyzer
from src.sentiment.portfolio_sentiment_scorer import PortfolioSentimentScorer
from src.sentiment.advanced_emotion_sentiment import EnhancedEmotionalAnalyzer
from src.sentiment.fear_greed_index import SentimentIndicator

# Try to import yfinance for sector performance data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    logging.warning("yfinance not available. Some sector analysis features will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class SectorEconomicAnalyzer:
    """
    Analyzes economic factors affecting different sectors through sentiment trends
    """
    def __init__(self, output_dir='results/sector_analysis'):
        """
        Initialize sector economic analyzer
        
        Parameters:
        -----------
        output_dir : str
            Directory to save analysis results
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize analyzers
        self.sector_classifier = SectorClassifier()
        self.sector_sentiment_analyzer = SectorSentimentAnalyzer(
            sector_classifier=self.sector_classifier,
            use_enhanced=True
        )
        
        # Initialize data storage
        self.results = None
        self.time_series_data = {}
        self.sector_correlations = {}
        self.economic_indicators = {}
        
        # Map sectors to ETFs that track them
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial Services': 'XLF',
            'Consumer Cyclical': 'XLY',
            'Energy': 'XLE',
            'Communication Services': 'XLC',
            'Consumer Defensive': 'XLP',
            'Industrials': 'XLI',
            'Basic Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Financial': 'XLF'  # Add alternate name
        }
        
        # Map sectors to major stocks in that sector
        self.sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Consumer Cyclical': ['AMZN', 'HD', 'NKE', 'SBUX', 'MCD'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA'],
            'Financial': ['JPM', 'BAC', 'WFC', 'C', 'GS']  # Add alternate name
        }
    
    def get_sector_performance(self, sector, period='1y'):
        """
        Get performance data for a sector
        
        Parameters:
        -----------
        sector : str
            Sector name to analyze
        period : str
            Time period for analysis (e.g., '1m', '3m', '6m', '1y', '3y')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector performance data
        """
        logger.info(f"Getting performance data for sector: {sector}")
        
        if sector not in self.sector_etfs:
            logger.warning(f"No ETF found for sector: {sector}")
            # Create empty DataFrame with minimum required columns
            return pd.DataFrame({
                'performance': [],
                'ytd_performance': [],
                'top_stock': []
            })
        
        try:
            # Get ETF data if yfinance is available
            if HAS_YFINANCE:
                etf_ticker = self.sector_etfs[sector]
                etf_data = yf.download(etf_ticker, period=period, progress=False)
                
                if etf_data.empty:
                    logger.warning(f"No data available for ETF: {etf_ticker}")
                    return self._create_default_sector_data(sector)
                
                # Calculate performance metrics
                etf_data['performance'] = etf_data['Close'].pct_change(20) * 100  # 20-day rolling return
                
                # Calculate YTD performance
                start_of_year = datetime(datetime.now().year, 1, 1)
                if etf_data.index[0].date() <= start_of_year.date():
                    start_price = etf_data.loc[etf_data.index >= start_of_year, 'Close'].iloc[0]
                    etf_data['ytd_performance'] = (etf_data['Close'] / start_price - 1) * 100
                else:
                    # If data doesn't go back to start of year, calculate from earliest date
                    etf_data['ytd_performance'] = (etf_data['Close'] / etf_data['Close'].iloc[0] - 1) * 100
                
                # Find top performing stock in the sector
                top_stock = self._find_top_performing_stock(sector, period)
                etf_data['top_stock'] = top_stock
                
                return etf_data
            else:
                # If yfinance not available, create simulated data
                logger.warning("yfinance not available, using simulated sector data")
                return self._create_simulated_sector_data(sector)
        
        except Exception as e:
            logger.error(f"Error getting sector performance: {e}")
            return self._create_default_sector_data(sector)
    
    def _find_top_performing_stock(self, sector, period='1y'):
        """Find the top performing stock in a sector"""
        if not HAS_YFINANCE:
            return "Unknown"
            
        sector_stocks = self.sector_stocks.get(sector, [])
        if not sector_stocks:
            return "Unknown"
            
        try:
            # Find top performing stock
            top_stock = None
            max_return = -float('inf')
            
            for stock in sector_stocks:
                try:
                    stock_data = yf.download(stock, period=period, progress=False)
                    if not stock_data.empty:
                        stock_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1) * 100
                        
                        if stock_return > max_return:
                            max_return = stock_return
                            top_stock = stock
                except Exception as e:
                    logger.debug(f"Error fetching data for {stock}: {e}")
            
            return top_stock if top_stock else "Unknown"
            
        except Exception as e:
            logger.error(f"Error finding top performing stock: {e}")
            return "Unknown"
    
    def _create_default_sector_data(self, sector):
        """Create default (empty) sector data"""
        # Create a date range for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create empty DataFrame with minimum required columns
        df = pd.DataFrame(index=date_range)
        df['performance'] = np.nan
        df['ytd_performance'] = np.nan
        df['top_stock'] = "Unknown"
        
        return df
    
    def _create_simulated_sector_data(self, sector):
        """Create simulated sector performance data"""
        # Create a date range for the past year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with simulated data
        df = pd.DataFrame(index=date_range)
        
        # Simulate closing prices based on sector
        # Different sectors have different trends and volatility
        sector_params = {
            'Technology': {'trend': 0.08, 'volatility': 0.015},
            'Healthcare': {'trend': 0.05, 'volatility': 0.010},
            'Financial Services': {'trend': 0.03, 'volatility': 0.012},
            'Consumer Cyclical': {'trend': 0.06, 'volatility': 0.014},
            'Energy': {'trend': 0.04, 'volatility': 0.018},
            'Communication Services': {'trend': 0.07, 'volatility': 0.013},
            'Consumer Defensive': {'trend': 0.02, 'volatility': 0.008},
            'Industrials': {'trend': 0.03, 'volatility': 0.011},
            'Basic Materials': {'trend': 0.02, 'volatility': 0.016},
            'Real Estate': {'trend': 0.01, 'volatility': 0.012},
            'Utilities': {'trend': 0.01, 'volatility': 0.007},
            'Financial': {'trend': 0.03, 'volatility': 0.012}  # Alternate name
        }
        
        params = sector_params.get(sector, {'trend': 0.04, 'volatility': 0.012})
        
        # Generate random price series with trend
        np.random.seed(hash(sector) % 10000)  # Use sector name as seed for reproducibility
        returns = np.random.normal(params['trend']/365, params['volatility'], len(date_range))
        prices = 100 * (1 + returns).cumprod()
        
        df['Close'] = prices
        df['Open'] = prices * (1 + np.random.normal(0, 0.002, len(date_range)))
        df['High'] = np.maximum(df['Close'], df['Open']) * (1 + np.abs(np.random.normal(0, 0.003, len(date_range))))
        df['Low'] = np.minimum(df['Close'], df['Open']) * (1 - np.abs(np.random.normal(0, 0.003, len(date_range))))
        df['Volume'] = np.random.normal(1000000, 200000, len(date_range))
        
        # Calculate performance metrics
        df['performance'] = df['Close'].pct_change(20) * 100  # 20-day rolling return
        df['ytd_performance'] = (df['Close'] / df['Close'][df.index.year == start_date.year][0] - 1) * 100
        df['top_stock'] = self.sector_stocks.get(sector, ["Unknown"])[0]
        
        return df
    
    def get_sector_comparison(self):
        """
        Compare performance across sectors
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sector comparison data
        """
        results = {}
        
        for sector, etf in self.sector_etfs.items():
            try:
                if HAS_YFINANCE:
                    data = yf.download(etf, period='1y', progress=False)
                    if not data.empty:
                        ytd_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                        results[sector] = ytd_return
                else:
                    # Generate random return if yfinance not available
                    np.random.seed(hash(sector) % 10000)  # Use sector name as seed
                    results[sector] = np.random.normal(5, 10)  # Mean 5%, std 10%
            except Exception as e:
                logger.error(f"Error fetching data for {sector} ({etf}): {e}")
        
        return pd.DataFrame(list(results.items()), columns=['Sector', 'YTD Return']).sort_values('YTD Return', ascending=False)

    def load_economic_indicators(self, indicator_data: Dict[str, pd.DataFrame]):
        """
        Load economic indicator data
        
        Parameters:
        -----------
        indicator_data : dict
            Dictionary mapping indicator names to DataFrames with time series data
        """
        self.economic_indicators = indicator_data
        logger.info(f"Loaded {len(indicator_data)} economic indicators")
    
    def analyze_texts_with_dates(self, texts: List[str], dates: List[datetime], 
                               tickers: Optional[List[str]] = None):
        """
        Analyze texts with dates for time series analysis
        
        Parameters:
        -----------
        texts : list of str
            List of financial texts to analyze
        dates : list of datetime
            Corresponding dates for each text
        tickers : list of str, optional
            Corresponding ticker symbols for each text if available
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with analysis results
        """
        if len(texts) != len(dates):
            raise ValueError("Number of texts and dates must match")
        
        # If tickers not provided, use None for all texts
        if tickers is None:
            tickers = [None] * len(texts)
        elif len(tickers) != len(texts):
            raise ValueError("Number of texts and tickers must match")
            
        # Create DataFrame with texts and metadata
        df = pd.DataFrame({
            'text': texts,
            'date': dates,
            'ticker': tickers
        })
        
        # Run sector and sentiment analysis
        analysis_results = self.sector_sentiment_analyzer.analyze_texts(df['text'])
        
        # Add metadata to results
        results = pd.concat([df, analysis_results], axis=1)
        
        # Cache results
        self.results = results
        
        # Create time series data by sector
        self._generate_time_series_data()
        
        return results
    
    def _generate_time_series_data(self):
        """Generate time series data from analysis results"""
        if self.results is None:
            logger.error("No analysis results available")
            return
            
        # Group by date and sector
        grouped = self.results.groupby([
            pd.Grouper(key='date', freq='D'), 
            self.sector_sentiment_analyzer.classification_col
        ])
        
        # Calculate daily sentiment metrics by sector
        daily_metrics = grouped.agg({
            self.sector_sentiment_analyzer.score_col: ['mean', 'count', 'std'],
            self.sector_sentiment_analyzer.sentiment_col: lambda x: (x == 'positive').mean() * 100
        })
        
        # Flatten multi-level columns
        daily_metrics.columns = ['_'.join(col).strip() for col in daily_metrics.columns.values]
        daily_metrics = daily_metrics.reset_index()
        
        # Store time series by sector
        sectors = daily_metrics[self.sector_sentiment_analyzer.classification_col].unique()
        
        for sector in sectors:
            sector_data = daily_metrics[
                daily_metrics[self.sector_sentiment_analyzer.classification_col] == sector
            ].sort_values('date')
            
            # Store in dictionary
            self.time_series_data[sector] = sector_data
            
        logger.info(f"Generated time series data for {len(sectors)} sectors")
    
    def analyze_sector_economic_factors(self, lookback_days=90):
        """
        Analyze economic factors affecting different sectors
        
        Parameters:
        -----------
        lookback_days : int
            Number of days to look back for analysis
            
        Returns:
        --------
        dict
            Dictionary with sector economic analysis results
        """
        if not self.time_series_data:
            logger.error("No time series data available. Run analyze_texts_with_dates first.")
            return {}
            
        # Determine analysis period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        sector_factors = {}
        
        for sector, sector_data in self.time_series_data.items():
            logger.info(f"Analyzing economic factors for sector: {sector}")
            
            # Filter to analysis period
            period_data = sector_data[
                (sector_data['date'] >= start_date) & 
                (sector_data['date'] <= end_date)
            ].copy()
            
            if len(period_data) < 10:
                logger.warning(f"Insufficient data for sector {sector} in specified period")
                continue
                
            # Check for trends in sentiment
            try:
                trend_analysis = self._analyze_sentiment_trends(period_data, sector)
                
                # Correlate with economic indicators
                indicator_correlations = self._correlate_with_economic_indicators(
                    period_data, sector
                )
                
                # Identify economic factors
                economic_factors = self._identify_economic_factors(
                    period_data, indicator_correlations, sector
                )
                
                # Combine results
                sector_factors[sector] = {
                    'trend_analysis': trend_analysis,
                    'indicator_correlations': indicator_correlations,
                    'economic_factors': economic_factors,
                    'data_points': len(period_data)
                }
                
            except Exception as e:
                logger.error(f"Error analyzing economic factors for {sector}: {e}")
        
        # Calculate sector correlations
        self._calculate_sector_correlations()
        
        # Save results
        self._save_economic_analysis(sector_factors)
        
        return sector_factors
    
    def _analyze_sentiment_trends(self, sector_data, sector_name):
        """Analyze sentiment trends for a sector"""
        # Resample to ensure continuous dates
        daily_data = sector_data.set_index('date').resample('D').mean().reset_index()
        daily_data = daily_data.fillna(method='ffill')
        
        # Calculate moving averages
        sentiment_col = f"{self.sector_sentiment_analyzer.score_col}_mean"
        daily_data['ma7'] = daily_data[sentiment_col].rolling(window=7).mean()
        daily_data['ma30'] = daily_data[sentiment_col].rolling(window=30).mean()
        
        # Calculate trend direction and strength
        recent_trend = None
        trend_strength = None
        
        if len(daily_data) >= 30:
            # Linear regression for last 30 days
            recent_data = daily_data.tail(30).reset_index()
            x = np.arange(len(recent_data))
            y = recent_data[sentiment_col].values
            
            # Check for NaNs
            valid_indices = ~np.isnan(y)
            if np.sum(valid_indices) > 10:
                x_valid = x[valid_indices]
                y_valid = y[valid_indices]
                
                coeffs = np.polyfit(x_valid, y_valid, 1)
                slope = coeffs[0]
                
                # Determine trend direction
                if slope > 0.01:
                    recent_trend = "improving"
                elif slope < -0.01:
                    recent_trend = "deteriorating"
                else:
                    recent_trend = "stable"
                    
                # Determine trend strength
                r_squared = np.corrcoef(x_valid, y_valid)[0, 1] ** 2
                if r_squared > 0.5:
                    trend_strength = "strong"
                elif r_squared > 0.2:
                    trend_strength = "moderate"
                else:
                    trend_strength = "weak"
        
        # Calculate volatility
        volatility = daily_data[sentiment_col].std()
        
        # Detect sentiment reversals
        reversals = []
        if len(daily_data) >= 14:
            for i in range(7, len(daily_data)):
                prev_week = daily_data.iloc[i-7:i][sentiment_col].mean()
                current = daily_data.iloc[i][sentiment_col]
                
                # Check for significant reversal
                if (prev_week < 0 and current > 0.2) or (prev_week > 0 and current < -0.2):
                    reversals.append({
                        'date': daily_data.iloc[i]['date'],
                        'from': 'negative' if prev_week < 0 else 'positive',
                        'to': 'positive' if current > 0 else 'negative',
                        'magnitude': abs(current - prev_week)
                    })
        
        # Compile trend analysis results
        trend_analysis = {
            'recent_trend': recent_trend,
            'trend_strength': trend_strength,
            'volatility': float(volatility) if not pd.isna(volatility) else None,
            'current_sentiment': float(daily_data[sentiment_col].iloc[-1]) if not daily_data.empty else None,
            'sentiment_reversals': reversals,
            'last_updated': datetime.now().strftime('%Y-%m-%d')
        }
        
        return trend_analysis
    
    def _correlate_with_economic_indicators(self, sector_data, sector_name):
        """Correlate sector sentiment with economic indicators"""
        correlations = {}
        
        if not self.economic_indicators:
            logger.warning("No economic indicators available for correlation analysis")
            return correlations
            
        # Prepare sector sentiment data
        sector_ts = sector_data.set_index('date')[f"{self.sector_sentiment_analyzer.score_col}_mean"]
        
        # Correlate with each economic indicator
        for indicator_name, indicator_df in self.economic_indicators.items():
            try:
                # Ensure indicator data has date index
                if 'date' in indicator_df.columns:
                    indicator_ts = indicator_df.set_index('date')
                else:
                    indicator_ts = indicator_df
                
                # Get indicator column (assume first numeric column if not specified)
                indicator_col = next((col for col in indicator_ts.columns 
                                     if pd.api.types.is_numeric_dtype(indicator_ts[col])), 
                                    indicator_ts.columns[0])
                
                # Align time series
                aligned_data = pd.merge(
                    sector_ts.reset_index(),
                    indicator_ts[indicator_col].reset_index(),
                    on='date',
                    how='inner'
                )
                
                if len(aligned_data) < 10:
                    logger.debug(f"Insufficient aligned data points for {indicator_name}")
                    continue
                
                # Calculate correlation
                correlation = aligned_data[f"{self.sector_sentiment_analyzer.score_col}_mean"].corr(
                    aligned_data[indicator_col]
                )
                
                correlations[indicator_name] = {
                    'correlation': float(correlation) if not pd.isna(correlation) else None,
                    'data_points': len(aligned_data),
                    'indicator_column': indicator_col
                }
                
            except Exception as e:
                logger.error(f"Error correlating with {indicator_name}: {e}")
        
        return correlations
    
    def _identify_economic_factors(self, sector_data, indicator_correlations, sector_name):
        """Identify key economic factors affecting the sector"""
        # Start with basic factors based on sentiment trends
        sentiment_col = f"{self.sector_sentiment_analyzer.score_col}_mean"
        current_sentiment = sector_data[sentiment_col].iloc[-1] if not sector_data.empty else 0
        
        # Base factors on sentiment
        if current_sentiment > 0.3:
            sentiment_factors = ["Strong positive market sentiment", "Investor optimism"]
        elif current_sentiment > 0:
            sentiment_factors = ["Mild positive sentiment", "Cautious optimism"]
        elif current_sentiment > -0.3:
            sentiment_factors = ["Mild negative sentiment", "Investor caution"]
        else:
            sentiment_factors = ["Significant negative sentiment", "Investor pessimism"]
            
        # Add factors based on indicator correlations
        correlation_factors = []
        if indicator_correlations:
            # Sort by absolute correlation
            sorted_correlations = sorted(
                indicator_correlations.items(),
                key=lambda x: abs(x[1]['correlation']) if x[1]['correlation'] is not None else 0,
                reverse=True
            )
            
            # Take top 3 most correlated indicators
            for indicator_name, info in sorted_correlations[:3]:
                corr = info['correlation']
                if corr is None or abs(corr) < 0.3:
                    continue
                    
                if corr > 0.7:
                    relation = f"strongly positively correlated with {indicator_name}"
                elif corr > 0.4:
                    relation = f"moderately positively correlated with {indicator_name}"
                elif corr > 0.3:
                    relation = f"weakly positively correlated with {indicator_name}"
                elif corr < -0.7:
                    relation = f"strongly negatively correlated with {indicator_name}"
                elif corr < -0.4:
                    relation = f"moderately negatively correlated with {indicator_name}"
                elif corr < -0.3:
                    relation = f"weakly negatively correlated with {indicator_name}"
                else:
                    continue
                    
                correlation_factors.append(f"Sector sentiment is {relation}")
        
        # Add sector-specific factors based on domain knowledge
        sector_specific_factors = self._get_sector_specific_factors(sector_name, current_sentiment)
        
        # Combine all factors
        all_factors = sentiment_factors + correlation_factors + sector_specific_factors
        
        return all_factors
    
    def _get_sector_specific_factors(self, sector_name, current_sentiment):
        """Get sector-specific economic factors based on domain knowledge"""
        sector_factors = {
            'Technology': [
                "Innovation and R&D spending trends",
                "Digital transformation investment",
                "Competitive landscape changes",
                "Consumer technology adoption rates"
            ],
            'Healthcare': [
                "Healthcare policy and regulation changes",
                "Drug pricing pressures",
                "Aging population demographics",
                "Medical innovation and breakthroughs"
            ],
            'Financial': [
                "Interest rate environment",
                "Regulatory compliance costs",
                "Consumer credit health",
                "Capital market activity"
            ],
            'Consumer Discretionary': [
                "Consumer confidence levels",
                "Disposable income trends",
                "E-commerce adoption rates",
                "Retail spending patterns"
            ],
            'Energy': [
                "Global oil and gas price trends",
                "Renewable energy adoption",
                "Environmental regulations",
                "Geopolitical supply disruptions"
            ],
            'Industrial': [
                "Manufacturing activity levels",
                "Supply chain constraints",
                "Infrastructure spending",
                "Trade policy impacts"
            ],
            'Materials': [
                "Commodity price volatility",
                "Construction activity",
                "Global demand for raw materials",
                "Tariff and trade restrictions"
            ],
            'Communication Services': [
                "Digital advertising market growth",
                "Streaming content competition",
                "Telecommunications infrastructure investment",
                "Social media user engagement trends"
            ],
            'Utilities': [
                "Regulatory rate case outcomes",
                "Renewable energy transition costs",
                "Weather pattern impacts",
                "Infrastructure aging and replacement needs"
            ],
            'Real Estate': [
                "Interest rate sensitivity",
                "Commercial occupancy trends",
                "Residential housing market strength",
                "Urban vs. suburban demand shifts"
            ]
        }
        
        # Get base factors for the sector (or use generic if sector not in dictionary)
        base_factors = sector_factors.get(sector_name, [
            "Overall economic activity",
            "Industry-specific competitive dynamics",
            "Regulatory environment changes"
        ])
        
        # Filter or prioritize factors based on sentiment
        if current_sentiment > 0.3:
            # For positive sentiment, emphasize growth factors
            return [f for f in base_factors if 'growth' in f.lower() or 'adoption' in f.lower() 
                   or 'investment' in f.lower()][:2]
        elif current_sentiment < -0.3:
            # For negative sentiment, emphasize risk factors
            return [f for f in base_factors if 'pressure' in f.lower() or 'cost' in f.lower() 
                   or 'regulation' in f.lower()][:2]
        else:
            # For neutral sentiment, return a balanced view
            return base_factors[:2]
    
    def _calculate_sector_correlations(self):
        """Calculate correlations between sector sentiments"""
        if not self.time_series_data:
            logger.error("No time series data available")
            return
            
        # Create a DataFrame with sentiment for each sector
        sentiment_col = f"{self.sector_sentiment_analyzer.score_col}_mean"
        all_sectors_data = {}
        
        for sector, sector_data in self.time_series_data.items():
            all_sectors_data[sector] = sector_data.set_index('date')[sentiment_col]
        
        # Combine into a single DataFrame
        combined_df = pd.DataFrame(all_sectors_data)
        
        # Calculate correlation matrix
        correlation_matrix = combined_df.corr()
        
        # Store correlation matrix
        self.sector_correlations = correlation_matrix
        
        logger.info(f"Calculated correlation matrix for {len(correlation_matrix)} sectors")
    
    def _save_economic_analysis(self, sector_factors):
        """Save economic analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save sector factors as JSON
        factors_filepath = os.path.join(self.output_dir, f"sector_economic_factors_{timestamp}.json")
        
        # Convert any non-serializable objects to strings
        serializable_factors = {}
        for sector, data in sector_factors.items():
            serializable_factors[sector] = {
                'trend_analysis': {
                    k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v)
                    for k, v in data['trend_analysis'].items()
                },
                'indicator_correlations': {
                    indicator: {
                        k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v)
                        for k, v in info.items()
                    }
                    for indicator, info in data['indicator_correlations'].items()
                },
                'economic_factors': data['economic_factors'],
                'data_points': data['data_points']
            }
        
        with open(factors_filepath, 'w') as f:
            json.dump(serializable_factors, f, indent=2)
        
        logger.info(f"Economic factors saved to {factors_filepath}")
        
        # Save sector correlations
        if not self.sector_correlations.empty:
            corr_filepath = os.path.join(self.output_dir, f"sector_correlations_{timestamp}.csv")
            self.sector_correlations.to_csv(corr_filepath)
            logger.info(f"Sector correlations saved to {corr_filepath}")
            
            # Plot heatmap
            self.plot_sector_correlation_heatmap(save_path=os.path.join(
                self.output_dir, f"sector_correlation_heatmap_{timestamp}.png"
            ))
    
    def plot_sentiment_trends_by_sector(self, top_n=5, days=90, figsize=(14, 8), save_path=None):
        """
        Plot sentiment trends by sector
        
        Parameters:
        -----------
        top_n : int
            Number of top sectors to include
        days : int
            Number of days to look back
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.time_series_data:
            logger.error("No time series data available")
            return
            
        # Calculate average sentiment for each sector
        sentiment_col = f"{self.sector_sentiment_analyzer.score_col}_mean"
        sector_avg_sentiment = {}
        
        for sector, sector_data in self.time_series_data.items():
            avg_sentiment = sector_data[sentiment_col].mean()
            total_mentions = sector_data[f"{self.sector_sentiment_analyzer.score_col}_count"].sum()
            sector_avg_sentiment[sector] = (avg_sentiment, total_mentions)
        
        # Sort sectors by average sentiment
        sorted_sectors = sorted(
            sector_avg_sentiment.items(),
            key=lambda x: x[1][0],
            reverse=True
        )
        
        # Select top and bottom sectors (with sufficient mentions)
        selected_sectors = []
        for sector, (sentiment, mentions) in sorted_sectors:
            if mentions >= 20:  # Minimum mentions threshold
                selected_sectors.append(sector)
                if len(selected_sectors) >= top_n:
                    break
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        for sector in selected_sectors:
            sector_data = self.time_series_data[sector]
            
            # Filter to date range
            filtered_data = sector_data[
                (sector_data['date'] >= start_date) & 
                (sector_data['date'] <= end_date)
            ]
            
            # Plot sentiment trend with 7-day moving average
            plt.plot(
                filtered_data['date'],
                filtered_data[sentiment_col].rolling(window=7).mean(),
                label=sector,
                linewidth=2
            )
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.title(f'Sentiment Trends by Sector (Past {days} Days)', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sentiment Score', fontsize=12)
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sector sentiment trends plot saved to {save_path}")
            
        return plt.gcf()
    
    def plot_sector_correlation_heatmap(self, figsize=(12, 10), save_path=None):
        """
        Plot correlation heatmap between sectors
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if self.sector_correlations.empty:
            logger.error("No sector correlations available")
            return
            
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        mask = np.triu(np.ones_like(self.sector_correlations, dtype=bool))
        sns.heatmap(
            self.sector_correlations, 
            annot=True, 
            fmt=".2f", 
            cmap='RdBu_r',
            mask=mask,
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True,
            linewidths=0.5
        )
        
        plt.title('Sector Sentiment Correlations', fontsize=16)
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sector correlation heatmap saved to {save_path}")
            
        return plt.gcf()
    
    def plot_economic_factor_impact(self, sector, figsize=(14, 8), save_path=None):
        """
        Plot impact of economic factors on sector sentiment
        
        Parameters:
        -----------
        sector : str
            Sector to analyze
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        if not self.economic_indicators:
            logger.error("No economic indicators available")
            return
            
        if sector not in self.time_series_data:
            logger.error(f"No data available for sector: {sector}")
            return
            
        # Get sector sentiment data
        sector_data = self.time_series_data[sector]
        sentiment_col = f"{self.sector_sentiment_analyzer.score_col}_mean"
        sector_sentiment = sector_data.set_index('date')[sentiment_col]
        
        # Find indicators with highest correlation
        correlations = {}
        
        for indicator_name, indicator_df in self.economic_indicators.items():
            try:
                # Ensure indicator data has date index
                if 'date' in indicator_df.columns:
                    indicator_ts = indicator_df.set_index('date')
                else:
                    indicator_ts = indicator_df
                
                # Get indicator column
                indicator_col = next((col for col in indicator_ts.columns 
                                     if pd.api.types.is_numeric_dtype(indicator_ts[col])), 
                                    indicator_ts.columns[0])
                
                # Align time series
                aligned_data = pd.merge(
                    sector_sentiment.reset_index(),
                    indicator_ts[indicator_col].reset_index(),
                    on='date',
                    how='inner'
                )
                
                if len(aligned_data) < 10:
                    continue
                
                # Calculate correlation
                correlation = aligned_data[sentiment_col].corr(aligned_data[indicator_col])
                
                correlations[indicator_name] = {
                    'correlation': correlation,
                    'df': aligned_data,
                    'indicator_col': indicator_col
                }
                
            except Exception as e:
                logger.error(f"Error analyzing indicator {indicator_name}: {e}")
        
        # Sort by absolute correlation
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]['correlation']) if not pd.isna(x[1]['correlation']) else 0,
            reverse=True
        )
        
        # Take top 4 indicators with highest correlation
        top_indicators = sorted_correlations[:4]
        
        if not top_indicators:
            logger.error("No indicators with sufficient correlation found")
            return
        
        # Create subplots
        fig, axes = plt.subplots(len(top_indicators), 1, figsize=figsize, sharex=True)
        
        if len(top_indicators) == 1:
            axes = [axes]
        
        for i, (indicator_name, info) in enumerate(top_indicators):
            ax = axes[i]
            df = info['df']
            indicator_col = info['indicator_col']
            correlation = info['correlation']
            
            # Plot sentiment
            color = 'tab:blue'
            ax.set_ylabel('Sentiment', color=color)
            ax.plot(df['date'], df[sentiment_col], color=color, label='Sentiment')
            ax.tick_params(axis='y', labelcolor=color)
            
            # Create second y-axis
            ax2 = ax.twinx()
            color = 'tab:red'
            ax2.set_ylabel(indicator_name, color=color)
            ax2.plot(df['date'], df[indicator_col], color=color, label=indicator_name)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add correlation in title
            ax.set_title(f"{indicator_name} vs. {sector} Sentiment (r = {correlation:.2f})")
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f"Economic Indicators Affecting {sector} Sector", fontsize=16, y=1.02)
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Economic factor impact plot saved to {save_path}")
            
        return plt.gcf()
    
    def generate_sector_economic_report(self, sector=None):
        """
        Generate a comprehensive economic report for a sector or all sectors
        
        Parameters:
        -----------
        sector : str, optional
            Specific sector to generate report for. If None, reports on all sectors.
            
        Returns:
        --------
        dict
            Dictionary with report content
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sectors': {}
        }
        
        if not self.time_series_data:
            logger.error("No sector data available. Run analyze_texts_with_dates first.")
            return report
            
        # Analyze economic factors if not already done
        if not hasattr(self, 'sector_economic_factors'):
            self.sector_economic_factors = self.analyze_sector_economic_factors()
            
        # Generate report for specific sector or all sectors
        sectors_to_report = [sector] if sector else self.time_series_data.keys()
        
        for sector_name in sectors_to_report:
            if sector_name not in self.time_series_data:
                logger.warning(f"No data available for sector: {sector_name}")
                continue
                
            # Get sector data
            sector_data = self.time_series_data[sector_name]
            sentiment_col = f"{self.sector_sentiment_analyzer.score_col}_mean"
            
            # Calculate basic metrics
            recent_data = sector_data.sort_values('date').tail(30)
            
            avg_sentiment = recent_data[sentiment_col].mean() if not recent_data.empty else None
            sentiment_trend = "improving" if avg_sentiment > 0.1 else ("deteriorating" if avg_sentiment < -0.1 else "stable")
            
            # Get economic factors for the sector
            economic_factors = self.sector_economic_factors.get(sector_name, {}).get('economic_factors', [])
            
            # Get correlated sectors
            if not self.sector_correlations.empty and sector_name in self.sector_correlations.index:
                sector_corrs = self.sector_correlations[sector_name].drop(sector_name)
                top_correlated = sector_corrs.nlargest(3).to_dict()
                bottom_correlated = sector_corrs.nsmallest(3).to_dict()
            else:
                top_correlated = {}
                bottom_correlated = {}
            
            # Generate sector report
            sector_report = {
                'current_sentiment': float(avg_sentiment) if avg_sentiment is not None else None,
                'sentiment_trend': sentiment_trend,
                'data_points': len(recent_data),
                'economic_factors': economic_factors,
                'positively_correlated_sectors': top_correlated,
                'negatively_correlated_sectors': bottom_correlated,
                'key_insights': self._generate_sector_insights(sector_name, avg_sentiment, economic_factors)
            }
            
            report['sectors'][sector_name] = sector_report
        
        # Add cross-sector insights
        report['cross_sector_insights'] = self._generate_cross_sector_insights()
        
        # Save report
        report_filepath = os.path.join(
            self.output_dir, 
            f"{'sector_' + sector if sector else 'all_sectors'}_economic_report_{timestamp}.json"
        )
        
        with open(report_filepath, 'w') as f:
            # Convert any non-serializable objects
            report_json = json.dumps(report, default=str, indent=2)
            f.write(report_json)
            
        logger.info(f"Economic report saved to {report_filepath}")
        
        return report
    
    def _generate_sector_insights(self, sector, sentiment, economic_factors):
        """Generate key insights for a sector"""
        insights = []
        
        # Sentiment-based insights
        if sentiment is not None:
            if sentiment > 0.3:
                insights.append(f"The {sector} sector is showing strong positive sentiment, suggesting favorable economic conditions.")
            elif sentiment > 0.1:
                insights.append(f"The {sector} sector is showing moderate positive sentiment, indicating relatively stable economic conditions.")
            elif sentiment > -0.1:
                insights.append(f"The {sector} sector is showing neutral sentiment, suggesting mixed economic signals.")
            elif sentiment > -0.3:
                insights.append(f"The {sector} sector is showing moderate negative sentiment, indicating economic headwinds.")
            else:
                insights.append(f"The {sector} sector is showing strong negative sentiment, suggesting significant economic challenges.")
        
        # Economic factor insights
        if economic_factors:
            insights.append(f"Key economic factors affecting the {sector} sector include: {', '.join(economic_factors[:3])}")
        
        # Sector-specific insights based on domain knowledge
        sector_specific_insights = {
            'Technology': [
                "Technology sector performance is often tied to innovation cycles, R&D spending, and adoption of new technologies.",
                "Changes in interest rates can significantly impact technology company valuations, especially for growth-oriented firms."
            ],
            'Healthcare': [
                "Healthcare sentiment is heavily influenced by regulatory developments, policy changes, and drug pricing pressures.",
                "Aging demographics and increased focus on preventative care represent long-term tailwinds for the sector."
            ],
            'Financial': [
                "Financial sector performance is closely tied to interest rate environments, yield curves, and credit quality trends.",
                "Regulatory changes and compliance costs remain significant factors affecting financial institution profitability."
            ],
            'Energy': [
                "Energy sector sentiment is driven by global supply-demand dynamics, geopolitical factors, and the pace of renewable energy transition.",
                "Commodity price volatility and environmental regulations are key factors affecting future capital investment decisions."
            ],
            'Consumer Discretionary': [
                "Consumer spending trends, inflation impacts on disposable income, and shifts in consumer preferences drive sector performance.",
                "E-commerce penetration and omnichannel capabilities are increasingly important differentiators within the sector."
            ],
            'Industrial': [
                "The industrial sector is sensitive to global manufacturing activity, supply chain dynamics, and infrastructure spending.",
                "Labor costs, material inflation, and automation trends are key factors affecting long-term sector profitability."
            ]
        }
        
        if sector in sector_specific_insights:
            insights.extend(sector_specific_insights[sector])
        
        return insights
    
    def _generate_cross_sector_insights(self):
        """Generate insights across sectors"""
        insights = []
        
        # Calculate average sentiment across all sectors
        all_sentiment = []
        for sector_data in self.time_series_data.values():
            sentiment_col = f"{self.sector_sentiment_analyzer.score_col}_mean"
            recent_data = sector_data.sort_values('date').tail(30)
            if not recent_data.empty:
                all_sentiment.append(recent_data[sentiment_col].mean())
        
        if all_sentiment:
            avg_sentiment = sum(all_sentiment) / len(all_sentiment)
            
            # Overall market sentiment insight
            if avg_sentiment > 0.2:
                insights.append("Overall market sentiment is strongly positive across sectors, suggesting broad-based economic strength.")
            elif avg_sentiment > 0.1:
                insights.append("Overall market sentiment is moderately positive, indicating economic expansion with some sector-specific challenges.")
            elif avg_sentiment > -0.1:
                insights.append("Market sentiment is neutral to slightly positive, suggesting steady but uneven economic conditions across sectors.")
            elif avg_sentiment > -0.2:
                insights.append("Market sentiment is moderately negative, indicating economic headwinds across multiple sectors.")
            else:
                insights.append("Market sentiment is strongly negative across sectors, suggesting significant economic challenges.")
        
        # Sector divergence analysis
        if len(all_sentiment) > 1:
            sentiment_variance = np.var(all_sentiment)
            
            if sentiment_variance > 0.1:
                insights.append("High variance in sector sentiment indicates significant economic divergence, with some sectors outperforming while others face challenges.")
            elif sentiment_variance > 0.05:
                insights.append("Moderate variance in sector sentiment suggests uneven economic conditions across industries.")
            else:
                insights.append("Low variance in sector sentiment indicates relatively uniform economic conditions across sectors.")
        
        # Correlation analysis
        if not self.sector_correlations.empty:
            avg_correlation = self.sector_correlations.values[np.triu_indices_from(self.sector_correlations.values, 1)].mean()
            
            if avg_correlation > 0.7:
                insights.append("Very high inter-sector correlations suggest systematic market-wide factors are dominating sector-specific influences.")
            elif avg_correlation > 0.5:
                insights.append("High inter-sector correlations indicate common economic factors affecting most sectors simultaneously.")
            elif avg_correlation > 0.3:
                insights.append("Moderate inter-sector correlations suggest a balance between market-wide factors and sector-specific dynamics.")
            else:
                insights.append("Low inter-sector correlations indicate sector-specific factors are dominating broader market trends.")
        
        # Add some general economic observations
        general_insights = [
            "Inflation expectations, monetary policy, and interest rate environments remain key cross-sector economic drivers.",
            "Supply chain dynamics, labor market trends, and geopolitical factors continue to affect sentiment across multiple sectors.",
            "Market liquidity and risk appetite are important factors influencing relative sector performance."
        ]
        
        insights.extend(general_insights)
        
        return insights


# Example usage
if __name__ == "__main__":
    # Sample texts with dates and tickers
    sample_texts = [
        "The technology sector is showing strong growth despite recent market volatility.",
        "Financial stocks are under pressure due to concerns about loan defaults.",
        "Healthcare companies reported better than expected earnings this quarter.",
        "Energy prices continue to rise, benefiting oil and gas producers.",
        "Retail sales declined unexpectedly, putting pressure on consumer discretionary stocks."
    ]
    
    sample_dates = [
        datetime.now() - timedelta(days=5),
        datetime.now() - timedelta(days=4),
        datetime.now() - timedelta(days=3),
        datetime.now() - timedelta(days=2),
        datetime.now() - timedelta(days=1)
    ]
    
    sample_tickers = [
        "AAPL",
        "JPM",
        "JNJ",
        "XOM",
        "AMZN"
    ]
    
    # Sample economic indicators
    sample_indicators = {
        "Interest Rates": pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)],
            "value": np.linspace(3.5, 4.5, 30) + np.random.normal(0, 0.1, 30)
        }),
        "Inflation": pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)],
            "CPI": np.linspace(4.2, 3.8, 30) + np.random.normal(0, 0.05, 30)
        }),
        "Unemployment": pd.DataFrame({
            "date": [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)],
            "rate": np.linspace(3.8, 3.6, 30) + np.random.normal(0, 0.02, 30)
        })
    }
    
    # Initialize analyzer
    analyzer = SectorEconomicAnalyzer()
    
    # Load economic indicators
    analyzer.load_economic_indicators(sample_indicators)
    
    # Analyze texts
    results = analyzer.analyze_texts_with_dates(sample_texts, sample_dates, sample_tickers)
    print("\nSector Analysis Results:")
    print(results[['text', 'ticker', 'sector_classification', 'sentiment_label', 'sentiment_score']].head())
    
    # Analyze sector economic factors
    factors = analyzer.analyze_sector_economic_factors(lookback_days=30)
    print("\nSector Economic Factors:")
    for sector, info in factors.items():
        print(f"\n{sector}:")
        print(f"Trend: {info['trend_analysis']['recent_trend']} ({info['trend_analysis']['trend_strength']})")
        print(f"Economic Factors: {', '.join(info['economic_factors'][:3])}")
    
    # Generate sector economic report
    report = analyzer.generate_sector_economic_report()
    print("\nGenerated Economic Report")
    
    # Plot sector sentiment trends
    analyzer.plot_sentiment_trends_by_sector(save_path="sector_sentiment_trends.png")
    print("\nPlotted sector sentiment trends")
    
    # Plot sector correlation heatmap
    analyzer.plot_sector_correlation_heatmap(save_path="sector_correlation_heatmap.png")
    print("\nPlotted sector correlation heatmap")