"""
Stock Dashboard Application
An interactive dashboard for stock data visualization and analysis
"""
import os
import sys
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc  # Using version 1.5.0
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# Import local modules
from dashboard.data_loader import StockDataLoader
from dashboard.visualization import StockVisualizer

# Try importing potentially problematic modules with fallbacks
try:
    from src.predictive_models.price_predictor import StockPricePredictor
    HAS_PREDICTOR = True
except Exception as e:
    logger.warning(f"Could not import StockPricePredictor: {e}")
    HAS_PREDICTOR = False

try:
    from src.sector_analysis.sector_economic_analyzer import SectorEconomicAnalyzer
    HAS_SECTOR_ANALYZER = True
except Exception as e:
    logger.warning(f"Could not import SectorEconomicAnalyzer: {e}")
    HAS_SECTOR_ANALYZER = False

try:
    from src.sentiment.sector_sentiment_analyzer import SectorSentimentAnalyzer
    HAS_SENTIMENT_ANALYZER = True
except Exception as e:
    logger.warning(f"Could not import SectorSentimentAnalyzer: {e}")
    HAS_SENTIMENT_ANALYZER = False
    
try:
    from src.sentiment.portfolio_sentiment_scorer import PortfolioSentimentScorer
    HAS_PORTFOLIO_SCORER = True
except Exception as e:
    logger.warning(f"Could not import PortfolioSentimentScorer: {e}")
    HAS_PORTFOLIO_SCORER = False

# Initialize components
data_loader = StockDataLoader(
    data_directory="data/historical",
    news_directory="data/news"
)
visualizer = StockVisualizer()

# Initialize optional components
if HAS_PREDICTOR:
    predictor = StockPricePredictor(
        data_dir='data',
        models_dir='models',
        results_dir='results'
    )
else:
    predictor = None

# Get available stocks
available_stocks = data_loader.available_stocks

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)
server = app.server
app.title = "Stock Analytics Dashboard"

# Define app layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Stock Analytics Dashboard", className="text-center p-3 mb-2 bg-primary text-white")
        ], width=12)
    ]),
    
    # Stock selection and date range row
    dbc.Row([
        # Stock selection dropdown
        dbc.Col([
            html.Label("Select Stock"),
            dcc.Dropdown(
                id="stock-selector",
                options=[{'label': ticker, 'value': ticker} for ticker in available_stocks],
                value=available_stocks[0] if available_stocks else None,
                clearable=False
            )
        ], width=3),
        
        # Date range selector
        dbc.Col([
            html.Label("Select Date Range"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=datetime(2010, 1, 1),
                max_date_allowed=datetime.now(),
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now(),
                display_format='YYYY-MM-DD'
            )
        ], width=5),
        
        # Comparison stock selector
        dbc.Col([
            html.Label("Compare With (Optional)"),
            dcc.Dropdown(
                id="comparison-selector",
                options=[{'label': ticker, 'value': ticker} for ticker in available_stocks],
                value=None,
                multi=True
            )
        ], width=4)
    ], className="mb-4"),
    
    # Tabs for different visualizations
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                # Price Tab with improved UI
                dbc.Tab(label="Price Chart", tab_id="price-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Price Chart Settings", className="mt-2 mb-3 text-primary"),
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Chart Type", className="mb-2"),
                                        dbc.Switch(
                                            id="candlestick-toggle",
                                            label="Show Candlestick Chart",
                                            value=False,
                                            className="mt-2 custom-switch"
                                        ),
                                    ])
                                ], className="mb-3"),
                                
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Technical Indicators", className="mb-2"),
                                        dbc.Switch(
                                            id="indicators-toggle",
                                            label="Show Moving Averages",
                                            value=True,
                                            className="mt-2 custom-switch"
                                        ),
                                        dbc.Switch(
                                            id="bollinger-toggle",
                                            label="Show Bollinger Bands",
                                            value=False,
                                            className="mt-2 custom-switch"
                                        ),
                                    ])
                                ], className="mb-3"),
                                
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Volume", className="mb-2"),
                                        dbc.Switch(
                                            id="volume-toggle",
                                            label="Show Volume Chart",
                                            value=True,
                                            className="mt-2 custom-switch"
                                        ),
                                    ])
                                ], className="mb-3"),
                                
                                dbc.Card([
                                    dbc.CardBody([
                                        html.H6("Display Period", className="mb-2"),
                                        dbc.RadioItems(
                                            id="timeframe-selector",
                                            options=[
                                                {"label": "1 Month", "value": 30},
                                                {"label": "3 Months", "value": 90},
                                                {"label": "6 Months", "value": 180},
                                                {"label": "1 Year", "value": 365},
                                                {"label": "All Data", "value": "all"},
                                            ],
                                            value=90,
                                            inline=True,
                                            className="mt-2"
                                        ),
                                    ])
                                ], className="mb-3"),
                                
                                html.Div([
                                    dbc.Button(
                                        "Apply Settings",
                                        id="apply-settings-button",
                                        color="primary",
                                        className="w-100"
                                    ),
                                ], className="d-grid gap-2"),
                                
                            ], className="p-3 border rounded bg-light")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H5("AAPL Key Statistics", id="ticker-stats-title", className="mt-2 mb-3 text-primary"),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H6("Current Price", className="card-subtitle text-muted mb-1"),
                                                html.H4(id="current-price", className="card-title text-primary mb-0"),
                                            ])
                                        ])
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H6("Daily Change", className="card-subtitle text-muted mb-1"),
                                                html.H4(id="daily-change", className="card-title mb-0"),
                                            ])
                                        ])
                                    ], width=6),
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H6("Volume", className="card-subtitle text-muted mb-1"),
                                                html.Div(id="volume-value", className="card-text"),
                                            ])
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H6("Daily Range", className="card-subtitle text-muted mb-1"),
                                                html.Div(id="daily-range", className="card-text"),
                                            ])
                                        ])
                                    ], width=4),
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H6("Period Range", className="card-subtitle text-muted mb-1"),
                                                html.Div(id="period-range", className="card-text"),
                                            ])
                                        ])
                                    ], width=4),
                                ], className="mb-3"),
                                
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Card([
                                            dbc.CardBody([
                                                html.H6("Additional Stats", className="card-subtitle text-muted mb-1"),
                                                html.Div(id="key-stats", className="card-text"),
                                            ])
                                        ])
                                    ], width=12),
                                ]),
                            ], className="p-3 border rounded bg-light")
                        ], width=9)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Loading(
                                id="loading-price-chart",
                                type="circle",
                                children=[
                                    dcc.Graph(id="price-chart", style={"height": "70vh"})
                                ]
                            )
                        ], width=12)
                    ], className="mt-3")
                ]),
                
                # Comparison Tab
                dbc.Tab(label="Comparison", tab_id="comparison-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Comparison Settings", className="mt-2"),
                                dbc.Switch(
                                    id="normalize-toggle",
                                    label="Normalize Prices",
                                    value=True,
                                    className="mt-2"
                                ),
                                html.Hr(),
                                html.H6("Selected Stocks:"),
                                html.Div(id="comparison-stocks-list")
                            ], className="p-3 border rounded")
                        ], width=3),
                        dbc.Col([
                            dcc.Graph(id="comparison-chart", style={"height": "50vh"})
                        ], width=9)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="correlation-heatmap", style={"height": "40vh"})
                        ], width=12)
                    ], className="mt-3")
                ]),
                
                # Analysis Tab
                dbc.Tab(label="Analysis", tab_id="analysis-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Analysis Type", className="mt-2"),
                                dbc.RadioItems(
                                    id="analysis-type",
                                    options=[
                                        {"label": "Returns", "value": "returns"},
                                        {"label": "Volatility", "value": "volatility"}
                                    ],
                                    value="returns",
                                    inline=True,
                                    className="mt-2"
                                )
                            ], className="p-3 border rounded")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H5("Analysis Statistics", className="mt-2"),
                                html.Div(id="analysis-stats")
                            ], className="p-3 border rounded")
                        ], width=9)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="analysis-chart", style={"height": "60vh"})
                        ], width=12)
                    ], className="mt-3")
                ]),
                
                # Prediction Tab
                dbc.Tab(label="Predictions", tab_id="prediction-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Prediction Settings", className="mt-2"),
                                html.Label("Days to Predict", className="mt-2"),
                                dcc.Slider(
                                    id="prediction-days",
                                    min=1,
                                    max=30,
                                    step=1,
                                    value=10,
                                    marks={i: str(i) for i in range(0, 31, 5)},
                                ),
                                html.Label("Model Type", className="mt-3"),
                                dbc.RadioItems(
                                    id="prediction-model",
                                    options=[
                                        {"label": "LSTM", "value": "lstm"},
                                        {"label": "Random Forest", "value": "random_forest"},
                                        {"label": "XGBoost", "value": "xgboost"}
                                    ],
                                    value="lstm",
                                    inline=True,
                                    className="mt-2"
                                ),
                                dbc.Button(
                                    "Generate Prediction", 
                                    id="prediction-button", 
                                    color="primary", 
                                    className="mt-3"
                                ),
                                html.Div(id="prediction-status", className="mt-2")
                            ], className="p-3 border rounded")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H5("Prediction Results", className="mt-2"),
                                html.Div(id="prediction-stats")
                            ], className="p-3 border rounded")
                        ], width=9)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="prediction-chart", style={"height": "60vh"})
                        ], width=12)
                    ], className="mt-3")
                ]),
                
                # News Tab
                dbc.Tab(label="News", tab_id="news-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("News Feed", className="mt-2"),
                                html.Div(id="news-feed", style={"maxHeight": "80vh", "overflow": "auto"})
                            ], className="p-3 border rounded")
                        ], width=12)
                    ], className="mt-3")
                ]),
                
                # Sector Analysis Tab
                dbc.Tab(label="Sector Analysis", tab_id="sector-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Sector Analysis Settings", className="mt-2"),
                                html.Label("Select Sector"),
                                dcc.Dropdown(
                                    id="sector-selector",
                                    options=[
                                        {"label": "Technology", "value": "Technology"},
                                        {"label": "Healthcare", "value": "Healthcare"},
                                        {"label": "Financial Services", "value": "Financial Services"},
                                        {"label": "Consumer Cyclical", "value": "Consumer Cyclical"},
                                        {"label": "Energy", "value": "Energy"},
                                        {"label": "Communication Services", "value": "Communication Services"},
                                        {"label": "Consumer Defensive", "value": "Consumer Defensive"},
                                        {"label": "Industrials", "value": "Industrials"},
                                        {"label": "Basic Materials", "value": "Basic Materials"},
                                        {"label": "Real Estate", "value": "Real Estate"},
                                        {"label": "Utilities", "value": "Utilities"}
                                    ],
                                    value="Technology",
                                    clearable=False
                                ),
                                html.Button("Analyze Sector", id="sector-analyze-button", className="btn btn-primary mt-3"),
                                html.Div(id="sector-status", className="mt-2"),
                                html.Hr(),
                                html.H6("Analysis Type"),
                                dbc.RadioItems(
                                    id="sector-analysis-type",
                                    options=[
                                        {"label": "Performance", "value": "performance"},
                                        {"label": "Comparison", "value": "comparison"},
                                        {"label": "Sentiment", "value": "sentiment"}
                                    ],
                                    value="performance",
                                    inline=True,
                                    className="mt-2"
                                )
                            ], className="p-3 border rounded")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H5("Sector Performance", className="mt-2"),
                                html.Div(id="sector-performance-stats")
                            ], className="p-3 border rounded")
                        ], width=9)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="sector-chart", style={"height": "60vh"})
                        ], width=12)
                    ], className="mt-3")
                ]),
                
                # Portfolio Analysis Tab
                dbc.Tab(label="Portfolio", tab_id="portfolio-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Portfolio Settings", className="mt-2"),
                                html.Label("Select Stocks for Portfolio"),
                                dcc.Dropdown(
                                    id="portfolio-stocks",
                                    options=[{'label': ticker, 'value': ticker} for ticker in available_stocks],
                                    value=[available_stocks[0]] if available_stocks else [],
                                    multi=True
                                ),
                                html.Hr(),
                                html.Label("Analysis Period"),
                                dcc.DatePickerRange(
                                    id='portfolio-date-range',
                                    min_date_allowed=datetime(2010, 1, 1),
                                    max_date_allowed=datetime.now(),
                                    start_date=datetime.now() - timedelta(days=365),
                                    end_date=datetime.now(),
                                    display_format='YYYY-MM-DD'
                                ),
                                html.Button("Analyze Portfolio", id="portfolio-analyze-button", className="btn btn-primary mt-3"),
                                html.Div(id="portfolio-status", className="mt-2")
                            ], className="p-3 border rounded")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H5("Portfolio Analytics", className="mt-2"),
                                html.Div(id="portfolio-stats")
                            ], className="p-3 border rounded")
                        ], width=9)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="portfolio-performance-chart", style={"height": "40vh"})
                        ], width=12)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="portfolio-composition-chart", style={"height": "40vh"})
                        ], width=12)
                    ], className="mt-3")
                ]),

                # Model Accuracy Tab
                dbc.Tab(label="Model Accuracy", tab_id="accuracy-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H5("Model Accuracy Settings", className="mt-2"),
                                html.Label("Select Model to Evaluate"),
                                dbc.RadioItems(
                                    id="accuracy-model",
                                    options=[
                                        {"label": "LSTM", "value": "lstm"},
                                        {"label": "Random Forest", "value": "random_forest"},
                                        {"label": "XGBoost", "value": "xgboost"}
                                    ],
                                    value="lstm",
                                    inline=True,
                                    className="mt-2"
                                ),
                                html.Label("Prediction Horizon (Days)", className="mt-3"),
                                dcc.Slider(
                                    id="accuracy-days",
                                    min=1,
                                    max=30,
                                    step=1,
                                    value=5,
                                    marks={i: str(i) for i in range(0, 31, 5)},
                                ),
                                html.Label("Number of Backtest Periods", className="mt-3"),
                                dcc.Slider(
                                    id="backtest-periods",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=3,
                                    marks={i: str(i) for i in range(1, 11)},
                                ),
                                dbc.Button(
                                    "Evaluate Model Accuracy", 
                                    id="accuracy-button", 
                                    color="primary", 
                                    className="mt-3"
                                ),
                                html.Div(id="accuracy-status", className="mt-2")
                            ], className="p-3 border rounded")
                        ], width=3),
                        dbc.Col([
                            html.Div([
                                html.H5("Accuracy Metrics", className="mt-2"),
                                html.Div(id="accuracy-metrics", className="mt-3")
                            ], className="p-3 border rounded")
                        ], width=9)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id="accuracy-chart", style={"height": "60vh"})
                        ], width=12)
                    ], className="mt-3")
                ]),
            ], id="tabs", active_tab="price-tab")
        ], width=12)
    ]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Stock Analytics Dashboard - Data as of " + datetime.now().strftime("%Y-%m-%d"),
                  className="text-center text-muted")
        ], width=12)
    ])
], fluid=True)

# Utility function for date parsing
def parse_date_string(date_str):
    """
    Parse date string that could be in various formats including ISO-8601
    
    Parameters:
    -----------
    date_str : str
        Date string to parse
        
    Returns:
    --------
    datetime
        Parsed datetime object
    """
    if not isinstance(date_str, str):
        return date_str
        
    try:
        # Try simple format first
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        try:
            # Try ISO format with T separator
            if 'T' in date_str:
                # Remove timezone info if present
                date_str = date_str.split('+')[0].split('Z')[0]
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            try:
                # Try without seconds
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
            except ValueError:
                pass
    
    # Log the problematic date string
    logger.warning(f"Could not parse date string: {date_str}")
    return date_str

# Callbacks

# Update date range when stock is selected
@app.callback(
    [Output('date-range', 'min_date_allowed'),
     Output('date-range', 'max_date_allowed'),
     Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    [Input('stock-selector', 'value')]
)
def update_date_range(ticker):
    if not ticker:
        today = datetime.now()
        return datetime(2010, 1, 1), today, today - timedelta(days=365), today
    
    # Load data and get date range
    data_loader.load_stock_data(ticker)
    start_date, end_date = data_loader.get_date_range(ticker)
    
    # Set default display range to last year
    display_start = max(end_date - timedelta(days=365), start_date)
    
    return start_date, end_date, display_start, end_date

# Update price chart
@app.callback(
    [Output('price-chart', 'figure'),
     Output('key-stats', 'children')],
    [Input('stock-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('candlestick-toggle', 'value'),
     Input('indicators-toggle', 'value'),
     Input('bollinger-toggle', 'value'),
     Input('volume-toggle', 'value')]
)
def update_price_chart(ticker, start_date, end_date, show_candlestick, show_indicators, show_bollinger, show_volume):
    if not ticker or not start_date or not end_date:
        return go.Figure(), "No data available"
    
    # Convert string dates to datetime
    start_date = parse_date_string(start_date)
    end_date = parse_date_string(end_date)
    
    # Load data
    df = data_loader.load_stock_data(ticker)
    if df is None:
        return go.Figure(), "Error loading data"
    
    # Filter date range
    df = data_loader.filter_date_range(ticker, start_date, end_date)
    
    # Create chart
    if show_candlestick:
        fig = visualizer.create_candlestick_chart(
            df, 
            title=f"{ticker} Stock Price", 
            include_volume=show_volume,
            include_indicators=show_indicators,
            include_bollinger=show_bollinger
        )
    else:
        fig = visualizer.create_price_chart(
            df, 
            title=f"{ticker} Stock Price",
            include_indicators=show_indicators
        )
    
    # Create key stats display
    if df is not None and len(df) > 0:
        latest_data = df.iloc[-1]
        change = latest_data['Close'] - df.iloc[-2]['Close'] if len(df) > 1 else 0
        pct_change = (change / df.iloc[-2]['Close'] * 100) if len(df) > 1 else 0
        
        stats = [
            html.H6(f"Price: ${latest_data['Close']:.2f}"),
            html.P([
                f"Change: ",
                html.Span(
                    f"${change:.2f} ({pct_change:.2f}%)",
                    style={'color': 'green' if change >= 0 else 'red'}
                )
            ]),
            html.P(f"Volume: {latest_data['Volume']:,}" if 'Volume' in latest_data else ""),
            html.Hr(),
            html.P(f"Range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}"),
            html.P(f"Avg Volume: {df['Volume'].mean():,.0f}" if 'Volume' in df else ""),
            html.P(f"Days Shown: {len(df)}")
        ]
    else:
        stats = html.P("No data available")
    
    return fig, stats

# Update ticker statistics title
@app.callback(
    Output('ticker-stats-title', 'children'),
    [Input('stock-selector', 'value')]
)
def update_ticker_title(ticker):
    if ticker:
        return f"{ticker} Key Statistics"
    return "Key Statistics"

# Update detailed price chart statistics
@app.callback(
    [Output('current-price', 'children'),
     Output('daily-change', 'children'),
     Output('daily-change', 'style'),
     Output('volume-value', 'children'),
     Output('daily-range', 'children'),
     Output('period-range', 'children')],
    [Input('stock-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_price_stats(ticker, start_date, end_date):
    if not ticker or not start_date or not end_date:
        return "$0.00", "0.00 (0.00%)", {'color': 'black'}, "0", "$0.00 - $0.00", "$0.00 - $0.00"
    
    # Convert string dates to datetime
    start_date = parse_date_string(start_date)
    end_date = parse_date_string(end_date)
    
    # Load data
    df = data_loader.load_stock_data(ticker)
    if df is None or df.empty:
        return "$0.00", "0.00 (0.00%)", {'color': 'black'}, "0", "$0.00 - $0.00", "$0.00 - $0.00"
    
    # Filter date range
    df = data_loader.filter_date_range(ticker, start_date, end_date)
    
    if df is None or df.empty:
        return "$0.00", "0.00 (0.00%)", {'color': 'black'}, "0", "$0.00 - $0.00", "$0.00 - $0.00"
    
    # Get latest data
    latest_data = df.iloc[-1]
    
    # Calculate price and change
    current_price = f"${latest_data['Close']:.2f}"
    
    if len(df) > 1:
        prev_close = df.iloc[-2]['Close']
        change = latest_data['Close'] - prev_close
        pct_change = (change / prev_close) * 100
        change_text = f"${change:.2f} ({pct_change:.2f}%)"
        change_style = {'color': 'green' if change >= 0 else 'red', 'fontWeight': 'bold'}
    else:
        change_text = "$0.00 (0.00%)"
        change_style = {'color': 'black'}
    
    # Calculate volume
    if 'Volume' in latest_data:
        volume_text = f"{latest_data['Volume']:,.0f}"
    else:
        volume_text = "N/A"
    
    # Calculate daily range (today's high/low)
    if 'High' in latest_data and 'Low' in latest_data:
        daily_range = f"${latest_data['Low']:.2f} - ${latest_data['High']:.2f}"
    else:
        daily_range = "N/A"
    
    # Calculate period range
    if 'High' in df.columns and 'Low' in df.columns:
        period_low = df['Low'].min()
        period_high = df['High'].max()
        period_range = f"${period_low:.2f} - ${period_high:.2f}"
    else:
        period_range = "N/A"
    
    return current_price, change_text, change_style, volume_text, daily_range, period_range

# Add callback for timeframe selector
@app.callback(
    [Output('date-range', 'start_date'),
     Output('date-range', 'end_date')],
    [Input('timeframe-selector', 'value'),
     Input('apply-settings-button', 'n_clicks')],
    [State('stock-selector', 'value'),
     State('date-range', 'min_date_allowed'),
     State('date-range', 'max_date_allowed')]
)
def update_timeframe(days_value, n_clicks, ticker, min_date, max_date):
    # Only trigger on button click
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] != 'apply-settings-button.n_clicks' or n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    if not ticker:
        raise dash.exceptions.PreventUpdate
    
    # Get current end date (usually today)
    end_date = parse_date_string(max_date)
    
    # If "all data" is selected
    if days_value == "all":
        start_date = parse_date_string(min_date)
        return start_date, end_date
    
    # Calculate start date based on selected period
    days = int(days_value)
    start_date = end_date - timedelta(days=days)
    
    # Make sure start date is not before min_date
    min_date = parse_date_string(min_date)
    if start_date < min_date:
        start_date = min_date
    
    return start_date, end_date

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)