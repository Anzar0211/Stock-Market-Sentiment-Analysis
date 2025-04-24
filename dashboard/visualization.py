"""
Visualization utilities for Stock Dashboard
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class StockVisualizer:
    """
    Class for creating interactive stock visualizations
    """
    def __init__(self):
        """Initialize the stock visualizer"""
        # Define enhanced color schemes
        self.colors = {
            'price': '#1f77b4',  # Blue
            'volume': '#2ca02c',  # Green
            'returns': '#ff7f0e',  # Orange
            'up': '#26a69a',  # Teal
            'down': '#ef5350',  # Red
            'sma_50': '#7986cb',  # Blue-purple
            'sma_200': '#5c6bc0',  # Deeper blue-purple
            'ema_20': '#f06292',  # Pink
            'bb_upper': '#9575cd',  # Purple
            'bb_middle': '#7e57c2',  # Deeper purple
            'bb_lower': '#9575cd',  # Purple
            'prediction': '#e040fb',  # Pink
            'background': '#f8f9fa',
            'grid': '#e6e9ec',
            'macd_line': '#26a69a',  # Teal
            'signal_line': '#ef5350',  # Red
            'histogram_up': '#26a69a',  # Teal
            'histogram_down': '#ef5350'  # Red
        }
        
        # Default chart layout with improved styling
        self.default_layout = {
            'template': 'plotly_white',
            'plot_bgcolor': self.colors['background'],
            'paper_bgcolor': self.colors['background'],
            'xaxis': {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            },
            'yaxis': {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            },
            'legend': {
                'orientation': 'h',
                'y': 1.02,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 12}
            },
            'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40},
            'hoverlabel': {'font_size': 12},
            'font': {'family': 'Arial, sans-serif'}
        }
    
    def create_candlestick_chart(self, df, title=None, include_volume=True, 
                                include_indicators=True, include_bollinger=False):
        """
        Create an enhanced interactive candlestick chart
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data
        title : str, optional
            Chart title
        include_volume : bool
            Whether to include volume subplot
        include_indicators : bool
            Whether to include technical indicators
        include_bollinger : bool
            Whether to include Bollinger Bands
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"DataFrame missing required columns: {required_cols}")
            return None
        
        # Prepare technical indicators if they don't exist
        if include_indicators and 'SMA_50' not in df.columns:
            try:
                # Calculate SMA indicators
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # Add EMA indicator
                df['EMA_20'] = df['Close'].ewm(span=20).mean()
            except Exception as e:
                logger.warning(f"Could not calculate technical indicators: {e}")
        
        # Calculate Bollinger Bands if they don't exist
        if include_bollinger and ('BB_upper' not in df.columns):
            try:
                # Calculate 20-day SMA and standard deviation
                sma_20 = df['Close'].rolling(window=20).mean()
                std_20 = df['Close'].rolling(window=20).std()
                
                # Calculate Bollinger Bands (20-day, 2 standard deviations)
                df['BB_upper'] = sma_20 + (std_20 * 2)
                df['BB_middle'] = sma_20
                df['BB_lower'] = sma_20 - (std_20 * 2)
            except Exception as e:
                logger.warning(f"Could not calculate Bollinger Bands: {e}")
        
        # Calculate MACD for additional indicators
        if include_indicators and 'MACD_line' not in df.columns:
            try:
                # Calculate MACD components
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                df['MACD_line'] = ema_12 - ema_26
                df['Signal_line'] = df['MACD_line'].ewm(span=9).mean()
                df['MACD_histogram'] = df['MACD_line'] - df['Signal_line']
            except Exception as e:
                logger.warning(f"Could not calculate MACD: {e}")
        
        # Create figure with secondary y-axis for volume
        if include_volume and 'Volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                specs=[[{"secondary_y": True}], [{}]]
            )
        else:
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color=self.colors['up'],
                decreasing_line_color=self.colors['down'],
                increasing_fillcolor=self.colors['up'],
                decreasing_fillcolor=self.colors['down'],
                hovertemplate='<b>Date</b>: %{x}<br>' +
                             '<b>Open</b>: $%{open:.2f}<br>' +
                             '<b>High</b>: $%{high:.2f}<br>' +
                             '<b>Low</b>: $%{low:.2f}<br>' +
                             '<b>Close</b>: $%{close:.2f}<br>'
            ),
            row=1, col=1
        )
        
        # Add volume as bar chart on second row
        if include_volume and 'Volume' in df.columns:
            # Get colors for volume bars based on price movement
            colors = np.where(df['Close'] >= df['Open'], self.colors['up'], self.colors['down'])
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.8,
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Volume</b>: %{y:,.0f}<br>'
                ),
                row=2, col=1
            )
        
        # Add technical indicators
        if include_indicators:
            # Add 50-day SMA
            if 'SMA_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_50'],
                        name='50-day SMA',
                        line=dict(color=self.colors['sma_50'], width=1.5),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>SMA (50)</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
            
            # Add 200-day SMA
            if 'SMA_200' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['SMA_200'],
                        name='200-day SMA',
                        line=dict(color=self.colors['sma_200'], width=1.5),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>SMA (200)</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
            
            # Add 20-day EMA
            if 'EMA_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['EMA_20'],
                        name='20-day EMA',
                        line=dict(color=self.colors['ema_20'], width=1.5, dash='dot'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>EMA (20)</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        if include_bollinger:
            if all(x in df.columns for x in ['BB_upper', 'BB_middle', 'BB_lower']):
                # Upper band
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_upper'],
                        name='Upper Bollinger Band',
                        line=dict(color=self.colors['bb_upper'], width=1.5, dash='dash'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>Upper BB</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
                
                # Middle band
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_middle'],
                        name='Middle Bollinger Band',
                        line=dict(color=self.colors['bb_middle'], width=1.5, dash='dash'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>Middle BB</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
                
                # Lower band with fill
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['BB_lower'],
                        name='Lower Bollinger Band',
                        line=dict(color=self.colors['bb_lower'], width=1.5, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(149, 117, 205, 0.15)',
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                     '<b>Lower BB</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
        
        # Update layout
        layout = self.default_layout.copy()
        layout['title'] = {
            'text': title or 'Stock Price Chart',
            'font': {'size': 18, 'color': '#444'}
        }
        
        if include_volume:
            layout['xaxis2'] = {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title': 'Date'
            }
            layout['yaxis2'] = {
                'title': 'Volume',
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            }
        
        # Apply improved layout
        fig.update_layout(**layout)
        
        # Set y-axis titles
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if include_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Add range slider with better configuration
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05,
            rangebreaks=[
                # Hide weekends
                dict(bounds=["sat", "mon"])
            ]
        )
        
        # Add watermark 
        fig.add_annotation(
            text="Stock Analytics Dashboard",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                family="Arial",
                size=30,
                color="rgba(150,150,150,0.1)"
            ),
            textangle=-30
        )
        
        return fig
    
    def create_price_chart(self, df, predictions_df=None, title=None, include_indicators=True, include_volume=False):
        """
        Create an enhanced interactive line chart for stock prices
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        predictions_df : pandas.DataFrame, optional
            DataFrame with price predictions
        title : str, optional
            Chart title
        include_indicators : bool
            Whether to include technical indicators
        include_volume : bool
            Whether to include volume subplot
            
        Returns:
        --------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        # Check required columns
        if 'Close' not in df.columns:
            logger.error("DataFrame missing 'Close' column")
            return None
        
        # Prepare technical indicators if they don't exist
        if include_indicators and 'SMA_50' not in df.columns:
            try:
                # Calculate SMA indicators
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
                df['SMA_200'] = df['Close'].rolling(window=200).mean()
                
                # Add EMA indicator
                df['EMA_20'] = df['Close'].ewm(span=20).mean()
            except Exception as e:
                logger.warning(f"Could not calculate technical indicators: {e}")
        
        # Create figure with volume subplot if requested
        if include_volume and 'Volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                specs=[[{"secondary_y": False}], [{}]]
            )
        else:
            fig = go.Figure()
        
        # Add close price line
        if include_volume and 'Volume' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Close Price',
                    line=dict(color=self.colors['price'], width=2.5),
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Close</b>: $%{y:.2f}<br>'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    name='Close Price',
                    line=dict(color=self.colors['price'], width=2.5),
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Close</b>: $%{y:.2f}<br>'
                )
            )
        
        # Add technical indicators
        if include_indicators:
            # Add 50-day SMA
            if 'SMA_50' in df.columns:
                if include_volume and 'Volume' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_50'],
                            name='50-day SMA',
                            line=dict(color=self.colors['sma_50'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (50)</b>: $%{y:.2f}<br>'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_50'],
                            name='50-day SMA',
                            line=dict(color=self.colors['sma_50'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (50)</b>: $%{y:.2f}<br>'
                        )
                    )
            
            # Add 200-day SMA
            if 'SMA_200' in df.columns:
                if include_volume and 'Volume' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_200'],
                            name='200-day SMA',
                            line=dict(color=self.colors['sma_200'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (200)</b>: $%{y:.2f}<br>'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['SMA_200'],
                            name='200-day SMA',
                            line=dict(color=self.colors['sma_200'], width=1.5),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>SMA (200)</b>: $%{y:.2f}<br>'
                        )
                    )
                    
            # Add 20-day EMA
            if 'EMA_20' in df.columns:
                if include_volume and 'Volume' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['EMA_20'],
                            name='20-day EMA',
                            line=dict(color=self.colors['ema_20'], width=1.5, dash='dot'),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>EMA (20)</b>: $%{y:.2f}<br>'
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['EMA_20'],
                            name='20-day EMA',
                            line=dict(color=self.colors['ema_20'], width=1.5, dash='dot'),
                            hovertemplate='<b>Date</b>: %{x}<br>' +
                                        '<b>EMA (20)</b>: $%{y:.2f}<br>'
                        )
                    )
        
        # Add volume bars if requested
        if include_volume and 'Volume' in df.columns:
            # Get colors for volume bars based on price changes
            df_temp = df.copy()
            df_temp['Price_Change'] = df_temp['Close'].pct_change()
            colors = np.where(df_temp['Price_Change'] >= 0, self.colors['up'], self.colors['down'])
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.8,
                    hovertemplate='<b>Date</b>: %{x}<br>' +
                                '<b>Volume</b>: %{y:,.0f}<br>'
                ),
                row=2, col=1
            )
        
        # Add predictions if available
        if predictions_df is not None and 'Close' in predictions_df.columns:
            if include_volume and 'Volume' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df.index,
                        y=predictions_df['Close'],
                        name='Predictions',
                        line=dict(color=self.colors['prediction'], width=2.5, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=7, symbol='circle'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                    '<b>Predicted Price</b>: $%{y:.2f}<br>'
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=predictions_df.index,
                        y=predictions_df['Close'],
                        name='Predictions',
                        line=dict(color=self.colors['prediction'], width=2.5, dash='dash'),
                        mode='lines+markers',
                        marker=dict(size=7, symbol='circle'),
                        hovertemplate='<b>Date</b>: %{x}<br>' +
                                    '<b>Predicted Price</b>: $%{y:.2f}<br>'
                    )
                )
            
            # Add shaded area for prediction region
            if len(df) > 0 and len(predictions_df) > 0:
                # Add vertical line at the transition point between historical and prediction
                last_historical_date = df.index[-1]
                first_prediction_date = predictions_df.index[0]
                
                # Check if there's a gap between historical and prediction
                if last_historical_date != first_prediction_date:
                    if include_volume and 'Volume' in df.columns:
                        fig.add_vline(
                            x=last_historical_date,
                            line_width=1,
                            line_dash="dash",
                            line_color="gray",
                            row=1, col=1
                        )
                    else:
                        fig.add_vline(
                            x=last_historical_date,
                            line_width=1,
                            line_dash="dash",
                            line_color="gray"
                        )
                
                # Add shaded area for prediction region
                if include_volume and 'Volume' in df.columns:
                    fig.add_vrect(
                        x0=last_historical_date,
                        x1=predictions_df.index[-1],
                        fillcolor="rgba(128, 0, 128, 0.05)",
                        layer="below",
                        line_width=0,
                        row=1, col=1
                    )
                else:
                    fig.add_vrect(
                        x0=last_historical_date,
                        x1=predictions_df.index[-1],
                        fillcolor="rgba(128, 0, 128, 0.05)",
                        layer="below",
                        line_width=0
                    )
        
        # Update layout
        layout = self.default_layout.copy()
        layout['title'] = {
            'text': title or 'Stock Price Chart',
            'font': {'size': 18, 'color': '#444'}
        }
        
        # Set axis titles and configure slider
        if include_volume and 'Volume' in df.columns:
            layout['xaxis2'] = {
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title': 'Date'
            }
            layout['yaxis2'] = {
                'title': 'Volume',
                'showgrid': True,
                'gridcolor': self.colors['grid'],
                'title_font': {'size': 14}
            }
            fig.update_layout(**layout)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            # Add range slider with better configuration
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeslider_thickness=0.05,
                rangebreaks=[
                    # Hide weekends
                    dict(bounds=["sat", "mon"])
                ]
            )
        else:
            fig.update_layout(**layout)
            fig.update_yaxes(title_text="Price ($)")
            
            # Add range slider
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeslider_thickness=0.05,
                rangebreaks=[
                    # Hide weekends
                    dict(bounds=["sat", "mon"])
                ]
            )
        
        # Add watermark 
        fig.add_annotation(
            text="Stock Analytics Dashboard",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                family="Arial",
                size=30,
                color="rgba(150,150,150,0.1)"
            ),
            textangle=-30
        )
        
        return fig