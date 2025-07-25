#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Visualization Module for StockTracker

This module provides advanced visualization capabilities for stock data analysis,
including interactive charts, candlestick plots, technical indicators, heatmaps,
3D visualizations, and animated charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class StockVisualizer:
    """Advanced stock data visualizer with interactive charts and advanced visualizations."""
    
    def __init__(self):
        """Initialize the StockVisualizer."""
        pass
    
    def plot_interactive_price_chart(self, data: pd.DataFrame, symbol: str = "",
                                   title: str = "Stock Price") -> go.Figure:
        """
        Create an interactive price chart using Plotly.
        
        Args:
            data: Stock data DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            symbol: Stock symbol for title
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: Interactive price chart
            
        Raises:
            TypeError: If data is not a pandas DataFrame
            ValueError: If required columns are missing from data
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Check for empty data
        if data.empty:
            raise ValueError("data cannot be empty")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add closing price line
        fig.add_trace(
            go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price',
                      line=dict(color='blue', width=2)),
            secondary_y=False,
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='lightblue'),
            secondary_y=True,
        )
        
        # Set axis titles
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Volume", secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title=f"{title} - {symbol}" if symbol else title,
            template="plotly_white",
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def plot_candlestick_chart(self, data: pd.DataFrame, symbol: str = "",
                              title: str = "Candlestick Chart") -> go.Figure:
        """
        Create a candlestick chart using Plotly.
        
        Args:
            data: Stock data DataFrame with columns ['open', 'high', 'low', 'close']
            symbol: Stock symbol for title
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: Candlestick chart
            
        Raises:
            TypeError: If data is not a pandas DataFrame
            ValueError: If required columns are missing from data or data is empty
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in data: {missing_columns}")
        
        # Check for empty data
        if data.empty:
            raise ValueError("data cannot be empty")
        
        # Check for NaN values in required columns
        if data[required_columns].isnull().any().any():
            raise ValueError("Data contains NaN values in required columns")
        
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Candlesticks"
        ))
        
        fig.update_layout(
            title=f"{title} - {symbol}" if symbol else title,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600
        )
        
        return fig
    
    def plot_technical_indicators(self, data: pd.DataFrame, indicators: Dict[str, pd.Series],
                                 symbol: str = "", title: str = "Technical Indicators",
                                 confidence_intervals: Optional[Dict[str, tuple]] = None) -> go.Figure:
        """
        Plot technical indicators with price data.
        
        Args:
            data: Stock data DataFrame with 'close' column
            indicators: Dictionary of indicator names and their Series data
            symbol: Stock symbol for title
            title: Chart title
            confidence_intervals: Dictionary of indicator names and their confidence interval tuples (lower, upper)
            
        Returns:
            plotly.graph_objects.Figure: Technical indicators chart
            
        Raises:
            TypeError: If data is not a pandas DataFrame or indicators is not a dict
            ValueError: If 'close' column is missing from data or data is empty
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(indicators, dict):
            raise TypeError("indicators must be a dictionary")
        
        # Check for required 'close' column
        if 'close' not in data.columns:
            raise ValueError("Missing required 'close' column in data")
        
        # Check for empty data
        if data.empty:
            raise ValueError("data cannot be empty")
        
        # Check for empty indicators
        if not indicators:
            raise ValueError("indicators cannot be empty")
        
        # Validate indicator data
        for name, indicator_data in indicators.items():
            if not isinstance(indicator_data, pd.Series):
                raise TypeError(f"Indicator '{name}' must be a pandas Series")
            if indicator_data.empty:
                raise ValueError(f"Indicator '{name}' cannot be empty")
        
        # Validate confidence intervals if provided
        if confidence_intervals is not None:
            if not isinstance(confidence_intervals, dict):
                raise TypeError("confidence_intervals must be a dictionary or None")
            for name, interval in confidence_intervals.items():
                if name not in indicators:
                    raise ValueError(f"Confidence interval provided for unknown indicator '{name}'")
                if not isinstance(interval, tuple) or len(interval) != 2:
                    raise TypeError(f"Confidence interval for '{name}' must be a tuple of two Series")
                lower, upper = interval
                if not isinstance(lower, pd.Series) or not isinstance(upper, pd.Series):
                    raise TypeError(f"Confidence interval for '{name}' must contain two pandas Series")
        
        fig = make_subplots(rows=len(indicators)+1, cols=1,
                           shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=['Price'] + list(indicators.keys()))
        
        # Add price data
        fig.add_trace(
            go.Scatter(x=data.index, y=data['close'], mode='lines', name='Close Price',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        # Add indicators
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, (name, indicator_data) in enumerate(indicators.items()):
            color_idx = i % len(colors)
            
            # Add indicator line
            fig.add_trace(
                go.Scatter(x=data.index, y=indicator_data, mode='lines', name=name,
                          line=dict(color=colors[color_idx])),
                row=i+2, col=1
            )
            
            # Add confidence intervals if provided
            if confidence_intervals and name in confidence_intervals:
                lower_bound, upper_bound = confidence_intervals[name]
                
                # Add upper bound
                fig.add_trace(go.Scatter(
                    x=upper_bound.index,
                    y=upper_bound,
                    mode='lines',
                    name=f'{name} Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ), row=i+2, col=1)
                
                # Add lower bound and fill area
                fig.add_trace(go.Scatter(
                    x=lower_bound.index,
                    y=lower_bound,
                    mode='lines',
                    name=f'{name} Lower Bound',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f'rgba({color_idx*20}, {255-color_idx*20}, {100+color_idx*10}, 0.2)',
                    showlegend=False
                ), row=i+2, col=1)
        
        fig.update_layout(
            title=f"{title} - {symbol}" if symbol else title,
            template="plotly_white",
            height=200*(len(indicators)+1),
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_heatmap(self, data_dict: Dict[str, pd.DataFrame],
                                metric: str = 'close',
                                cluster: bool = False) -> go.Figure:
        """
        Create a correlation heatmap for multiple stocks.
        
        Args:
            data_dict: Dictionary of stock data with symbol as key and DataFrame as value
            metric: Column to use for correlation calculation
            cluster: Whether to cluster stocks by correlation similarity
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
            
        Raises:
            TypeError: If data_dict is not a dictionary or metric is not a string
            ValueError: If data_dict is empty or required metric column is missing
        """
        # Validate inputs
        if not isinstance(data_dict, dict):
            raise TypeError("data_dict must be a dictionary")
        if not isinstance(metric, str):
            raise TypeError("metric must be a string")
        if not isinstance(cluster, bool):
            raise TypeError("cluster must be a boolean")
        
        # Check for empty data
        if not data_dict:
            raise ValueError("data_dict cannot be empty")
        
        # Validate each DataFrame in data_dict
        for symbol, data in data_dict.items():
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"Data for symbol '{symbol}' must be a pandas DataFrame")
            if metric not in data.columns:
                raise ValueError(f"Missing required column '{metric}' in data for symbol '{symbol}'")
            if data.empty:
                raise ValueError(f"Data for symbol '{symbol}' cannot be empty")
        
        # Create correlation matrix
        symbols = list(data_dict.keys())
        closes = pd.DataFrame({symbol: data[metric] for symbol, data in data_dict.items()})
        correlation_matrix = closes.corr()
        
        # Handle case where correlation matrix is empty or has NaN values
        if correlation_matrix.empty or correlation_matrix.isnull().all().all():
            raise ValueError("Unable to compute correlation matrix - insufficient data")
        
        # Cluster stocks by correlation if requested
        if cluster:
            try:
                from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
                from scipy.spatial.distance import squareform
                
                # Convert correlation to distance
                distance_matrix = 1 - abs(correlation_matrix)
                condensed_distance = squareform(distance_matrix)
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(condensed_distance, method='ward')
                
                # Reorder correlation matrix based on clustering
                order = leaves_list(linkage_matrix)
                reordered_symbols = [correlation_matrix.index[i] for i in order]
                correlation_matrix = correlation_matrix.reindex(index=reordered_symbols, columns=reordered_symbols)
            except ImportError:
                # If scipy is not available, just use regular correlation matrix
                pass
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        # Update layout based on clustering
        if cluster:
            fig.update_layout(
                title="Stock Correlation Matrix with Clustering",
                template="plotly_white",
                xaxis_title="Stock Symbols",
                yaxis_title="Stock Symbols",
                height=600
            )
        else:
            fig.update_layout(
                title="Stock Correlation Matrix",
                template="plotly_white",
                xaxis_title="Stock Symbols",
                yaxis_title="Stock Symbols",
                height=500
            )
        
        return fig
    
    def plot_industry_performance_heatmap(self, industry_data: Dict[str, float],
                                          title: str = "Industry Performance Heatmap") -> go.Figure:
        """
        Create a heatmap for industry performance visualization.
        
        Args:
            industry_data: Dictionary of industry names and their performance metrics
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: Industry performance heatmap
            
        Raises:
            TypeError: If industry_data is not a dictionary or title is not a string
            ValueError: If industry_data is empty
        """
        # Validate inputs
        if not isinstance(industry_data, dict):
            raise TypeError("industry_data must be a dictionary")
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        
        # Check for empty data
        if not industry_data:
            raise ValueError("industry_data cannot be empty")
        
        # Validate industry data values
        for industry, value in industry_data.items():
            if not isinstance(industry, str):
                raise TypeError("All industry names must be strings")
            if not isinstance(value, (int, float)):
                raise TypeError("All performance metrics must be numeric")
        
        # Prepare data for heatmap
        industries = list(industry_data.keys())
        performances = list(industry_data.values())
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[performances],
            x=industries,
            y=['Performance'],
            colorscale='RdYlGn',
            zmid=0,
            text=[performances],
            texttemplate="%{text:.2f}%",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            xaxis_title="Industries",
            yaxis_title="",
            height=300
        )
        
        return fig
    
    def plot_3d_risk_return(self, portfolio_data: pd.DataFrame,
                           x_col: str = 'return', y_col: str = 'risk', z_col: str = 'sharpe_ratio',
                           title: str = "Risk-Return-Performance 3D Visualization",
                           color_col: Optional[str] = None, size_col: Optional[str] = None,
                           hover_data: Optional[List[str]] = None) -> go.Figure:
        """
        Create a 3D visualization of risk, return, and performance metrics.
        
        Args:
            portfolio_data: Portfolio data with risk, return, and performance metrics
            x_col: Column for x-axis (typically return)
            y_col: Column for y-axis (typically risk)
            z_col: Column for z-axis (typically Sharpe ratio or another metric)
            title: Chart title
            color_col: Column to use for marker color (defaults to z_col)
            size_col: Column to use for marker size
            hover_data: Additional columns to include in hover information
            
        Returns:
            plotly.graph_objects.Figure: 3D scatter plot
            
        Raises:
            TypeError: If portfolio_data is not a pandas DataFrame
            ValueError: If required columns are missing from portfolio_data or data is empty
        """
        # Validate inputs
        if not isinstance(portfolio_data, pd.DataFrame):
            raise TypeError("portfolio_data must be a pandas DataFrame")
        
        required_columns = [x_col, y_col, z_col]
        if color_col is not None:
            required_columns.append(color_col)
        if size_col is not None:
            required_columns.append(size_col)
        if hover_data is not None:
            if not isinstance(hover_data, list):
                raise TypeError("hover_data must be a list or None")
            required_columns.extend(hover_data)
            
        missing_columns = [col for col in required_columns if col not in portfolio_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in portfolio_data: {missing_columns}")
        
        # Check for empty data
        if portfolio_data.empty:
            raise ValueError("portfolio_data cannot be empty")
        
        # Check for NaN values in required columns
        if portfolio_data[required_columns].isnull().any().any():
            raise ValueError("Data contains NaN values in required columns")
        
        # Set default color column
        if color_col is None:
            color_col = z_col
            
        # Create hover template
        hover_template = f'<b>%{{text}}</b><br>' + \
                        f'{x_col.replace("_", " ").title()}: %{{x:.4f}}<br>' + \
                        f'{y_col.replace("_", " ").title()}: %{{y:.4f}}<br>' + \
                        f'{z_col.replace("_", " ").title()}: %{{z:.4f}}'
        
        # Add additional hover data if provided
        if hover_data:
            for col in hover_data:
                hover_template += f'<br>{col.replace("_", " ").title()}: %{{customdata[{hover_data.index(col)}]:.4f}}'
        
        hover_template += '<extra></extra>'
        
        # Create marker size array
        marker_size = 8
        if size_col is not None:
            # Normalize size values to a reasonable range
            size_data = portfolio_data[size_col]
            min_size, max_size = size_data.min(), size_data.max()
            if max_size != min_size:
                marker_size = 5 + 15 * (size_data - min_size) / (max_size - min_size)
            else:
                marker_size = [10] * len(size_data)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=portfolio_data[x_col],
            y=portfolio_data[y_col],
            z=portfolio_data[z_col],
            mode='markers',
            marker=dict(
                size=marker_size,
                color=portfolio_data[color_col],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_col.replace('_', ' ').title())
            ),
            text=portfolio_data.index,
            customdata=portfolio_data[hover_data].values if hover_data else None,
            hovertemplate=hover_template
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                zaxis_title=z_col.replace('_', ' ').title()
            ),
            template="plotly_white",
            height=700
        )
        
        return fig
    
    def plot_animated_price(self, data: pd.DataFrame, symbol: str = "",
                           title: str = "Animated Price Movement",
                           mode: str = 'lines', speed: int = 50,
                           show_volume: bool = False) -> go.Figure:
        """
        Create an animated price movement chart.
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol for title
            title: Chart title
            mode: Display mode ('lines', 'markers', 'lines+markers')
            speed: Animation speed in milliseconds
            show_volume: Whether to show volume bars in a subplot
            
        Returns:
            plotly.graph_objects.Figure: Animated price chart
            
        Raises:
            TypeError: If data is not a pandas DataFrame
            ValueError: If 'close' column is missing from data or data is empty
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(mode, str) or mode not in ['lines', 'markers', 'lines+markers']:
            raise ValueError("mode must be one of 'lines', 'markers', 'lines+markers'")
        if not isinstance(speed, int) or speed <= 0:
            raise ValueError("speed must be a positive integer")
        if not isinstance(show_volume, bool):
            raise TypeError("show_volume must be a boolean")
        
        # Check for required 'close' column
        if 'close' not in data.columns:
            raise ValueError("Missing required 'close' column in data")
        
        # Check for empty data
        if data.empty:
            raise ValueError("data cannot be empty")
        
        # Check for minimum data points required for animation
        if len(data) < 2:
            raise ValueError("data must contain at least 2 data points for animation")
        
        # Create figure with subplots if volume is requested
        if show_volume and 'volume' in data.columns:
            fig = make_subplots(rows=2, cols=1,
                               shared_xaxes=True,
                               vertical_spacing=0.05,
                               row_heights=[0.7, 0.3])
        else:
            fig = go.Figure()
        
        # Add initial trace
        fig.add_trace(go.Scatter(
            x=data.index[:1],
            y=data['close'][:1],
            mode=mode,
            line=dict(width=3, color='blue'),
            name='Price'
        ), row=1, col=1)
        
        # Add volume bars if requested
        if show_volume and 'volume' in data.columns:
            fig.add_trace(go.Bar(
                x=data.index[:1],
                y=data['volume'][:1],
                name='Volume',
                marker_color='lightblue'
            ), row=2, col=1)
        
        # Create frames for animation
        frames = []
        for i in range(1, len(data)+1):
            frame_data = []
            
            # Add price data to frame
            frame_data.append(go.Scatter(
                x=data.index[:i],
                y=data['close'][:i],
                mode=mode,
                line=dict(width=3, color='blue'),
                name='Price'
            ))
            
            # Add volume data to frame if requested
            if show_volume and 'volume' in data.columns:
                frame_data.append(go.Bar(
                    x=data.index[:i],
                    y=data['volume'][:i],
                    name='Volume',
                    marker_color='lightblue',
                    showlegend=False
                ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=f'frame{i}'
            ))
        
        fig.update(frames=frames)
        
        # Update layout
        fig.update_layout(
            title=f"{title} - {symbol}" if symbol else title,
            template="plotly_white",
            height=700 if show_volume else 600,
            updatemenus=[dict(
                type='buttons',
                buttons=[dict(
                    label='Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=speed, redraw=True), fromcurrent=True)]
                )]
            )]
        )
        
        # Set axis titles and ranges
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        if show_volume and 'volume' in data.columns:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        # Set axis ranges
        fig.update_xaxes(range=[data.index[0], data.index[-1]])
        fig.update_yaxes(range=[data['close'].min() * 0.95, data['close'].max() * 1.05], row=1, col=1)
        
        if show_volume and 'volume' in data.columns:
            fig.update_yaxes(range=[0, data['volume'].max() * 1.1], row=2, col=1)
        
        return fig
    
    def plot_prediction_with_confidence(self, historical_data: pd.DataFrame,
                                       predictions: pd.Series,
                                       confidence_interval: Optional[tuple] = None,
                                       symbol: str = "",
                                       title: str = "Price Prediction with Confidence Interval") -> go.Figure:
        """
        Plot stock price predictions with confidence intervals.
        
        Args:
            historical_data: Historical stock data
            predictions: Predicted prices
            confidence_interval: Tuple of (lower_bound, upper_bound) series
            symbol: Stock symbol for title
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: Prediction chart with confidence interval
            
        Raises:
            TypeError: If historical_data is not a pandas DataFrame or predictions is not a pandas Series
            ValueError: If 'close' column is missing from historical_data or data is empty
        """
        # Validate inputs
        if not isinstance(historical_data, pd.DataFrame):
            raise TypeError("historical_data must be a pandas DataFrame")
        if not isinstance(predictions, pd.Series):
            raise TypeError("predictions must be a pandas Series")
        if confidence_interval is not None and not isinstance(confidence_interval, tuple):
            raise TypeError("confidence_interval must be a tuple of Series or None")
        
        # Check for required 'close' column
        if 'close' not in historical_data.columns:
            raise ValueError("Missing required 'close' column in historical_data")
        
        # Check for empty data
        if historical_data.empty:
            raise ValueError("historical_data cannot be empty")
        if predictions.empty:
            raise ValueError("predictions cannot be empty")
        
        # Validate confidence interval if provided
        if confidence_interval is not None:
            if len(confidence_interval) != 2:
                raise ValueError("confidence_interval must be a tuple of two Series")
            lower_bound, upper_bound = confidence_interval
            if not isinstance(lower_bound, pd.Series) or not isinstance(upper_bound, pd.Series):
                raise TypeError("confidence_interval must contain two pandas Series")
            if lower_bound.empty or upper_bound.empty:
                raise ValueError("confidence_interval Series cannot be empty")
        
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        ))
        
        # Add predictions
        fig.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions,
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence interval if provided
        if confidence_interval:
            lower_bound, upper_bound = confidence_interval
            
            # Add upper bound
            fig.add_trace(go.Scatter(
                x=upper_bound.index,
                y=upper_bound,
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            # Add lower bound and fill area
            fig.add_trace(go.Scatter(
                x=lower_bound.index,
                y=lower_bound,
                mode='lines',
                name='Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"{title} - {symbol}" if symbol else title,
            template="plotly_white",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600
        )
        
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, Any]],
                             metric: str = "price_change_percent",
                             title: str = "Model Performance Comparison") -> go.Figure:
        """
        Compare performance of different models.
        
        Args:
            model_results: Dictionary of model results
            metric: Metric to compare (price_change_percent, predicted_price, etc.)
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: Model comparison chart
            
        Raises:
            TypeError: If model_results is not a dictionary or metric is not a string
            ValueError: If model_results is empty or metric is not found in model results
        """
        # Validate inputs
        if not isinstance(model_results, dict):
            raise TypeError("model_results must be a dictionary")
        if not isinstance(metric, str):
            raise TypeError("metric must be a string")
        
        # Check for empty data
        if not model_results:
            raise ValueError("model_results cannot be empty")
        
        # Validate each model result
        for model_name, result in model_results.items():
            if not isinstance(result, dict):
                raise TypeError(f"Result for model '{model_name}' must be a dictionary")
            if metric not in result:
                raise ValueError(f"Metric '{metric}' not found in results for model '{model_name}'")
        
        # Extract data for comparison
        models = list(model_results.keys())
        values = [result[metric] for result in model_results.values()]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition='auto',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title=title,
            template="plotly_white",
            xaxis_title="Model",
            yaxis_title=metric.replace('_', ' ').title(),
            height=500
        )
        
        return fig
    
    def plot_risk_metrics(self, risk_data: Dict[str, Any], symbol: str = "",
                         title: str = "Risk Metrics Visualization") -> go.Figure:
        """
        Visualize risk metrics for a stock.
        
        Args:
            risk_data: Risk assessment data
            symbol: Stock symbol for title
            title: Chart title
            
        Returns:
            plotly.graph_objects.Figure: Risk metrics visualization
            
        Raises:
            TypeError: If risk_data is not a dictionary
        """
        # Validate inputs
        if not isinstance(risk_data, dict):
            raise TypeError("risk_data must be a dictionary")
        
        # Prepare data for radar chart
        categories = ['Volatility', 'Max Drawdown', 'VaR', 'Beta', 'Sharpe Ratio']
        values = [
            risk_data.get('volatility', 0) * 10,  # Scale for better visualization
            abs(risk_data.get('max_drawdown', 0)) * 100,  # Convert to percentage
            abs(risk_data.get('var_historical', 0)) * 100,  # Convert to percentage
            risk_data.get('beta', 0) * 10,  # Scale for better visualization
            max(0, risk_data.get('sharpe_ratio', 0)) * 10  # Scale and ensure non-negative
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Risk Metrics'
        ))
        
        fig.update_layout(
            title=f"{title} - {symbol}" if symbol else title,
            template="plotly_white",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1]
                )
            ),
            height=500
        )
        
        return fig


# Dashboard functions
def create_dashboard_summary(stock_data: pd.DataFrame, prediction_data: Dict[str, Any],
                           risk_data: Dict[str, Any]) -> go.Figure:
    """
    Create a comprehensive dashboard summary with key metrics.
    
    Args:
        stock_data: Stock data DataFrame
        prediction_data: Prediction results
        risk_data: Risk assessment data
        
    Returns:
        plotly.graph_objects.Figure: Dashboard summary
        
    Raises:
        TypeError: If stock_data is not a pandas DataFrame or prediction_data/risk_data are not dictionaries
        ValueError: If 'close' column is missing from stock_data or stock_data is empty
    """
    # Validate inputs
    if not isinstance(stock_data, pd.DataFrame):
        raise TypeError("stock_data must be a pandas DataFrame")
    if not isinstance(prediction_data, dict):
        raise TypeError("prediction_data must be a dictionary")
    if not isinstance(risk_data, dict):
        raise TypeError("risk_data must be a dictionary")
        
    # Check for required 'close' column
    if 'close' not in stock_data.columns:
        raise ValueError("Missing required 'close' column in stock_data")
        
    # Check for empty data
    if stock_data.empty:
        raise ValueError("stock_data cannot be empty")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Trend', 'Prediction', 'Risk Metrics', 'Key Statistics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "indicator"}, {"type": "table"}]]
    )
    
    # Add price trend
    fig.add_trace(
        go.Scatter(x=stock_data.index[-30:], y=stock_data['close'][-30:],
                  mode='lines+markers', name='Price Trend'),
        row=1, col=1
    )
    
    # Add prediction comparison
    current_price = stock_data['close'].iloc[-1]
    predicted_price = prediction_data.get('predicted_price', current_price)
    fig.add_trace(
        go.Bar(x=['Current', 'Predicted'], y=[current_price, predicted_price],
              name='Price Comparison'),
        row=1, col=2
    )
    
    # Add risk metrics indicators
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=risk_data.get('volatility', 0) * 100,
            title={'text': "Volatility (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 50]}}
        ),
        row=2, col=1
    )
    
    # Add key statistics table
    stats_data = {
        'Metric': ['Current Price', 'Predicted Price', 'Price Change (%)', 'Volatility',
                  'Sharpe Ratio', 'Max Drawdown', 'Beta'],
        'Value': [
            f"¥{current_price:.2f}",
            f"¥{predicted_price:.2f}",
            f"{prediction_data.get('price_change_percent', 0):.2f}%",
            f"{risk_data.get('volatility', 0) * 100:.2f}%",
            f"{risk_data.get('sharpe_ratio', 0):.2f}",
            f"{risk_data.get('max_drawdown', 0) * 100:.2f}%",
            f"{risk_data.get('beta', 0):.2f}"
        ]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_data.keys()),
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[stats_data[k] for k in stats_data.keys()],
                      fill_color='lavender',
                      align='left')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Stock Analysis Dashboard",
        template="plotly_white",
        height=800
    )
    
    return fig


def create_comprehensive_dashboard(stock_data: pd.DataFrame, prediction_data: Dict[str, Any],
                                 risk_data: Dict[str, Any], portfolio_data: Optional[pd.DataFrame] = None,
                                 backtest_results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple visualization components.
    
    Args:
        stock_data: Stock data DataFrame
        prediction_data: Prediction results
        risk_data: Risk assessment data
        portfolio_data: Portfolio data (optional)
        backtest_results: Backtest results (optional)
        
    Returns:
        plotly.graph_objects.Figure: Comprehensive dashboard
        
    Raises:
        TypeError: If inputs are not of expected types
        ValueError: If required data is missing or empty
    """
    # Validate inputs
    if not isinstance(stock_data, pd.DataFrame):
        raise TypeError("stock_data must be a pandas DataFrame")
    if not isinstance(prediction_data, dict):
        raise TypeError("prediction_data must be a dictionary")
    if not isinstance(risk_data, dict):
        raise TypeError("risk_data must be a dictionary")
    if portfolio_data is not None and not isinstance(portfolio_data, pd.DataFrame):
        raise TypeError("portfolio_data must be a pandas DataFrame or None")
    if backtest_results is not None and not isinstance(backtest_results, dict):
        raise TypeError("backtest_results must be a dictionary or None")
        
    # Check for required 'close' column
    if 'close' not in stock_data.columns:
        raise ValueError("Missing required 'close' column in stock_data")
        
    # Check for empty data
    if stock_data.empty:
        raise ValueError("stock_data cannot be empty")
    
    # Determine the number of subplots needed
    rows = 3
    if portfolio_data is not None:
        rows += 1
    if backtest_results is not None:
        rows += 1
    
    # Create subplots
    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=('Price Trend', 'Prediction', 'Risk Metrics', 'Key Statistics',
                       'Technical Indicators', 'Correlation Matrix'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "indicator"}, {"type": "table"}],
               [{"secondary_y": False}, {"secondary_y": False}]] +
              ([[{"secondary_y": False}, {"secondary_y": False}]] if portfolio_data is not None else []) +
              ([[{"secondary_y": False}, {"secondary_y": False}]] if backtest_results is not None else []),
        vertical_spacing=0.08
    )
    
    # Add price trend
    fig.add_trace(
        go.Scatter(x=stock_data.index[-30:], y=stock_data['close'][-30:],
                  mode='lines+markers', name='Price Trend'),
        row=1, col=1
    )
    
    # Add prediction comparison
    current_price = stock_data['close'].iloc[-1]
    predicted_price = prediction_data.get('predicted_price', current_price)
    fig.add_trace(
        go.Bar(x=['Current', 'Predicted'], y=[current_price, predicted_price],
              name='Price Comparison'),
        row=1, col=2
    )
    
    # Add risk metrics indicators
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=risk_data.get('volatility', 0) * 100,
            title={'text': "Volatility (%)"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 50]}}
        ),
        row=2, col=1
    )
    
    # Add key statistics table
    stats_data = {
        'Metric': ['Current Price', 'Predicted Price', 'Price Change (%)', 'Volatility',
                  'Sharpe Ratio', 'Max Drawdown', 'Beta'],
        'Value': [
            f"¥{current_price:.2f}",
            f"¥{predicted_price:.2f}",
            f"{prediction_data.get('price_change_percent', 0):.2f}%",
            f"{risk_data.get('volatility', 0) * 100:.2f}%",
            f"{risk_data.get('sharpe_ratio', 0):.2f}",
            f"{risk_data.get('max_drawdown', 0) * 100:.2f}%",
            f"{risk_data.get('beta', 0):.2f}"
        ]
    }
    
    fig.add_trace(
        go.Table(
            header=dict(values=list(stats_data.keys()),
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[stats_data[k] for k in stats_data.keys()],
                      fill_color='lavender',
                      align='left')
        ),
        row=2, col=2
    )
    
    # Add technical indicators (if available)
    row_offset = 3
    try:
        # Calculate some technical indicators for display
        sma_20 = stock_data['close'].rolling(20).mean()
        rsi = np.random.rand(len(stock_data)) * 100  # Placeholder for actual RSI calculation
        
        fig.add_trace(
            go.Scatter(x=stock_data.index[-60:], y=sma_20[-60:],
                      mode='lines', name='SMA 20'),
            row=row_offset, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=stock_data.index[-60:], y=rsi[-60:],
                      mode='lines', name='RSI'),
            row=row_offset, col=2
        )
    except Exception:
        # If technical indicators fail, just add empty traces
        fig.add_trace(go.Scatter(x=[], y=[], name='SMA 20'), row=row_offset, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], name='RSI'), row=row_offset, col=2)
    
    # Add portfolio analysis if provided
    if portfolio_data is not None:
        row_offset += 1
        try:
            fig.add_trace(
                go.Scatter(x=portfolio_data.index, y=portfolio_data['value'],
                          mode='lines', name='Portfolio Value'),
                row=row_offset, col=1
            )
            
            # Add portfolio weights pie chart
            if 'weights' in portfolio_data.columns:
                fig.add_trace(
                    go.Pie(labels=portfolio_data.columns[1:], values=portfolio_data['weights'].iloc[0],
                          name="Portfolio Weights"),
                    row=row_offset, col=2
                )
        except Exception:
            # If portfolio analysis fails, just add empty traces
            fig.add_trace(go.Scatter(x=[], y=[], name='Portfolio Value'), row=row_offset, col=1)
            fig.add_trace(go.Scatter(x=[], y=[], name='Portfolio Weights'), row=row_offset, col=2)
    
    # Add backtest results if provided
    if backtest_results is not None:
        row_offset += 1
        try:
            portfolio_value = backtest_results.get('portfolio_value', pd.Series())
            benchmark_value = backtest_results.get('benchmark_value', pd.Series())
            
            fig.add_trace(
                go.Scatter(x=portfolio_value.index, y=portfolio_value,
                          mode='lines', name='Portfolio'),
                row=row_offset, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=benchmark_value.index, y=benchmark_value,
                          mode='lines', name='Benchmark'),
                row=row_offset, col=1
            )
        except Exception:
            # If backtest results fail, just add empty traces
            fig.add_trace(go.Scatter(x=[], y=[], name='Portfolio'), row=row_offset, col=1)
            fig.add_trace(go.Scatter(x=[], y=[], name='Benchmark'), row=row_offset, col=1)
    
    fig.update_layout(
        title="Comprehensive Stock Analysis Dashboard",
        template="plotly_white",
        height=400 * rows,
        showlegend=True
    )
    
    return fig


def plot_multi_stock_comparison(stock_data_dict: Dict[str, pd.DataFrame],
                               metric: str = 'close') -> go.Figure:
    """
    Compare multiple stocks on the same chart.
    
    Args:
        stock_data_dict: Dictionary of stock data with symbol as key
        metric: Metric to compare
        
    Returns:
        plotly.graph_objects.Figure: Multi-stock comparison chart
        
    Raises:
        TypeError: If stock_data_dict is not a dictionary or metric is not a string
        ValueError: If stock_data_dict is empty or required metric column is missing
    """
    # Validate inputs
    if not isinstance(stock_data_dict, dict):
        raise TypeError("stock_data_dict must be a dictionary")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string")
        
    # Check for empty data
    if not stock_data_dict:
        raise ValueError("stock_data_dict cannot be empty")
    
    # Validate each DataFrame in stock_data_dict
    for symbol, data in stock_data_dict.items():
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Data for symbol '{symbol}' must be a pandas DataFrame")
        if metric not in data.columns:
            raise ValueError(f"Missing required column '{metric}' in data for symbol '{symbol}'")
        if data.empty:
            raise ValueError(f"Data for symbol '{symbol}' cannot be empty")
    
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (symbol, data) in enumerate(stock_data_dict.items()):
        color_idx = i % len(colors)
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[metric],
            mode='lines',
            name=symbol,
            line=dict(color=colors[color_idx])
        ))
    
    fig.update_layout(
        title=f"Multi-Stock {metric.title()} Comparison",
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title=metric.title(),
        height=600
    )
    
    return fig


def plot_portfolio_analysis(portfolio_weights: List[float],
                           stock_returns: pd.DataFrame,
                           title: str = "Portfolio Analysis") -> go.Figure:
    """
    Visualize portfolio analysis results.
    
    Args:
        portfolio_weights: Portfolio weights for each stock
        stock_returns: Stock returns data
        title: Chart title
        
    Returns:
        plotly.graph_objects.Figure: Portfolio analysis visualization
        
    Raises:
        TypeError: If portfolio_weights is not a list, stock_returns is not a pandas DataFrame, or title is not a string
        ValueError: If portfolio_weights or stock_returns are empty, or if their lengths don't match
    """
    # Validate inputs
    if not isinstance(portfolio_weights, list):
        raise TypeError("portfolio_weights must be a list")
    if not isinstance(stock_returns, pd.DataFrame):
        raise TypeError("stock_returns must be a pandas DataFrame")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
        
    # Check for empty data
    if not portfolio_weights:
        raise ValueError("portfolio_weights cannot be empty")
    if stock_returns.empty:
        raise ValueError("stock_returns cannot be empty")
        
    # Check for length mismatch
    if len(portfolio_weights) != len(stock_returns.columns):
        raise ValueError("Length of portfolio_weights must match number of columns in stock_returns")
    
    # Validate each weight
    for i, weight in enumerate(portfolio_weights):
        if not isinstance(weight, (int, float)):
            raise TypeError(f"Weight at index {i} must be a number")
    
    # Validate stock_returns DataFrame
    for col in stock_returns.columns:
        if not isinstance(col, str):
            raise TypeError("All column names in stock_returns must be strings")
    
    # Calculate portfolio returns
    portfolio_returns = (stock_returns * portfolio_weights).sum(axis=1)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Portfolio Returns', 'Cumulative Returns', 'Weights', 'Risk Contribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"secondary_y": False}]]
    )
    
    # Add portfolio returns
    fig.add_trace(
        go.Scatter(x=portfolio_returns.index, y=portfolio_returns,
                  mode='lines', name='Portfolio Returns'),
        row=1, col=1
    )
    
    # Add cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    fig.add_trace(
        go.Scatter(x=cumulative_returns.index, y=cumulative_returns,
                  mode='lines', name='Cumulative Returns'),
        row=1, col=2
    )
    
    # Add weights pie chart
    fig.add_trace(
        go.Pie(labels=stock_returns.columns, values=portfolio_weights, name="Weights"),
        row=2, col=1
    )
    
    # Add risk contribution (simplified)
    risk_contrib = np.abs(portfolio_weights)  # Simplified risk contribution
    fig.add_trace(
        go.Bar(x=stock_returns.columns, y=risk_contrib, name='Risk Contribution'),
        row=2, col=2
    )
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=800
    )
    
    return fig


def plot_backtest_results(backtest_results: Dict[str, Any],
                         title: str = "Backtest Results") -> go.Figure:
    """
    Visualize backtest results.
    
    Args:
        backtest_results: Backtest results data
        title: Chart title
        
    Returns:
        plotly.graph_objects.Figure: Backtest results visualization
        
    Raises:
        TypeError: If backtest_results is not a dictionary or title is not a string
        ValueError: If required keys are missing from backtest_results
    """
    # Validate inputs
    if not isinstance(backtest_results, dict):
        raise TypeError("backtest_results must be a dictionary")
    if not isinstance(title, str):
        raise TypeError("title must be a string")
        
    # Check for required keys
    required_keys = ['portfolio_value', 'benchmark_value']
    missing_keys = [key for key in required_keys if key not in backtest_results]
    if missing_keys:
        raise ValueError(f"Missing required keys in backtest_results: {missing_keys}")
        
    # Extract data
    portfolio_value = backtest_results.get('portfolio_value', pd.Series())
    benchmark_value = backtest_results.get('benchmark_value', pd.Series())
    trades = backtest_results.get('trades', [])
    
    # Validate extracted data
    if not isinstance(portfolio_value, pd.Series):
        raise TypeError("portfolio_value must be a pandas Series")
    if not isinstance(benchmark_value, pd.Series):
        raise TypeError("benchmark_value must be a pandas Series")
    if not isinstance(trades, list):
        raise TypeError("trades must be a list")
        
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio vs Benchmark', 'Drawdown'),
        row_heights=[0.7, 0.3]
    )
    
    # Add portfolio value
    fig.add_trace(
        go.Scatter(x=portfolio_value.index, y=portfolio_value,
                  mode='lines', name='Portfolio', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add benchmark value
    fig.add_trace(
        go.Scatter(x=benchmark_value.index, y=benchmark_value,
                  mode='lines', name='Benchmark', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # Add trade markers
    if trades:
        buy_dates = [trade['date'] for trade in trades if trade['type'] == 'buy']
        sell_dates = [trade['date'] for trade in trades if trade['type'] == 'sell']
        
        if buy_dates:
            buy_values = [portfolio_value.loc[date] for date in buy_dates if date in portfolio_value.index]
            fig.add_trace(
                go.Scatter(x=buy_dates, y=buy_values, mode='markers',
                          marker=dict(symbol='triangle-up', size=10, color='green'),
                          name='Buy'),
                row=1, col=1
            )
        
        if sell_dates:
            sell_values = [portfolio_value.loc[date] for date in sell_dates if date in portfolio_value.index]
            fig.add_trace(
                go.Scatter(x=sell_dates, y=sell_values, mode='markers',
                          marker=dict(symbol='triangle-down', size=10, color='red'),
                          name='Sell'),
                row=1, col=1
            )
    
    # Calculate and add drawdown
    if not portfolio_value.empty:
        cumulative_returns = portfolio_value / portfolio_value.iloc[0] - 1
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / (running_max + 1)
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown,
                      mode='lines', name='Drawdown', line=dict(color='red', width=1),
                      fill='tozeroy'),
            row=2, col=1
        )
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=700
    )
    
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    
    return fig


# Real-time visualization functions
def initialize_realtime_chart(symbol: str, window_size: int = 100) -> go.Figure:
    """
    Initialize a real-time chart for streaming data.
    
    Args:
        symbol: Stock symbol
        window_size: Number of data points to display
        
    Returns:
        plotly.graph_objects.Figure: Real-time chart
        
    Raises:
        TypeError: If symbol is not a string or window_size is not an integer
        ValueError: If window_size is not positive
    """
    # Validate inputs
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
        
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode='lines+markers',
        name='Real-time Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"Real-time Price - {symbol}",
        template="plotly_white",
        xaxis_title="Time",
        yaxis_title="Price",
        height=500,
        showlegend=True,
        xaxis=dict(
            type='date'
        )
    )
    
    # Store window size in figure for later use
    fig.layout.window_size = window_size
    
    return fig


def update_realtime_chart(fig: go.Figure, new_data: Dict[str, Any]) -> go.Figure:
    """
    Update real-time chart with new data point.
    
    Args:
        fig: Existing figure to update
        new_data: New data point with 'time' and 'price' keys
        
    Returns:
        plotly.graph_objects.Figure: Updated chart
        
    Raises:
        ValueError: If required keys are missing from new_data
        TypeError: If fig is not a go.Figure or new_data is not a dict
    """
    # Validate inputs
    if not isinstance(fig, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dictionary")
    if 'time' not in new_data or 'price' not in new_data:
        raise ValueError("new_data must contain 'time' and 'price' keys")
    
    # Get existing data
    trace = fig.data[0]
    x_data = list(trace.x)
    y_data = list(trace.y)
    
    # Add new data point
    x_data.append(new_data['time'])
    y_data.append(new_data['price'])
    
    # Limit data points to window size
    window_size = getattr(fig.layout, 'window_size', 100)
    if len(x_data) > window_size:
        x_data = x_data[-window_size:]
        y_data = y_data[-window_size:]
    
    # Update the trace
    fig.data[0].x = x_data
    fig.data[0].y = y_data
    
    # Update x-axis range to show the latest data
    if len(x_data) > 1:
        fig.update_xaxes(range=[x_data[0], x_data[-1]])
    
    # Update y-axis range to fit data
    if y_data:
        y_min, y_max = min(y_data), max(y_data)
        y_range = y_max - y_min
        # Add some padding to the y-axis range
        y_padding = y_range * 0.05 if y_range > 0 else 0.1
        fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding])
    
    return fig


def initialize_multi_metric_realtime_chart(symbol: str, metrics: List[str], window_size: int = 100) -> go.Figure:
    """
    Initialize a real-time chart for streaming data with multiple metrics.
    
    Args:
        symbol: Stock symbol
        metrics: List of metric names to display
        window_size: Number of data points to display
        
    Returns:
        plotly.graph_objects.Figure: Real-time chart with multiple metrics
        
    Raises:
        TypeError: If symbol is not a string, metrics is not a list, or window_size is not an integer
        ValueError: If metrics is empty or window_size is not positive
    """
    # Validate inputs
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    if not isinstance(metrics, list):
        raise TypeError("metrics must be a list")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
        
    # Check for empty metrics
    if not metrics:
        raise ValueError("metrics cannot be empty")
        
    # Validate each metric
    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            raise TypeError(f"Metric at index {i} must be a string")
    
    # Create subplots for each metric
    fig = make_subplots(
        rows=len(metrics), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=metrics
    )
    
    # Add traces for each metric
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for i, metric in enumerate(metrics):
        color_idx = i % len(colors)
        fig.add_trace(go.Scatter(
            x=[], y=[],
            mode='lines+markers',
            name=metric,
            line=dict(color=colors[color_idx], width=2)
        ), row=i+1, col=1)
    
    fig.update_layout(
        title=f"Real-time Multi-Metric Chart - {symbol}",
        template="plotly_white",
        height=200 * len(metrics),
        showlegend=True,
        xaxis=dict(type='date')
    )
    
    # Store window size and metrics in figure for later use
    fig.layout.window_size = window_size
    fig.layout.metrics = metrics
    
    return fig


def update_multi_metric_realtime_chart(fig: go.Figure, new_data: Dict[str, Any]) -> go.Figure:
    """
    Update real-time chart with new data points for multiple metrics.
    
    Args:
        fig: Existing figure to update
        new_data: New data point with 'time' key and metric keys
        
    Returns:
        plotly.graph_objects.Figure: Updated chart
        
    Raises:
        ValueError: If required keys are missing from new_data
        TypeError: If fig is not a go.Figure or new_data is not a dict
    """
    # Validate inputs
    if not isinstance(fig, go.Figure):
        raise TypeError("fig must be a plotly.graph_objects.Figure")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dictionary")
    if 'time' not in new_data:
        raise ValueError("new_data must contain 'time' key")
    
    # Get stored metrics and window size
    metrics = getattr(fig.layout, 'metrics', [])
    window_size = getattr(fig.layout, 'window_size', 100)
    
    # Validate that all required metrics are in new_data
    for metric in metrics:
        if metric not in new_data:
            raise ValueError(f"new_data must contain '{metric}' key")
    
    # Update each trace
    for i, metric in enumerate(metrics):
        # Get existing data for this trace
        trace = fig.data[i]
        x_data = list(trace.x)
        y_data = list(trace.y)
        
        # Add new data point
        x_data.append(new_data['time'])
        y_data.append(new_data[metric])
        
        # Limit data points to window size
        if len(x_data) > window_size:
            x_data = x_data[-window_size:]
            y_data = y_data[-window_size:]
        
        # Update the trace
        fig.data[i].x = x_data
        fig.data[i].y = y_data
    
    # Update x-axis range to show the latest data
    if len(fig.data[0].x) > 1:
        fig.update_xaxes(range=[fig.data[0].x[0], fig.data[0].x[-1]])
    
    # Update y-axis ranges to fit data for each subplot
    for i in range(len(metrics)):
        y_data = list(fig.data[i].y)
        if y_data:
            y_min, y_max = min(y_data), max(y_data)
            y_range = y_max - y_min
            # Add some padding to the y-axis range
            y_padding = y_range * 0.05 if y_range > 0 else 0.1
            fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding], row=i+1, col=1)
    
    return fig

# Utility functions
def save_plot(fig: go.Figure, filename: str, format: str = 'html') -> None:
    """
    Save a plot to a file.
    
    Args:
        fig: Plotly figure to save
        filename: Output filename
        format: Output format ('html', 'png', 'jpeg', 'pdf', 'svg')
    """
    if format == 'html':
        fig.write_html(filename)
    else:
        fig.write_image(filename, format=format)


def show_plot(fig: go.Figure) -> None:
    """
    Display a plot.
    
    Args:
        fig: Plotly figure to display
    """
    fig.show()


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100) * 10 + 100,
        'high': np.random.randn(100) * 12 + 105,
        'low': np.random.randn(100) * 8 + 95,
        'close': np.random.randn(100) * 10 + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Initialize visualizer
    visualizer = StockVisualizer()
    
    # Test interactive price chart
    fig1 = visualizer.plot_interactive_price_chart(sample_data, "TEST", "Interactive Price Chart")
    print("Created interactive price chart")
    
    # Test candlestick chart
    fig2 = visualizer.plot_candlestick_chart(sample_data, "TEST", "Candlestick Chart")
    print("Created candlestick chart")
    
    # Test technical indicators
    sma = sample_data['close'].rolling(20).mean()
    rsi = np.random.rand(100) * 100
    indicators = {
        'SMA 20': sma,
        'RSI': pd.Series(rsi, index=dates)
    }
    fig3 = visualizer.plot_technical_indicators(sample_data, indicators, "TEST", "Technical Indicators")
    print("Created technical indicators chart")
    
    # Test correlation heatmap
    data_dict = {
        'STOCK1': sample_data,
        'STOCK2': sample_data * 1.1,
        'STOCK3': sample_data * 0.9
    }
    fig4 = visualizer.plot_correlation_heatmap(data_dict)
    print("Created correlation heatmap")
    
    # Test industry performance heatmap
    industry_data = {
        'Technology': 5.2,
        'Finance': -2.1,
        'Healthcare': 3.8,
        'Energy': -1.5,
        'Consumer': 2.7
    }
    fig5 = visualizer.plot_industry_performance_heatmap(industry_data)
    print("Created industry performance heatmap")
    
    # Test 3D visualization
    portfolio_df = pd.DataFrame({
        'return': np.random.rand(50) * 0.2,
        'risk': np.random.rand(50) * 0.3,
        'sharpe_ratio': np.random.rand(50) * 2
    })
    fig6 = visualizer.plot_3d_risk_return(portfolio_df)
    print("Created 3D risk-return visualization")
    
    # Test animated chart
    fig7 = visualizer.plot_animated_price(sample_data.head(30), "TEST", "Animated Price")
    print("Created animated price chart")
    
    print("All visualization tests completed successfully!")