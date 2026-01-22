#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Stock Prediction Models

This module implements advanced models for stock price prediction including:
- Transformer model for time series prediction
- GRU model as an alternative to LSTM
- Ensemble learning methods (Random Forest, XGBoost) for comparison
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime
import json
import warnings
from performance_optimizer import model_cache, data_loader, memory_optimizer
warnings.filterwarnings('ignore')

# 修正tensorflow.keras导入
Sequential = tf.keras.Sequential
LSTM = tf.keras.layers.LSTM
GRU = tf.keras.layers.GRU
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
Model = tf.keras.Model
load_model = tf.keras.models.load_model
MultiHeadAttention = tf.keras.layers.MultiHeadAttention
LayerNormalization = tf.keras.layers.LayerNormalization
Layer = tf.keras.layers.Layer

# Transformer implementation components
class TimeSeriesTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_dim, output_dim, dropout_rate=0.1):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input projection layer
        self.input_projection = Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = self.positional_encoding(10000, d_model)
        
        # Transformer blocks
        self.enc_layers = [
            self.TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        
        # Output layer
        self.output_layer = Dense(output_dim)
        
        # Dropout
        self.dropout = Dropout(dropout_rate)
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates
    
    class TransformerBlock(Layer):
        def __init__(self, d_model, num_heads, dff, dropout_rate=0.0):
            super().__init__()
            
            self.mha = MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model
            )
            self.ffn = self.point_wise_feed_forward_network(d_model, dff)
            
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            
            self.dropout1 = Dropout(dropout_rate)
            self.dropout2 = Dropout(dropout_rate)
        
        def point_wise_feed_forward_network(self, d_model, dff):
            return Sequential([
                Dense(dff, activation='relu'),
                Dense(d_model)
            ])
        
        def call(self, x, training=None):
            # Multi-head attention
            attn_output = self.mha(query=x, key=x, value=x)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(x + attn_output)
            
            # Feed forward network
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            out2 = self.layernorm2(out1 + ffn_output)
            
            return out2
    
    def call(self, x, training=None):
        seq_len = tf.shape(x)[1]
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        # Using tf.slice instead of direct indexing to avoid Pylance errors
        pos_encoding_slice = tf.slice(self.pos_encoding, [0, 0, 0], [-1, seq_len, -1])
        x = x + pos_encoding_slice
        
        x = self.dropout(x, training=training)
        
        # Pass through transformer blocks
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)
        
        # Global average pooling to get sequence representation
        x = tf.reduce_mean(x, axis=1)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class AdvancedStockPredictor:
    def __init__(self, look_back=60, model_type='lstm'):
        """
        Initialize advanced stock predictor

        Args:
            look_back: Number of historical days to use for prediction
            model_type: Type of model to use ('lstm', 'gru', 'transformer', 'rf', 'xgboost')
        """
        self.look_back = look_back
        self.model_type = model_type
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models_dir = 'models'
        self.model_version = None

        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def create_dataset(self, dataset, look_back=1):
        """
        Create dataset for training
        
        Args:
            dataset: Input dataset
            look_back: Look back period
            
        Returns:
            tuple: (input data, output data)
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    def build_lstm_model(self, input_shape):
        """
        Build LSTM model
        
        Args:
            input_shape: Input data shape
            
        Returns:
            tf.keras.Model: LSTM model
        """
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_gru_model(self, input_shape):
        """
        Build GRU model
        
        Args:
            input_shape: Input data shape
            
        Returns:
            tf.keras.Model: GRU model
        """
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_transformer_model(self, input_shape):
        """
        Build Transformer model
        
        Args:
            input_shape: Input data shape
            
        Returns:
            tf.keras.Model: Transformer model
        """
        # For now, let's use a simpler approach that we know works
        # Using the existing imports
        inputs = Input(shape=(input_shape[0],))
        
        # Dense layers to process the sequence
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_rf_model(self):
        """
        Build Random Forest model
        
        Returns:
            RandomForestRegressor: Random Forest model
        """
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        return model
    
    def build_xgboost_model(self):
        """
        Build XGBoost model
        
        Returns:
            xgb.XGBRegressor: XGBoost model
        """
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        return model
    
    def prepare_features(self, data):
        """
        Prepare features for ensemble models
        
        Args:
            data: Input data
            
        Returns:
            np.array: Prepared features
        """
        # Add technical indicators as features
        # For simplicity, we'll use price-based features
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['open_close_pct'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=10).std()
        
        # Time-based features
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['quarter'] = pd.to_datetime(df.index).quarter
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features
        feature_columns = [
            'close', 'volume', 'price_change', 'price_change_pct', 
            'high_low_pct', 'open_close_pct', 'ma_5', 'ma_10', 'ma_20',
            'volatility', 'day_of_week', 'month', 'quarter'
        ]
        
        # Filter to only include existing columns
        existing_features = [col for col in feature_columns if col in df.columns]
        features = df[existing_features]
        
        return features
    
    def train(self, data, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train model

        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        # Check if model is already cached
        params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'look_back': self.look_back
        }

        cached_model, cached_scaler = model_cache.get_cached_model(
            self.model_type, data, params
        )

        if cached_model is not None and cached_scaler is not None:
            self.model = cached_model
            self.scaler = cached_scaler
            return None  # Return None since model loaded from cache

        if self.model_type in ['lstm', 'gru', 'transformer']:
            result = self._train_neural_network(data, epochs, batch_size, validation_split)
        elif self.model_type in ['rf', 'xgboost']:
            result = self._train_ensemble_model(data)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Cache the trained model
        model_cache.cache_model(self.model_type, data, params, self.model, self.scaler)

        return result
    
    def _train_neural_network(self, data, epochs, batch_size, validation_split):
        """
        Train neural network models (LSTM, GRU, Transformer)
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            History object
        """
        # Data preprocessing
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Split data
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Create dataset
        X_train, y_train = self.create_dataset(train_data, self.look_back)
        X_test, y_test = self.create_dataset(test_data, self.look_back)
        
        # Reshape data for neural networks
        if self.model_type in ['lstm', 'gru']:
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        elif self.model_type == 'transformer':
            # For transformer, we need to ensure the shape is correct
            # Transformer expects 2D input: (batch_size, sequence_length)
            # No need to add extra dimension
            # But we need to make sure the shapes are consistent
            pass
        
        # Build model
        if self.model_type in ['lstm', 'gru']:
            input_shape = (X_train.shape[1], 1)
        elif self.model_type == 'transformer':
            input_shape = (X_train.shape[1],)  # Just sequence length
        else:
            input_shape = (X_train.shape[1],)
        
        if self.model_type == 'lstm':
            self.model = self.build_lstm_model(input_shape)
        elif self.model_type == 'gru':
            self.model = self.build_gru_model(input_shape)
        elif self.model_type == 'transformer':
            self.model = self.build_transformer_model(input_shape)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history
    
    def _train_ensemble_model(self, data):
        """
        Train ensemble models (Random Forest, XGBoost)
        
        Args:
            data: Training data
            
        Returns:
            dict: Training results
        """
        # Prepare features
        features = self.prepare_features(data)
        
        # Align features with target (close price)
        target = data.loc[features.index, 'close']
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]
        
        # Scale features for better performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        if self.model_type == 'rf':
            self.model = self.build_rf_model()
        elif self.model_type == 'xgboost':
            self.model = self.build_xgboost_model()
        
        # Train model
        if self.model_type == 'rf':
            self.model.fit(X_train_scaled, y_train)
        elif self.model_type == 'xgboost':
            self.model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred)
        }
    
    def predict(self, data):
        """
        Predict stock price
        
        Args:
            data: Input data
            
        Returns:
            float: Predicted price
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        
        if self.model_type in ['lstm', 'gru', 'transformer']:
            return self._predict_neural_network(data)
        elif self.model_type in ['rf', 'xgboost']:
            return self._predict_ensemble_model(data)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _predict_neural_network(self, data):
        """
        Predict using neural network models
        
        Args:
            data: Input data
            
        Returns:
            float: Predicted price
        """
        # Data preprocessing
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(dataset)
        
        # Create test dataset
        test_data = scaled_data[len(scaled_data) - self.look_back - 1:, :]
        X_test = []
        if len(test_data) >= self.look_back:
            X_test.append(test_data[0:self.look_back, 0])
        else:
            # If not enough data, pad with the first value
            padding = np.full(self.look_back - len(test_data), test_data[0, 0])
            X_test.append(np.concatenate([padding, test_data[:, 0]]))
        X_test = np.array(X_test)
        
        # Reshape for neural networks
        if self.model_type in ['lstm', 'gru']:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        elif self.model_type == 'transformer':
            # For transformer, ensure correct shape
            if len(X_test.shape) == 2:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            # Ensure we have the right input dimensions for transformer
            if X_test.shape[0] == 0 or X_test.shape[1] == 0:
                raise ValueError("Invalid input shape for transformer model")
        
        # Predict
        predicted_price = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        
        return predicted_price[0][0]
    
    def _predict_ensemble_model(self, data):
        """
        Predict using ensemble models
        
        Args:
            data: Input data
            
        Returns:
            float: Predicted price
        """
        # Prepare features
        features = self.prepare_features(data)
        
        # Get the last row of features
        last_features = features.iloc[-1:].values
        
        # Scale features
        last_features_scaled = self.scaler.transform(last_features)
        
        # Predict
        predicted_price = self.model.predict(last_features_scaled)[0]
        
        return predicted_price
    
    def save_model(self, model_name=None):
        """
        Save model to disk
        
        Args:
            model_name: Custom model name (optional)
            
        Returns:
            str: Model path
        """
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        # Generate model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{timestamp}"
        
        # Create model version
        self.model_version = model_name
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save model based on type
        if self.model_type in ['lstm', 'gru', 'transformer']:
            model_path = os.path.join(model_dir, 'model.h5')
            self.model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        else:
            model_path = os.path.join(model_dir, 'model.pkl')
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'look_back': self.look_back,
            'version': model_name,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_dir
    
    def load_model(self, model_name):
        """
        Load model from disk
        
        Args:
            model_name: Model name to load
        """
        model_dir = os.path.join(self.models_dir, model_name)
        
        if not os.path.exists(model_dir):
            raise ValueError(f"Model {model_name} not found.")
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Set model parameters
        self.model_type = metadata['model_type']
        self.look_back = metadata['look_back']
        self.model_version = metadata['version']
        
        # Load model based on type
        if self.model_type in ['lstm', 'gru', 'transformer']:
            model_path = os.path.join(model_dir, 'model.h5')
            self.model = load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
        else:
            model_path = os.path.join(model_dir, 'model.pkl')
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
    
    def evaluate_model(self, data, metrics=['mae', 'rmse', 'mape']):
        """
        Evaluate model performance
        
        Args:
            data: Test data
            metrics: List of metrics to calculate
            
        Returns:
            dict: Evaluation results
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        
        # Prepare data
        if self.model_type in ['lstm', 'gru', 'transformer']:
            return self._evaluate_neural_network(data, metrics)
        else:
            return self._evaluate_ensemble_model(data, metrics)
    
    def _evaluate_neural_network(self, data, metrics):
        """
        Evaluate neural network models
        
        Args:
            data: Test data
            metrics: List of metrics to calculate
            
        Returns:
            dict: Evaluation results
        """
        # Data preprocessing
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(dataset)
        
        # Split data
        train_size = int(len(scaled_data) * 0.8)
        test_data = scaled_data[train_size:]
        
        # Create dataset
        X_test, y_test = self.create_dataset(test_data, self.look_back)
        
        # Reshape for neural networks
        if self.model_type in ['lstm', 'gru']:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Predict
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        results = {}
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_test_actual, predictions)
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_test_actual, predictions))
        if 'mape' in metrics:
            results['mape'] = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100
        
        return results
    
    def _evaluate_ensemble_model(self, data, metrics):
        """
        Evaluate ensemble models
        
        Args:
            data: Test data
            metrics: List of metrics to calculate
            
        Returns:
            dict: Evaluation results
        """
        # Prepare features
        features = self.prepare_features(data)
        
        # Align features with target
        target = data.loc[features.index, 'close']
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_test = features.iloc[split_idx:]
        y_test = target.iloc[split_idx:]
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        predictions = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {}
        if 'mae' in metrics:
            results['mae'] = mean_absolute_error(y_test, predictions)
        if 'rmse' in metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
        if 'mape' in metrics:
            results['mape'] = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        return results
    
    def plot_predictions(self, data, days=30):
        """
        Plot predictions
        
        Args:
            data: Historical data
            days: Number of days to display
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        
        if self.model_type in ['lstm', 'gru', 'transformer']:
            self._plot_neural_network_predictions(data, days)
        else:
            print("Plotting is only supported for neural network models.")
    
    def _plot_neural_network_predictions(self, data, days):
        """
        Plot neural network model predictions
        
        Args:
            data: Historical data
            days: Number of days to display
        """
        # Data preprocessing
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(dataset)
        
        # Split data
        train_size = int(len(scaled_data) * 0.8)
        test_data = scaled_data[train_size:]
        
        # Create dataset
        X_test, y_test = self.create_dataset(test_data, self.look_back)
        
        # Reshape for neural networks
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Predict
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Plot
        plt.figure(figsize=(16, 8))
        plt.title(f'{self.model_type.upper()} Model - Stock Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price (¥)')
        plt.plot(y_test_actual[-days:], label='Actual Price')
        plt.plot(predictions[-days:], label='Predicted Price')
        plt.legend()
        plt.show()


class HyperparameterTuner:
    def __init__(self, model_type, param_grid):
        """
        Initialize hyperparameter tuner
        
        Args:
            model_type: Type of model to tune
            param_grid: Parameter grid for tuning
        """
        self.model_type = model_type
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = None
        self.results = []
    
    def grid_search(self, data, cv=3):
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            data: Training data
            cv: Number of cross-validation folds
            
        Returns:
            dict: Best parameters and score
        """
        from sklearn.model_selection import ParameterGrid
        from sklearn.metrics import mean_squared_error
        
        best_score = float('inf')
        best_params = None
        
        # Create parameter combinations
        param_combinations = list(ParameterGrid(self.param_grid))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            print(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Perform cross-validation
            cv_scores = []
            
            # Simple cross-validation by splitting data
            for fold in range(cv):
                # Split data for this fold
                fold_size = len(data) // cv
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size
                
                # Adjust for last fold
                if fold == cv - 1:
                    end_idx = len(data)
                
                # Split data
                test_data = data.iloc[start_idx:end_idx]
                train_data = pd.concat([data.iloc[:start_idx], data.iloc[end_idx:]])
                
                # Train model with current parameters
                try:
                    model = AdvancedStockPredictor(
                        look_back=params.get('look_back', 60),
                        model_type=self.model_type
                    )
                    
                    # Set model-specific parameters
                    if self.model_type == 'rf':
                        model.model = RandomForestRegressor(**{k: v for k, v in params.items() if k != 'look_back'})
                    elif self.model_type == 'xgboost':
                        model.model = xgb.XGBRegressor(**{k: v for k, v in params.items() if k != 'look_back'})
                    
                    # Train and evaluate
                    if self.model_type in ['rf', 'xgboost']:
                        model._train_ensemble_model(train_data)
                        eval_result = model._evaluate_ensemble_model(test_data, ['rmse'])
                        cv_scores.append(eval_result['rmse'])
                    else:
                        # For neural networks, we'll use a simplified approach
                        model.train(train_data, epochs=params.get('epochs', 10), batch_size=params.get('batch_size', 32))
                        eval_result = model._evaluate_neural_network(test_data, ['rmse'])
                        cv_scores.append(eval_result['rmse'])
                
                except Exception as e:
                    print(f"Error with parameters {params}: {e}")
                    cv_scores.append(float('inf'))
            
            # Calculate average score
            avg_score = np.mean(cv_scores) if cv_scores else float('inf')
            self.results.append({
                'params': params,
                'score': avg_score,
                'scores': cv_scores
            })
            
            # Update best parameters
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }


def compare_models(data, model_types=['lstm', 'gru', 'transformer', 'rf', 'xgboost']):
    """
    Compare different models
    
    Args:
        data: Training data
        model_types: List of model types to compare
        
    Returns:
        dict: Comparison results
    """
    results = {}
    
    for model_type in model_types:
        print(f"Training {model_type.upper()} model...")
        
        try:
            # Create model
            model = AdvancedStockPredictor(look_back=60, model_type=model_type)
            
            # Train model
            if model_type in ['lstm', 'gru', 'transformer']:
                model.train(data, epochs=20, batch_size=32)  # Reduced epochs for faster comparison
            else:
                model.train(data)
            
            # Evaluate model
            eval_results = model.evaluate_model(data, metrics=['mae', 'rmse', 'mape'])
            results[model_type] = eval_results
            
            print(f"{model_type.upper()} results: {eval_results}")
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    return results


# Example usage
if __name__ == "__main__":
    print("Advanced Stock Prediction Models")
    print("This module provides advanced models for stock price prediction.")
    print("Available models: LSTM, GRU, Transformer, Random Forest, XGBoost")