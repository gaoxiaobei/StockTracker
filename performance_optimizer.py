#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Optimizer for StockTracker

This module provides performance optimizations for the StockTracker system,
including model caching, batch processing, and resource management.
"""

import os
import gc
import joblib
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib


class ModelCache:
    """
    A cache system for ML models to avoid repeated training
    """
    def __init__(self, cache_dir=".model_cache", max_age_hours=24):
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        self._ensure_cache_dir_exists()
    
    def _ensure_cache_dir_exists(self):
        """Create cache directory if it doesn't exist."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def _generate_cache_key(self, model_type: str, data_hash: str, params: Dict[str, Any]) -> str:
        """Generate a unique cache key for the model."""
        cache_str = f"{model_type}_{data_hash}_{str(sorted(params.items()))}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """Get the full path for the cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.joblib")
    
    def _is_cache_valid(self, file_path: str) -> bool:
        """Check if cache file exists and is not older than max_age_hours."""
        if not os.path.exists(file_path):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        return (datetime.now() - file_time).total_seconds() < self.max_age_hours * 3600
    
    def get_cached_model(self, model_type: str, data: pd.DataFrame, params: Dict[str, Any]):
        """Try to retrieve a cached model."""
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        cache_key = self._generate_cache_key(model_type, data_hash, params)
        cache_file_path = self._get_cache_file_path(cache_key)
        
        if self._is_cache_valid(cache_file_path):
            try:
                with open(cache_file_path, 'rb') as f:
                    model_data = joblib.load(f)
                print(f"从缓存加载 {model_type} 模型")
                return model_data['model'], model_data['scaler']
            except Exception:
                pass  # If cache loading fails, return None to trigger retraining
        return None, None
    
    def cache_model(self, model_type: str, data: pd.DataFrame, params: Dict[str, Any], 
                    model, scaler):
        """Cache a trained model."""
        data_hash = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()
        cache_key = self._generate_cache_key(model_type, data_hash, params)
        cache_file_path = self._get_cache_file_path(cache_key)
        
        try:
            model_data = {
                'model': model,
                'scaler': scaler,
                'timestamp': datetime.now(),
                'data_hash': data_hash
            }
            with open(cache_file_path, 'wb') as f:
                joblib.dump(model_data, f)
            print(f"{model_type} 模型已缓存")
        except Exception as e:
            print(f"缓存模型时出错: {str(e)}")


class DataLoader:
    """
    Optimized data loader with preprocessing pipeline
    """
    def __init__(self, look_back=60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def preprocess_data(self, data: pd.DataFrame, column: str = 'close'):
        """Optimized preprocessing pipeline."""
        # Extract the target column
        dataset = data[column].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)
        
        return scaled_data
    
    def create_sequences(self, data: np.ndarray):
        """Create sequences for time series prediction."""
        X, y = [], []
        for i in range(len(data) - self.look_back - 1):
            X.append(data[i:(i + self.look_back), 0])
            y.append(data[i + self.look_back, 0])
        return np.array(X), np.array(y)
    
    def prepare_training_data(self, data: pd.DataFrame, column: str = 'close', 
                           train_ratio: float = 0.8):
        """Prepare training and test data."""
        scaled_data = self.preprocess_data(data, column)
        
        # Split data
        train_size = int(len(scaled_data) * train_ratio)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)
        
        # Reshape for LSTM/GRU
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, y_train, X_test, y_test


class MemoryOptimizer:
    """
    Memory optimization utilities
    """
    @staticmethod
    def clear_session():
        """Clear TensorFlow session to free memory."""
        tf.keras.backend.clear_session()
        gc.collect()
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != "object":
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        return df


def optimize_tensorflow():
    """Configure TensorFlow for better performance."""
    # Configure GPU memory growth if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for better performance (if supported and GPU present)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("TensorFlow: 已启用 GPU 混合精度优化")
        except RuntimeError as e:
            print(f"GPU配置错误: {e}")
    else:
        print("TensorFlow: 未检测到 GPU，使用默认精度设置")


def batch_predict(data: pd.DataFrame, model_func, batch_size: int = 1000):
    """
    Perform batch prediction to handle large datasets efficiently.
    """
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        batch_result = model_func(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
    
    return results


# Singleton instance for global use
model_cache = ModelCache()
data_loader = DataLoader()
memory_optimizer = MemoryOptimizer()


if __name__ == "__main__":
    print("Performance Optimizer initialized")
    print("Optimizing TensorFlow...")
    optimize_tensorflow()
    print("Done!")