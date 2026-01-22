import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import sys
import os
# Add the parent directory to the path so we can import from performance_optimizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from performance_optimizer import model_cache

# 修正tensorflow.keras导入
Sequential = tf.keras.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout


class StockPredictor:
    def __init__(self, look_back=60):
        """
        初始化股票预测器
        
        Args:
            look_back: 用于预测的历史数据天数
        """
        self.look_back = look_back
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def create_dataset(self, dataset, look_back=1):
        """
        创建数据集
        
        Args:
            dataset: 输入数据集
            look_back: 回看天数
            
        Returns:
            tuple: (输入数据, 输出数据)
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    def build_model(self, input_shape):
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入数据的形状
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
        self.model = model
        return model
    
    def train(self, data, epochs=100, batch_size=32):
        """
        训练模型
        
        Args:
            data: 训练数据
            epochs: 训练轮数
            batch_size: 批次大小
        """
        # Check if model is already cached
        params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'look_back': self.look_back
        }

        cached_model, cached_scaler = model_cache.get_cached_model(
            "base_lstm", data, params
        )

        if cached_model is not None and cached_scaler is not None:
            self.model = cached_model
            self.scaler = cached_scaler
            return None

        # 数据预处理
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(dataset)
        
        # 划分训练集和测试集
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # 创建数据集
        X_train, y_train = self.create_dataset(train_data, self.look_back)
        X_test, y_test = self.create_dataset(test_data, self.look_back)
        
        # 重塑数据以适应LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 构建模型
        if self.model is None:
            self.model = self.build_model((X_train.shape[1], 1))
        
        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Cache the trained model
        model_cache.cache_model("base_lstm", data, params, self.model, self.scaler)
        
        return history
    
    def predict(self, data):
        """
        预测股票价格
        
        Args:
            data: 输入数据
            
        Returns:
            numpy.array: 预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 数据预处理
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(dataset)
        
        # 创建测试数据集
        test_data = scaled_data[len(scaled_data) - self.look_back - 1:, :]
        X_test = []
        X_test.append(test_data[0:self.look_back, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 预测
        predicted_price = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price)
        
        return predicted_price[0][0]
    
    def plot_predictions(self, data, days=30):
        """
        绘制预测结果
        
        Args:
            data: 历史数据
            days: 显示的天数
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 数据预处理
        dataset = data['close'].values.reshape(-1, 1)
        scaled_data = self.scaler.transform(dataset)
        
        # 获取最近的数据用于预测
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # 创建测试数据集
        X_test, y_test = self.create_dataset(test_data, self.look_back)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 预测
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # 绘图
        plt.figure(figsize=(16, 8))
        plt.title('股票价格预测')
        plt.xlabel('日期')
        plt.ylabel('价格 (¥)')
        plt.plot(y_test_actual[-days:], label='实际价格')
        plt.plot(predictions[-days:], label='预测价格')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # 这里可以添加测试代码
    print("股票预测模型类已定义")