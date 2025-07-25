# 使用指南

本指南将介绍如何使用StockTracker项目。

## 命令行工具

`main.py` 是项目的主程序入口，提供了命令行工具，方便用户进行快速的股票分析和预测。

### 查看帮助信息

```bash
python main.py --help
```

### 预测股票价格

```bash
python main.py predict --symbol 002607 --model_type lstm
```

### 评估股票风险

```bash
python main.py assess_risk --symbol 002607
```

### 运行策略回测

```bash
python main.py backtest --symbol 002607 --strategy_type ma_crossover
```

## Web界面

项目提供了一个基于Streamlit的Web界面，方便用户进行交互式的股票分析和预测。

### 启动Web界面

```bash
streamlit run app.py
```

启动后，Web界面将在默认浏览器中打开，通常地址为 http://localhost:8501

### Web界面功能

Web界面提供以下功能模块：

1. **首页** - 项目介绍和快速开始指南
2. **股票分析** - 股票数据加载和基本信息查看
3. **技术指标** - 各种技术指标的计算和可视化
4. **价格预测** - 使用机器学习模型进行股票价格预测
5. **风险评估** - 全面的风险指标计算和评估
6. **投资组合** - 投资组合分析、优化和模拟
7. **回测分析** - 交易策略的回测和性能评估
8. **参数设置** - 用户偏好设置
9. **帮助文档** - 使用说明和键盘快捷键

## API调用

`models/predictors.py` 模块提供了一系列高级API，方便用户在自己的代码中进行调用。

### 预测股票价格

```python
import models.predictors as predictor

# 使用默认的LSTM模型预测指定股票的价格
result = predictor.predict_stock_price("002607")
print(result)
```

### 评估股票风险

```python
import models.predictors as predictor

# 评估股票风险
risk_result = predictor.assess_stock_risk("002607")
print(risk_result)
```

### 策略回测

```python
import models.predictors as predictor

# 运行移动平均线交叉策略回测
backtest_result = predictor.run_strategy_backtest("002607", strategy_type="ma_crossover", short_window=10, long_window=30)
print(backtest_result)