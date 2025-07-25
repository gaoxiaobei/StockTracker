# StockTracker - 股票价格预测系统

StockTracker 是一个基于机器学习的股票价格预测系统，使用 akshare 获取股票数据，支持多种先进的机器学习模型进行预测，包括 LSTM、GRU、Transformer 以及集成学习模型（随机森林、XGBoost）。

## 功能特性

- 使用 akshare 获取实时股票数据
- 支持多种先进的机器学习模型进行股票价格预测：
  - LSTM（长短期记忆网络）
  - GRU（门控循环单元）
  - Transformer（变压器模型）
  - 随机森林（Random Forest）
  - XGBoost（极端梯度提升）
- 支持超参数调优（网格搜索、贝叶斯优化）
- 支持模型持久化和版本控制
- 支持特征工程（技术指标、时间特征等）
- 支持模型验证和评估（MAE、RMSE、MAPE等指标）
- 支持回测功能评估模型历史表现
- 支持策略回测功能，包括多种交易策略
- 支持交易成本模拟（佣金、滑点等）
- 支持业绩评估指标（累计收益、年化收益、最大回撤、夏普比率等）
- 支持基准比较功能
- 支持仓位管理功能
- 支持止损和止盈机制
- 支持参数优化功能（网格搜索、遗传算法）
- 支持蒙特卡洛模拟评估策略稳健性
- 支持股票价格趋势预测
- 提供简单的投资建议
- 支持可视化股票价格预测结果
- 提供高级可视化功能：
  - 交互式图表（使用Plotly）
  - K线图（蜡烛图）可视化
  - 技术指标叠加图（支持置信区间）
  - 热力图（相关性矩阵、行业表现等，支持聚类）
  - 3D可视化（风险-收益-时间立方体，支持颜色和大小映射）
  - 动画图表（价格变化动画，支持成交量显示）
  - 置信区间可视化
  - 模型性能对比图表
  - 风险指标可视化
  - 实时数据可视化
  - 综合仪表板
- 提供常用技术指标计算功能
- 提供全面的风险评估功能：
  - 波动率（标准差）计算
  - VaR（风险价值）计算 - 历史模拟法和参数法
  - 最大回撤（Max Drawdown）计算
  - 夏普比率计算
  - 贝塔系数计算（相对于市场指数）
  - Alpha值计算
  - 风险等级评估和投资建议
  - 蒙特卡洛模拟预测潜在损失
  - 相关性分析（股票与市场、股票间相关性）
- 提供投资组合分析功能：
  - 投资组合构建和权重分配
  - 投资组合预期收益和风险计算
  - 现代投资组合理论（MPT）相关计算
  - 有效前沿计算和可视化
  - 资本资产定价模型（CAPM）相关计算
  - 夏普比率最大化优化算法
  - 投资组合风险贡献分析
  - 投资组合优化（均值-方差优化、最小方差组合、风险平价组合、Black-Litterman模型）
  - 投资组合绩效评估（Alpha、Beta、信息比率、特雷诺比率、业绩归因分析）
  - 蒙特卡洛模拟用于投资组合优化和情景分析

## 项目结构

```
StockTracker/
├── analysis/           # 分析模块
│   ├── __init__.py     # 分析模块初始化文件
│   ├── backtest.py     # 策略回测模块
│   ├── portfolio.py    # 投资组合分析模块
│   ├── risk.py         # 风险评估模块
│   └── technical.py    # 技术指标计算模块
├── configs/            # 配置模块
│   └── __init__.py     # 配置模块初始化文件
├── data/               # 数据模块
│   ├── __init__.py     # 数据模块初始化文件
│   └── fetcher.py      # 数据获取模块
├── docs/               # 文档目录
│   └── __init__.py     # 文档模块初始化文件
├── examples/           # 示例脚本
│   ├── __init__.py     # 示例模块初始化文件
│   └── demo.py         # 演示脚本
├── models/             # 模型模块
│   ├── __init__.py     # 模型模块初始化文件
│   ├── advanced.py     # 高级机器学习模型模块
│   ├── base.py         # 基础机器学习模型模块
│   └── predictors.py   # 预测器模块
├── tests/              # 测试模块
│   ├── __init__.py     # 测试模块初始化文件
│   ├── test_all.py     # 综合测试脚本
│   ├── test_fixes.py   # 修复测试脚本
│   ├── test_portfolio.py # 投资组合测试脚本
│   └── test_transformer.py # Transformer模型测试脚本
├── ui/                 # 用户界面模块
│   ├── __init__.py     # 用户界面模块初始化文件
│   └── web.py          # Web界面应用
├── utils/              # 工具模块
│   └── __init__.py     # 工具模块初始化文件
├── visualization/      # 可视化模块
│   ├── __init__.py     # 可视化模块初始化文件
│   └── charts.py       # 图表可视化模块
├── .gitignore          # Git忽略文件
├── .python-version     # Python版本文件
├── app.py              # Web应用入口
├── main.py             # 主程序入口
├── pyproject.toml      # 项目依赖配置
├── README.md           # 项目说明文档
├── requirements.txt    # 项目依赖列表
└── uv.lock             # uv锁定文件
```

## 安装依赖

项目使用 `uv` 作为包管理器，依赖项已在 `pyproject.toml` 中配置。

```bash
# 安装依赖
uv sync
```

## 使用方法

### 1. 运行主程序

```bash
uv run main.py
```

### 2. 运行Web界面

```bash
uv run streamlit run app.py
```
启动后，Web界面将在默认浏览器中打开，通常地址为 http://localhost:8501

### 3. API使用示例

`models/predictors.py` 模块提供了一系列高级API，方便用户进行调用。

#### 预测股票价格

```python
import models.predictors as predictor

# 使用默认的LSTM模型预测指定股票的价格
result = predictor.predict_stock_price("002607")
print(result)

# 使用GRU模型预测
result = predictor.predict_stock_price("002607", model_type="gru")
print(result)```

#### 评估股票风险

```python
import models.predictors as predictor

# 评估股票风险
risk_result = predictor.assess_stock_risk("002607")
print(risk_result)```

#### 策略回测

```python
import models.predictors as predictor

# 运行移动平均线交叉策略回测
backtest_result = predictor.run_strategy_backtest("002607", strategy_type="ma_crossover", short_window=10, long_window=30)
print(backtest_result)
```

## 模块说明

### `analysis` - 分析模块
- `backtest.py`: 策略回测模块
- `portfolio.py`: 投资组合分析模块
- `risk.py`: 风险评估模块
- `technical.py`: 技术指标计算模块

### `data` - 数据模块
- `fetcher.py`: 数据获取模块，封装了akshare的数据接口。

### `models` - 模型模块
- `base.py`: 基础机器学习模型模块。
- `advanced.py`: 高级机器学习模型模块，包括LSTM, GRU, Transformer等。
- `predictors.py`: 预测器模块，提供了高级API。

### `ui` - 用户界面模块
- `web.py`: 基于Streamlit的Web界面。

### `visualization` - 可视化模块
- `charts.py`: 图表可视化模块，基于Plotly。

## 依赖项

主要依赖项请参考 `pyproject.toml` 文件。
- `akshare`
- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `xgboost`
- `joblib`
- `plotly`
- `streamlit`

## 注意事项

1. 股票预测仅用于学习和研究目的，不构成投资建议
2. 机器学习模型的预测结果可能存在误差
3. 投资有风险，入市需谨慎

## 许可证

本项目仅供学习和研究使用。