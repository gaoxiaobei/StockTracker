# Configuration Guide

This document provides a comprehensive guide to configuring the StockTracker application. It covers all aspects of configuration including environment variables, data sources, models, visualization, and web interface settings.

## 1. Configuration Overview

### Purpose of the Configs Module

The `configs` module in StockTracker is designed to manage all application configuration settings. While the module exists in the project structure, the current implementation relies on in-code configuration and parameter passing rather than external configuration files.

### How Configuration Works in the Project

StockTracker uses a combination of approaches for configuration:

1. **In-code configuration**: Most settings are defined directly in the code
2. **Function parameters**: Many settings can be customized through function parameters
3. **User interface settings**: Web interface allows users to configure certain parameters
4. **Model persistence**: Trained models and their configurations can be saved and loaded

### Configuration File Formats Supported

While StockTracker doesn't currently use external configuration files extensively, it supports:

- Python modules for programmatic configuration
- JSON for model metadata and some data export
- TOML for project dependencies (pyproject.toml)
- CSV for data import/export

## 2. Environment Variables

StockTracker currently doesn't rely heavily on environment variables for configuration. However, when deploying in different environments, you might consider using environment variables for:

| Environment Variable | Default Value | Usage | Security Considerations |
|---------------------|---------------|-------|-------------------------|
| `STOCKTRACKER_DATA_DIR` | `./data` | Directory for storing downloaded stock data | Low - Path information only |
| `STOCKTRACKER_MODELS_DIR` | `./models` | Directory for saving trained models | Low - Path information only |
| `STOCKTRACKER_LOG_LEVEL` | `INFO` | Application logging level | Low - Configuration only |
| `STOCKTRACKER_PROXY` | None | HTTP proxy for data fetching | Medium - Network configuration |

### Usage Examples

```bash
# Set data directory
export STOCKTRACKER_DATA_DIR=/path/to/data

# Set logging level
export STOCKTRACKER_LOG_LEVEL=DEBUG

# Set proxy for data fetching
export STOCKTRACKER_PROXY=http://proxy.company.com:8080
```

### Security Considerations

- Most configuration in StockTracker is not security-sensitive
- When deploying in production, ensure file system permissions are properly set
- If adding API keys for data sources in the future, use environment variables and never commit them to version control
- For web deployments, follow security best practices for Streamlit applications

## 3. Data Source Configuration

### Akshare Settings

StockTracker uses the `akshare` library for fetching stock data. The configuration is primarily done through function parameters:

```python
# In data/fetcher.py
def get_stock_data(symbol: str, period: str = "daily", start_date: Optional[str] = None,
                   end_date: Optional[str] = None, adjust: str = "qfq") -> pd.DataFrame:
```

Key parameters:
- `period`: Data frequency ("daily", "weekly", "monthly")
- `adjust`: Adjustment type ("qfq": 前复权, "hfq": 后复权, "": 不复权)
- `timeout`: Request timeout (default 30 seconds)

### Data Update Frequencies

StockTracker doesn't have automatic data update mechanisms. Data is fetched on-demand when requested through the UI or API. For production deployments, consider:

- Implementing a scheduled task to update data regularly
- Using a database to cache frequently accessed data
- Setting up appropriate cache expiration policies

### Cache Settings

Currently, StockTracker doesn't implement data caching. All data is fetched fresh from akshare on each request. For improved performance in production:

```python
# Example cache implementation
import functools
import time

@functools.lru_cache(maxsize=128)
def get_stock_data_cached(symbol, period="daily", start_date=None):
    # Implementation would go here
    pass
```

### Proxy Configuration

To configure a proxy for data fetching:

```python
# In data/fetcher.py, you could modify the akshare calls to use a proxy
import akshare as ak
import os

proxy = os.getenv('STOCKTRACKER_PROXY')
if proxy:
    # Configure proxy for requests (akshare is built on requests)
    proxies = {'http': proxy, 'https': proxy}
    # Pass proxies to akshare functions if supported
```

## 4. Model Configuration

### Default Model Parameters

StockTracker supports multiple models with different default parameters:

#### LSTM/GRU Models
```python
# In models/advanced.py
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
```

Default parameters:
- `look_back`: 60 (historical days used for prediction)
- `units`: 50 (neurons in LSTM layers)
- `dropout`: 0.2 (regularization)
- `epochs`: 100 (training iterations)
- `batch_size`: 32 (training batch size)

#### Transformer Model
```python
# Simplified transformer implementation
inputs = Input(shape=(input_shape[0],))
x = Dense(64, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(1)(x)
```

#### Ensemble Models (Random Forest, XGBoost)
```python
# Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

# XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
```

### Hyperparameter Tuning Options

StockTracker includes a `HyperparameterTuner` class for optimizing model parameters:

```python
# Example usage
param_grid = {
    'look_back': [30, 60, 90],
    'epochs': [50, 100, 200],
    'batch_size': [16, 32, 64]
}

tuner = HyperparameterTuner('lstm', param_grid)
results = tuner.grid_search(data)
```

Supported tuning methods:
- Grid search
- Future support for Bayesian optimization

### Model Persistence Settings

Models can be saved and loaded using the built-in persistence features:

```python
# Save model
predictor = AdvancedStockPredictor(look_back=60, model_type='lstm')
predictor.train(data)
model_path = predictor.save_model('my_lstm_model')

# Load model
new_predictor = AdvancedStockPredictor()
new_predictor.load_model('my_lstm_model')
```

Models are saved in:
- HDF5 format for neural networks (TensorFlow/Keras)
- Pickle format for ensemble models (scikit-learn, XGBoost)

### GPU Configuration

StockTracker uses TensorFlow for neural network models, which automatically utilizes available GPUs. To configure GPU usage:

```python
# In models/advanced.py
import tensorflow as tf

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

For multi-GPU setups:
```python
# Configure multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # Build and compile model here
    model = build_model()
    model.compile(...)
```

## 5. Visualization Settings

### Chart Themes

StockTracker uses Plotly for visualizations with the default "plotly_white" theme. To change themes:

```python
# In visualization/charts.py
fig.update_layout(
    template="plotly_dark",  # Alternative themes
    # template="ggplot2",
    # template="seaborn",
    # template="simple_white",
    # template="plotly_white",
)
```

### Default Colors

Visualization colors are defined in each plotting function:

```python
# In visualization/charts.py
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
```

To customize colors globally, you can modify the color arrays in each visualization function or create a centralized color configuration.

### Export Formats

Charts can be exported in multiple formats:

```python
# In visualization/charts.py
def save_plot(fig: go.Figure, filename: str, format: str = 'html') -> None:
    if format == 'html':
        fig.write_html(filename)
    else:
        fig.write_image(filename, format=format)
```

Supported formats:
- HTML (interactive charts)
- PNG
- JPEG
- PDF
- SVG

### Real-time Update Settings

For real-time charts, the window size can be configured:

```python
# In visualization/charts.py
def initialize_realtime_chart(symbol: str, window_size: int = 100) -> go.Figure:
    # window_size determines how many data points to display
    pass
```

## 6. Web Interface Configuration

### Streamlit Settings

Streamlit configuration is set at the beginning of the web application:

```python
# In ui/web.py and app.py
st.set_page_config(
    page_title="StockTracker - 股票价格预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Port Configuration

To change the port when running the Streamlit app:

```bash
streamlit run app.py --server.port 8502
```

Other Streamlit server options:
```bash
# Set server address
streamlit run app.py --server.address 0.0.0.0

# Disable browser auto-opening
streamlit run app.py --server.headless true

# Set maximum upload size (for data uploads)
streamlit run app.py --server.maxUploadSize 200
```

### Authentication

StockTracker currently doesn't implement authentication. For production deployments, consider:

1. Adding basic authentication with `streamlit-authenticator`
2. Implementing OAuth with services like Google or GitHub
3. Using reverse proxy authentication (nginx, Apache)

Example with streamlit-authenticator:
```python
import streamlit_authenticator as stauth

# User credentials would be stored securely
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')
```

### Performance Settings

To optimize performance for the web interface:

```bash
# Increase message size limits
streamlit run app.py --server.maxMessageSize 200

# Enable caching
streamlit run app.py --global.cache true

# Set log level
streamlit run app.py --logger.level info
```

## 7. Custom Configuration

### How to Create Custom Config Files

While StockTracker doesn't currently use external configuration files, you can add them:

1. Create a `config.py` file in the `configs` directory
2. Define configuration classes or dictionaries
3. Import and use in your modules

Example `configs/app_config.py`:
```python
class AppConfig:
    # Data settings
    DEFAULT_PERIOD = "daily"
    DEFAULT_ADJUST = "qfq"
    
    # Model settings
    DEFAULT_LOOKBACK = 60
    DEFAULT_EPOCHS = 100
    
    # Visualization settings
    DEFAULT_THEME = "plotly_white"
    DEFAULT_WINDOW_SIZE = 100
    
    # Paths
    DATA_DIR = "./data"
    MODELS_DIR = "./models"
```

### Configuration Inheritance

For complex configurations, you can implement inheritance:

```python
class BaseConfig:
    DEBUG = False
    TESTING = False

class DevelopmentConfig(BaseConfig):
    DEBUG = True
    DATA_DIR = "./data/dev"

class ProductionConfig(BaseConfig):
    DATA_DIR = "/var/lib/stocktracker/data"
```

### Validation Rules

When implementing custom configuration, add validation:

```python
def validate_config(config):
    """Validate configuration settings"""
    if config.DEFAULT_LOOKBACK <= 0:
        raise ValueError("DEFAULT_LOOKBACK must be positive")
    
    if config.DEFAULT_EPOCHS <= 0:
        raise ValueError("DEFAULT_EPOCHS must be positive")
    
    # Add more validation rules as needed
    return True
```

### Best Practices

1. **Separate configuration from code**: Use external files for environment-specific settings
2. **Use environment variables for secrets**: Never hardcode sensitive information
3. **Validate configuration at startup**: Check for invalid values early
4. **Document all configuration options**: Provide clear explanations for each setting
5. **Use sensible defaults**: Ensure the application works with minimal configuration
6. **Support configuration reloading**: Allow updating settings without restarting the application
7. **Group related settings**: Organize configuration in logical sections
8. **Version configuration files**: Track changes to configuration over time

## Conclusion

StockTracker's configuration is currently simple and code-based, which works well for a research project.