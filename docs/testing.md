# Testing Documentation

This document provides comprehensive information about testing in the StockTracker project, including how to run tests, write new tests, and maintain test quality.

## Table of Contents

1. [Testing Overview](#testing-overview)
2. [Running Tests](#running-tests)
3. [Test Structure](#test-structure)
4. [Writing Tests](#writing-tests)
5. [Test Coverage](#test-coverage)
6. [Performance Testing](#performance-testing)
7. [Debugging Tests](#debugging-tests)

## Testing Overview

### Types of Tests Used

StockTracker uses a comprehensive testing approach that includes:

- **Functional Tests**: Verify that features work as expected
- **Integration Tests**: Test interactions between different components
- **Regression Tests**: Ensure that bug fixes remain fixed
- **Model Tests**: Validate machine learning model performance
- **Performance Tests**: Check system performance under various conditions

### Testing Framework

StockTracker uses a custom testing framework implemented in Python rather than traditional frameworks like pytest or unittest. The testing approach is based on:

- Direct function calls to test components
- Custom test runners in individual test files
- Comprehensive result reporting with pass/fail status
- JSON-based test reports for detailed analysis

Key test files include:
- `tests/test_all.py`: Comprehensive test suite covering all major functionality
- `tests/test_portfolio.py`: Specific tests for portfolio analysis features
- `tests/test_transformer.py`: Tests for the Transformer model
- `tests/test_fixes.py`: Regression tests for bug fixes

### Test Coverage Goals

The project aims for comprehensive test coverage with the following goals:

- **Core functionality**: 100% coverage for critical components like data fetching, prediction models, and risk assessment
- **UI components**: 80% coverage for visualization and web interface features
- **Edge cases**: 70% coverage for error handling and boundary conditions
- **Performance**: Regular benchmarking of key operations

### Continuous Integration

While the project doesn't currently have a configured CI/CD pipeline, tests are designed to be easily integrated with CI systems. All test files can be run independently and produce consistent results.

## Running Tests

### How to Run All Tests

To run the complete test suite:

```bash
python tests/test_all.py
```

This command executes all tests in the comprehensive test suite and generates a detailed report including:
- Total number of tests executed
- Number of passed and failed tests
- Pass rate percentage
- Detailed failure information
- JSON report saved to a timestamped file

### Running Specific Test Modules

Individual test modules can be run separately:

```bash
# Run portfolio-specific tests
python tests/test_portfolio.py

# Run Transformer model tests
python tests/test_transformer.py

# Run regression tests
python tests/test_fixes.py
```

### Running Individual Tests

To run specific test functions, you can modify the test files or create temporary scripts. For example, to test just the data fetching functionality:

```python
import data.fetcher as data_fetcher

def test_single_function():
    try:
        stock_data = data_fetcher.get_stock_data('000001', period='daily', start_date='20240101', adjust='qfq')
        if not stock_data.empty:
            print("✓ Data fetching test passed")
        else:
            print("✗ Data fetching test failed - Empty data returned")
    except Exception as e:
        print(f"✗ Data fetching test failed with exception: {e}")

if __name__ == "__main__":
    test_single_function()
```

### Test Output Formats

Tests produce output in multiple formats:

1. **Console Output**: Real-time pass/fail status with detailed messages
2. **JSON Reports**: Detailed test results saved to timestamped JSON files
3. **Summary Reports**: Final pass/fail statistics with failure details

Example JSON report structure:
```json
{
  "timestamp": "2025-07-25T10:30:45.123456",
  "total_tests": 15,
  "passed_tests": 13,
  "failed_tests": 2,
  "test_details": [
    {
      "test_name": "Data Fetching Function",
      "success": true,
      "details": "Successfully retrieved 250 records",
      "timestamp": "2025-07-25T10:30:45.123456"
    }
  ]
}
```

### CI/CD Integration

To integrate tests with a CI/CD pipeline, you can use the exit codes from test scripts:

```bash
# Run tests and check exit code
python tests/test_all.py
if [ $? -eq 0 ]; then
  echo "All tests passed"
else
  echo "Some tests failed"
  exit 1
fi
```

## Test Structure

### Directory Organization

The test directory follows a logical structure:

```
tests/
├── test_all.py          # Comprehensive test suite
├── test_fixes.py        # Regression tests for bug fixes
├── test_portfolio.py    # Portfolio analysis tests
├── test_transformer.py  # Transformer model tests
└── __init__.py          # Package initialization
```

### Test File Naming Conventions

Test files follow these naming conventions:

- `test_all.py`: Main comprehensive test suite
- `test_<feature>.py`: Tests for specific features or components
- `test_fixes.py`: Regression tests for bug fixes
- All test files should start with `test_` prefix

### Test Class Structure

The main test suite uses a class-based approach in `tests/test_all.py`:

```python
class StockTrackerTestSuite:
    def __init__(self):
        """Initialize test suite with result tracking"""
        self.test_results = []
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
    
    def log_result(self, test_name: str, success: bool, details: str = ""):
        """Log test results with detailed information"""
        # Implementation details...
    
    def test_data_fetching(self):
        """Test data fetching functionality"""
        # Test implementation...
```

### Fixture Usage

StockTracker uses a custom approach to test fixtures rather than traditional fixture frameworks:

1. **Test Data Generation**: Functions to create sample data for testing
2. **Mock Objects**: Custom implementations to simulate external dependencies
3. **Setup/Teardown**: Manual initialization and cleanup in test functions

Example of test data generation:
```python
def create_test_data():
    """Create sample stock data for testing"""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    test_data = pd.DataFrame({
        'open': np.random.rand(30) * 10 + 100,
        'high': np.random.rand(30) * 12 + 105,
        'low': np.random.rand(30) * 8 + 95,
        'close': np.random.rand(30) * 10 + 100,
        'volume': np.random.randint(1000, 10000, 30)
    }, index=dates)
    return test_data
```

## Writing Tests

### Test Structure Guidelines

When writing new tests, follow these guidelines:

1. **Descriptive Test Names**: Use clear, descriptive names that indicate what is being tested
2. **Single Responsibility**: Each test should focus on one specific functionality
3. **Setup and Teardown**: Properly initialize and clean up test resources
4. **Assertions**: Use clear assertions to verify expected outcomes
5. **Error Handling**: Test both success and failure scenarios

Example test structure:
```python
def test_new_feature():
    """Test the new feature functionality."""
    print("\n=== Testing New Feature ===")
    
    try:
        # Setup test data
        test_data = create_test_data()
        
        # Execute function
        result = new_feature_function(test_data)
        
        # Validate result
        assert isinstance(result, pd.Series)
        assert len(result) > 0
        assert not result.isna().all()
        
        print("✓ New feature test passed")
        return True
        
    except Exception as e:
        print(f"✗ New feature test failed: {e}")
        return False
```

### Mocking External Dependencies

For external dependencies like APIs or databases, use mock data or stub implementations:

```python
def mock_data_fetcher(symbol, **kwargs):
    """Mock data fetcher for testing"""
    # Return predefined test data instead of calling external API
    return create_test_data()

# In your test
original_fetcher = data_fetcher.get_stock_data
data_fetcher.get_stock_data = mock_data_fetcher

try:
    # Run test with mocked dependency
    result = function_that_uses_data_fetcher()
    # Validate result
finally:
    # Restore original function
    data_fetcher.get_stock_data = original_fetcher
```

### Data Generation for Tests

Create reusable functions for generating test data:

```python
def generate_stock_data(days=100, base_price=100.0):
    """Generate realistic stock data for testing"""
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    prices = [base_price]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(10000, 100000, days)
    }, index=dates)
    
    return data
```

### Testing ML Models

For machine learning models, focus on:

1. **Training Verification**: Ensure models can train without errors
2. **Prediction Validation**: Verify predictions are reasonable (not NaN or infinite)
3. **Performance Benchmarks**: Compare against baseline performance
4. **Edge Cases**: Test with insufficient or anomalous data

Example ML model test:
```python
def test_lstm_model():
    """Test LSTM model training and prediction"""
    # Get test data
    stock_data = data_fetcher.get_stock_data('000001', period='daily', 
                                           start_date='20240101', adjust='qfq')
    
    # Create model instance
    model = LSTMStockPredictor(look_back=60)
    
    # Test training
    history = model.train(stock_data)
    assert history is not None
    
    # Test prediction
    prediction = model.predict(stock_data)
    assert isinstance(prediction, (int, float))
    assert not np.isnan(prediction)
    assert prediction > 0  # Price should be positive
```

### Testing Visualizations

For visualization components:

1. **Chart Creation**: Verify charts can be created without errors
2. **Data Integration**: Ensure visualizations properly display data
3. **Interactive Features**: Test interactive components when applicable

Example visualization test:
```python
def test_chart_creation():
    """Test chart creation functionality"""
    try:
        # Create test data
        test_data = generate_stock_data(30)
        
        # Initialize visualizer
        visualizer = visualization.StockVisualizer()
        
        # Test chart creation
        fig = visualizer.plot_interactive_price_chart(test_data, "TEST", "Test Chart")
        assert fig is not None
        
        print("✓ Chart creation test passed")
        return True
    except Exception as e:
        print(f"✗ Chart creation test failed: {e}")
        return False
```

## Test Coverage

### How to Measure Coverage

Currently, StockTracker doesn't use automated coverage tools. Coverage is measured manually by:

1. **Feature Mapping**: Ensuring each major feature has corresponding tests
2. **Code Path Analysis**: Verifying different execution paths are tested
3. **Edge Case Testing**: Checking boundary conditions and error handling

### Coverage Reporting

Test results are reported in two ways:

1. **Console Output**: Real-time pass/fail status during test execution
2. **JSON Reports**: Detailed reports saved to files with comprehensive information

Example console output:
```
=== 测试数据获取功能 ===
✓ 数据获取功能: 通过
✓ 股票信息获取功能: 通过

=== 测试股票价格预测功能 ===
  测试 LSTM 模型...
✓ LSTM预测模型: 通过 - 预测价格: 12.34
  测试 GRU 模型...
✓ GRU预测模型: 通过 - 预测价格: 12.36
```

### Coverage Requirements

The project aims for the following coverage targets:

- **Core Components**: 100% test coverage for critical data processing and prediction functions
- **UI Components**: 80% coverage for visualization and web interface features
- **Error Handling**: 70% coverage for exception handling and edge cases
- **New Features**: 100% coverage for all newly added functionality

### Exclusions

Some components may be excluded from coverage requirements:

1. **External API Calls**: Direct calls to external services (use mocking instead)
2. **UI Elements**: Purely cosmetic UI elements without functional logic
3. **Temporary Debug Code**: Code that is clearly marked for temporary use only
4. **Platform-Specific Code**: Code that only runs on specific platforms

## Performance Testing

### Benchmark Tests

Performance benchmarks are implemented as part of the regular test suite:

```python
import time

def benchmark_data_fetching():
    """Benchmark data fetching performance"""
    start_time = time.time()
    
    # Execute function to benchmark
    stock_data = data_fetcher.get_stock_data('000001', period='daily', 
                                           start_date='20200101', adjust='qfq')
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Data fetching took {execution_time:.2f} seconds for {len(stock_data)} records")
    
    # Performance criteria
    assert execution_time < 10.0  # Should complete within 10 seconds
    return execution_time
```

### Load Testing

Load testing focuses on:

1. **Concurrent Operations**: Testing multiple simultaneous requests
2. **Large Dataset Processing**: Handling large volumes of data
3. **Memory Usage**: Monitoring memory consumption during operations

Example load test:
```python
def test_concurrent_predictions():
    """Test concurrent prediction requests"""
    import threading
    import time
    
    results = []
    errors = []
    
    def worker(symbol):
        try:
            result = predictor.predict_stock_price(symbol, days=1, model_type='lstm')
            results.append(result)
        except Exception as e:
            errors.append(str(e))
    
    # Create multiple threads
    symbols = ['000001', '000002', '000003', '000004', '000005']
    threads = []
    
    start_time = time.time()
    
    for symbol in symbols:
        thread = threading.Thread(target=worker, args=(symbol,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    
    print(f"Processed {len(symbols)} concurrent requests in {end_time - start_time:.2f} seconds")
    print(f"Successful: {len(results)}, Errors: {len(errors)}")
    
    assert len(errors) == 0, f"Errors occurred: {errors}"
```

### Memory Usage Testing

Monitor memory usage during critical operations:

```python
import psutil
import os

def test_memory_usage():
    """Test memory usage during operations"""
    process = psutil.Process(os.getpid())
    
    # Measure initial memory
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Perform memory-intensive operation
    large_dataset = generate_stock_data(days=10000)  # Large dataset
    result = complex_analysis_function(large_dataset)
    
    # Measure final memory
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (increase: {memory_increase:.1f}MB)")
    
    # Memory usage should be reasonable
    assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f}MB"
```

### Model Performance Evaluation

For ML models, evaluate performance metrics:

```python
def test_model_accuracy():
    """Test model prediction accuracy"""
    # Get historical data
    stock_data = data_fetcher.get_stock_data('000001', period='daily', 
                                           start_date='20230101', 
                                           end_date='20240101', 
                                           adjust='qfq')
    
    # Use part of data for training
    train_data = stock_data.iloc[:-30]  # All but last 30 days
    test_data = stock_data.iloc[-30:]   # Last 30 days for testing
    
    # Train model
    model = LSTMStockPredictor(look_back=60)
    model.train(train_data)
    
    # Make predictions for test period
    predictions = []
    actual_prices = test_data['close'].values
    
    for i in range(len(test_data)):
        # Predict one day at a time
        pred_data = train_data.append(test_data.iloc[:i+1])
        prediction = model.predict(pred_data)
        predictions.append(prediction)
    
    # Calculate accuracy metrics
    mae = np.mean(np.abs(np.array(predictions) - actual_prices))
    rmse = np.sqrt(np.mean((np.array(predictions) - actual_prices) ** 2))
    
    print(f"Model Accuracy - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Accuracy should be within reasonable bounds
    assert mae < actual_prices.mean() * 0.1, f"MAE too high: {mae}"
```

## Debugging Tests

### Common Test Failures

Common test failure patterns and solutions:

1. **Data Fetching Failures**:
   - Check network connectivity
   - Verify stock symbol validity
   - Ensure date formats are correct
   - Handle API rate limiting

2. **Model Training Errors**:
   - Check data quality and format
   - Verify sufficient data for training
   - Ensure model parameters are valid
   - Handle memory constraints

3. **Visualization Issues**:
   - Verify data structure and types
   - Check required libraries are installed
   - Handle empty or invalid data

### Debugging Techniques

1. **Verbose Logging**: Add detailed logging to trace execution flow
2. **Intermediate Results**: Print or save intermediate values for inspection
3. **Step-by-Step Execution**: Run tests in debug mode to step through code
4. **Minimal Test Cases**: Create simplified test cases to isolate issues

Example debugging approach:
```python
def debug_test_function():
    """Debug a failing test function"""
    print("Starting debug test...")
    
    # Step 1: Verify input data
    stock_data = data_fetcher.get_stock_data('000001', period='daily', 
                                           start_date='20240101', adjust='qfq')
    print(f"Retrieved {len(stock_data)} records")
    print(f"Data columns: {stock_data.columns.tolist()}")
    print(f"Data types: {stock_data.dtypes}")
    
    # Step 2: Check data quality
    print(f"Missing values: {stock_data.isnull().sum().sum()}")
    print(f"Data range: {stock_data.index.min()} to {stock_data.index.max()}")
    
    # Step 3: Test function with verbose output
    try:
        result = problematic_function(stock_data, debug=True)
        print(f"Function completed successfully: {result}")
    except Exception as e:
        print(f"Function failed with error: {e}")
        import traceback
        traceback.print_exc()
```

### Logging in Tests

Use structured logging for better debugging:

```python
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('StockTrackerTests')

def test_with_logging():
    """Test function with detailed logging"""
    logger.info("Starting data fetching test")
    
    try:
        stock_data = data_fetcher.get_stock_data('000001', period='daily', 
                                               start_date='20240101', adjust='qfq')
        logger.info(f"Successfully fetched {len(stock_data)} records")
        
        # Validate data
        required_columns = ['close', 'open', 'high', 'low', 'volume']
        missing_columns = [col for col in required_columns if col not in stock_data.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        logger.info("Data validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return False
```

### Test Isolation

Ensure tests are isolated to prevent interference:

1. **Independent Data**: Use separate test data for each test
2. **Clean State**: Reset any modified state between tests
3. **Resource Management**: Properly close files, connections, etc.

Example of test isolation:
```python
def test_with_isolation():
    """Test with proper isolation"""
    # Create isolated test environment
    test_data = generate_stock_data(100)
    
    # Store original state
    original_config = get_current_config()
    
    try:
        # Modify configuration for test
        set_test_config()
        
        # Run test
        result = function_under_test(test_data)
        
        # Validate result
        assert result is not None
        return True
        
    finally:
        # Restore original state
        restore_config(original_config)
        
        # Clean up test data if needed
        cleanup_test_data()
```

---

This testing documentation provides a comprehensive guide for testing in the StockTracker project. As the project evolves, this documentation should be updated to reflect new testing practices and requirements.