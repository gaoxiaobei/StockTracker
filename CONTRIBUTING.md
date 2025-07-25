# Contributing to StockTracker

Thank you for your interest in contributing to StockTracker! This document provides guidelines and instructions for contributing to the project. By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Development Environment Setup](#development-environment-setup)
- [Code Contributions](#code-contributions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Reporting Issues](#reporting-issues)

## How to Contribute

There are several ways you can contribute to StockTracker:

1. **Code Contributions**: Implement new features, fix bugs, or improve existing functionality
2. **Documentation**: Improve documentation, add examples, or translate content
3. **Testing**: Write tests, report bugs, or help verify fixes
4. **Community**: Answer questions, participate in discussions, or help others

## Development Environment Setup

To set up your development environment for StockTracker, follow these steps:

### Prerequisites

- Python 3.12+
- uv package manager
- Git

### Installation Steps

1. **Fork and Clone the Repository**

   ```bash
   git clone https://github.com/your-username/StockTracker.git
   cd StockTracker
   ```

2. **Install Dependencies**

   StockTracker uses `uv` as the package manager. Install dependencies with:

   ```bash
   uv sync
   ```

   If you don't have `uv` installed, install it first:

   ```bash
   pip install uv
   ```

3. **Verify Installation**

   Run the main program to verify everything is set up correctly:

   ```bash
   python main.py --help
   ```

4. **Run Tests**

   To ensure your environment is properly set up, run the test suite:

   ```bash
   python -m pytest tests/
   ```

## Code Contributions

We welcome code contributions that improve StockTracker. Before starting work on a significant feature or change, please:

1. Check existing issues to see if your idea is already being discussed
2. Open a new issue to discuss your proposed changes
3. Wait for feedback from maintainers before implementing

### Development Workflow

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our coding standards

3. Add or update tests as necessary

4. Run tests to ensure everything works:
   ```bash
   python -m pytest tests/
   ```

5. Commit your changes with a clear, descriptive commit message

6. Push your branch and open a pull request

## Pull Request Process

1. Ensure your code follows our [coding standards](#coding-standards)
2. Include tests for any new functionality
3. Update documentation as needed
4. Verify all tests pass
5. Submit your pull request with a clear description of the changes
6. Respond to any feedback during the review process

### Pull Request Guidelines

- Keep pull requests focused on a single feature or bug fix
- Include a clear description of what changes were made and why
- Reference any related issues using GitHub's issue linking
- Ensure your branch is up to date with the main branch before submitting

## Coding Standards

### Python Code Style

StockTracker follows PEP 8 style guidelines. Key points include:

- Use 4 spaces for indentation (no tabs)
- Limit lines to 88 characters (to match Black's default)
- Use descriptive variable and function names
- Write docstrings for all public functions, classes, and modules
- Follow naming conventions:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Documentation

- Write clear, concise docstrings for all public functions and classes
- Use Google-style docstrings
- Include type hints for function parameters and return values
- Update README.md and other documentation when adding new features

### Example Docstring

```python
def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate the moving average of a data series.
    
    Args:
        data: The time series data to calculate the moving average for.
        window: The number of periods to include in the moving average.
        
    Returns:
        A pandas Series containing the moving average values.
        
    Raises:
        ValueError: If window is not a positive integer.
    """
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Window must be a positive integer")
    
    return data.rolling(window=window).mean()
```

## Reporting Issues

### Bug Reports

When reporting a bug, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. Your environment information (Python version, OS, etc.)
5. Any relevant error messages or logs
6. Screenshots if applicable

### Feature Requests

For feature requests, please include:

1. A clear description of the proposed feature
2. The problem it would solve or benefit it would provide
3. Any implementation ideas or considerations
4. Examples of how it would be used

### Security Issues

Please do not report security vulnerabilities through public GitHub issues. Instead, please contact the maintainers directly.

## Additional Resources

- [README.md](README.md) - Project overview and usage instructions
- [docs/](docs/) - Additional documentation
- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Usage Guide](docs/usage.md) - Comprehensive usage documentation

Thank you for contributing to StockTracker!