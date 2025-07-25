# 安装指南

本指南将引导您完成StockTracker项目的安装和配置。

## 环境要求

- Python 3.12+
- uv

## 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/your-username/StockTracker.git
   cd StockTracker
   ```

2. **安装依赖**

   项目使用 `uv` 作为包管理器，依赖项已在 `pyproject.toml` 中配置。

   ```bash
   uv sync
   ```

   如果未安装 `uv`，请先使用 `pip` 安装：

   ```bash
   pip install uv
   ```

3. **验证安装**

   运行主程序，查看是否能正常输出帮助信息：

   ```bash
   python main.py --help
   ```

   如果看到帮助信息，说明安装成功。