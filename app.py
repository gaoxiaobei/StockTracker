#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StockTracker - 股票价格预测系统 Web界面
使用Streamlit构建的交互式Web应用
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
import time
import json
import base64
from io import StringIO

# Import our existing modules
import models.predictors as predictor
import data.fetcher as data_fetcher
import analysis.technical as indicators
import models.advanced as advanced_model
import analysis.risk as risk_assessment
import analysis.portfolio as portfolio
import analysis.backtest as backtest
import visualization.charts as visualization

# Set page configuration
st.set_page_config(
    page_title="StockTracker - 股票价格预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design and keyboard shortcuts info
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.2rem;
        color: #ff7f0e;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .warning {
        color: #ffc107;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .section-header {
            font-size: 1.3rem;
        }
    }
    /* Keyboard shortcuts info */
    .keyboard-shortcuts {
        background-color: #e9ecef;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 4px 4px 0;
    }
    .keyboard-shortcuts h4 {
        margin-top: 0;
        color: #1f77b4;
    }
    .shortcut-item {
        margin: 0.5rem 0;
    }
    .shortcut-key {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-family: monospace;
        margin-right: 0.5rem;
    }
</style>
""")

# Session state initialization
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'current_symbol' not in st.session_state:
    st.session_state.current_symbol = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None
if 'portfolio_result' not in st.session_state:
    st.session_state.portfolio_result = None
if 'backtest_result' not in st.session_state:
    st.session_state.backtest_result = None
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

def main():
    """Main application function"""
    # App header
    st.title("📈 StockTracker 股票价格预测系统")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("导航")
        page = st.selectbox(
            "选择功能页面",
            [
                "🏠 首页",
                "🔍 股票分析",
                "📊 技术指标",
                "🔮 价格预测",
                "⚠️ 风险评估",
                "💼 投资组合",
                "📈 回测分析",
                "⚙️ 参数设置",
                "ℹ️ 帮助文档"
            ]
        )
        
        # Display current stock info in sidebar
        st.markdown("---")
        st.subheader("当前股票")
        if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
            current_price = st.session_state.stock_data['close'].iloc[-1]
            st.metric("当前价格", f"¥{current_price:.2f}")
        else:
            st.info("请先选择股票")
    
    # Route to appropriate page
    if page == "🏠 首页":
        show_home_page()
    elif page == "🔍 股票分析":
        show_stock_analysis_page()
    elif page == "📊 技术指标":
        show_technical_indicators_page()
    elif page == "🔮 价格预测":
        show_prediction_page()
    elif page == "⚠️ 风险评估":
        show_risk_assessment_page()
    elif page == "💼 投资组合":
        show_portfolio_page()
    elif page == "📈 回测分析":
        show_backtest_page()
    elif page == "⚙️ 参数设置":
        show_settings_page()
    elif page == "ℹ️ 帮助文档":
        show_help_page()

def show_home_page():
    """Display home page"""
    st.header("欢迎使用StockTracker")
    
    # Add data upload section
    st.subheader("上传数据")
    uploaded_file = st.file_uploader("上传股票数据文件 (CSV格式)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read uploaded CSV file
            dataframe = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if all(col in dataframe.columns for col in required_columns):
                # Convert date column to datetime
                dataframe['date'] = pd.to_datetime(dataframe['date'])
                dataframe.set_index('date', inplace=True)
                
                # Store in session state
                st.session_state.stock_data = dataframe
                st.success("数据上传成功！请前往'🔍 股票分析'页面查看数据。")
                st.info("💡 提示：您可以在'🔍 股票分析'页面查看上传的数据详情和进行进一步分析。")
            else:
                st.error(f"CSV文件缺少必要列: {', '.join(required_columns)}")
                st.warning("请确保CSV文件包含以下列：date, open, high, low, close, volume")
        except pd.errors.EmptyDataError:
            st.error("上传的文件为空，请检查文件内容。")
        except pd.errors.ParserError:
            st.error("CSV文件格式不正确，请检查文件内容。")
        except Exception as e:
            st.error(f"数据上传失败: {str(e)}")
            st.warning("请确保上传的文件是有效的CSV格式，并包含正确的数据结构。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 系统功能
        
        StockTracker 是一个基于机器学习的股票价格预测系统，提供以下功能：
        
        - **📈 股票价格预测** - 使用多种机器学习模型预测股票价格
        - **📊 技术指标分析** - 计算和可视化各种技术指标
        - **⚠️ 风险评估** - 全面的风险指标计算和评估
        - **💼 投资组合分析** - 投资组合构建、优化和分析
        - **📈 回测分析** - 策略回测和性能评估
        - **🎨 高级可视化** - 交互式图表和仪表板
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 快速开始
        
        1. 在左侧导航栏选择"🔍 股票分析"
        2. 输入股票代码（如：002607）
        3. 选择分析功能页面
        4. 查看分析结果和图表
        
        ### 📊 支持的模型
        
        - LSTM（长短期记忆网络）
        - GRU（门控循环单元）
        - Transformer（变压器模型）
        - 随机森林
        - XGBoost
        """)
        
        # Quick stock input
        st.markdown("### ⚡ 快速分析")
        quick_symbol = st.text_input("输入股票代码快速分析", placeholder="例如：002607")
        if st.button("快速分析") and quick_symbol:
            with st.spinner("正在获取股票数据..."):
                stock_data = data_fetcher.get_stock_data(quick_symbol, period="daily", start_date="20200101", adjust="qfq")
                if not stock_data.empty:
                    st.session_state.stock_data = stock_data
                    st.session_state.current_symbol = quick_symbol
                    st.success(f"已加载股票 {quick_symbol} 的数据")
                else:
                    st.error("无法获取股票数据，请检查股票代码")

def show_stock_analysis_page():
    """Display stock analysis page"""
    st.header("🔍 股票分析")
    
    # Stock input section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input("股票代码", value="002607", help="输入股票代码，如：002607")
    
    with col2:
        start_date = st.date_input("开始日期", value=pd.to_datetime("2020-01-01"))
    
    with col3:
        st.markdown("")
        st.markdown("")
        load_data = st.button("加载数据")
    
    if load_data and symbol:
        with st.spinner("正在获取股票数据..."):
            try:
                stock_data = data_fetcher.get_stock_data(
                    symbol,
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    adjust="qfq"
                )
                
                if not stock_data.empty:
                    st.session_state.stock_data = stock_data
                    st.success(f"成功加载 {symbol} 的股票数据，共 {len(stock_data)} 条记录")
                    st.info("💡 提示：您可以在页面下方查看股票数据详情、图表和进行技术指标分析。")
                    
                    # Get stock info
                    stock_info = data_fetcher.get_stock_info(symbol)
                    if stock_info:
                        st.subheader("股票基本信息")
                        info_cols = st.columns(4)
                        info_cols[0].metric("股票名称", stock_info.get("股票简称", "未知"))
                        info_cols[1].metric("行业", stock_info.get("行业", "未知"))
                        info_cols[2].metric("总市值", stock_info.get("总市值", "未知"))
                        info_cols[3].metric("流通市值", stock_info.get("流通市值", "未知"))
                else:
                    st.error("无法获取股票数据，请检查股票代码和日期")
                    st.warning("💡 提示：请确保输入的股票代码正确，并且网络连接正常。")
            except Exception as e:
                st.error(f"获取股票数据时出错: {str(e)}")
                st.warning("💡 提示：请检查网络连接，或稍后再试。如果问题持续存在，请联系技术支持。")
    
    # Display stock data if available
    if st.session_state.stock_data is not None and not st.session_state.stock_data.empty:
        st.subheader("股票价格数据")
        
        # Display basic info
        latest_data = st.session_state.stock_data.iloc[-1]
        metrics_cols = st.columns(5)
        metrics_cols[0].metric("当前价格", f"¥{latest_data['close']:.2f}")
        metrics_cols[1].metric("开盘价", f"¥{latest_data['open']:.2f}")
        metrics_cols[2].metric("最高价", f"¥{latest_data['high']:.2f}")
        metrics_cols[3].metric("最低价", f"¥{latest_data['low']:.2f}")
        metrics_cols[4].metric("成交量", f"{latest_data['volume']:,}")
        
        # Display price chart
        st.subheader("价格走势")
        fig = px.line(st.session_state.stock_data.tail(120), y='close', title=f"{symbol} 股票价格走势")
        fig.update_layout(xaxis_title="日期", yaxis_title="价格 (¥)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.subheader("历史数据")
        st.dataframe(st.session_state.stock_data.tail(20))
        
        # Export data
        csv = st.session_state.stock_data.to_csv().encode('utf-8')
        st.download_button(
            label="下载CSV数据",
            data=csv,
            file_name=f"{symbol}_stock_data.csv",
            mime="text/csv"
        )

def show_technical_indicators_page():
    """Display technical indicators page"""
    st.header("📊 技术指标分析")
    
    if st.session_state.stock_data is None or st.session_state.stock_data.empty:
        st.warning("请先在'🔍 股票分析'页面加载股票数据")
        return
    
    # Select indicators to calculate
    st.subheader("选择技术指标")
    indicators_selected = st.multiselect(
        "选择要计算的技术指标",
        ["移动平均线", "指数移动平均线", "相对强弱指数(RSI)", "MACD", "布林带", "随机指标", "能量潮(OBV)"],
        ["移动平均线", "相对强弱指数(RSI)"]
    )
    
    if st.button("计算技术指标") and indicators_selected:
        with st.spinner("正在计算技术指标..."):
            try:
                stock_data = st.session_state.stock_data
                indicator_results = {}
                
                # Calculate selected indicators
                if "移动平均线" in indicators_selected:
                    with st.spinner("计算移动平均线..."):
                        sma_20 = indicators.simple_moving_average(stock_data, period=20)
                        sma_50 = indicators.simple_moving_average(stock_data, period=50)
                        indicator_results["SMA 20"] = sma_20
                        indicator_results["SMA 50"] = sma_50
                
                if "指数移动平均线" in indicators_selected:
                    with st.spinner("计算指数移动平均线..."):
                        ema_20 = indicators.exponential_moving_average(stock_data, period=20)
                        ema_50 = indicators.exponential_moving_average(stock_data, period=50)
                        indicator_results["EMA 20"] = ema_20
                        indicator_results["EMA 50"] = ema_50
                
                if "相对强弱指数(RSI)" in indicators_selected:
                    with st.spinner("计算相对强弱指数..."):
                        rsi = indicators.relative_strength_index(stock_data, period=14)
                        indicator_results["RSI"] = rsi
                
                if "MACD" in indicators_selected:
                    with st.spinner("计算MACD..."):
                        macd_data = indicators.moving_average_convergence_divergence(stock_data)
                        indicator_results["MACD"] = macd_data['macd_line']
                        indicator_results["Signal"] = macd_data['signal_line']
                
                if "布林带" in indicators_selected:
                    with st.spinner("计算布林带..."):
                        bb_data = indicators.bollinger_bands(stock_data, period=20)
                        indicator_results["Upper Band"] = bb_data['upper_band']
                        indicator_results["Middle Band"] = bb_data['middle_band']
                        indicator_results["Lower Band"] = bb_data['lower_band']
                
                if "随机指标" in indicators_selected:
                    with st.spinner("计算随机指标..."):
                        stoch_data = indicators.stochastic_oscillator(stock_data, k_period=14, d_period=3)
                        indicator_results["Stoch %K"] = stoch_data['k_percent']
                        indicator_results["Stoch %D"] = stoch_data['d_percent']
                
                if "能量潮(OBV)" in indicators_selected:
                    with st.spinner("计算能量潮..."):
                        obv = indicators.on_balance_volume(stock_data)
                        indicator_results["OBV"] = obv
                
                # Display results
                if indicator_results:
                    st.subheader("技术指标图表")
                    
                    # Plot price with selected indicators
                    with st.spinner("生成技术指标图表..."):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['close'],
                            mode='lines',
                            name='收盘价',
                            line=dict(color='blue')
                        ))
                        
                        # Add indicator traces
                        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                        color_idx = 0
                        
                        for name, data in indicator_results.items():
                            if name in ['SMA 20', 'SMA 50', 'EMA 20', 'EMA 50', 'Middle Band']:
                                fig.add_trace(go.Scatter(
                                    x=stock_data.index[-len(data):],
                                    y=data,
                                    mode='lines',
                                    name=name,
                                    line=dict(color=colors[color_idx % len(colors)])
                                ))
                                color_idx += 1
                        
                        fig.update_layout(
                            title="价格与技术指标",
                            xaxis_title="日期",
                            yaxis_title="价格",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display indicator values
                    st.subheader("最新指标值")
                    indicator_cols = st.columns(len(indicator_results))
                    for i, (name, data) in enumerate(indicator_results.items()):
                        if len(data) > 0:
                            latest_value = data.iloc[-1]
                            indicator_cols[i].metric(name, f"{latest_value:.2f}")
                
            except Exception as e:
                st.error(f"计算技术指标时出错: {str(e)}")

def show_prediction_page():
    """Display prediction page"""
    st.header("🔮 股票价格预测")
    
    if st.session_state.stock_data is None or st.session_state.stock_data.empty:
        st.warning("请先在'🔍 股票分析'页面加载股票数据")
        return
    
    # Model selection
    st.subheader("模型设置")
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "选择预测模型",
            ["lstm", "gru", "transformer", "rf", "xgboost"],
            format_func=lambda x: {
                "lstm": "LSTM (长短期记忆网络)",
                "gru": "GRU (门控循环单元)",
                "transformer": "Transformer (变压器模型)",
                "rf": "随机森林",
                "xgboost": "XGBoost"
            }[x]
        )
    
    with col2:
        days = st.slider("预测天数", 1, 30, 5)
    
    if st.button("开始预测"):
        with st.spinner(f"正在使用 {model_type.upper()} 模型进行预测..."):
            try:
                # Get prediction
                result = predictor.predict_stock_price(
                    st.session_state.stock_data.index.name or "symbol",
                    days=days,
                    model_type=model_type
                )
                
                if "error" not in result:
                    st.session_state.prediction_result = result
                    st.success("预测完成！")
                    st.info("💡 提示：预测结果仅供参考，不构成投资建议。投资有风险，入市需谨慎。")
                    
                    # Display results
                    st.subheader("预测结果")
                    pred_cols = st.columns(4)
                    pred_cols[0].metric("当前价格", f"¥{result['current_price']:.2f}")
                    pred_cols[1].metric("预测价格", f"¥{result['predicted_price']:.2f}")
                    pred_cols[2].metric(
                        "价格变化",
                        f"¥{result['price_change']:.2f}",
                        f"{result['price_change_percent']:.2f}%"
                    )
                    
                    # Color-coded change indicator
                    if result['price_change_percent'] > 0:
                        pred_cols[3].metric(
                            "变化方向",
                            "上涨",
                            f"{result['price_change_percent']:.2f}%",
                            delta_color="normal"
                        )
                    else:
                        pred_cols[3].metric(
                            "变化方向",
                            "下跌",
                            f"{result['price_change_percent']:.2f}%",
                            delta_color="inverse"
                        )
                    
                    # Investment suggestion
                    st.subheader("投资建议")
                    if result['price_change_percent'] > 5:
                        st.success("📈 强烈买入 - 预测价格上涨超过5%")
                    elif result['price_change_percent'] > 2:
                        st.success("📈 买入 - 预测价格上涨超过2%")
                    elif result['price_change_percent'] > 0:
                        st.info("➡️ 持有 - 预测价格略有上涨")
                    elif result['price_change_percent'] > -2:
                        st.info("↔️ 持有 - 预测价格基本持平")
                    elif result['price_change_percent'] > -5:
                        st.warning("📉 减持 - 预测价格略有下跌")
                    else:
                        st.error("🚨 卖出 - 预测价格大幅下跌超过5%")
                    
                    # Export prediction results
                    st.subheader("导出结果")
                    pred_json = json.dumps(result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="下载预测结果(JSON)",
                        data=pred_json,
                        file_name=f"{result['symbol']}_prediction_{model_type}.json",
                        mime="application/json"
                    )
                    
                    # Generate and download report
                    if st.button("生成预测报告"):
                        try:
                            # Create a simple report
                            report_content = f"""
# 股票预测报告

## 基本信息
- 股票代码: {result['symbol']}
- 股票名称: {result['stock_name']}
- 预测模型: {model_type.upper()}
- 预测天数: {days}

## 预测结果
- 当前价格: ¥{result['current_price']:.2f}
- 预测价格: ¥{result['predicted_price']:.2f}
- 价格变化: ¥{result['price_change']:.2f} ({result['price_change_percent']:.2f}%)

## 投资建议
"""
                            if result['price_change_percent'] > 5:
                                report_content += "强烈买入 - 预测价格上涨超过5%"
                            elif result['price_change_percent'] > 2:
                                report_content += "买入 - 预测价格上涨超过2%"
                            elif result['price_change_percent'] > 0:
                                report_content += "持有 - 预测价格略有上涨"
                            elif result['price_change_percent'] > -2:
                                report_content += "持有 - 预测价格基本持平"
                            elif result['price_change_percent'] > -5:
                                report_content += "减持 - 预测价格略有下跌"
                            else:
                                report_content += "卖出 - 预测价格大幅下跌超过5%"
                            
                            report_content += f"""

## 报告生成时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 免责声明
本报告仅用于学习和研究目的，不构成投资建议。机器学习模型的预测结果可能存在误差，投资有风险，入市需谨慎。
"""
                            
                            st.download_button(
                                label="下载预测报告",
                                data=report_content,
                                file_name=f"{result['symbol']}_prediction_report.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"生成报告时出错: {str(e)}")
                    
                    # Visualization
                    st.subheader("预测可视化")
                    try:
                        with st.spinner("正在生成预测图表..."):
                            predictor.plot_prediction_with_confidence_interval(
                                st.session_state.stock_data.index.name or "symbol",
                                model_type=model_type,
                                days=days
                            )
                    except Exception as e:
                        st.info("可视化功能需要在本地环境中运行以显示图表")
                        st.warning("💡 提示：预测图表需要在本地环境中运行才能显示。")
                        
                else:
                    st.error(f"预测失败: {result['error']}")
                    st.warning("💡 提示：请检查网络连接，或尝试使用其他预测模型。")
            
            except Exception as e:
                st.error(f"预测过程中出错: {str(e)}")
                st.warning("💡 提示：预测过程出现异常，请稍后再试或联系技术支持。")

def show_risk_assessment_page():
    """Display risk assessment page"""
    st.header("⚠️ 风险评估")
    
    if st.session_state.stock_data is None or st.session_state.stock_data.empty:
        st.warning("请先在'🔍 股票分析'页面加载股票数据")
        return
    
    # Risk assessment parameters
    st.subheader("评估参数")
    market_symbol = st.text_input("市场指数代码", value="sh000001", help="用于计算贝塔系数等指标")
    
    if st.button("开始风险评估"):
        with st.spinner("正在进行风险评估..."):
            try:
                # Perform risk assessment
                result = predictor.assess_stock_risk(
                    st.session_state.stock_data.index.name or "symbol",
                    market_symbol=market_symbol
                )
                
                if "error" not in result:
                    st.session_state.risk_result = result
                    st.success("风险评估完成！")
                    st.info("💡 提示：风险评估结果基于历史数据计算，仅供参考。投资有风险，决策需谨慎。")
                    
                    # Display risk metrics
                    st.subheader("风险指标")
                    risk_cols = st.columns(4)
                    risk_cols[0].metric("波动率", f"{result['volatility']:.4f}")
                    risk_cols[1].metric("历史VaR(95%)", f"{result['var_historical']:.4f}")
                    risk_cols[2].metric("最大回撤", f"{result['max_drawdown']:.4f}")
                    risk_cols[3].metric("夏普比率", f"{result['sharpe_ratio']:.4f}")
                    
                    # Additional metrics
                    st.subheader("更多指标")
                    more_cols = st.columns(4)
                    more_cols[0].metric("贝塔系数", f"{result['beta']:.4f}")
                    more_cols[1].metric("Alpha值", f"{result['alpha']:.4f}")
                    more_cols[2].metric("市场相关性", f"{result['correlation_with_market']:.4f}")
                    more_cols[3].metric("数据点数", result['data_points'])
                    
                    # Risk level assessment
                    st.subheader("风险评级")
                    risk_level = result['risk_level']
                    if risk_level['risk_level'] == "低风险":
                        st.success(f"🟢 {risk_level['risk_level']}: {risk_level['explanation']}")
                        st.info(f"投资建议: {risk_level['investment_advice']}")
                    elif risk_level['risk_level'] == "中等风险":
                        st.warning(f"🟡 {risk_level['risk_level']}: {risk_level['explanation']}")
                        st.info(f"投资建议: {risk_level['investment_advice']}")
                    else:
                        st.error(f"🔴 {risk_level['risk_level']}: {risk_level['explanation']}")
                        st.info(f"投资建议: {risk_level['investment_advice']}")
                    
                    # Monte Carlo simulation results
                    st.subheader("蒙特卡洛模拟")
                    mc_results = result['monte_carlo_simulation']
                    mc_cols = st.columns(5)
                    mc_cols[0].metric("预期损失", f"{mc_results['expected_loss']:.4f}")
                    mc_cols[1].metric("VaR 95%", f"{mc_results['var_95']:.4f}")
                    mc_cols[2].metric("VaR 99%", f"{mc_results['var_99']:.4f}")
                    mc_cols[3].metric("最小损失", f"{mc_results['min_loss']:.4f}")
                    mc_cols[4].metric("最大损失", f"{mc_results['max_loss']:.4f}")
                    
                    # Export risk assessment results
                    st.subheader("导出结果")
                    risk_json = json.dumps(result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="下载风险评估结果(JSON)",
                        data=risk_json,
                        file_name=f"{st.session_state.stock_data.index.name or 'symbol'}_risk_assessment.json",
                        mime="application/json"
                    )
                    
                    # Generate and download report
                    if st.button("生成风险评估报告"):
                        try:
                            # Create a simple report
                            report_content = f"""
# 股票风险评估报告

## 基本信息
- 股票代码: {st.session_state.stock_data.index.name or 'symbol'}
- 市场指数: {market_symbol}

## 风险指标
- 波动率: {result['volatility']:.4f}
- 历史VaR(95%): {result['var_historical']:.4f}
- 最大回撤: {result['max_drawdown']:.4f}
- 夏普比率: {result['sharpe_ratio']:.4f}

## 更多指标
- 贝塔系数: {result['beta']:.4f}
- Alpha值: {result['alpha']:.4f}
- 市场相关性: {result['correlation_with_market']:.4f}
- 数据点数: {result['data_points']}

## 风险评级
- 风险等级: {risk_level['risk_level']}
- 风险解释: {risk_level['explanation']}
- 投资建议: {risk_level['investment_advice']}

## 蒙特卡洛模拟结果
- 预期损失: {mc_results['expected_loss']:.4f}
- VaR 95%: {mc_results['var_95']:.4f}
- VaR 99%: {mc_results['var_99']:.4f}
- 最小损失: {mc_results['min_loss']:.4f}
- 最大损失: {mc_results['max_loss']:.4f}

## 报告生成时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 免责声明
本报告仅用于学习和研究目的，不构成投资建议。机器学习模型的预测结果可能存在误差，投资有风险，入市需谨慎。
"""
                            
                            st.download_button(
                                label="下载风险评估报告",
                                data=report_content,
                                file_name=f"{st.session_state.stock_data.index.name or 'symbol'}_risk_assessment_report.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"生成报告时出错: {str(e)}")
                    
                else:
                    st.error(f"风险评估失败: {result['error']}")
                    st.warning("💡 提示：请检查网络连接，或稍后再试。")
            
            except Exception as e:
                st.error(f"风险评估过程中出错: {str(e)}")
                st.warning("💡 提示：风险评估过程出现异常，请稍后再试或联系技术支持。")

def show_portfolio_page():
    """Display portfolio analysis page"""
    st.header("💼 投资组合分析")
    
    # Portfolio setup
    st.subheader("投资组合设置")
    
    # Initialize portfolio in session state if not exists
    if 'portfolio_stocks' not in st.session_state:
        st.session_state.portfolio_stocks = [
            {"symbol": "002607", "name": "中公教育", "weight": 0.4},
            {"symbol": "000001", "name": "平安银行", "weight": 0.3},
            {"symbol": "600036", "name": "招商银行", "weight": 0.3}
        ]
    
    # Display current portfolio
    st.markdown("#### 当前投资组合")
    portfolio_df = pd.DataFrame(st.session_state.portfolio_stocks)
    edited_df = st.data_editor(
        portfolio_df,
        num_rows="dynamic",
        use_container_width=True
    )
    
    # Update portfolio
    if st.button("更新投资组合"):
        # Store previous portfolio for undo functionality
        previous_portfolio = st.session_state.portfolio_stocks.copy()
        st.session_state.portfolio_stocks = edited_df.to_dict('records')
        st.success("投资组合已更新")
        
        # Add undo button
        if st.button("撤销更新"):
            st.session_state.portfolio_stocks = previous_portfolio
            st.success("投资组合更新已撤销")
    
    # Portfolio analysis actions
    st.subheader("分析选项")
    analysis_action = st.selectbox(
        "选择分析类型",
        ["投资组合分析", "投资组合优化", "蒙特卡洛模拟"]
    )
    
    if analysis_action == "投资组合分析":
        if st.button("分析投资组合"):
            with st.spinner("正在分析投资组合..."):
                try:
                    # Prepare stocks dict
                    stocks_dict = {
                        stock['symbol']: {"symbol": stock['symbol'], "name": stock['name']}
                        for stock in st.session_state.portfolio_stocks
                    }
                    
                    # Analyze portfolio
                    with st.spinner("计算投资组合指标..."):
                        result = predictor.analyze_portfolio(stocks_dict)
                    
                    if "error" not in result and result.get("success"):
                        st.session_state.portfolio_result = result
                        st.success("投资组合分析完成！")
                        st.info("💡 提示：投资组合分析结果基于历史数据计算，仅供参考。投资有风险，决策需谨慎。")
                        
                        # Display results
                        metrics = result["metrics"]
                        st.subheader("投资组合指标")
                        metric_cols = st.columns(3)
                        metric_cols[0].metric("预期收益", f"{metrics['expected_return']:.4f}")
                        metric_cols[1].metric("风险(波动率)", f"{metrics['volatility']:.4f}")
                        metric_cols[2].metric("夏普比率", f"{metrics['sharpe_ratio']:.4f}")
                        
                        # Risk contribution
                        st.subheader("风险贡献分析")
                        risk_contrib = result["risk_contribution"]
                        if "error" not in risk_contrib:
                            contrib_data = pd.DataFrame({
                                "股票": risk_contrib["symbols"],
                                "风险贡献(%)": [f"{p:.2f}%" for p in risk_contrib["percentage_contributions"]]
                            })
                            st.table(contrib_data)
                            
                            # Export portfolio analysis results
                            st.subheader("导出结果")
                            portfolio_json = json.dumps(result, ensure_ascii=False, indent=2)
                            st.download_button(
                                label="下载投资组合分析结果(JSON)",
                                data=portfolio_json,
                                file_name="portfolio_analysis.json",
                                mime="application/json"
                            )
                            
                            # Generate and download report
                            if st.button("生成投资组合分析报告"):
                                try:
                                    # Create a simple report
                                    report_content = f"""
# 投资组合分析报告

## 投资组合指标
- 预期收益: {metrics['expected_return']:.4f}
- 风险(波动率): {metrics['volatility']:.4f}
- 夏普比率: {metrics['sharpe_ratio']:.4f}

## 风险贡献分析
"""
                                    if "error" not in risk_contrib:
                                        for i, symbol in enumerate(risk_contrib["symbols"]):
                                            report_content += f"- {symbol}: {risk_contrib['percentage_contributions'][i]:.2f}%\n"
                                    
                                    report_content += f"""

## 报告生成时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 免责声明
本报告仅用于学习和研究目的，不构成投资建议。机器学习模型的预测结果可能存在误差，投资有风险，入市需谨慎。
"""
                                    
                                    st.download_button(
                                        label="下载投资组合分析报告",
                                        data=report_content,
                                        file_name="portfolio_analysis_report.txt",
                                        mime="text/plain"
                                    )
                                except Exception as e:
                                    st.error(f"生成报告时出错: {str(e)}")
                    else:
                        st.error(f"投资组合分析失败: {result.get('error', '未知错误')}")
                        st.warning("💡 提示：请检查网络连接，或稍后再试。")
                
                except Exception as e:
                    st.error(f"投资组合分析过程中出错: {str(e)}")
                    st.warning("💡 提示：投资组合分析过程出现异常，请稍后再试或联系技术支持。")
    
    elif analysis_action == "投资组合优化":
        optimization_method = st.selectbox(
            "选择优化方法",
            ["mean_variance", "minimum_variance"],
            format_func=lambda x: {
                "mean_variance": "均值-方差优化",
                "minimum_variance": "最小方差组合优化"
            }[x]
        )
        
        if st.button("优化投资组合"):
            with st.spinner("正在优化投资组合..."):
                try:
                    # Prepare stocks dict
                    stocks_dict = {
                        stock['symbol']: {"symbol": stock['symbol'], "name": stock['name']}
                        for stock in st.session_state.portfolio_stocks
                    }
                    
                    # Optimize portfolio
                    with st.spinner("优化投资组合..."):
                        result = predictor.optimize_portfolio(stocks_dict, method=optimization_method)
                    
                    if "error" not in result and result.get("success"):
                        st.success("投资组合优化完成！")
                        st.info("💡 提示：投资组合优化结果基于历史数据计算，仅供参考。投资有风险，决策需谨慎。")
                        
                        # Display results
                        st.subheader("优化结果")
                        opt_cols = st.columns(3)
                        opt_cols[0].metric("优化后预期收益", f"{result['expected_return']:.4f}")
                        opt_cols[1].metric("优化后风险", f"{result['volatility']:.4f}")
                        opt_cols[2].metric("优化后夏普比率", f"{result['sharpe_ratio']:.4f}")
                        
                        # Optimized weights
                        st.subheader("优化后权重")
                        weights_data = pd.DataFrame({
                            "股票": result["symbols"],
                            "优化权重(%)": [f"{w*100:.2f}%" for w in result["weights"]]
                        })
                        st.table(weights_data)
                    else:
                        st.error(f"投资组合优化失败: {result.get('error', '未知错误')}")
                        st.warning("💡 提示：请检查网络连接，或稍后再试。")
                
                except Exception as e:
                    st.error(f"投资组合优化过程中出错: {str(e)}")
                    st.warning("💡 提示：投资组合优化过程出现异常，请稍后再试或联系技术支持。")
    
    elif analysis_action == "蒙特卡洛模拟":
        n_simulations = st.slider("模拟次数", 1000, 10000, 5000, step=1000)
        
        if st.button("运行蒙特卡洛模拟"):
            with st.spinner("正在进行蒙特卡洛模拟..."):
                try:
                    # Prepare stocks dict
                    stocks_dict = {
                        stock['symbol']: {"symbol": stock['symbol'], "name": stock['name']}
                        for stock in st.session_state.portfolio_stocks
                    }
                    
                    # Run Monte Carlo simulation
                    with st.spinner("运行蒙特卡洛模拟..."):
                        result = predictor.monte_carlo_portfolio_simulation(stocks_dict, n_simulations=n_simulations)
                    
                    if "error" not in result:
                        st.success("蒙特卡洛模拟完成！")
                        
                        # Display results
                        st.subheader("模拟结果")
                        sim_cols = st.columns(3)
                        sim_cols[0].metric("最大夏普比率", f"{result['max_sharpe_ratio']:.4f}")
                        sim_cols[1].metric("最小波动率", f"{result['min_volatility']:.4f}")
                        sim_cols[2].metric("模拟次数", result['n_simulations'])
                        
                        # Best portfolio
                        st.subheader("最优投资组合")
                        best_cols = st.columns(len(result['symbols']))
                        for i, (symbol, weight) in enumerate(zip(result['symbols'], result['weights_for_max_sharpe'])):
                            best_cols[i].metric(f"{symbol}", f"{weight*100:.1f}%")
                    else:
                        st.error(f"蒙特卡洛模拟失败: {result['error']}")
                        st.warning("💡 提示：请检查网络连接，或稍后再试。")
                
                except Exception as e:
                    st.error(f"蒙特卡洛模拟过程中出错: {str(e)}")
                    st.warning("💡 提示：蒙特卡洛模拟过程出现异常，请稍后再试或联系技术支持。")

def show_backtest_page():
    """Display backtest page"""
    st.header("📈 策略回测分析")
    
    if st.session_state.stock_data is None or st.session_state.stock_data.empty:
        st.warning("请先在'🔍 股票分析'页面加载股票数据")
        return
    
    # Strategy selection
    st.subheader("策略设置")
    strategy_type = st.selectbox(
        "选择回测策略",
        ["ma_crossover", "rsi"],
        format_func=lambda x: {
            "ma_crossover": "移动平均线交叉策略",
            "rsi": "RSI超买超卖策略"
        }[x]
    )
    
    # Strategy parameters
    if strategy_type == "ma_crossover":
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.slider("短期窗口", 5, 50, 20)
        with col2:
            long_window = st.slider("长期窗口", 30, 100, 50)
        strategy_params = {"short_window": short_window, "long_window": long_window}
    
    elif strategy_type == "rsi":
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.slider("RSI周期", 5, 30, 14)
        with col2:
            overbought = st.slider("超买阈值", 70, 90, 70)
        with col3:
            oversold = st.slider("超卖阈值", 10, 30, 30)
        strategy_params = {"period": rsi_period, "overbought": overbought, "oversold": oversold}
    else:
        strategy_params = {}
    
    if st.button("运行回测"):
        with st.spinner("正在进行策略回测..."):
            try:
                # Run backtest
                result = predictor.run_strategy_backtest(
                    st.session_state.stock_data.index.name or "symbol",
                    strategy_type=strategy_type,
                    start_date="20200101",  # Add start_date parameter
                    **strategy_params
                )
                
                if "error" not in result and result.get("success"):
                    st.session_state.backtest_result = result
                    st.success("回测完成！")
                    
                    # Display results
                    st.subheader("回测结果")
                    metrics = result["result"]["metrics"]
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("累计收益", f"{metrics.get('cumulative_return', 0)*100:.2f}%")
                    metric_cols[1].metric("年化收益", f"{metrics.get('annualized_return', 0)*100:.2f}%")
                    metric_cols[2].metric("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    metric_cols[3].metric("最大回撤", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
                    
                    # Additional metrics
                    st.subheader("更多指标")
                    more_cols = st.columns(4)
                    more_cols[0].metric("交易次数", len(result["result"]["engine"].trades))
                    more_cols[1].metric("胜率", f"{metrics.get('win_rate', 0)*100:.2f}%")
                    more_cols[2].metric("盈亏比", f"{metrics.get('profit_loss_ratio', 0):.2f}")
                    more_cols[3].metric("最大单笔收益", f"{metrics.get('max_trade_return', 0)*100:.2f}%")
                    
                    # Export backtest results
                    st.subheader("导出结果")
                    backtest_json = json.dumps(result, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="下载回测结果(JSON)",
                        data=backtest_json,
                        file_name=f"backtest_{strategy_type}.json",
                        mime="application/json"
                    )
                    
                    # Generate and download report
                    if st.button("生成回测报告"):
                        try:
                            # Create a simple report
                            report_content = f"""
# 策略回测报告

## 基本信息
- 策略类型: {result['strategy_name']}
- 股票代码: {st.session_state.stock_data.index.name or 'symbol'}

## 回测结果
- 累计收益: {metrics.get('cumulative_return', 0)*100:.2f}%
- 年化收益: {metrics.get('annualized_return', 0)*100:.2f}%
- 夏普比率: {metrics.get('sharpe_ratio', 0):.2f}
- 最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%

## 更多指标
- 交易次数: {len(result["result"]["engine"].trades)}
- 胜率: {metrics.get('win_rate', 0)*100:.2f}%
- 盈亏比: {metrics.get('profit_loss_ratio', 0):.2f}
- 最大单笔收益: {metrics.get('max_trade_return', 0)*100:.2f}%

## 报告生成时间
{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 免责声明
本报告仅用于学习和研究目的，不构成投资建议。机器学习模型的预测结果可能存在误差，投资有风险，入市需谨慎。
"""
                            
                            st.download_button(
                                label="下载回测报告",
                                data=report_content,
                                file_name=f"backtest_{strategy_type}_report.txt",
                                mime="text/plain"
                            )
                        except Exception as e:
                            st.error(f"生成报告时出错: {str(e)}")
                    
                    # Visualization with loading indicator
                    st.subheader("回测图表")
                    try:
                        with st.spinner("正在生成回测图表..."):
                            predictor.plot_backtest_results_chart(
                                st.session_state.stock_data.index.name or "symbol",
                                strategy_type=strategy_type,
                                **strategy_params
                            )
                    except Exception as e:
                        st.info("可视化功能需要在本地环境中运行以显示图表")
                        
                else:
                    st.error(f"回测失败: {result.get('error', '未知错误')}")
                    st.warning("💡 提示：请检查网络连接，或稍后再试。")
            
            except Exception as e:
                st.error(f"回测过程中出错: {str(e)}")
                st.warning("💡 提示：回测过程出现异常，请稍后再试或联系技术支持。")

def show_settings_page():
    """Display settings page"""
    st.header("⚙️ 参数设置")
    
    # User preferences
    st.subheader("用户偏好设置")
    
    # Default stock symbol
    default_symbol = st.text_input(
        "默认股票代码", 
        value=st.session_state.user_preferences.get("default_symbol", "002607")
    )
    
    # Date range
    default_start_date = st.date_input(
        "默认开始日期",
        value=pd.to_datetime(st.session_state.user_preferences.get("default_start_date", "2020-01-01"))
    )
    
    # Chart preferences
    st.subheader("图表设置")
    chart_theme = st.selectbox(
        "图表主题",
        ["默认", "暗色", "亮色"],
        index=["默认", "暗色", "亮色"].index(st.session_state.user_preferences.get("chart_theme", "默认"))
    )
    
    # Save preferences
    if st.button("保存设置"):
        # Store previous preferences for undo functionality
        previous_preferences = st.session_state.user_preferences.copy()
        st.session_state.user_preferences = {
            "default_symbol": default_symbol,
            "default_start_date": default_start_date.strftime("%Y-%m-%d"),
            "chart_theme": chart_theme
        }
        st.success("设置已保存")
        
        # Add undo button
        if st.button("撤销设置"):
            st.session_state.user_preferences = previous_preferences
            st.success("设置已撤销")

def show_help_page():
    """Display help page"""
    st.header("ℹ️ 帮助文档")
    
    st.markdown("""
    ### 📖 使用说明
    
    StockTracker 是一个功能强大的股票分析和预测系统。以下是各功能模块的使用说明：
    
    #### 🔍 股票分析
    - 输入股票代码（如：002607）和日期范围
    - 点击"加载数据"获取股票历史数据
    - 查看股票基本信息和价格走势
    
    #### 📊 技术指标
    - 在已加载股票数据的基础上进行技术指标分析
    - 选择需要计算的技术指标
    - 查看指标数值和可视化图表
    
    #### 🔮 价格预测
    - 选择预测模型（LSTM、GRU、Transformer、随机森林、XGBoost）
    - 设置预测天数
    - 查看预测结果和投资建议
    
    #### ⚠️ 风险评估
    - 基于历史数据进行全面风险评估
    - 查看波动率、VaR、最大回撤等风险指标
    - 获取风险评级和投资建议
    
    #### 💼 投资组合
    - 自定义投资组合股票和权重
    - 进行投资组合分析、优化和蒙特卡洛模拟
    - 查看风险贡献和最优权重分配
    
    #### 📈 回测分析
    - 选择交易策略（移动平均线交叉、RSI等）
    - 设置策略参数
    - 查看回测结果和性能指标
    
    ### ⌨️ 键盘快捷键
    
    StockTracker 支持以下键盘快捷键以提高操作效率：
    """)
    
    # Display keyboard shortcuts
    st.markdown("""
#### 导航快捷键
- **Alt + 1** : 跳转到首页
- **Alt + 2** : 跳转到股票分析页面
- **Alt + 3** : 跳转到技术指标页面
- **Alt + 4** : 跳转到价格预测页面
- **Alt + 5** : 跳转到风险评估页面
- **Alt + 6** : 跳转到投资组合页面
- **Alt + 7** : 跳转到回测分析页面
- **Alt + 8** : 跳转到参数设置页面
- **Alt + 9** : 跳转到帮助文档页面

#### 通用快捷键
- **Ctrl + R** : 刷新当前页面
- **F1** : 显示帮助文档
- **Esc** : 关闭当前对话框或弹出窗口
    """)
    
    st.markdown("""
    ### ⚠️ 注意事项
    
    1. 股票预测仅用于学习和研究目的，不构成投资建议
    2. 机器学习模型的预测结果可能存在误差
    3. 投资有风险，入市需谨慎
    4. 请确保网络连接正常以获取实时股票数据
    5. 键盘快捷键在不同浏览器和操作系统中可能有所不同
    
    ### 📞 联系方式
    
    如有任何问题或建议，请联系：
    - 邮箱：support@stocktracker.com
    - GitHub：https://github.com/stocktracker/stocktracker
    """)

if __name__ == "__main__":
    main()