# 故障排除指南

本指南旨在帮助用户解决在使用StockTracker过程中可能遇到的常见问题。如果遇到未在此列出的问题，请查看项目的GitHub Issues页面或联系技术支持。

## 目录

1. [安装问题](#安装问题)
2. [数据获取问题](#数据获取问题)
3. [模型训练问题](#模型训练问题)
4. [Web界面问题](#web界面问题)
5. [可视化问题](#可视化问题)
6. [性能问题](#性能问题)

---

## 安装问题

### uv sync失败

**问题描述**: 在运行 `uv sync` 命令时出现错误。

**常见错误信息**:
```
error: Failed to download distribution
error: Failed to parse requirement
```

**解决方案**:
1. 确保已正确安装uv:
   ```bash
   pip install uv
   ```
2. 检查网络连接，确保可以访问PyPI镜像源
3. 尝试更换镜像源:
   ```bash
   uv pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ uv
   ```
4. 清理缓存后重试:
   ```bash
   uv cache clean
   uv sync
   ```

**预防措施**:
- 定期更新uv到最新版本
- 确保Python版本符合要求(3.12+)

### 依赖冲突

**问题描述**: 安装依赖时出现版本冲突。

**常见错误信息**:
```
error: ResolutionImpossible
error: conflict resolution
```

**解决方案**:
1. 删除现有的虚拟环境:
   ```bash
   rm -rf .venv
   ```
2. 重新创建环境并安装依赖:
   ```bash
   uv sync
   ```
3. 如果问题持续存在，尝试手动安装冲突的包:
   ```bash
   uv pip install 包名==版本号
   ```

**预防措施**:
- 定期更新 `pyproject.toml` 文件中的依赖版本
- 避免手动修改锁定文件 `uv.lock`

### Python版本兼容性

**问题描述**: Python版本不兼容导致安装或运行失败。

**常见错误信息**:
```
error: Python version requirement not satisfied
```

**解决方案**:
1. 检查当前Python版本:
   ```bash
   python --version
   ```
2. 如果版本低于3.12，请升级Python
3. 使用pyenv管理Python版本:
   ```bash
   pyenv install 3.12.0
   pyenv global 3.12.0
   ```

**预防措施**:
- 在项目根目录检查 `.python-version` 文件
- 使用虚拟环境隔离项目依赖

### 包安装错误

**问题描述**: 特定包安装失败。

**常见错误信息**:
```
error: Failed building wheel
error: Microsoft Visual C++ 14.0 is required
```

**解决方案**:
1. 对于编译问题，安装Microsoft C++ Build Tools
2. 对于特定包，尝试使用预编译版本:
   ```bash
   uv pip install --only-binary=all 包名
   ```
3. 检查包的系统要求和依赖

**预防措施**:
- 在干净的环境中安装依赖
- 定期更新系统和构建工具

---

## 数据获取问题

### akshare连接超时

**问题描述**: 获取股票数据时连接超时。

**常见错误信息**:
```
TimeoutError: [Errno 60] Operation timed out
ConnectionError: ('Connection aborted.', TimeoutError(60, 'Operation timed out'))
```

**解决方案**:
1. 检查网络连接是否正常
2. 增加超时时间:
   ```python
   stock_df = ak.stock_zh_a_hist(symbol=symbol, timeout=60)
   ```
3. 尝试使用代理或更换网络环境
4. 稍后重试，可能是服务器临时问题

**预防措施**:
- 在代码中实现重试机制
- 使用本地缓存减少重复请求

### 无效股票代码

**问题描述**: 输入的股票代码无法获取数据。

**常见错误信息**:
```
Warning: 股票 XXXXXX 没有返回任何数据
ValueError: Empty data received for symbol
```

**解决方案**:
1. 验证股票代码格式是否正确(如: "002607")
2. 检查股票代码是否存在
3. 确认股票在指定日期范围内有交易数据
4. 尝试其他股票代码进行测试

**预防措施**:
- 在输入股票代码时添加验证逻辑
- 提供股票代码查询功能

### 网络连接问题

**问题描述**: 无法连接到数据源。

**常见错误信息**:
```
ConnectionError: HTTPSConnectionPool
NewConnectionError: Failed to establish a new connection
```

**解决方案**:
1. 检查防火墙设置是否阻止了连接
2. 验证DNS解析是否正常:
   ```bash
   nslookup akshare.com
   ```
3. 尝试使用代理服务器
4. 检查公司或学校网络限制

**预防措施**:
- 实现网络连接检测机制
- 提供离线模式或本地数据支持

### API速率限制

**问题描述**: 请求过于频繁导致被限制访问。

**常见错误信息**:
```
HTTPError: 429 Client Error: Too Many Requests
RateLimitError: API rate limit exceeded
```

**解决方案**:
1. 在请求之间添加延迟:
   ```python
   import time
   time.sleep(1)  # 1秒延迟
   ```
2. 实现指数退避重试机制
3. 减少批量请求的数据量
4. 联系数据提供商了解API限制

**预防措施**:
- 实现请求频率控制
- 使用本地缓存避免重复请求

---

## 模型训练问题

### 训练数据不足

**问题描述**: 数据量不足以训练模型。

**常见错误信息**:
```
ValueError: Not enough data points for training
Warning: 股票 XXXXXX 数据记录太少
```

**解决方案**:
1. 选择有更长交易历史的股票
2. 延长数据获取的时间范围:
   ```python
   stock_data = data_fetcher.get_stock_data(symbol, start_date="20150101")
   ```
3. 使用数据增强技术
4. 降低模型复杂度

**预防措施**:
- 在训练前检查数据量
- 设置最小数据量阈值

### 训练时内存错误

**问题描述**: 模型训练过程中出现内存不足。

**常见错误信息**:
```
MemoryError
CUDA out of memory
Killed
```

**解决方案**:
1. 减小批次大小(batch_size):
   ```python
   model.train(data, batch_size=16)  # 从32减小到16
   ```
2. 减少训练周期(epochs)
3. 降低模型复杂度(减少层数或神经元数量)
4. 使用CPU训练替代GPU训练

**预防措施**:
- 监控系统资源使用情况
- 根据硬件配置调整训练参数

### 收敛问题

**问题描述**: 模型训练无法收敛或收敛缓慢。

**常见错误信息**:
```
Loss is not decreasing
Training accuracy is not improving
```

**解决方案**:
1. 调整学习率:
   ```python
   model.compile(optimizer=Adam(learning_rate=0.001))
   ```
2. 检查数据预处理是否正确
3. 尝试不同的优化器
4. 增加训练周期
5. 检查是否存在数据泄露

**预防措施**:
- 实现训练监控和可视化
- 使用学习率调度器

### GPU/CUDA问题

**问题描述**: GPU加速相关问题。

**常见错误信息**:
```
CUDA not available
No GPU/TPU found
ImportError: libcuda.so.1
```

**解决方案**:
1. 检查CUDA驱动是否正确安装
2. 验证TensorFlow GPU版本是否正确安装:
   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
3. 安装CUDA和cuDNN库
4. 如果不需要GPU加速，使用CPU版本:
   ```bash
   uv pip install tensorflow-cpu
   ```

**预防措施**:
- 在安装时明确选择CPU或GPU版本
- 提供环境检测脚本

---

## Web界面问题

### Streamlit未启动

**问题描述**: Web界面无法启动。

**常见错误信息**:
```
ModuleNotFoundError: No module named 'streamlit'
Command 'streamlit' not found
```

**解决方案**:
1. 确保已安装Streamlit:
   ```bash
   uv pip install streamlit
   ```
2. 使用Python模块方式启动:
   ```bash
   python -m streamlit run app.py
   ```
3. 检查Python环境是否正确激活

**预防措施**:
- 确保所有依赖都已正确安装
- 使用虚拟环境隔离项目

### 端口冲突

**问题描述**: Web服务端口被占用。

**常见错误信息**:
```
OSError: [Errno 48] Address already in use
```

**解决方案**:
1. 更换端口启动:
   ```bash
   streamlit run app.py --server.port 8502
   ```
2. 查找并终止占用端口的进程:
   ```bash
   lsof -i :8501
   kill -9 PID
   ```
3. 重启网络服务

**预防措施**:
- 在启动时检查端口可用性
- 提供端口配置选项

### 浏览器兼容性问题

**问题描述**: Web界面在某些浏览器中显示异常。

**常见错误信息**:
```
JavaScript error
CSS not loading
```

**解决方案**:
1. 使用现代浏览器(Chrome, Firefox, Edge)
2. 清除浏览器缓存和Cookie
3. 禁用浏览器扩展程序
4. 检查浏览器控制台错误信息

**预防措施**:
- 使用标准Web技术确保兼容性
- 定期测试不同浏览器

### 加载缓慢

**问题描述**: Web界面加载速度慢。

**常见错误信息**:
```
Loading...
TimeoutError
```

**解决方案**:
1. 检查网络连接速度
2. 优化数据加载逻辑
3. 实现数据缓存机制
4. 减少初始加载的数据量
5. 使用异步加载技术

**预防措施**:
- 实现数据预加载和缓存
- 优化前端资源加载

---

## 可视化问题

### 图表无法显示

**问题描述**: 图表无法正常渲染或显示。

**常见错误信息**:
```
Plotly error
Figure not showing
```

**解决方案**:
1. 确保已安装Plotly:
   ```bash
   uv pip install plotly
   ```
2. 在Jupyter环境中使用:
   ```python
   fig.show()
   ```
3. 在Web应用中使用:
   ```python
   st.plotly_chart(fig)
   ```
4. 检查数据格式是否正确

**预防措施**:
- 验证图表库是否正确安装
- 提供图表显示测试功能

### 缺少plotly库

**问题描述**: 缺少图表库依赖。

**常见错误信息**:
```
ModuleNotFoundError: No module named 'plotly'
ImportError: cannot import name 'graph_objects'
```

**解决方案**:
1. 安装Plotly:
   ```bash
   uv pip install plotly
   ```
2. 安装额外依赖:
   ```bash
   uv pip install plotly kaleido
   ```
3. 检查版本兼容性

**预防措施**:
- 在依赖文件中包含所有必需的图表库
- 定期更新依赖版本

### 实时图表更新失败

**问题描述**: 实时图表无法更新数据。

**常见错误信息**:
```
Update failed
Invalid data format
```

**解决方案**:
1. 检查数据格式是否正确
2. 验证时间戳格式
3. 确保数据更新频率合理
4. 检查网络连接是否稳定

**预防措施**:
- 实现数据格式验证
- 提供更新失败的错误处理

---

## 性能问题

### 预测速度慢

**问题描述**: 模型预测耗时过长。

**常见错误信息**:
```
Prediction timeout
Slow response
```

**解决方案**:
1. 使用更简单的模型(如随机森林替代LSTM)
2. 减少预测天数
3. 实现模型缓存
4. 使用模型量化技术
5. 并行处理多个预测请求

**预防措施**:
- 监控预测响应时间
- 提供性能配置选项

### 内存使用过高

**问题描述**: 应用程序占用过多内存。

**常见错误信息**:
```
Memory usage high
System slow
```

**解决方案**:
1. 优化数据结构，及时释放不需要的对象
2. 使用生成器替代大列表
3. 实现内存监控和垃圾回收
4. 减少同时加载的数据量

**预防措施**:
- 定期进行内存性能分析
- 实现内存使用监控

### 训练时间过长

**问题描述**: 模型训练耗时过长。

**常见错误信息**:
```
Training timeout
Process killed
```

**解决方案**:
1. 减少训练周期(epochs)
2. 减小数据集大小
3. 使用更简单的模型架构
4. 启用早停机制(Early Stopping)
5. 使用GPU加速训练

**预防措施**:
- 实现训练进度监控
- 提供训练时间估算功能

---

## 获取帮助

如果以上解决方案都无法解决问题，请尝试以下方式获取帮助:

1. **查看GitHub Issues**: 访问项目GitHub页面查看已知问题和解决方案
2. **提交Issue**: 如果发现新问题，请详细描述问题和环境信息
3. **社区支持**: 在相关技术社区寻求帮助
4. **联系维护者**: 通过项目提供的联系方式获取支持

**提交Issue时请包含以下信息**:
- 错误信息和堆栈跟踪
- 操作系统和Python版本
- 项目版本信息
- 重现问题的步骤
- 相关配置文件和代码片段