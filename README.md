# 🌊 发达水FatherSwim - 智能洪灾仿真系统

## 系统简介

发达水FatherSwim是一个基于多智能体建模(ABM)的智能洪灾仿真系统，专为洪灾场景下的应急管理和决策支持而设计。系统集成了先进的可视化技术、实时仿真引擎和交互式地图编辑功能。

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 8GB+ 内存
- 现代浏览器(Chrome/Firefox/Safari)

### 安装步骤

#### 方法一：一键安装(推荐)
```bash
# 赋予执行权限
chmod +x setup.sh
# 运行安装脚本
./setup.sh
```

#### 方法二：手动安装
```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动系统
python app.py
```

### 启动系统
```bash
# 使用默认端口8080
python app.py

# 或指定端口
python app.py --port 8080
```

## 📋 功能特性

### 🎯 核心功能
- **智能体仿真**: 基于差序格局理论的社会网络建模
- **洪灾模拟**: 动态洪水蔓延和风险评估
- **交互式地图**: 支持洪水区/安全区绘制和编辑
- **实时可视化**: 多维度智能体状态显示和动画效果
- **时间轴控制**: 精确的仿真进度控制和回放功能

### 🔧 高级特性
- **多场景支持**: 预设多种洪灾场景模板
- **参数优化**: 智能参数调节和性能优化
- **数据导出**: 支持仿真结果和可视化数据导出
- **安全机制**: 完整的输入验证和安全防护

## 🎮 使用指南

### 1. 系统访问
启动后访问: `http://localhost:8080`

### 2. 主要界面
- **交互仿真**: `/interactive_simulation` - 主要仿真界面
- **基础版**: `/dashboard` - 简化版仪表板
- **增强版**: `/enhanced_dashboard` - 高级分析界面
- **项目展示**: `/project_showcase` - 系统展示页面

### 3. 操作流程

#### 基础仿真
1. 进入交互仿真界面
2. 设置仿真参数(智能体数量、洪灾强度等)
3. 点击"开始仿真"按钮
4. 使用时间轴控制仿真进度
5. 观察智能体行为和洪水蔓延

#### 地图编辑
1. 点击"编辑模式"按钮
2. 选择绘制工具(洪水区/安全区)
3. 在地图上绘制区域
4. 保存地图配置

#### 智能体分析
1. 点击地图上的智能体标记
2. 查看详细信息弹窗
3. 使用"聚焦"功能跟踪特定智能体
4. 分析智能体决策和移动轨迹

## 🔧 配置说明

### 环境变量
创建`.env`文件(可选):
```
SECRET_KEY=your-secret-key
DEBUG=True
LOG_LEVEL=INFO
```

### 系统参数
主要配置文件:
- `config/security_config.py` - 安全配置
- `config/logging_config.py` - 日志配置
- `src/core/` - 核心仿真参数

## 📊 数据格式

### 智能体数据
```json
{
  "id": "agent_001",
  "position": [113.2644, 23.1291],
  "type": "resident",
  "status": "safe",
  "resources": 0.8,
  "risk_level": 0.2
}
```

### 仿真结果
系统自动保存仿真数据到`results/`目录:
- `simulation_agents.json` - 智能体状态数据
- `maps/` - 地图和风险数据
- `visualizations/` - 可视化结果

## 🚨 故障排除

### 常见问题

**Q: 系统启动失败**
A: 检查Python版本和依赖安装，确保端口8080未被占用

**Q: 地图无法加载**
A: 检查网络连接，确保可以访问地图服务API

**Q: 仿真运行缓慢**
A: 减少智能体数量或降低仿真精度，检查系统内存使用

**Q: 浏览器兼容性问题**
A: 推荐使用Chrome 90+或Firefox 88+，启用JavaScript

### 日志查看
```bash
# 查看系统日志
tail -f logs/simulation_run.log

# 查看错误日志
tail -f logs/errors.log
```

## 📞 技术支持

### 系统要求
- **最低配置**: 4GB内存，双核CPU
- **推荐配置**: 8GB+内存，四核CPU
- **浏览器**: Chrome 90+, Firefox 88+, Safari 14+

### 性能优化
- 智能体数量建议控制在1000以内
- 定期清理`cache/`和`logs/`目录
- 生产环境建议使用Gunicorn部署

## 📄 许可证

本系统仅供学术研究和教育用途使用。

---

**发达水FatherSwim团队**  
版本: 1.0.0  
更新日期: 2025年1月