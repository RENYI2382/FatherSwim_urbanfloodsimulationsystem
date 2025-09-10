#!/bin/bash
# 基于差序格局的洪灾ABM仿真系统启动脚本

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  基于差序格局的洪灾ABM仿真系统  ${NC}"
echo -e "${BLUE}=======================================${NC}"

# 检查Python版本
echo -e "${YELLOW}检查Python版本...${NC}"
python3 --version
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 未找到Python3，请确保已安装Python 3.8或更高版本${NC}"
    exit 1
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 无法创建虚拟环境，请确保已安装venv模块${NC}"
        exit 1
    fi
fi

# 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 无法激活虚拟环境${NC}"
    exit 1
fi

# 安装依赖
echo -e "${YELLOW}安装依赖...${NC}"

# 检查Python版本，针对Python 3.13+版本特殊处理
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
    echo -e "${YELLOW}检测到Python 3.13+版本，使用兼容性安装方式...${NC}"
    
    # 先安装基础工具
    pip install --upgrade pip setuptools wheel
    
    # 单独安装可能有问题的包
    echo -e "${YELLOW}安装核心依赖...${NC}"
    pip install flask==2.2.5 flask-cors==4.0.0
    pip install requests==2.31.0
    
    # 尝试安装PyYAML
    echo -e "${YELLOW}安装PyYAML...${NC}"
    pip install pyyaml==6.0.1 || pip install --pre pyyaml
    
    # 安装其他依赖
    echo -e "${YELLOW}安装其他依赖...${NC}"
    pip install openai numpy pandas matplotlib networkx aiohttp async-timeout python-dotenv tqdm colorama
else
    # 正常安装所有依赖
    pip install -r requirements.txt
fi

# 检查关键依赖是否安装成功
python3 -c "import flask; import requests; print('核心依赖检查通过')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}警告: 核心依赖安装失败，尝试使用pip直接安装...${NC}"
    pip install flask requests
    
    # 再次检查
    python3 -c "import flask; import requests" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 无法安装核心依赖，系统可能无法正常运行${NC}"
        echo -e "${YELLOW}建议: 请尝试使用Python 3.10或3.11版本${NC}"
    else
        echo -e "${GREEN}核心依赖安装成功${NC}"
    fi
fi

# 检查API配置
echo -e "${YELLOW}检查API配置...${NC}"
if [ -f "config/reasoning_model_config.yaml" ]; then
    echo -e "${GREEN}API配置文件已找到${NC}"
else
    echo -e "${RED}警告: 未找到API配置文件，可能会影响系统功能${NC}"
fi

# 运行API测试
echo -e "${YELLOW}运行API测试...${NC}"
python api_evidence_test.py
if [ $? -ne 0 ]; then
    echo -e "${RED}警告: API测试失败，但将继续尝试运行系统${NC}"
fi

# 启动系统
echo -e "${YELLOW}启动系统...${NC}"
echo -e "${GREEN}系统将在http://localhost:8080上运行${NC}"
python app.py

# 如果系统退出，提示用户
echo -e "${RED}系统已退出${NC}"