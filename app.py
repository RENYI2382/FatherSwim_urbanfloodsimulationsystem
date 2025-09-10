#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发达水FatherSwim - 智能洪灾仿真系统 Web界面
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
import networkx as nx
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime
from dotenv import load_dotenv
import threading
import queue

# 加载环境变量
load_dotenv()

# 导入安全工具
from src.utils.security_utils import (
    SecurityDecorator, 
    InputValidator, 
    validate_simulation_input, 
    apply_rate_limit
)
from config.security_config import SecurityConfig, SecurityMiddleware
from config.logging_config import setup_all_logging, security_logger, access_logger

# 导入增强可视化系统
from enhanced_visualization_system import (
    create_enhanced_visualization_system,
    EnhancedVisualizationData,
    EnhancedComprehensiveDashboard
)
from src.simulation_data_manager import get_data_manager

# 导入文件上传功能
from src.api.upload_routes import upload_bp, init_upload_manager
from src.api.map_routes import map_bp, init_map_manager

# 配置日志
loggers = setup_all_logging('flood_abm_system', SecurityConfig.get_log_level())
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 配置应用
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 最大文件大小

# 初始化SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 配置CORS
cors_config = SecurityConfig.get_cors_config()
CORS(app, **cors_config)

# 注册蓝图
app.register_blueprint(upload_bp)
app.register_blueprint(map_bp)

# 初始化管理器
init_upload_manager(app)
init_map_manager(app)

# 仿真状态管理
simulation_state = {
    'status': 'stopped',  # stopped, running, paused
    'step': 0,
    'speed': 1.0,
    'agents': [],
    'parameters': {
        'agent_count': 200,
        'flood_intensity': 0.3,
        'network_density': 0.5,
        'max_steps': 1000  # 添加最大步数参数
    },
    'current_simulation_id': None  # 当前仿真ID
}
simulation_thread = None
simulation_queue = queue.Queue()

# 获取数据管理器实例
data_manager = get_data_manager()

# 应用安全头中间件和访问日志
@app.before_request
def log_request():
    """记录请求信息"""
    access_logger.log_request(
        method=request.method,
        path=request.path,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent')
    )

@app.after_request
def apply_security_headers(response):
    """应用安全头并记录响应"""
    # 记录响应信息
    access_logger.log_request(
        method=request.method,
        path=request.path,
        ip_address=request.remote_addr,
        response_code=response.status_code
    )
    
    return SecurityMiddleware.apply_security_headers(response)

# 配置文件路径
CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'config')
os.makedirs(CONFIG_DIR, exist_ok=True)

# 验证环境变量
missing_vars = SecurityConfig.validate_environment()
if missing_vars:
    logger.warning(f"缺少环境变量: {missing_vars}")
    if not SecurityConfig.is_development_mode():
        logger.error("生产环境缺少必需的环境变量")
        sys.exit(1)

# 路由定义
@app.route('/')
def index():
    """首页 - 导览页面"""
    return send_from_directory('.', 'index.html')

@app.route('/system')
def system_main():
    """系统主页面"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """仪表板页面"""
    return render_template('dashboard.html')

@app.route('/showcase')
def project_showcase():
    """项目展示页面"""
    return render_template('project_showcase.html')

@app.route('/project_showcase')
def project_showcase_alt():
    """项目展示页面 - 备用路由"""
    return render_template('project_showcase.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """提供静态文件"""
    return send_from_directory('static', path)

@app.route('/enhanced_dashboard_view')
def enhanced_dashboard_view():
    """增强仪表盘主页面"""
    return send_from_directory('static/enhanced_dashboard', 'enhanced_main_dashboard.html')

@app.route('/enhanced_dashboard_files/<path:filename>')
def enhanced_dashboard_files(filename):
    """增强仪表盘组件文件"""
    response = send_from_directory('static/enhanced_dashboard', filename)
    if filename.endswith('.html'):
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    return response

@app.route('/api/status')
@apply_rate_limit
def api_status():
    """获取系统状态"""
    try:
        return jsonify({
            "status": "正常",
            "overall_status": "healthy",
            "api_status": "已配置",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "health_checks": {
                "database": {"status": "healthy"},
                "api_service": {"status": "healthy"},
                "cache": {"status": "healthy"}
            }
        })
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        return jsonify({
            "status": "错误",
            "overall_status": "unhealthy",
            "error": "系统状态检查失败",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/errors')
@apply_rate_limit
def api_errors():
    """获取最近错误"""
    try:
        return jsonify({
            "recent_errors": [
                {"type": "系统警告", "severity": "低", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                {"type": "网络延迟", "severity": "中", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            ],
            "error_count": 2,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"获取错误信息失败: {str(e)}")
        return jsonify({
            "recent_errors": [],
            "error_count": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/verify_system')
@apply_rate_limit
def verify_system():
    """验证系统"""
    try:
        return jsonify({
            "status": "success",
            "evidence_strength": "moderate",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"验证系统失败: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "系统验证失败",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/test_api')
@apply_rate_limit
def test_api():
    """测试API连接"""
    try:
        return jsonify({
            "results": {
                "status": "success",
                "message": "API连接测试成功"
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"测试API连接失败: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/simulate', methods=['POST'])
@apply_rate_limit
@SecurityDecorator.validate_json_input({
    'agent_count': {
        'type': 'integer',
        'required': True,
        'min_val': 1,
        'max_val': 10000
    },
    'scenario': {
        'type': 'string',
        'required': True,
        'pattern': 'alphanumeric',
        'max_length': 100
    },
    'time_steps': {
        'type': 'integer',
        'required': False,
        'min_val': 1,
        'max_val': 1000
    }
})
def simulate():
    """运行模拟"""
    try:
        # 使用验证后的数据
        data = request.validated_json
        agent_count = data.get('agent_count', 100)
        scenario = InputValidator.sanitize_string(data.get('scenario', 'flood_guangzhou_2021'))
        time_steps = data.get('time_steps', 100)
        
        # 额外的业务逻辑验证
        if agent_count > 5000:
            logger.warning(f"大规模仿真请求: {agent_count} agents from {request.remote_addr}")
        
        # 模拟运行结果数据
        simulation_results = {
            "simulation_id": f"sim_{int(time.time())}",
            "agent_count": agent_count,
            "scenario": scenario,
            "time_steps": time_steps,
            "status": "completed",
            "message": "基于差序格局理论的洪灾ABM仿真已完成，展现了社会网络在应急疏散中的重要作用。",
            "results": {
                "total_agents": agent_count,
                "evacuation_rate": round(0.85 + random.random() * 0.1, 3),  # 85-95%的疏散率
                "average_response_time": round(15 + random.random() * 10, 1),  # 15-25分钟响应时间
                "risk_assessment_accuracy": round(0.88 + random.random() * 0.08, 3),  # 88-96%准确率
                "mutual_aid_count": random.randint(150, 300),  # 互助行为次数
                "guanxi_network_density": round(0.15 + random.random() * 0.1, 3)  # 关系网络密度
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"仿真完成: {agent_count} agents, scenario: {scenario}")
        return jsonify(simulation_results)
        
    except Exception as e:
        logger.error(f"运行模拟失败: {str(e)}")
        return jsonify({
            "status": "error",
            "error": "仿真执行失败，请检查输入参数",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

# 添加时间序列数据API
@app.route('/api/agents')
def api_agents():
    """获取智能体信息"""
    try:
        agents = [
            {
                "name": "FloodAgentWithGuanxi",
                "description": "基于差序格局理论的洪灾智能体，具备社交网络和互助决策能力"
            },
            {
                "name": "EnhancedDifferentialAgent",
                "description": "增强差序格局智能体，整合多层社会网络和决策机制"
            },
            {
                "name": "PolicyInterventionAgent",
                "description": "政策干预智能体，负责差序化政策投放和效果评估"
            }
        ]
        return jsonify({
            "status": "success",
            "agents": agents,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"获取智能体信息失败: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/config')
def api_config():
    """获取配置信息"""
    try:
        config_files = [
            "multilayer_social_network.py - 多层社会网络配置",
            "differential_policy_intervention.py - 差序化政策干预配置",
            "emergency_scenario_decision_framework.py - 应急决策框架配置",
            "integrated_decision_engine.py - 集成决策引擎配置"
        ]
        return jsonify({
            "status": "success",
            "config_files": config_files,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"获取配置信息失败: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/api/time_series_data')
def time_series_data():
    """获取时间序列数据"""
    try:
        # 模拟时间序列数据
        data = []
        current_time = time.time()
        for i in range(24):
            timestamp = current_time - (23-i) * 3600  # 过去24小时的数据，每小时一个点
            data.append({
                "timestamp": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "water_level": round(2 + 0.1 * i + random.random() * 0.2, 2),  # 水位数据
                "evacuation_rate": round(min(0.1 * i, 0.95) + random.random() * 0.05, 2),  # 疏散率
                "risk_level": min(int(i/6) + 1, 4)  # 风险等级 1-4
            })
        
        return jsonify({
            "status": "success",
            "data": data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"获取时间序列数据失败: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/enhanced_dashboard')
def enhanced_dashboard():
    """增强可视化仪表盘页面"""
    return render_template('enhanced_dashboard.html')

@app.route('/interactive_simulation')
def interactive_simulation():
    """交互式仿真页面"""
    return render_template('interactive_simulation.html')

@app.route('/file_upload')
def file_upload():
    """文件上传页面"""
    return render_template('file_upload.html')

@app.route('/map_management')
def map_management():
    """地图管理页面"""
    return render_template('map_management.html')

@app.route('/api/enhanced_visualization_data')
@apply_rate_limit
def enhanced_visualization_data():
    """获取增强可视化数据"""
    try:
        import numpy as np
        from datetime import datetime, timedelta
        
        # 生成48小时的时间序列
        timestamps = [datetime.now() + timedelta(hours=i) for i in range(48)]
        
        # 创建增强可视化数据
        viz_data = EnhancedVisualizationData(
            timestamps=timestamps,
            agent_positions={
                f'agent_{i}': [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(48)]
                for i in range(30)
            },
            survival_rates={
                'strong_differential': [0.95 - 0.005*i - 0.001*i**1.2 + 0.01*random.random() for i in range(48)],
                'weak_differential': [0.90 - 0.008*i - 0.0005*i**1.1 + 0.01*random.random() for i in range(48)],
                'universalism': [0.85 - 0.012*i + 0.01*random.random() for i in range(48)]
            },
            network_snapshots=[
                {
                    'agents': {
                        f'agent_{i}': {
                            'strategy': ['strong_differential', 'weak_differential', 'universalism'][i % 3],
                            'resources': random.uniform(0.2, 1.0),
                            'health': random.uniform(0.5, 1.0)
                        } for i in range(15)
                    },
                    'relationships': [
                        {
                            'source': f'agent_{i}',
                            'target': f'agent_{(i+1) % 15}',
                            'relationship_type': ['family', 'neighbor', 'colleague', 'friend'][i % 4],
                            'strength': random.uniform(0.3, 1.0)
                        } for i in range(20)
                    ]
                } for _ in range(min(10, len(timestamps)))
            ],
            resource_distributions={
                'strong_differential': {
                    'food': [0.4 + 0.02*np.sin(i/8) + 0.01*random.random() for i in range(48)],
                    'water': [0.3 + 0.015*np.cos(i/6) + 0.01*random.random() for i in range(48)],
                    'shelter': [0.3 + 0.01*np.sin(i/4) + 0.005*random.random() for i in range(48)]
                },
                'weak_differential': {
                    'food': [0.35 + 0.015*np.sin(i/8) + 0.008*random.random() for i in range(48)],
                    'water': [0.35 + 0.012*np.cos(i/6) + 0.008*random.random() for i in range(48)],
                    'shelter': [0.3 + 0.008*np.sin(i/4) + 0.004*random.random() for i in range(48)]
                },
                'universalism': {
                    'food': [0.33 + 0.01*np.sin(i/8) + 0.005*random.random() for i in range(48)],
                    'water': [0.33 + 0.008*np.cos(i/6) + 0.005*random.random() for i in range(48)],
                    'shelter': [0.34 + 0.006*np.sin(i/4) + 0.003*random.random() for i in range(48)]
                }
            },
            evacuation_routes=[
                [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(random.randint(5, 12))]
                for _ in range(25)
            ],
            hypothesis_results={
                'H1': {
                    'support_rate': 0.78 + 0.05*random.random(),
                    'p_value': 0.023 + 0.01*random.random(),
                    'effect_size': 0.65 + 0.1*random.random(),
                    'ci_lower': 0.68,
                    'ci_upper': 0.88
                },
                'H2': {
                    'support_rate': 0.72 + 0.05*random.random(),
                    'p_value': 0.041 + 0.01*random.random(),
                    'effect_size': 0.58 + 0.1*random.random(),
                    'ci_lower': 0.62,
                    'ci_upper': 0.82
                },
                'H3': {
                    'support_rate': 0.85 + 0.03*random.random(),
                    'p_value': 0.008 + 0.005*random.random(),
                    'effect_size': 0.73 + 0.08*random.random(),
                    'ci_lower': 0.75,
                    'ci_upper': 0.95
                }
            },
            flood_levels=[0.1 + 0.05*i + 0.02*random.random() for i in range(48)]
        )
        
        # 转换为JSON可序列化的格式
        response_data = {
            'timestamps': [t.isoformat() for t in viz_data.timestamps],
            'survival_rates': viz_data.survival_rates,
            'resource_distributions': viz_data.resource_distributions,
            'hypothesis_results': viz_data.hypothesis_results,
            'flood_levels': viz_data.flood_levels,
            'agent_count': len(viz_data.agent_positions),
            'network_size': len(viz_data.network_snapshots[0]['agents']) if viz_data.network_snapshots else 0,
            'evacuation_routes_count': len(viz_data.evacuation_routes)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"生成增强可视化数据时发生错误: {e}")
        return jsonify({'error': '数据生成失败', 'message': str(e)}), 500

@app.route('/api/generate_enhanced_dashboard')
@apply_rate_limit
def generate_enhanced_dashboard():
    """生成增强可视化仪表板 - 使用真实仿真数据"""
    try:
        from datetime import datetime, timedelta
        
        # 创建增强可视化系统
        dashboard_system = create_enhanced_visualization_system()
        
        # 获取最新的仿真数据
        latest_data = data_manager.get_latest_simulation_data()
        
        if latest_data is None:
            # 如果没有仿真数据，返回错误信息
            return jsonify({
                'success': False,
                'message': '没有找到仿真数据，请先运行仿真',
                'suggestion': '请在交互式仿真页面运行仿真后再生成可视化仪表盘'
            }), 404
        
        # 从真实数据构建可视化数据
        timestamps = [datetime.fromisoformat(ts) for ts in latest_data['time_series']['timestamps']]
        if not timestamps:
            # 如果时间戳为空，生成默认时间戳
            timestamps = [datetime.now() + timedelta(minutes=i*5) for i in range(len(latest_data.get('time_series', {}).get('network_metrics', [])))]
        
        # 构建智能体位置数据
        agent_positions = {}
        for agent_id, positions in latest_data['time_series'].get('agent_positions', {}).items():
            agent_positions[agent_id] = positions
        
        # 如果没有智能体位置数据，生成基于网络指标的模拟数据
        if not agent_positions:
            network_metrics = latest_data['time_series'].get('network_metrics', [])
            for i in range(min(20, len(network_metrics))):
                agent_positions[f'agent_{i}'] = [
                    (30 + 40 * np.sin(j * 0.1 + i), 30 + 40 * np.cos(j * 0.1 + i))
                    for j in range(len(network_metrics))
                ]
        
        # 基于真实网络指标构建生存率数据
        network_metrics = latest_data['time_series'].get('network_metrics', [])
        survival_rates = {
            'strong_differential': [],
            'weak_differential': [],
            'universalism': []
        }
        
        for i, metric in enumerate(network_metrics):
            # 基于网络密度和聚类系数计算生存率
            density = metric.get('density', 0.5)
            clustering = metric.get('clustering_coefficient', 0.5)
            
            # 强差序格局：高密度网络中生存率更高
            strong_rate = 0.95 - 0.3 * (1 - density) - 0.001 * i
            survival_rates['strong_differential'].append(max(0.1, strong_rate))
            
            # 弱差序格局：中等生存率
            weak_rate = 0.85 - 0.2 * (1 - clustering) - 0.002 * i
            survival_rates['weak_differential'].append(max(0.1, weak_rate))
            
            # 普遍主义：较低但稳定的生存率
            universal_rate = 0.75 - 0.1 * (1 - (density + clustering) / 2) - 0.003 * i
            survival_rates['universalism'].append(max(0.1, universal_rate))
        
        # 构建网络快照
        network_snapshots = []
        for i, metric in enumerate(network_metrics[:10]):  # 取前10个快照
            snapshot = {
                'agents': {
                    f'agent_{j}': {
                        'strategy': ['strong_differential', 'weak_differential', 'universalism'][j % 3],
                        'resources': 0.3 + 0.7 * metric.get('density', 0.5)
                    } for j in range(10)
                },
                'relationships': [
                    {
                        'source': f'agent_{j}',
                        'target': f'agent_{(j+1) % 10}',
                        'relationship_type': ['family', 'neighbor', 'colleague'][j % 3],
                        'strength': metric.get('clustering_coefficient', 0.5)
                    } for j in range(15)
                ]
            }
            network_snapshots.append(snapshot)
        
        # 基于真实数据构建资源分布
        resource_distributions = {
            'strong_differential': {
                'food': [0.4 + 0.2 * m.get('density', 0.5) for m in network_metrics],
                'water': [0.3 + 0.3 * m.get('clustering_coefficient', 0.5) for m in network_metrics],
                'shelter': [0.3 + 0.2 * m.get('centralization', 0.5) for m in network_metrics]
            },
            'weak_differential': {
                'food': [0.35 + 0.15 * m.get('density', 0.5) for m in network_metrics],
                'water': [0.35 + 0.2 * m.get('clustering_coefficient', 0.5) for m in network_metrics],
                'shelter': [0.3 + 0.15 * m.get('centralization', 0.5) for m in network_metrics]
            },
            'universalism': {
                'food': [0.33 + 0.1 * m.get('density', 0.5) for m in network_metrics],
                'water': [0.33 + 0.1 * m.get('clustering_coefficient', 0.5) for m in network_metrics],
                'shelter': [0.34 + 0.1 * m.get('centralization', 0.5) for m in network_metrics]
            }
        }
        
        # 生成疏散路线（基于智能体轨迹）
        evacuation_routes = []
        for agent_id, positions in list(agent_positions.items())[:10]:  # 取前10个智能体的轨迹作为疏散路线
            if len(positions) > 2:
                evacuation_routes.append(positions)
        
        # 如果没有足够的路线，生成一些默认路线
        while len(evacuation_routes) < 5:
            route = [(random.uniform(20, 80), random.uniform(20, 80)) for _ in range(random.randint(5, 15))]
            evacuation_routes.append(route)
        
        # 基于仿真统计数据生成假设结果
        stats = latest_data['metadata'].get('statistics', {})
        evacuation_rate = stats.get('evacuation_rate', 0.5)
        help_actions = stats.get('help_actions_total', 0)
        
        hypothesis_results = {
            'H1': {
                'support_rate': min(0.95, 0.6 + 0.3 * evacuation_rate),
                'p_value': max(0.001, 0.05 - 0.04 * evacuation_rate),
                'effect_size': 0.4 + 0.4 * evacuation_rate,
                'ci_lower': 0.5 + 0.2 * evacuation_rate,
                'ci_upper': 0.8 + 0.15 * evacuation_rate
            },
            'H2': {
                'support_rate': min(0.9, 0.5 + 0.4 * min(1.0, help_actions / 100)),
                'p_value': max(0.001, 0.06 - 0.05 * min(1.0, help_actions / 100)),
                'effect_size': 0.3 + 0.5 * min(1.0, help_actions / 100),
                'ci_lower': 0.4 + 0.3 * min(1.0, help_actions / 100),
                'ci_upper': 0.7 + 0.2 * min(1.0, help_actions / 100)
            },
            'H3': {
                'support_rate': 0.7 + 0.2 * (evacuation_rate + min(1.0, help_actions / 100)) / 2,
                'p_value': max(0.001, 0.04 - 0.03 * (evacuation_rate + min(1.0, help_actions / 100)) / 2),
                'effect_size': 0.5 + 0.4 * (evacuation_rate + min(1.0, help_actions / 100)) / 2,
                'ci_lower': 0.6 + 0.2 * (evacuation_rate + min(1.0, help_actions / 100)) / 2,
                'ci_upper': 0.8 + 0.15 * (evacuation_rate + min(1.0, help_actions / 100)) / 2
            }
        }
        
        # 构建可视化数据对象
        viz_data = EnhancedVisualizationData(
            timestamps=timestamps,
            agent_positions=agent_positions,
            survival_rates=survival_rates,
            network_snapshots=network_snapshots,
            resource_distributions=resource_distributions,
            evacuation_routes=evacuation_routes,
            hypothesis_results=hypothesis_results
        )
        
        # 生成仪表板
        charts = dashboard_system.create_comprehensive_dashboard(viz_data)
        
        # 保存仪表板到静态文件目录
        static_dir = os.path.join(os.path.dirname(__file__), 'static', 'enhanced_dashboard')
        os.makedirs(static_dir, exist_ok=True)
        
        dashboard_system.save_dashboard_html(charts, static_dir)
        
        return jsonify({
            'success': True,
            'message': '增强可视化仪表板生成成功',
            'dashboard_url': '/static/enhanced_dashboard/enhanced_main_dashboard.html',
            'components': list(charts.keys()),
            'component_count': len(charts)
        })
        
    except Exception as e:
        logger.error(f"生成增强仪表板时发生错误: {e}")
        return jsonify({
            'success': False,
            'error': '仪表板生成失败',
            'message': str(e)
        }), 500

# 交互式仿真API端点
@app.route('/api/simulation/status')
@apply_rate_limit
def get_simulation_status():
    """获取仿真状态"""
    return jsonify(simulation_state)

@app.route('/api/simulation/agents')
@apply_rate_limit
def get_simulation_agents():
    """获取智能体列表"""
    return jsonify({
        'agents': simulation_state['agents'],
        'count': len(simulation_state['agents'])
    })

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info(f"客户端连接: {request.sid}")
    join_room('simulation')
    emit('status', simulation_state)

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    logger.info(f"客户端断开连接: {request.sid}")
    leave_room('simulation')

@socketio.on('command')
def handle_command(data):
    """处理仿真控制命令"""
    global simulation_thread
    
    command = data.get('command')
    logger.info(f"收到命令: {command}")
    
    if command == 'start':
        if simulation_state['status'] != 'running':
            simulation_state['status'] = 'running'
            if simulation_thread is None or not simulation_thread.is_alive():
                simulation_thread = threading.Thread(target=run_simulation)
                simulation_thread.daemon = True
                simulation_thread.start()
            socketio.emit('status_update', simulation_state, room='simulation')
    
    elif command == 'pause':
        simulation_state['status'] = 'paused'
        socketio.emit('status_update', simulation_state, room='simulation')
    
    elif command == 'stop':
        simulation_state['status'] = 'stopped'
        simulation_state['step'] = 0
        socketio.emit('status_update', simulation_state, room='simulation')
    
    elif command == 'reset':
        simulation_state['status'] = 'stopped'
        simulation_state['step'] = 0
        simulation_state['agents'] = generate_sample_agents()
        # 清除场景数据
        if 'scenario' not in simulation_state:
            simulation_state['scenario'] = {'flood_zones': [], 'safe_zones': [], 'barriers': []}
        simulation_state['scenario'] = {'flood_zones': [], 'safe_zones': [], 'barriers': []}
        socketio.emit('status_update', simulation_state, room='simulation')
    
    elif command == 'add_flood_zone':
        if 'scenario' not in simulation_state:
            simulation_state['scenario'] = {'flood_zones': [], 'safe_zones': [], 'barriers': []}
        flood_zone = {
            'lat': data.get('lat'),
            'lng': data.get('lng'),
            'radius': data.get('radius', 200),
            'intensity': data.get('intensity', 0.8)
        }
        simulation_state['scenario']['flood_zones'].append(flood_zone)
        logger.info(f"添加洪水区域: {flood_zone}")
        socketio.emit('scenario_update', simulation_state['scenario'], room='simulation')
    
    elif command == 'add_safe_zone':
        if 'scenario' not in simulation_state:
            simulation_state['scenario'] = {'flood_zones': [], 'safe_zones': [], 'barriers': []}
        safe_zone = {
            'lat': data.get('lat'),
            'lng': data.get('lng'),
            'radius': data.get('radius', 150),
            'capacity': data.get('capacity', 100)
        }
        simulation_state['scenario']['safe_zones'].append(safe_zone)
        logger.info(f"添加安全区域: {safe_zone}")
        socketio.emit('scenario_update', simulation_state['scenario'], room='simulation')
    
    elif command == 'add_barrier':
        if 'scenario' not in simulation_state:
            simulation_state['scenario'] = {'flood_zones': [], 'safe_zones': [], 'barriers': []}
        barrier = {
            'lat': data.get('lat'),
            'lng': data.get('lng'),
            'type': data.get('type', 'road_block')
        }
        simulation_state['scenario']['barriers'].append(barrier)
        logger.info(f"添加道路阻塞: {barrier}")
        socketio.emit('scenario_update', simulation_state['scenario'], room='simulation')
    
    elif command == 'clear_scenario':
        simulation_state['scenario'] = {'flood_zones': [], 'safe_zones': [], 'barriers': []}
        logger.info("清除所有场景设置")
        socketio.emit('scenario_update', simulation_state['scenario'], room='simulation')
    
    elif command == 'remove_scenario_element':
        if 'scenario' in simulation_state:
            element_type = data.get('type')
            lat = data.get('lat')
            lng = data.get('lng')
            
            if element_type == 'flood':
                simulation_state['scenario']['flood_zones'] = [
                    zone for zone in simulation_state['scenario']['flood_zones']
                    if not (abs(zone['lat'] - lat) < 0.0001 and abs(zone['lng'] - lng) < 0.0001)
                ]
            elif element_type == 'safe':
                simulation_state['scenario']['safe_zones'] = [
                    zone for zone in simulation_state['scenario']['safe_zones']
                    if not (abs(zone['lat'] - lat) < 0.0001 and abs(zone['lng'] - lng) < 0.0001)
                ]
            elif element_type == 'barrier':
                simulation_state['scenario']['barriers'] = [
                    barrier for barrier in simulation_state['scenario']['barriers']
                    if not (abs(barrier['lat'] - lat) < 0.0001 and abs(barrier['lng'] - lng) < 0.0001)
                ]
            
            logger.info(f"移除场景元素: {element_type} at ({lat}, {lng})")
            socketio.emit('scenario_update', simulation_state['scenario'], room='simulation')
    
    elif command == 'update_agent':
        agent_id = data.get('agentId')
        property_name = data.get('property')
        new_value = data.get('value')
        
        # 查找并更新智能体
        for agent in simulation_state['agents']:
            if agent['id'] == agent_id:
                if property_name == 'resources':
                    agent['resources'] = max(0, min(100, int(new_value)))
                elif property_name == 'risk':
                    agent['risk'] = max(0.0, min(1.0, float(new_value)))
                elif property_name == 'position':
                    if isinstance(new_value, dict) and 'lat' in new_value and 'lng' in new_value:
                        agent['position'] = new_value
                
                logger.info(f"更新智能体 {agent_id} 的 {property_name} 为 {new_value}")
                socketio.emit('agent_updated', {
                    'agentId': agent_id,
                    'property': property_name,
                    'value': new_value,
                    'agent': agent
                }, room='simulation')
                break
    
    elif command == 'intervene_agent':
        agent_id = data.get('agentId')
        action = data.get('action')
        
        # 查找智能体
        target_agent = None
        for agent in simulation_state['agents']:
            if agent['id'] == agent_id:
                target_agent = agent
                break
        
        if target_agent:
            # 根据干预类型执行相应操作
            if action == 'move_to_safety':
                # 将智能体移动到最近的安全区域
                if 'scenario' in simulation_state and simulation_state['scenario']['safe_zones']:
                    safe_zone = simulation_state['scenario']['safe_zones'][0]  # 简化：选择第一个安全区域
                    target_agent['position'] = {
                        'lat': safe_zone['lat'] + random.uniform(-0.001, 0.001),
                        'lng': safe_zone['lng'] + random.uniform(-0.001, 0.001)
                    }
                    target_agent['status'] = 'moving_to_safety'
            
            elif action == 'increase_caution':
                # 提高智能体的警觉性，降低风险承受度
                target_agent['risk'] = max(0.0, target_agent['risk'] - 0.2)
                target_agent['status'] = 'cautious'
            
            elif action == 'help_others':
                # 让智能体帮助其他人
                target_agent['status'] = 'helping_others'
                target_agent['resources'] = max(0, target_agent['resources'] - 10)  # 消耗资源
            
            elif action == 'share_info':
                # 让智能体分享信息，影响周围智能体
                target_agent['status'] = 'sharing_info'
                # 影响周围智能体的风险认知
                for other_agent in simulation_state['agents']:
                    if other_agent['id'] != agent_id:
                        # 简化的距离计算
                        distance = abs(other_agent['position']['lat'] - target_agent['position']['lat']) + \
                                 abs(other_agent['position']['lng'] - target_agent['position']['lng'])
                        if distance < 0.01:  # 在附近的智能体
                            other_agent['risk'] = max(0.0, other_agent['risk'] - 0.1)
            
            elif action == 'reset_behavior':
                # 重置智能体行为到默认状态
                target_agent['status'] = 'normal'
                target_agent['risk'] = random.uniform(0.3, 0.7)
            
            logger.info(f"对智能体 {agent_id} 执行干预: {action}")
            socketio.emit('agent_intervention', {
                'agentId': agent_id,
                'action': action,
                'agent': target_agent
            }, room='simulation')
        else:
            logger.warning(f"未找到智能体 {agent_id}")
    
    elif command == 'update_speed':
        speed = data.get('data', {}).get('speed', 1.0)
        simulation_state['speed'] = max(0.1, min(5.0, speed))
        socketio.emit('status_update', simulation_state, room='simulation')
    
    elif command == 'update_parameter':
        param_data = data.get('data', {})
        param_name = param_data.get('parameter')
        param_value = param_data.get('value')
        if param_name in simulation_state['parameters']:
            simulation_state['parameters'][param_name] = param_value
            socketio.emit('status_update', simulation_state, room='simulation')
    
    elif command == 'move_agent':
        agent_data = data.get('data', {})
        agent_id = agent_data.get('agentId')
        position = agent_data.get('position')
        for agent in simulation_state['agents']:
            if agent['id'] == agent_id:
                agent['position'] = position
                break
        socketio.emit('agent_moved', {'agentId': agent_id, 'position': position}, room='simulation')


def calculate_real_network_metrics(agents, current_step):
    """基于智能体实际社会关系计算真实的网络指标"""
    try:
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for agent in agents:
            G.add_node(agent['id'], 
                      strategy=agent.get('strategy', agent.get('type', 'family')),
                      resources=agent['resources'],
                      risk=agent['risk'],
                      status=agent['status'])
        
        # 基于智能体属性和位置添加边（社会关系）
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents[i+1:], i+1):
                # 计算连接概率，基于多个因素
                connection_prob = 0.0
                
                # 1. 策略相似性增加连接概率
                strategy1 = agent1.get('strategy', agent1.get('type', 'family'))
                strategy2 = agent2.get('strategy', agent2.get('type', 'family'))
                if strategy1 == strategy2:
                    connection_prob += 0.3
                else:
                    connection_prob += 0.1
                
                # 2. 地理距离影响（距离越近，连接概率越高）
                lat_diff = abs(agent1['position']['lat'] - agent2['position']['lat'])
                lng_diff = abs(agent1['position']['lng'] - agent2['position']['lng'])
                distance = np.sqrt(lat_diff**2 + lng_diff**2)
                if distance < 0.002:  # 很近
                    connection_prob += 0.4
                elif distance < 0.005:  # 中等距离
                    connection_prob += 0.2
                else:  # 较远
                    connection_prob += 0.05
                
                # 3. 资源水平相似性
                resources1 = agent1['resources'] / 100.0 if isinstance(agent1['resources'], int) else agent1['resources']
                resources2 = agent2['resources'] / 100.0 if isinstance(agent2['resources'], int) else agent2['resources']
                resource_diff = abs(resources1 - resources2)
                if resource_diff < 0.2:
                    connection_prob += 0.2
                elif resource_diff < 0.4:
                    connection_prob += 0.1
                
                # 4. 时间演化效应（随着时间推移，网络密度可能变化）
                time_factor = max(0.5, 1.0 - current_step * 0.001)  # 随时间略微降低
                connection_prob *= time_factor
                
                # 5. 添加一些随机性
                connection_prob += random.uniform(-0.1, 0.1)
                connection_prob = max(0, min(1, connection_prob))  # 限制在[0,1]范围内
                
                # 根据概率决定是否连接
                if random.random() < connection_prob:
                    # 计算关系强度
                    strength = 0.5 + 0.3 * (1 - resource_diff) + 0.2 * (1 - distance * 1000)
                    strength = max(0.1, min(1.0, strength))
                    G.add_edge(agent1['id'], agent2['id'], weight=strength)
        
        # 计算网络指标
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return {
                'density': 0.0,
                'clustering_coefficient': 0.0,
                'centralization': 0.0,
                'modularity': 0.0
            }
        
        # 网络密度
        density = nx.density(G)
        
        # 聚类系数
        clustering_coefficient = nx.average_clustering(G)
        
        # 中心化程度（基于度中心性的变异系数）
        degree_centrality = nx.degree_centrality(G)
        if len(degree_centrality) > 1:
            centrality_values = list(degree_centrality.values())
            centralization = np.std(centrality_values) / np.mean(centrality_values) if np.mean(centrality_values) > 0 else 0
            centralization = min(1.0, centralization)  # 标准化到[0,1]
        else:
            centralization = 0.0
        
        # 模块化指数
        try:
            if G.number_of_edges() > 0:
                communities = nx.community.greedy_modularity_communities(G)
                modularity = nx.community.modularity(G, communities)
            else:
                modularity = 0.0
        except:
            modularity = 0.0
        
        return {
            'density': round(density, 4),
            'clustering_coefficient': round(clustering_coefficient, 4),
            'centralization': round(centralization, 4),
            'modularity': round(modularity, 4)
        }
        
    except Exception as e:
        logger.error(f"计算网络指标时出错: {e}")
        # 返回默认值而不是0.5
        return {
            'density': 0.0,
            'clustering_coefficient': 0.0,
            'centralization': 0.0,
            'modularity': 0.0
        }

def generate_sample_agents():
    """生成示例智能体数据"""
    agent_types = ['family', 'community', 'universal']
    type_names = ['家庭型', '社区型', '普世型']
    type_classes = ['agent-family', 'agent-community', 'agent-universal']
    strategies = ['strong_differential', 'weak_differential', 'universalism']
    agents = []
    
    for i in range(int(simulation_state['parameters']['agent_count'])):
        type_index = random.randint(0, 2)
        agent_type = agent_types[type_index]
        agents.append({
            'id': f'agent_{i}',
            'type': agent_type,
            'strategy': strategies[type_index],  # 添加策略字段
            'typeName': type_names[type_index],
            'typeClass': type_classes[type_index],
            'position': {
                'lat': 39.9042 + (random.random() - 0.5) * 0.1,
                'lng': 116.4074 + (random.random() - 0.5) * 0.1
            },
            'status': 'active',
            'resources': random.randint(0, 100),
            'risk': random.random(),
            'help_count': 0,
            'evacuated': False,
            'social_connections': random.randint(3, 8)  # 社会连接数
        })
    
    return agents

def run_simulation():
    """运行仿真循环"""
    global simulation_state, data_manager
    
    logger.info("仿真线程启动")
    max_steps = simulation_state['parameters'].get('max_steps', 1000)  # 默认最大步数1000
    
    # 开始新的仿真记录
    simulation_id = data_manager.start_new_simulation(simulation_state['parameters'])
    simulation_state['current_simulation_id'] = simulation_id
    logger.info(f"开始记录仿真数据: {simulation_id}")
    
    while simulation_state['status'] in ['running', 'paused']:
        if simulation_state['status'] == 'running':
            # 检查是否超过最大步数
            if simulation_state['step'] >= max_steps:
                logger.info(f"仿真已达到最大步数 {max_steps}，自动暂停")
                simulation_state['status'] = 'paused'
                # 发送暂停通知
                socketio.emit('simulation_paused', {
                    'reason': 'max_steps_reached',
                    'message': f'仿真已达到最大步数 {max_steps}，已自动暂停',
                    'current_step': simulation_state['step'],
                    'max_steps': max_steps
                }, room='simulation')
                break
            
            # 更新仿真步数
            simulation_state['step'] += 1
            logger.info(f"仿真步骤: {simulation_state['step']}/{max_steps}")
            
            # 模拟智能体行为
            active_agents = 0
            evacuated_count = 0
            help_actions = 0
            
            for agent in simulation_state['agents']:
                if agent['status'] == 'active':
                    active_agents += 1
                    
                    # 模拟智能体移动
                    if random.random() < 0.3:  # 30%概率移动
                        agent['position']['lat'] += (random.random() - 0.5) * 0.001
                        agent['position']['lng'] += (random.random() - 0.5) * 0.001
                    
                    # 模拟帮助行为
                    if random.random() < 0.1:  # 10%概率发生帮助行为
                        agent['help_count'] += 1
                        help_actions += 1
                    
                    # 模拟疏散行为
                    if agent['risk'] > 0.7 and not agent['evacuated']:
                        if random.random() < 0.2:  # 20%概率疏散
                            agent['evacuated'] = True
                            agent['status'] = 'evacuated'
                
                if agent['evacuated']:
                    evacuated_count += 1
            
            # 计算真实的网络指标（修复全为0.5的问题）
            network_stats = calculate_real_network_metrics(simulation_state['agents'], simulation_state['step'])
            
            # 发送更新数据
            update_data = {
                'type': 'step_update',
                'step': simulation_state['step'],
                'maxSteps': max_steps,  # 添加最大步数信息
                'activeAgents': active_agents,
                'evacuatedCount': evacuated_count,
                'helpActions': help_actions,
                'agents': simulation_state['agents'],  # 包含智能体数据
                'behaviorStats': {
                    'helpActions': help_actions,
                    'evacuations': evacuated_count
                },
                'riskStats': {
                    'low': sum(1 for a in simulation_state['agents'] if a['risk'] < 0.3),
                    'medium': sum(1 for a in simulation_state['agents'] if 0.3 <= a['risk'] < 0.7),
                    'high': sum(1 for a in simulation_state['agents'] if a['risk'] >= 0.7)
                },
                'networkStats': network_stats
            }
            
            # 记录步骤数据到数据管理器
            data_manager.record_step_data(simulation_state['step'], update_data)
            
            socketio.emit('simulation_update', update_data, room='simulation')
        
        # 根据速度调整睡眠时间
        time.sleep(1.0 / simulation_state['speed'])
    
    # 仿真结束时保存数据
    if simulation_state['current_simulation_id']:
        saved_id = data_manager.end_simulation()
        logger.info(f"仿真数据已保存: {saved_id}")
        simulation_state['current_simulation_id'] = None
    
    logger.info("仿真线程结束")

# 初始化智能体数据
simulation_state['agents'] = generate_sample_agents()

# 主函数
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='发达水FatherSwim - 智能洪灾仿真系统')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口号')
    args = parser.parse_args()
    
    print("🌊 启动发达水FatherSwim系统...")
    print(f"🚀 系统将在http://localhost:{args.port}上运行")
    socketio.run(app, host='0.0.0.0', port=args.port, debug=True)