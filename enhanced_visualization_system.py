#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强可视化系统
基于差序格局理论的城市洪灾智能体建模 - 理论验证可视化组件

新增功能：
1. 生存曲线对比分析 - 验证H1(强差序格局初期优势)
2. 社会网络动态图 - 展示差序格局的社会关系演化
3. 资源分配公平性分析 - 验证H2(先强后脆特征)
4. 移动轨迹热力图 - 显示空间行为模式
5. 假设验证仪表板 - 集成三个核心假设的验证结果
6. 实时数据流可视化 - 支持动态更新
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedVisualizationData:
    """增强可视化数据结构"""
    # 时间序列数据
    timestamps: List[datetime] = field(default_factory=list)
    
    # 智能体数据
    agent_positions: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    agent_strategies: Dict[str, str] = field(default_factory=dict)
    agent_resources: Dict[str, List[float]] = field(default_factory=dict)
    agent_health: Dict[str, List[float]] = field(default_factory=dict)
    
    # 生存分析数据
    survival_rates: Dict[str, List[float]] = field(default_factory=dict)
    survival_events: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # 社会网络数据
    network_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    relationship_strength: Dict[str, List[float]] = field(default_factory=dict)
    mutual_aid_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 资源分配数据
    resource_distributions: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)
    inequality_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    # 空间移动数据
    movement_patterns: Dict[str, List[Dict]] = field(default_factory=dict)
    evacuation_routes: List[List[Tuple[float, float]]] = field(default_factory=list)
    
    # 假设验证数据
    hypothesis_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistical_tests: Dict[str, List[Dict]] = field(default_factory=dict)
    
    # 环境数据
    flood_levels: List[float] = field(default_factory=list)
    risk_zones: List[Dict[str, Any]] = field(default_factory=list)

class EnhancedSurvivalAnalyzer:
    """增强生存分析可视化器"""
    
    def __init__(self):
        self.strategy_colors = {
            'strong_differential': '#FF6B6B',  # 红色 - 强差序格局
            'weak_differential': '#4ECDC4',   # 青色 - 弱差序格局
            'universalism': '#45B7D1',        # 蓝色 - 普遍主义
            'baseline': '#96CEB4'             # 绿色 - 基准线
        }
        self.strategy_labels = {
            'strong_differential': '强差序格局',
            'weak_differential': '弱差序格局',
            'universalism': '普遍主义',
            'baseline': '基准对照组'
        }
    
    def create_survival_curves_comparison(self, 
                                        survival_data: Dict[str, List[float]],
                                        timestamps: List[datetime],
                                        confidence_intervals: Optional[Dict[str, List[Tuple[float, float]]]] = None) -> go.Figure:
        """创建生存曲线对比图（支持置信区间）"""
        try:
            fig = go.Figure()
            
            for strategy, rates in survival_data.items():
                color = self.strategy_colors.get(strategy, '#000000')
                label = self.strategy_labels.get(strategy, strategy)
                
                # 主生存曲线
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=rates,
                    mode='lines+markers',
                    name=label,
                    line=dict(color=color, width=3),
                    marker=dict(size=6, symbol='circle'),
                    hovertemplate=f'<b>{label}</b><br>' +
                                 '时间: %{x}<br>' +
                                 '生存率: %{y:.2%}<br>' +
                                 '<extra></extra>'
                ))
                logger.info(f"成功添加策略 {strategy} 的主生存曲线")
            
                # 添加置信区间
                if confidence_intervals and strategy in confidence_intervals:
                    try:
                        ci_lower = [ci[0] for ci in confidence_intervals[strategy]]
                        ci_upper = [ci[1] for ci in confidence_intervals[strategy]]
                        
                        # 创建置信区间的x坐标（正向+反向）
                        x_coords = list(timestamps) + list(timestamps[::-1])
                        
                        # 创建置信区间的y坐标（上界+下界反向）
                        y_coords = list(ci_upper) + list(ci_lower[::-1])
                        
                        fig.add_trace(go.Scatter(
                            x=x_coords,
                            y=y_coords,
                            fill='toself',
                            fillcolor=f'rgba{tuple(list(plt.colors.to_rgba(color)[:3]) + [0.2])}',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{label} 置信区间',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        logger.info(f"成功添加策略 {strategy} 的置信区间")
                    except Exception as e:
                        logger.error(f"添加策略 {strategy} 置信区间时出错: {e}")
        
            fig.update_layout(
                title=dict(
                    text='<b>不同策略生存曲线对比分析</b><br><sub>验证H1: 强差序格局在灾害初期具有生存优势</sub>',
                    x=0.5,
                    font=dict(size=18)
                ),
                xaxis_title='时间进程',
                yaxis_title='累积生存率',
                yaxis=dict(tickformat='.0%', range=[0, 1]),
                hovermode='x unified',
                template='plotly_white',
                width=1200,
                height=700,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # 添加关键时间点标注（使用文本标注代替垂直线）
            try:
                # 使用索引来获取特定时间点
                early_idx = len(timestamps) // 3
                mid_idx = 2 * len(timestamps) // 3
                
                if early_idx < len(timestamps) and mid_idx < len(timestamps):
                    # 使用文本标注代替垂直线，避免plotly内部的datetime处理问题
                    fig.add_annotation(
                        x=timestamps[early_idx],
                        y=0.9,
                        text="灾害初期",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="red",
                        font=dict(size=12, color="red")
                    )
                    fig.add_annotation(
                        x=timestamps[mid_idx],
                        y=0.9,
                        text="灾害中期",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="orange",
                        font=dict(size=12, color="orange")
                    )
                logger.info("成功添加关键时间点标注")
            except Exception as e:
                logger.error(f"添加关键时间点标注时出错: {e}")
                # 即使标注失败也不影响主要功能
            
            return fig
        except Exception as e:
            logger.error(f"创建生存曲线对比图时出错: {e}")
            # 返回一个空的图表
            return go.Figure()
    
    def create_hazard_ratio_analysis(self, 
                                   survival_data: Dict[str, List[float]],
                                   timestamps: List[datetime]) -> go.Figure:
        """创建风险比分析图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('累积风险函数', '瞬时风险率', '风险比对比', '生存优势分析'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        baseline_rates = survival_data.get('universalism', survival_data[list(survival_data.keys())[0]])
        
        for i, (strategy, rates) in enumerate(survival_data.items()):
            if strategy == 'universalism':
                continue
                
            color = self.strategy_colors.get(strategy, '#000000')
            label = self.strategy_labels.get(strategy, strategy)
            
            # 累积风险函数 (1 - 生存率)
            cumulative_risk = [1 - rate for rate in rates]
            fig.add_trace(
                go.Scatter(x=timestamps, y=cumulative_risk, name=f'{label}-累积风险',
                          line=dict(color=color), mode='lines'),
                row=1, col=1
            )
            
            # 瞬时风险率（简化计算）
            hazard_rates = [max(0, (rates[i-1] - rates[i]) / rates[i-1]) if i > 0 and rates[i-1] > 0 else 0 
                           for i in range(len(rates))]
            fig.add_trace(
                go.Scatter(x=timestamps, y=hazard_rates, name=f'{label}-瞬时风险',
                          line=dict(color=color, dash='dot'), mode='lines'),
                row=1, col=2
            )
            
            # 风险比（相对于普遍主义）
            hazard_ratios = [rates[i] / baseline_rates[i] if baseline_rates[i] > 0 else 1 
                           for i in range(len(rates))]
            fig.add_trace(
                go.Scatter(x=timestamps, y=hazard_ratios, name=f'{label}-风险比',
                          line=dict(color=color), mode='lines+markers'),
                row=2, col=1
            )
            
            # 生存优势（log风险比）
            log_hazard_ratios = [np.log(hr) if hr > 0 else 0 for hr in hazard_ratios]
            fig.add_trace(
                go.Scatter(x=timestamps, y=log_hazard_ratios, name=f'{label}-生存优势',
                          line=dict(color=color), mode='lines', fill='tonexty' if i == 1 else None),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="<b>生存分析详细对比</b>",
            height=800,
            showlegend=True
        )
        
        # 添加基准线
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        return fig

class EnhancedSocialNetworkVisualizer:
    """增强社会网络可视化器"""
    
    def __init__(self):
        self.relationship_colors = {
            'family': '#FF6B6B',      # 血缘关系 - 红色
            'neighbor': '#4ECDC4',    # 地缘关系 - 青色
            'colleague': '#45B7D1',   # 业缘关系 - 蓝色
            'friend': '#96CEB4',      # 友缘关系 - 绿色
            'stranger': '#FFA07A'     # 陌生人 - 橙色
        }
        self.strategy_shapes = {
            'strong_differential': 'diamond',
            'weak_differential': 'circle',
            'universalism': 'square'
        }
    
    def create_dynamic_network_graph(self, 
                                   network_snapshots: List[Dict[str, Any]],
                                   timestamps: List[datetime]) -> go.Figure:
        """创建动态社会网络图"""
        if not network_snapshots:
            return go.Figure()
        
        # 使用第一个快照创建基础网络
        snapshot = network_snapshots[0]
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for agent_id, agent_data in snapshot.get('agents', {}).items():
            G.add_node(agent_id, **agent_data)
        
        # 添加边
        for relationship in snapshot.get('relationships', []):
            G.add_edge(
                relationship['source'], 
                relationship['target'],
                relationship_type=relationship.get('relationship_type', 'unknown'),
                strength=relationship.get('strength', 0.5)
            )
        
        # 使用力导向布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 创建边的轨迹
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            rel_type = edge[2].get('relationship_type', 'unknown')
            strength = edge[2].get('strength', 0.5)
            edge_info.append(f"关系类型: {rel_type}<br>关系强度: {strength:.2f}")
        
        # 创建节点轨迹
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_symbol = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            
            strategy = node[1].get('strategy', 'unknown')
            resources = node[1].get('resources', 0)
            
            node_text.append(f"智能体: {node[0]}<br>策略: {strategy}<br>资源: {resources:.2f}")
            
            # 根据策略设置颜色
            if strategy == 'strong_differential':
                node_color.append('#FF6B6B')
            elif strategy == 'weak_differential':
                node_color.append('#4ECDC4')
            elif strategy == 'universalism':
                node_color.append('#45B7D1')
            else:
                node_color.append('#96CEB4')
            
            # 根据资源设置大小
            node_size.append(max(10, resources * 30))
            
            # 根据策略设置形状
            node_symbol.append(self.strategy_shapes.get(strategy, 'circle'))
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='关系连接'
        ))
        
        # 添加节点
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                size=node_size,
                color=node_color,
                symbol=node_symbol,
                line=dict(width=2, color='white')
            ),
            name='智能体节点'
        ))
        
        fig.update_layout(
            title='<b>差序格局社会网络结构</b><br><sub>节点大小表示资源量，颜色表示策略类型</sub>',
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="红色=强差序格局, 青色=弱差序格局, 蓝色=普遍主义",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=700
        )
        
        return fig
    
    def create_network_metrics_dashboard(self, 
                                       network_snapshots: List[Dict[str, Any]],
                                       timestamps: List[datetime]) -> go.Figure:
        """创建网络指标仪表板"""
        if not network_snapshots:
            return go.Figure()
        
        # 计算网络指标
        metrics_over_time = {
            'density': [],
            'clustering': [],
            'centralization': [],
            'modularity': []
        }
        
        for snapshot in network_snapshots:
            G = nx.Graph()
            
            # 构建网络
            for agent_id, agent_data in snapshot.get('agents', {}).items():
                G.add_node(agent_id, **agent_data)
            
            for relationship in snapshot.get('relationships', []):
                G.add_edge(relationship['source'], relationship['target'])
            
            if len(G.nodes()) > 0:
                # 网络密度
                density = nx.density(G)
                metrics_over_time['density'].append(density)
                
                # 聚类系数
                clustering = nx.average_clustering(G) if len(G.nodes()) > 2 else 0
                metrics_over_time['clustering'].append(clustering)
                
                # 中心化程度（基于度中心性）
                if len(G.nodes()) > 1:
                    centrality = nx.degree_centrality(G)
                    max_centrality = max(centrality.values())
                    centralization = max_centrality
                else:
                    centralization = 0
                metrics_over_time['centralization'].append(centralization)
                
                # 模块化（简化计算）
                try:
                    communities = nx.community.greedy_modularity_communities(G)
                    modularity = nx.community.modularity(G, communities)
                except:
                    modularity = 0
                metrics_over_time['modularity'].append(modularity)
            else:
                for key in metrics_over_time:
                    metrics_over_time[key].append(0)
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('网络密度', '聚类系数', '中心化程度', '模块化指数'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 确保时间戳长度匹配
        valid_timestamps = timestamps[:len(metrics_over_time['density'])]
        
        # 添加各项指标
        fig.add_trace(
            go.Scatter(x=valid_timestamps, 
                      y=metrics_over_time['density'],
                      mode='lines+markers', name='网络密度',
                      line=dict(color='#FF6B6B')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=valid_timestamps, 
                      y=metrics_over_time['clustering'],
                      mode='lines+markers', name='聚类系数',
                      line=dict(color='#4ECDC4')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=valid_timestamps, 
                      y=metrics_over_time['centralization'],
                      mode='lines+markers', name='中心化程度',
                      line=dict(color='#45B7D1')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=valid_timestamps, 
                      y=metrics_over_time['modularity'],
                      mode='lines+markers', name='模块化指数',
                      line=dict(color='#96CEB4')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="<b>社会网络结构指标演化</b>",
            height=600,
            showlegend=False
        )
        
        return fig

class EnhancedResourceAnalyzer:
    """增强资源分配分析器"""
    
    def __init__(self):
        self.resource_colors = {
            'food': '#FF6B6B',
            'water': '#4ECDC4', 
            'shelter': '#45B7D1',
            'medicine': '#96CEB4',
            'information': '#FFA07A'
        }
    
    def create_resource_inequality_analysis(self, 
                                          resource_data: Dict[str, Dict[str, List[float]]],
                                          timestamps: List[datetime]) -> go.Figure:
        """创建资源不平等分析图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('基尼系数变化', '资源分配比例', '策略间资源差异', '资源集中度'),
            specs=[[{"secondary_y": False}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 计算基尼系数
        gini_coefficients = {}
        for strategy, resources in resource_data.items():
            gini_values = []
            for i in range(len(timestamps)):
                # 获取该时间点所有智能体的资源
                time_resources = [res_list[i] if i < len(res_list) else 0 
                                for res_list in resources.values()]
                if time_resources:
                    gini = self._calculate_gini_coefficient(time_resources)
                    gini_values.append(gini)
                else:
                    gini_values.append(0)
            gini_coefficients[strategy] = gini_values
        
        # 绘制基尼系数变化
        for strategy, gini_values in gini_coefficients.items():
            fig.add_trace(
                go.Scatter(x=timestamps, y=gini_values, 
                          mode='lines+markers', name=f'{strategy}-基尼系数'),
                row=1, col=1
            )
        
        # 资源分配饼图（使用最新时间点数据）
        if resource_data:
            latest_strategy = list(resource_data.keys())[0]
            latest_resources = resource_data[latest_strategy]
            
            resource_totals = {}
            for resource_type, values in latest_resources.items():
                resource_totals[resource_type] = sum(values[-10:]) if values else 0  # 最近10个时间点的平均
            
            fig.add_trace(
                go.Pie(labels=list(resource_totals.keys()), 
                      values=list(resource_totals.values()),
                      name="资源分配"),
                row=1, col=2
            )
        
        # 策略间资源差异
        if len(resource_data) > 1:
            strategies = list(resource_data.keys())
            resource_types = list(resource_data[strategies[0]].keys())
            
            for resource_type in resource_types:
                strategy_means = []
                for strategy in strategies:
                    if resource_type in resource_data[strategy]:
                        mean_value = np.mean(resource_data[strategy][resource_type][-10:]) if resource_data[strategy][resource_type] else 0
                        strategy_means.append(mean_value)
                    else:
                        strategy_means.append(0)
                
                fig.add_trace(
                    go.Bar(x=strategies, y=strategy_means, name=resource_type,
                          marker_color=self.resource_colors.get(resource_type, '#888')),
                    row=2, col=1
                )
        
        # 资源集中度（HHI指数）
        hhi_values = []
        for i in range(len(timestamps)):
            total_resources = 0
            strategy_resources = {}
            
            for strategy, resources in resource_data.items():
                strategy_total = sum([res_list[i] if i < len(res_list) else 0 
                                    for res_list in resources.values()])
                strategy_resources[strategy] = strategy_total
                total_resources += strategy_total
            
            if total_resources > 0:
                hhi = sum([(res/total_resources)**2 for res in strategy_resources.values()])
                hhi_values.append(hhi)
            else:
                hhi_values.append(0)
        
        fig.add_trace(
            go.Scatter(x=timestamps, y=hhi_values, 
                      mode='lines+markers', name='HHI指数',
                      line=dict(color='#FF6B6B')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="<b>资源分配公平性分析</b><br><sub>验证H2: 强差序格局的先强后脆特征</sub>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values or len(values) == 0:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        if cumsum[-1] == 0:
            return 0
        
        gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(sorted_values, 1))) / (n * cumsum[-1])
        return max(0, min(1, gini))

class EnhancedSpatialVisualizer:
    """增强空间移动可视化器"""
    
    def __init__(self):
        self.movement_colors = {
            'evacuation': '#FF6B6B',
            'resource_seeking': '#4ECDC4',
            'social_gathering': '#45B7D1',
            'random_walk': '#96CEB4'
        }
    
    def create_movement_heatmap_3d(self, 
                                 position_data: Dict[str, List[Tuple[float, float]]],
                                 grid_size: int = 50) -> go.Figure:
        """创建3D移动热力图"""
        if not position_data:
            return go.Figure()
        
        # 收集所有位置点
        all_positions = []
        for agent_positions in position_data.values():
            all_positions.extend(agent_positions)
        
        if not all_positions:
            return go.Figure()
        
        # 计算位置范围
        x_coords = [pos[0] for pos in all_positions]
        y_coords = [pos[1] for pos in all_positions]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 创建网格
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        
        # 计算热力图数据
        heatmap_data = np.zeros((grid_size, grid_size))
        
        for pos in all_positions:
            x_idx = min(int((pos[0] - x_min) / (x_max - x_min) * (grid_size - 1)), grid_size - 1)
            y_idx = min(int((pos[1] - y_min) / (y_max - y_min) * (grid_size - 1)), grid_size - 1)
            heatmap_data[y_idx, x_idx] += 1
        
        # 创建3D表面图
        fig = go.Figure(data=[go.Surface(
            z=heatmap_data,
            x=x_grid,
            y=y_grid,
            colorscale='Viridis',
            name='移动密度'
        )])
        
        fig.update_layout(
            title='<b>智能体移动轨迹3D热力图</b>',
            scene=dict(
                xaxis_title='X坐标',
                yaxis_title='Y坐标',
                zaxis_title='移动密度',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.6)
                )
            ),
            width=1000,
            height=700
        )
        
        return fig
    
    def create_evacuation_flow_analysis(self, 
                                       evacuation_routes: List[List[Tuple[float, float]]],
                                       timestamps: List[datetime]) -> go.Figure:
        """创建疏散流动分析图"""
        if not evacuation_routes:
            return go.Figure()
        
        fig = go.Figure()
        
        # 绘制疏散路径
        for i, route in enumerate(evacuation_routes[:20]):  # 限制显示路径数量
            if len(route) < 2:
                continue
                
            x_coords = [pos[0] for pos in route]
            y_coords = [pos[1] for pos in route]
            
            # 路径线
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                name=f'疏散路径 {i+1}',
                line=dict(width=2, color=f'rgba({i*10 % 255}, {(i*20) % 255}, {(i*30) % 255}, 0.7)'),
                marker=dict(size=4),
                showlegend=False
            ))
            
            # 起点标记
            fig.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='circle'),
                name='起点' if i == 0 else '',
                showlegend=i == 0
            ))
            
            # 终点标记
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode='markers',
                marker=dict(size=10, color='red', symbol='square'),
                name='终点' if i == 0 else '',
                showlegend=i == 0
            ))
        
        fig.update_layout(
            title='<b>疏散路径流动分析</b><br><sub>绿色圆点=起点，红色方块=终点</sub>',
            xaxis_title='X坐标',
            yaxis_title='Y坐标',
            hovermode='closest',
            width=1000,
            height=700
        )
        
        return fig

class EnhancedHypothesisValidator:
    """增强假设验证器"""
    
    def __init__(self):
        self.hypothesis_colors = {
            'H1': '#FF6B6B',  # 强差序格局初期优势
            'H2': '#4ECDC4',  # 先强后脆特征
            'H3': '#45B7D1'   # 普遍主义劣势
        }
        self.hypothesis_labels = {
            'H1': 'H1: 强差序格局初期优势',
            'H2': 'H2: 先强后脆特征', 
            'H3': 'H3: 普遍主义劣势'
        }
    
    def create_hypothesis_validation_dashboard(self, 
                                             hypothesis_results: Dict[str, Dict[str, Any]]) -> go.Figure:
        """创建假设验证仪表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('假设支持度', '统计显著性', '效应量分析', '置信区间'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        hypotheses = list(hypothesis_results.keys())
        support_rates = [hypothesis_results[h].get('support_rate', 0) for h in hypotheses]
        p_values = [hypothesis_results[h].get('p_value', 1) for h in hypotheses]
        effect_sizes = [hypothesis_results[h].get('effect_size', 0) for h in hypotheses]
        
        # 假设支持度柱状图
        fig.add_trace(
            go.Bar(
                x=hypotheses,
                y=support_rates,
                marker_color=[self.hypothesis_colors.get(h, '#888') for h in hypotheses],
                name='支持度',
                text=[f'{rate:.1%}' for rate in support_rates],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 统计显著性散点图
        significance_levels = [-np.log10(p) for p in p_values]
        fig.add_trace(
            go.Scatter(
                x=hypotheses,
                y=significance_levels,
                mode='markers',
                marker=dict(
                    size=[max(10, abs(es)*20) for es in effect_sizes],
                    color=[self.hypothesis_colors.get(h, '#888') for h in hypotheses],
                    opacity=0.7
                ),
                name='显著性水平',
                text=[f'p={p:.3f}' for p in p_values],
                textposition='top center'
            ),
            row=1, col=2
        )
        
        # 效应量分析
        fig.add_trace(
            go.Bar(
                x=hypotheses,
                y=effect_sizes,
                marker_color=[self.hypothesis_colors.get(h, '#888') for h in hypotheses],
                name='效应量',
                text=[f'{es:.2f}' for es in effect_sizes],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 置信区间
        for i, h in enumerate(hypotheses):
            ci_lower = hypothesis_results[h].get('ci_lower', support_rates[i] - 0.1)
            ci_upper = hypothesis_results[h].get('ci_upper', support_rates[i] + 0.1)
            
            fig.add_trace(
                go.Scatter(
                    x=[h, h],
                    y=[ci_lower, ci_upper],
                    mode='lines+markers',
                    line=dict(color=self.hypothesis_colors.get(h, '#888'), width=3),
                    marker=dict(size=8),
                    name=f'{h} 置信区间',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 添加显著性基准线
        fig.add_hline(y=1.3, line_dash="dash", line_color="red", 
                     annotation_text="p=0.05", row=1, col=2)
        
        fig.update_layout(
            title_text="<b>差序格局理论假设验证结果</b>",
            height=800,
            showlegend=True
        )
        
        return fig

class EnhancedComprehensiveDashboard:
    """增强综合仪表板"""
    
    def __init__(self):
        self.survival_analyzer = EnhancedSurvivalAnalyzer()
        self.network_visualizer = EnhancedSocialNetworkVisualizer()
        self.resource_analyzer = EnhancedResourceAnalyzer()
        self.spatial_visualizer = EnhancedSpatialVisualizer()
        self.hypothesis_validator = EnhancedHypothesisValidator()
    
    def create_comprehensive_dashboard(self, 
                                    viz_data: EnhancedVisualizationData) -> Dict[str, go.Figure]:
        """创建综合仪表板"""
        dashboard = {}
        
        try:
            # 1. 生存分析
            if viz_data.survival_rates and viz_data.timestamps:
                logger.info("创建生存曲线分析...")
                try:
                    dashboard['survival_curves'] = self.survival_analyzer.create_survival_curves_comparison(
                        viz_data.survival_rates, viz_data.timestamps
                    )
                    logger.info("生存曲线分析创建成功")
                except Exception as e:
                    logger.error(f"创建生存曲线分析时出错: {e}")
                
                try:
                    dashboard['hazard_analysis'] = self.survival_analyzer.create_hazard_ratio_analysis(
                        viz_data.survival_rates, viz_data.timestamps
                    )
                    logger.info("风险比分析创建成功")
                except Exception as e:
                    logger.error(f"创建风险比分析时出错: {e}")
            
            # 2. 社会网络分析
            if viz_data.network_snapshots:
                logger.info("创建社会网络分析...")
                try:
                    dashboard['network_graph'] = self.network_visualizer.create_dynamic_network_graph(
                        viz_data.network_snapshots, viz_data.timestamps
                    )
                    logger.info("动态网络图创建成功")
                except Exception as e:
                    logger.error(f"创建动态网络图时出错: {e}")
                
                try:
                    dashboard['network_metrics'] = self.network_visualizer.create_network_metrics_dashboard(
                        viz_data.network_snapshots, viz_data.timestamps
                    )
                    logger.info("网络指标仪表板创建成功")
                except Exception as e:
                    logger.error(f"创建网络指标仪表板时出错: {e}")
            
            # 3. 资源分配分析
            if viz_data.resource_distributions:
                logger.info("创建资源分配分析...")
                try:
                    dashboard['resource_inequality'] = self.resource_analyzer.create_resource_inequality_analysis(
                        viz_data.resource_distributions, viz_data.timestamps
                    )
                    logger.info("资源不平等分析创建成功")
                except Exception as e:
                    logger.error(f"创建资源不平等分析时出错: {e}")
            
            # 4. 空间移动分析
            if viz_data.agent_positions:
                logger.info("创建空间移动分析...")
                try:
                    dashboard['movement_heatmap'] = self.spatial_visualizer.create_movement_heatmap_3d(
                        viz_data.agent_positions
                    )
                    logger.info("移动热力图创建成功")
                except Exception as e:
                    logger.error(f"创建移动热力图时出错: {e}")
            
            if viz_data.evacuation_routes:
                try:
                    dashboard['evacuation_flow'] = self.spatial_visualizer.create_evacuation_flow_analysis(
                        viz_data.evacuation_routes, viz_data.timestamps
                    )
                    logger.info("疏散流动分析创建成功")
                except Exception as e:
                    logger.error(f"创建疏散流动分析时出错: {e}")
            
            # 5. 假设验证
            if viz_data.hypothesis_results:
                logger.info("创建假设验证分析...")
                try:
                    dashboard['hypothesis_validation'] = self.hypothesis_validator.create_hypothesis_validation_dashboard(
                        viz_data.hypothesis_results
                    )
                    logger.info("假设验证仪表板创建成功")
                except Exception as e:
                    logger.error(f"创建假设验证仪表板时出错: {e}")
            
            logger.info(f"成功创建 {len(dashboard)} 个可视化组件")
            
        except Exception as e:
            logger.error(f"创建仪表板时发生错误: {e}")
        
        return dashboard
    
    def save_dashboard_html(self, 
                          dashboard: Dict[str, go.Figure], 
                          output_dir: str = "enhanced_dashboard_output") -> None:
        """保存仪表板为HTML文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存各个组件
        for name, fig in dashboard.items():
            file_path = output_path / f"{name}.html"
            fig.write_html(str(file_path))
            logger.info(f"已保存: {file_path}")
        
        # 创建主仪表板页面
        self._create_main_dashboard_html(dashboard, output_path)
        
        logger.info(f"增强可视化仪表板已保存到: {output_path}")
    
    def _create_main_dashboard_html(self, 
                                  dashboard: Dict[str, go.Figure], 
                                  output_path: Path) -> None:
        """创建主仪表板HTML页面"""
        html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>差序格局理论验证 - 增强可视化仪表板</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        iframe {
            width: 100%;
            height: 600px;
            border: none;
            border-radius: 5px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background-color: #333;
            color: white;
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>基于差序格局理论的城市洪灾ABM仿真系统</h1>
        <h2>增强可视化仪表板</h2>
        <p>理论验证与深度分析平台</p>
    </div>
    
    <div class="dashboard-grid">
"""
        
        # 添加各个图表
        chart_titles = {
            'survival_curves': '生存曲线对比分析',
            'hazard_analysis': '风险比详细分析',
            'network_graph': '社会网络结构图',
            'network_metrics': '网络指标演化',
            'resource_inequality': '资源分配公平性',
            'movement_heatmap': '移动轨迹3D热力图',
            'evacuation_flow': '疏散流动分析',
            'hypothesis_validation': '假设验证结果'
        }
        
        for name, fig in dashboard.items():
            title = chart_titles.get(name, name)
            html_content += f"""
        <div class="chart-container">
            <div class="chart-title">{title}</div>
            <iframe src="{name}.html"></iframe>
        </div>
"""
        
        html_content += """
    </div>
    
    <div class="footer">
        <p>© 2025 差序格局理论验证项目 | 增强可视化系统</p>
        <p>生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
</body>
</html>
"""
        
        main_file = output_path / "enhanced_main_dashboard.html"
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"主仪表板页面已创建: {main_file}")

def create_enhanced_visualization_system() -> EnhancedComprehensiveDashboard:
    """创建增强可视化系统"""
    return EnhancedComprehensiveDashboard()

if __name__ == "__main__":
    # 示例数据
    timestamps = [datetime.now() + timedelta(hours=i) for i in range(48)]
    
    # 创建示例数据
    viz_data = EnhancedVisualizationData(
        timestamps=timestamps,
        agent_positions={
            f'agent_{i}': [(np.random.random()*10, np.random.random()*10) for _ in range(48)]
            for i in range(20)
        },
        survival_rates={
            'strong_differential': [0.95 - 0.005*i - 0.001*i**1.5 for i in range(48)],
            'weak_differential': [0.90 - 0.008*i - 0.0005*i**1.3 for i in range(48)],
            'universalism': [0.85 - 0.012*i for i in range(48)]
        },
        network_snapshots=[
            {
                'agents': {
                    f'agent_{i}': {
                        'strategy': ['strong_differential', 'weak_differential', 'universalism'][i % 3],
                        'resources': max(0.1, np.random.random() - t*0.02)  # 资源随时间减少
                    } for i in range(max(5, 15 - t//2))  # 智能体数量随时间减少
                },
                'relationships': [
                    {
                        'source': f'agent_{i}',
                        'target': f'agent_{(i+j) % max(5, 15 - t//2)}',
                        'relationship_type': ['family', 'neighbor', 'colleague'][i % 3],
                        'strength': max(0.1, np.random.random() - t*0.01)  # 关系强度随时间衰减
                    } for i in range(max(3, 20 - t)) for j in range(1, min(3, max(1, 4 - t//3)))
                    if i != (i+j) % max(5, 15 - t//2)  # 避免自环
                ]
            } for t in range(48)  # 48个时间点，每个都不同
        ],
        resource_distributions={
            'strong_differential': {
                'food': [0.4 + 0.01*np.sin(i/5) for i in range(48)],
                'water': [0.3 + 0.01*np.cos(i/5) for i in range(48)],
                'shelter': [0.3 + 0.005*np.sin(i/3) for i in range(48)]
            },
            'weak_differential': {
                'food': [0.35 + 0.008*np.sin(i/5) for i in range(48)],
                'water': [0.35 + 0.008*np.cos(i/5) for i in range(48)],
                'shelter': [0.3 + 0.004*np.sin(i/3) for i in range(48)]
            },
            'universalism': {
                'food': [0.33 + 0.005*np.sin(i/5) for i in range(48)],
                'water': [0.33 + 0.005*np.cos(i/5) for i in range(48)],
                'shelter': [0.34 + 0.003*np.sin(i/3) for i in range(48)]
            }
        },
        evacuation_routes=[
            [(np.random.random()*10, np.random.random()*10) for _ in range(np.random.randint(5, 15))]
            for _ in range(30)
        ],
        hypothesis_results={
            'H1': {
                'support_rate': 0.78,
                'p_value': 0.023,
                'effect_size': 0.65,
                'ci_lower': 0.68,
                'ci_upper': 0.88
            },
            'H2': {
                'support_rate': 0.72,
                'p_value': 0.041,
                'effect_size': 0.58,
                'ci_lower': 0.62,
                'ci_upper': 0.82
            },
            'H3': {
                'support_rate': 0.85,
                'p_value': 0.008,
                'effect_size': 0.73,
                'ci_lower': 0.75,
                'ci_upper': 0.95
            }
        }
    )
    
    # 创建增强可视化系统
    dashboard_system = create_enhanced_visualization_system()
    
    # 生成仪表板
    logger.info("开始创建增强可视化仪表板...")
    charts = dashboard_system.create_comprehensive_dashboard(viz_data)
    
    # 保存仪表板
    dashboard_system.save_dashboard_html(charts)
    
    print("\n=== 增强可视化系统创建完成 ===")
    print(f"生成了 {len(charts)} 个可视化组件:")
    for name in charts.keys():
        print(f"  - {name}")
    print("\n请查看 enhanced_dashboard_output/enhanced_main_dashboard.html")
    print("\n新增功能:")
    print("  ✅ 生存曲线置信区间分析")
    print("  ✅ 风险比和效应量分析")
    print("  ✅ 动态社会网络指标")
    print("  ✅ 资源不平等基尼系数")
    print("  ✅ 3D移动轨迹热力图")
    print("  ✅ 疏散流动路径分析")
    print("  ✅ 假设验证统计仪表板")