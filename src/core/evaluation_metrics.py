"""评估指标体系

实现基于差序格局理论的多维度评估指标：
1. 群体韧性指标：生存率、恢复能力、适应性等
2. 社会网络指标：网络密度、中心性、聚类系数等
3. 资源分配指标：公平性、效率、集中度等
4. 空间行为指标：移动模式、聚集度、疏散效率等

作者: 城市智能体项目组
日期: 2024-12-30
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, spatial
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """指标类别枚举"""
    RESILIENCE = "resilience"  # 群体韧性
    NETWORK = "network"  # 社会网络
    RESOURCE = "resource"  # 资源分配
    SPATIAL = "spatial"  # 空间行为
    TEMPORAL = "temporal"  # 时间动态
    DIFFERENTIAL = "differential"  # 差序格局特征


class MetricLevel(Enum):
    """指标层级枚举"""
    INDIVIDUAL = "individual"  # 个体层级
    GROUP = "group"  # 群体层级
    SYSTEM = "system"  # 系统层级


@dataclass
class MetricResult:
    """指标计算结果"""
    metric_name: str
    category: MetricCategory
    level: MetricLevel
    value: Union[float, Dict[str, float], List[float]]
    timestamp: datetime
    description: str
    unit: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResilienceMetrics:
    """群体韧性指标计算器"""
    
    def __init__(self):
        self.historical_data: List[Dict[str, Any]] = []
    
    def calculate_survival_resilience(self, agents: Dict[str, Any], 
                                    initial_population: int) -> MetricResult:
        """计算生存韧性指标"""
        alive_agents = sum(1 for agent in agents.values() if agent.get('status') == 'alive')
        current_population = len(agents)
        
        # 生存率
        survival_rate = alive_agents / max(initial_population, 1)
        
        # 人口保持率
        population_retention = current_population / max(initial_population, 1)
        
        # 综合生存韧性
        survival_resilience = (survival_rate * 0.7 + population_retention * 0.3)
        
        return MetricResult(
            metric_name="survival_resilience",
            category=MetricCategory.RESILIENCE,
            level=MetricLevel.SYSTEM,
            value={
                'survival_rate': survival_rate,
                'population_retention': population_retention,
                'overall_resilience': survival_resilience
            },
            timestamp=datetime.now(),
            description="群体生存韧性，包括生存率和人口保持率",
            unit="ratio"
        )
    
    def calculate_resource_resilience(self, agents: Dict[str, Any], 
                                    initial_resources: Dict[str, float]) -> MetricResult:
        """计算资源韧性指标"""
        current_resources = defaultdict(float)
        
        for agent in agents.values():
            resources = agent.get('resources', {})
            if isinstance(resources, dict):
                for resource_type, amount in resources.items():
                    current_resources[resource_type] += amount
            else:
                current_resources['total'] += resources
        
        # 资源保持率
        resource_retention = {}
        for resource_type, initial_amount in initial_resources.items():
            current_amount = current_resources.get(resource_type, 0)
            retention = current_amount / max(initial_amount, 1)
            resource_retention[resource_type] = retention
        
        # 平均资源保持率
        avg_retention = np.mean(list(resource_retention.values())) if resource_retention else 0
        
        # 资源分布均匀度
        resource_values = [agent.get('resources', 0) for agent in agents.values()]
        if isinstance(resource_values[0], dict):
            resource_values = [sum(r.values()) if isinstance(r, dict) else r for r in resource_values]
        
        resource_gini = self._calculate_gini_coefficient(resource_values)
        resource_evenness = 1.0 - resource_gini
        
        return MetricResult(
            metric_name="resource_resilience",
            category=MetricCategory.RESILIENCE,
            level=MetricLevel.SYSTEM,
            value={
                'resource_retention': resource_retention,
                'average_retention': avg_retention,
                'resource_evenness': resource_evenness,
                'gini_coefficient': resource_gini
            },
            timestamp=datetime.now(),
            description="资源韧性，包括资源保持率和分布均匀度",
            unit="ratio"
        )
    
    def calculate_adaptive_resilience(self, agents: Dict[str, Any], 
                                    time_series_data: List[Dict[str, Any]]) -> MetricResult:
        """计算适应性韧性指标"""
        if len(time_series_data) < 2:
            return MetricResult(
                metric_name="adaptive_resilience",
                category=MetricCategory.RESILIENCE,
                level=MetricLevel.SYSTEM,
                value=0.0,
                timestamp=datetime.now(),
                description="适应性韧性（数据不足）",
                unit="ratio"
            )
        
        # 计算策略变化率
        strategy_changes = 0
        total_agents = len(agents)
        
        for agent_id, agent_data in agents.items():
            strategy_history = agent_data.get('strategy_history', [])
            if len(strategy_history) > 1:
                changes = sum(1 for i in range(1, len(strategy_history)) 
                            if strategy_history[i] != strategy_history[i-1])
                strategy_changes += changes
        
        adaptation_rate = strategy_changes / max(total_agents, 1)
        
        # 计算行为多样性
        recent_actions = []
        for agent in agents.values():
            recent_actions.extend(agent.get('recent_actions', []))
        
        action_diversity = len(set(recent_actions)) / max(len(recent_actions), 1) if recent_actions else 0
        
        # 计算学习效率
        learning_scores = [agent.get('learning_score', 0) for agent in agents.values()]
        avg_learning = np.mean(learning_scores) if learning_scores else 0
        
        # 综合适应性韧性
        adaptive_resilience = (adaptation_rate * 0.4 + action_diversity * 0.3 + avg_learning * 0.3)
        
        return MetricResult(
            metric_name="adaptive_resilience",
            category=MetricCategory.RESILIENCE,
            level=MetricLevel.SYSTEM,
            value={
                'adaptation_rate': adaptation_rate,
                'action_diversity': action_diversity,
                'learning_efficiency': avg_learning,
                'overall_adaptiveness': adaptive_resilience
            },
            timestamp=datetime.now(),
            description="适应性韧性，包括策略变化、行为多样性和学习效率",
            unit="ratio"
        )
    
    def calculate_recovery_resilience(self, agents: Dict[str, Any], 
                                    disaster_impact_time: datetime) -> MetricResult:
        """计算恢复韧性指标"""
        current_time = datetime.now()
        time_since_disaster = (current_time - disaster_impact_time).total_seconds() / 3600  # 小时
        
        # 功能恢复率
        functional_agents = sum(1 for agent in agents.values() 
                              if agent.get('functional_status', True))
        function_recovery_rate = functional_agents / max(len(agents), 1)
        
        # 健康恢复率
        healthy_agents = sum(1 for agent in agents.values() 
                           if agent.get('health_status', 'healthy') in ['healthy', 'recovering'])
        health_recovery_rate = healthy_agents / max(len(agents), 1)
        
        # 社交网络恢复率
        active_connections = 0
        total_possible_connections = 0
        
        for agent in agents.values():
            connections = agent.get('social_connections', [])
            active_connections += len([c for c in connections if c.get('active', True)])
            total_possible_connections += len(connections)
        
        network_recovery_rate = active_connections / max(total_possible_connections, 1)
        
        # 恢复速度（基于时间）
        expected_recovery_time = 24  # 预期24小时内恢复
        recovery_speed = max(0, 1 - (time_since_disaster / expected_recovery_time))
        
        # 综合恢复韧性
        recovery_resilience = (function_recovery_rate * 0.3 + 
                             health_recovery_rate * 0.3 + 
                             network_recovery_rate * 0.2 + 
                             recovery_speed * 0.2)
        
        return MetricResult(
            metric_name="recovery_resilience",
            category=MetricCategory.RESILIENCE,
            level=MetricLevel.SYSTEM,
            value={
                'function_recovery_rate': function_recovery_rate,
                'health_recovery_rate': health_recovery_rate,
                'network_recovery_rate': network_recovery_rate,
                'recovery_speed': recovery_speed,
                'overall_recovery': recovery_resilience,
                'time_since_disaster_hours': time_since_disaster
            },
            timestamp=datetime.now(),
            description="恢复韧性，包括功能、健康、网络恢复和恢复速度",
            unit="ratio"
        )
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values or all(v == 0 for v in values):
            return 0.0
        
        sorted_values = sorted([max(0, v) for v in values])
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(sorted_values))


class NetworkMetrics:
    """社会网络指标计算器"""
    
    def __init__(self):
        pass
    
    def calculate_network_structure_metrics(self, network: nx.Graph) -> MetricResult:
        """计算网络结构指标"""
        if network.number_of_nodes() == 0:
            return MetricResult(
                metric_name="network_structure",
                category=MetricCategory.NETWORK,
                level=MetricLevel.SYSTEM,
                value={},
                timestamp=datetime.now(),
                description="网络结构指标（空网络）",
                unit="various"
            )
        
        # 基本网络指标
        density = nx.density(network)
        
        # 聚类系数
        clustering = nx.average_clustering(network)
        
        # 平均路径长度
        if nx.is_connected(network):
            avg_path_length = nx.average_shortest_path_length(network)
        else:
            # 对于非连通图，计算最大连通分量的平均路径长度
            largest_cc = max(nx.connected_components(network), key=len)
            subgraph = network.subgraph(largest_cc)
            avg_path_length = nx.average_shortest_path_length(subgraph) if len(largest_cc) > 1 else 0
        
        # 连通分量数量
        num_components = nx.number_connected_components(network)
        
        # 网络直径
        if nx.is_connected(network):
            diameter = nx.diameter(network)
        else:
            diameter = max([nx.diameter(network.subgraph(c)) for c in nx.connected_components(network) if len(c) > 1], default=0)
        
        # 度分布统计
        degrees = [d for n, d in network.degree()]
        degree_stats = {
            'mean': np.mean(degrees),
            'std': np.std(degrees),
            'max': max(degrees) if degrees else 0,
            'min': min(degrees) if degrees else 0
        }
        
        return MetricResult(
            metric_name="network_structure",
            category=MetricCategory.NETWORK,
            level=MetricLevel.SYSTEM,
            value={
                'density': density,
                'clustering_coefficient': clustering,
                'average_path_length': avg_path_length,
                'diameter': diameter,
                'num_components': num_components,
                'degree_statistics': degree_stats
            },
            timestamp=datetime.now(),
            description="网络结构指标，包括密度、聚类系数、路径长度等",
            unit="various"
        )
    
    def calculate_centrality_metrics(self, network: nx.Graph) -> MetricResult:
        """计算中心性指标"""
        if network.number_of_nodes() == 0:
            return MetricResult(
                metric_name="centrality_metrics",
                category=MetricCategory.NETWORK,
                level=MetricLevel.INDIVIDUAL,
                value={},
                timestamp=datetime.now(),
                description="中心性指标（空网络）",
                unit="ratio"
            )
        
        # 度中心性
        degree_centrality = nx.degree_centrality(network)
        
        # 介数中心性
        betweenness_centrality = nx.betweenness_centrality(network)
        
        # 接近中心性
        closeness_centrality = nx.closeness_centrality(network)
        
        # 特征向量中心性
        try:
            eigenvector_centrality = nx.eigenvector_centrality(network, max_iter=1000)
        except:
            eigenvector_centrality = {node: 0 for node in network.nodes()}
        
        # 计算中心性分布统计
        centrality_stats = {}
        for centrality_name, centrality_values in [
            ('degree', degree_centrality),
            ('betweenness', betweenness_centrality),
            ('closeness', closeness_centrality),
            ('eigenvector', eigenvector_centrality)
        ]:
            values = list(centrality_values.values())
            centrality_stats[centrality_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': max(values) if values else 0,
                'top_nodes': sorted(centrality_values.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        return MetricResult(
            metric_name="centrality_metrics",
            category=MetricCategory.NETWORK,
            level=MetricLevel.INDIVIDUAL,
            value={
                'degree_centrality': degree_centrality,
                'betweenness_centrality': betweenness_centrality,
                'closeness_centrality': closeness_centrality,
                'eigenvector_centrality': eigenvector_centrality,
                'centrality_statistics': centrality_stats
            },
            timestamp=datetime.now(),
            description="节点中心性指标，包括度、介数、接近和特征向量中心性",
            unit="ratio"
        )
    
    def calculate_differential_network_metrics(self, network: nx.Graph, 
                                             node_attributes: Dict[str, Dict[str, Any]]) -> MetricResult:
        """计算差序格局网络指标"""
        if network.number_of_nodes() == 0:
            return MetricResult(
                metric_name="differential_network",
                category=MetricCategory.DIFFERENTIAL,
                level=MetricLevel.SYSTEM,
                value={},
                timestamp=datetime.now(),
                description="差序格局网络指标（空网络）",
                unit="various"
            )
        
        # 按关系类型分析连接
        relationship_analysis = defaultdict(lambda: {'count': 0, 'strength': 0})
        
        for edge in network.edges(data=True):
            node1, node2, edge_data = edge
            relationship_type = edge_data.get('relationship_type', 'unknown')
            strength = edge_data.get('strength', 1.0)
            
            relationship_analysis[relationship_type]['count'] += 1
            relationship_analysis[relationship_type]['strength'] += strength
        
        # 计算关系类型分布
        total_edges = network.number_of_edges()
        relationship_distribution = {}
        for rel_type, data in relationship_analysis.items():
            relationship_distribution[rel_type] = {
                'proportion': data['count'] / max(total_edges, 1),
                'average_strength': data['strength'] / max(data['count'], 1)
            }
        
        # 计算差序格局指数（亲密关系占比）
        intimate_relationships = ['family', 'close_friend', 'relative']
        intimate_edges = sum(relationship_analysis[rel]['count'] for rel in intimate_relationships)
        differential_index = intimate_edges / max(total_edges, 1)
        
        # 计算网络同质性（相似节点连接倾向）
        homophily_scores = {}
        for attribute in ['strategy_type', 'age_group', 'education_level']:
            if attribute in node_attributes.get(list(network.nodes())[0], {}):
                homophily = self._calculate_homophily(network, node_attributes, attribute)
                homophily_scores[attribute] = homophily
        
        # 计算互惠性指标
        reciprocity = self._calculate_network_reciprocity(network)
        
        return MetricResult(
            metric_name="differential_network",
            category=MetricCategory.DIFFERENTIAL,
            level=MetricLevel.SYSTEM,
            value={
                'relationship_distribution': relationship_distribution,
                'differential_index': differential_index,
                'homophily_scores': homophily_scores,
                'reciprocity': reciprocity,
                'total_relationships': len(relationship_analysis)
            },
            timestamp=datetime.now(),
            description="差序格局网络特征，包括关系分布、差序指数、同质性等",
            unit="various"
        )
    
    def _calculate_homophily(self, network: nx.Graph, node_attributes: Dict[str, Dict[str, Any]], 
                           attribute: str) -> float:
        """计算网络同质性"""
        same_attribute_edges = 0
        total_edges = 0
        
        for edge in network.edges():
            node1, node2 = edge
            if (node1 in node_attributes and node2 in node_attributes and 
                attribute in node_attributes[node1] and attribute in node_attributes[node2]):
                
                total_edges += 1
                if node_attributes[node1][attribute] == node_attributes[node2][attribute]:
                    same_attribute_edges += 1
        
        return same_attribute_edges / max(total_edges, 1)
    
    def _calculate_network_reciprocity(self, network: nx.Graph) -> float:
        """计算网络互惠性"""
        if network.is_directed():
            return nx.reciprocity(network)
        else:
            # 对于无向图，所有边都是互惠的
            return 1.0


class ResourceMetrics:
    """资源分配指标计算器"""
    
    def __init__(self):
        pass
    
    def calculate_resource_distribution_metrics(self, agents: Dict[str, Any]) -> MetricResult:
        """计算资源分配指标"""
        # 提取资源数据
        resource_data = []
        for agent in agents.values():
            resources = agent.get('resources', 0)
            if isinstance(resources, dict):
                total_resources = sum(resources.values())
            else:
                total_resources = resources
            resource_data.append(total_resources)
        
        if not resource_data:
            return MetricResult(
                metric_name="resource_distribution",
                category=MetricCategory.RESOURCE,
                level=MetricLevel.SYSTEM,
                value={},
                timestamp=datetime.now(),
                description="资源分配指标（无数据）",
                unit="various"
            )
        
        # 基本统计
        mean_resources = np.mean(resource_data)
        std_resources = np.std(resource_data)
        
        # 基尼系数（不平等程度）
        gini_coefficient = self._calculate_gini_coefficient(resource_data)
        
        # 泰尔指数（另一种不平等度量）
        theil_index = self._calculate_theil_index(resource_data)
        
        # 20/80规则检验（帕累托分布）
        sorted_resources = sorted(resource_data, reverse=True)
        top_20_percent_count = max(1, len(sorted_resources) // 5)
        top_20_percent_resources = sum(sorted_resources[:top_20_percent_count])
        total_resources = sum(sorted_resources)
        pareto_ratio = top_20_percent_resources / max(total_resources, 1)
        
        # 资源集中度（HHI指数）
        resource_shares = [r / max(total_resources, 1) for r in resource_data]
        hhi_index = sum(share ** 2 for share in resource_shares)
        
        # 资源充足性（满足基本需求的比例）
        basic_need_threshold = mean_resources * 0.5  # 假设平均值的50%为基本需求
        adequacy_rate = sum(1 for r in resource_data if r >= basic_need_threshold) / len(resource_data)
        
        return MetricResult(
            metric_name="resource_distribution",
            category=MetricCategory.RESOURCE,
            level=MetricLevel.SYSTEM,
            value={
                'mean_resources': mean_resources,
                'std_resources': std_resources,
                'gini_coefficient': gini_coefficient,
                'theil_index': theil_index,
                'pareto_ratio': pareto_ratio,
                'hhi_index': hhi_index,
                'adequacy_rate': adequacy_rate,
                'total_agents': len(resource_data)
            },
            timestamp=datetime.now(),
            description="资源分配指标，包括不平等度、集中度、充足性等",
            unit="various"
        )
    
    def calculate_resource_efficiency_metrics(self, agents: Dict[str, Any], 
                                            resource_transactions: List[Dict[str, Any]]) -> MetricResult:
        """计算资源效率指标"""
        if not resource_transactions:
            return MetricResult(
                metric_name="resource_efficiency",
                category=MetricCategory.RESOURCE,
                level=MetricLevel.SYSTEM,
                value={'efficiency': 0, 'utilization': 0},
                timestamp=datetime.now(),
                description="资源效率指标（无交易数据）",
                unit="ratio"
            )
        
        # 资源利用率
        total_available = sum(agent.get('initial_resources', 0) for agent in agents.values())
        total_used = sum(transaction.get('amount', 0) for transaction in resource_transactions)
        utilization_rate = total_used / max(total_available, 1)
        
        # 交易效率（成功交易比例）
        successful_transactions = sum(1 for t in resource_transactions if t.get('success', False))
        transaction_efficiency = successful_transactions / len(resource_transactions)
        
        # 资源流动性（交易频率）
        unique_participants = set()
        for transaction in resource_transactions:
            unique_participants.add(transaction.get('giver'))
            unique_participants.add(transaction.get('receiver'))
        
        participation_rate = len(unique_participants) / max(len(agents), 1)
        
        # 资源配置效率（需求匹配度）
        demand_satisfaction = self._calculate_demand_satisfaction(agents, resource_transactions)
        
        # 综合效率分数
        overall_efficiency = (utilization_rate * 0.3 + 
                            transaction_efficiency * 0.3 + 
                            participation_rate * 0.2 + 
                            demand_satisfaction * 0.2)
        
        return MetricResult(
            metric_name="resource_efficiency",
            category=MetricCategory.RESOURCE,
            level=MetricLevel.SYSTEM,
            value={
                'utilization_rate': utilization_rate,
                'transaction_efficiency': transaction_efficiency,
                'participation_rate': participation_rate,
                'demand_satisfaction': demand_satisfaction,
                'overall_efficiency': overall_efficiency,
                'total_transactions': len(resource_transactions)
            },
            timestamp=datetime.now(),
            description="资源效率指标，包括利用率、交易效率、参与率等",
            unit="ratio"
        )
    
    def calculate_differential_resource_metrics(self, agents: Dict[str, Any], 
                                              resource_transactions: List[Dict[str, Any]]) -> MetricResult:
        """计算差序格局资源分配指标"""
        # 按关系类型分析资源流动
        relationship_flows = defaultdict(lambda: {'total_amount': 0, 'transaction_count': 0})
        
        for transaction in resource_transactions:
            relationship = transaction.get('relationship_type', 'unknown')
            amount = transaction.get('amount', 0)
            
            relationship_flows[relationship]['total_amount'] += amount
            relationship_flows[relationship]['transaction_count'] += 1
        
        # 计算差序格局资源分配模式
        intimate_relationships = ['family', 'close_friend', 'relative']
        distant_relationships = ['acquaintance', 'stranger', 'unknown']
        
        intimate_flow = sum(relationship_flows[rel]['total_amount'] for rel in intimate_relationships)
        distant_flow = sum(relationship_flows[rel]['total_amount'] for rel in distant_relationships)
        total_flow = intimate_flow + distant_flow
        
        differential_resource_ratio = intimate_flow / max(total_flow, 1)
        
        # 计算资源分配偏向性
        allocation_bias = {}
        for relationship, data in relationship_flows.items():
            if data['transaction_count'] > 0:
                avg_amount = data['total_amount'] / data['transaction_count']
                allocation_bias[relationship] = avg_amount
        
        # 计算资源集中度（按策略类型）
        strategy_resources = defaultdict(list)
        for agent in agents.values():
            strategy = agent.get('strategy_type', 'unknown')
            resources = agent.get('resources', 0)
            if isinstance(resources, dict):
                resources = sum(resources.values())
            strategy_resources[strategy].append(resources)
        
        strategy_concentration = {}
        for strategy, resources in strategy_resources.items():
            if resources:
                strategy_concentration[strategy] = {
                    'mean': np.mean(resources),
                    'std': np.std(resources),
                    'gini': self._calculate_gini_coefficient(resources)
                }
        
        return MetricResult(
            metric_name="differential_resource",
            category=MetricCategory.DIFFERENTIAL,
            level=MetricLevel.SYSTEM,
            value={
                'relationship_flows': dict(relationship_flows),
                'differential_resource_ratio': differential_resource_ratio,
                'allocation_bias': allocation_bias,
                'strategy_concentration': strategy_concentration,
                'intimate_vs_distant_ratio': intimate_flow / max(distant_flow, 1)
            },
            timestamp=datetime.now(),
            description="差序格局资源分配特征，包括关系偏向、策略集中度等",
            unit="various"
        )
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """计算基尼系数"""
        if not values or all(v == 0 for v in values):
            return 0.0
        
        sorted_values = sorted([max(0, v) for v in values])
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(sorted_values))
    
    def _calculate_theil_index(self, values: List[float]) -> float:
        """计算泰尔指数"""
        if not values or all(v == 0 for v in values):
            return 0.0
        
        mean_value = np.mean(values)
        n = len(values)
        
        theil = 0
        for value in values:
            if value > 0:
                theil += (value / mean_value) * np.log(value / mean_value)
        
        return theil / n
    
    def _calculate_demand_satisfaction(self, agents: Dict[str, Any], 
                                     transactions: List[Dict[str, Any]]) -> float:
        """计算需求满足度"""
        # 简化版本：假设所有成功交易都满足了需求
        successful_transactions = sum(1 for t in transactions if t.get('success', False))
        total_demand_requests = len([agent for agent in agents.values() 
                                   if agent.get('has_unmet_needs', False)])
        
        if total_demand_requests == 0:
            return 1.0  # 没有未满足需求
        
        return min(1.0, successful_transactions / total_demand_requests)


class SpatialMetrics:
    """空间行为指标计算器"""
    
    def __init__(self):
        pass
    
    def calculate_mobility_metrics(self, agents: Dict[str, Any], 
                                 movement_history: List[Dict[str, Any]]) -> MetricResult:
        """计算移动性指标"""
        if not movement_history:
            return MetricResult(
                metric_name="mobility_metrics",
                category=MetricCategory.SPATIAL,
                level=MetricLevel.INDIVIDUAL,
                value={},
                timestamp=datetime.now(),
                description="移动性指标（无移动数据）",
                unit="various"
            )
        
        # 计算移动距离统计
        movement_distances = []
        movement_speeds = []
        
        for movement in movement_history:
            distance = movement.get('distance', 0)
            duration = movement.get('duration', 1)
            speed = distance / max(duration, 0.1)
            
            movement_distances.append(distance)
            movement_speeds.append(speed)
        
        # 移动性统计
        mobility_stats = {
            'total_movements': len(movement_history),
            'average_distance': np.mean(movement_distances),
            'total_distance': sum(movement_distances),
            'average_speed': np.mean(movement_speeds),
            'max_distance': max(movement_distances) if movement_distances else 0,
            'distance_std': np.std(movement_distances)
        }
        
        # 计算移动模式
        movement_patterns = self._analyze_movement_patterns(movement_history)
        
        # 计算空间覆盖度
        unique_locations = set()
        for movement in movement_history:
            unique_locations.add(tuple(movement.get('start_location', (0, 0))))
            unique_locations.add(tuple(movement.get('end_location', (0, 0))))
        
        spatial_coverage = len(unique_locations)
        
        return MetricResult(
            metric_name="mobility_metrics",
            category=MetricCategory.SPATIAL,
            level=MetricLevel.INDIVIDUAL,
            value={
                'mobility_statistics': mobility_stats,
                'movement_patterns': movement_patterns,
                'spatial_coverage': spatial_coverage,
                'unique_locations': len(unique_locations)
            },
            timestamp=datetime.now(),
            description="移动性指标，包括距离、速度、模式、覆盖度等",
            unit="various"
        )
    
    def calculate_clustering_metrics(self, agents: Dict[str, Any]) -> MetricResult:
        """计算空间聚集指标"""
        # 提取智能体位置
        positions = []
        agent_ids = []
        
        for agent_id, agent_data in agents.items():
            location = agent_data.get('location', (0, 0))
            if isinstance(location, (list, tuple)) and len(location) >= 2:
                positions.append([location[0], location[1]])
                agent_ids.append(agent_id)
        
        if len(positions) < 2:
            return MetricResult(
                metric_name="clustering_metrics",
                category=MetricCategory.SPATIAL,
                level=MetricLevel.SYSTEM,
                value={'clustering_coefficient': 0, 'num_clusters': 0},
                timestamp=datetime.now(),
                description="空间聚集指标（位置数据不足）",
                unit="ratio"
            )
        
        positions = np.array(positions)
        
        # 使用K-means聚类分析
        optimal_k = min(max(2, len(positions) // 10), 10)  # 动态确定聚类数
        
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(positions)
            
            # 计算轮廓系数
            silhouette_avg = silhouette_score(positions, cluster_labels)
            
            # 计算聚类统计
            cluster_stats = {}
            for i in range(optimal_k):
                cluster_positions = positions[cluster_labels == i]
                if len(cluster_positions) > 0:
                    cluster_stats[f'cluster_{i}'] = {
                        'size': len(cluster_positions),
                        'center': kmeans.cluster_centers_[i].tolist(),
                        'spread': np.std(cluster_positions, axis=0).tolist()
                    }
        
        except Exception as e:
            logger.warning(f"聚类分析失败: {e}")
            silhouette_avg = 0
            cluster_stats = {}
            optimal_k = 1
        
        # 计算最近邻距离分布
        distances = spatial.distance.pdist(positions)
        nearest_neighbor_distances = []
        
        for i, pos in enumerate(positions):
            other_positions = np.delete(positions, i, axis=0)
            if len(other_positions) > 0:
                distances_to_others = spatial.distance.cdist([pos], other_positions)[0]
                nearest_neighbor_distances.append(min(distances_to_others))
        
        # 计算空间自相关（Moran's I）
        moran_i = self._calculate_morans_i(positions)
        
        return MetricResult(
            metric_name="clustering_metrics",
            category=MetricCategory.SPATIAL,
            level=MetricLevel.SYSTEM,
            value={
                'silhouette_score': silhouette_avg,
                'num_clusters': optimal_k,
                'cluster_statistics': cluster_stats,
                'average_nearest_neighbor': np.mean(nearest_neighbor_distances) if nearest_neighbor_distances else 0,
                'spatial_autocorrelation': moran_i,
                'total_agents': len(positions)
            },
            timestamp=datetime.now(),
            description="空间聚集指标，包括聚类质量、最近邻距离、空间自相关等",
            unit="various"
        )
    
    def calculate_evacuation_efficiency_metrics(self, agents: Dict[str, Any], 
                                              evacuation_data: List[Dict[str, Any]]) -> MetricResult:
        """计算疏散效率指标"""
        if not evacuation_data:
            return MetricResult(
                metric_name="evacuation_efficiency",
                category=MetricCategory.SPATIAL,
                level=MetricLevel.SYSTEM,
                value={'efficiency': 0},
                timestamp=datetime.now(),
                description="疏散效率指标（无疏散数据）",
                unit="ratio"
            )
        
        # 疏散时间分析
        evacuation_times = [data.get('evacuation_time', 0) for data in evacuation_data]
        successful_evacuations = sum(1 for data in evacuation_data if data.get('success', False))
        
        # 基本效率指标
        success_rate = successful_evacuations / len(evacuation_data)
        average_evacuation_time = np.mean(evacuation_times)
        
        # 疏散路径效率
        path_efficiencies = []
        for data in evacuation_data:
            actual_distance = data.get('actual_distance', 0)
            optimal_distance = data.get('optimal_distance', actual_distance)
            if optimal_distance > 0:
                efficiency = optimal_distance / actual_distance
                path_efficiencies.append(min(1.0, efficiency))
        
        average_path_efficiency = np.mean(path_efficiencies) if path_efficiencies else 0
        
        # 拥堵分析
        congestion_levels = [data.get('congestion_level', 0) for data in evacuation_data]
        average_congestion = np.mean(congestion_levels)
        
        # 疏散目标达成率
        target_locations = set(data.get('target_location') for data in evacuation_data if data.get('success'))
        location_diversity = len(target_locations)
        
        # 时间分布分析
        time_percentiles = {
            '25th': np.percentile(evacuation_times, 25),
            '50th': np.percentile(evacuation_times, 50),
            '75th': np.percentile(evacuation_times, 75),
            '95th': np.percentile(evacuation_times, 95)
        }
        
        # 综合疏散效率
        overall_efficiency = (success_rate * 0.4 + 
                            average_path_efficiency * 0.3 + 
                            (1 - average_congestion) * 0.2 + 
                            (1 / max(average_evacuation_time, 1)) * 0.1)
        
        return MetricResult(
            metric_name="evacuation_efficiency",
            category=MetricCategory.SPATIAL,
            level=MetricLevel.SYSTEM,
            value={
                'success_rate': success_rate,
                'average_evacuation_time': average_evacuation_time,
                'average_path_efficiency': average_path_efficiency,
                'average_congestion': average_congestion,
                'location_diversity': location_diversity,
                'time_percentiles': time_percentiles,
                'overall_efficiency': overall_efficiency,
                'total_evacuations': len(evacuation_data)
            },
            timestamp=datetime.now(),
            description="疏散效率指标，包括成功率、时间、路径效率、拥堵等",
            unit="various"
        )
    
    def _analyze_movement_patterns(self, movement_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析移动模式"""
        # 移动方向分析
        directions = []
        for movement in movement_history:
            start = movement.get('start_location', (0, 0))
            end = movement.get('end_location', (0, 0))
            
            if start != end:
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = np.arctan2(dy, dx)
                directions.append(angle)
        
        # 移动时间模式
        time_patterns = defaultdict(int)
        for movement in movement_history:
            timestamp = movement.get('timestamp')
            if timestamp:
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except:
                        continue
                hour = timestamp.hour
                time_patterns[f'hour_{hour}'] += 1
        
        # 移动频率分析
        agent_movements = defaultdict(int)
        for movement in movement_history:
            agent_id = movement.get('agent_id')
            if agent_id:
                agent_movements[agent_id] += 1
        
        movement_frequency_stats = {
            'mean': np.mean(list(agent_movements.values())) if agent_movements else 0,
            'std': np.std(list(agent_movements.values())) if agent_movements else 0,
            'max': max(agent_movements.values()) if agent_movements else 0
        }
        
        return {
            'direction_variance': np.var(directions) if directions else 0,
            'time_patterns': dict(time_patterns),
            'movement_frequency_stats': movement_frequency_stats,
            'total_unique_movers': len(agent_movements)
        }
    
    def _calculate_morans_i(self, positions: np.ndarray) -> float:
        """计算Moran's I空间自相关指数"""
        if len(positions) < 3:
            return 0.0
        
        try:
            # 构建空间权重矩阵（基于距离）
            distances = spatial.distance.cdist(positions, positions)
            
            # 使用反距离权重，避免除零
            weights = 1.0 / (distances + 1e-10)
            np.fill_diagonal(weights, 0)  # 自身权重为0
            
            # 标准化权重
            row_sums = weights.sum(axis=1)
            weights = weights / row_sums[:, np.newaxis]
            weights[np.isnan(weights)] = 0
            
            # 计算属性值（这里使用x坐标作为示例）
            values = positions[:, 0]
            mean_value = np.mean(values)
            
            # 计算Moran's I
            numerator = 0
            denominator = 0
            
            for i in range(len(positions)):
                for j in range(len(positions)):
                    if i != j:
                        numerator += weights[i, j] * (values[i] - mean_value) * (values[j] - mean_value)
                
                denominator += (values[i] - mean_value) ** 2
            
            if denominator == 0:
                return 0.0
            
            n = len(positions)
            moran_i = (n / weights.sum()) * (numerator / denominator)
            
            return moran_i
        
        except Exception as e:
            logger.warning(f"Moran's I计算失败: {e}")
            return 0.0


class ComprehensiveEvaluationSystem:
    """综合评估系统"""
    
    def __init__(self):
        self.resilience_metrics = ResilienceMetrics()
        self.network_metrics = NetworkMetrics()
        self.resource_metrics = ResourceMetrics()
        self.spatial_metrics = SpatialMetrics()
        
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def run_comprehensive_evaluation(self, 
                                   agents: Dict[str, Any],
                                   network: nx.Graph,
                                   movement_history: List[Dict[str, Any]],
                                   resource_transactions: List[Dict[str, Any]],
                                   evacuation_data: List[Dict[str, Any]],
                                   initial_conditions: Dict[str, Any]) -> Dict[str, MetricResult]:
        """运行综合评估"""
        results = {}
        
        try:
            # 韧性指标
            results['survival_resilience'] = self.resilience_metrics.calculate_survival_resilience(
                agents, initial_conditions.get('initial_population', len(agents))
            )
            
            results['resource_resilience'] = self.resilience_metrics.calculate_resource_resilience(
                agents, initial_conditions.get('initial_resources', {})
            )
            
            results['adaptive_resilience'] = self.resilience_metrics.calculate_adaptive_resilience(
                agents, initial_conditions.get('time_series_data', [])
            )
            
            results['recovery_resilience'] = self.resilience_metrics.calculate_recovery_resilience(
                agents, initial_conditions.get('disaster_start_time', datetime.now())
            )
            
            # 网络指标
            results['network_structure'] = self.network_metrics.calculate_network_structure_metrics(network)
            results['centrality_metrics'] = self.network_metrics.calculate_centrality_metrics(network)
            
            node_attributes = {agent_id: agent_data for agent_id, agent_data in agents.items()}
            results['differential_network'] = self.network_metrics.calculate_differential_network_metrics(
                network, node_attributes
            )
            
            # 资源指标
            results['resource_distribution'] = self.resource_metrics.calculate_resource_distribution_metrics(agents)
            results['resource_efficiency'] = self.resource_metrics.calculate_resource_efficiency_metrics(
                agents, resource_transactions
            )
            results['differential_resource'] = self.resource_metrics.calculate_differential_resource_metrics(
                agents, resource_transactions
            )
            
            # 空间指标
            results['mobility_metrics'] = self.spatial_metrics.calculate_mobility_metrics(agents, movement_history)
            results['clustering_metrics'] = self.spatial_metrics.calculate_clustering_metrics(agents)
            results['evacuation_efficiency'] = self.spatial_metrics.calculate_evacuation_efficiency_metrics(
                agents, evacuation_data
            )
            
        except Exception as e:
            logger.error(f"综合评估过程中出现错误: {e}")
        
        # 记录评估历史
        evaluation_record = {
            'timestamp': datetime.now(),
            'results': results,
            'agent_count': len(agents),
            'network_size': network.number_of_nodes(),
            'evaluation_id': f"eval_{len(self.evaluation_history)}"
        }
        self.evaluation_history.append(evaluation_record)
        
        return results
    
    def calculate_overall_performance_score(self, results: Dict[str, MetricResult]) -> Dict[str, float]:
        """计算综合性能分数"""
        category_scores = defaultdict(list)
        
        # 按类别收集分数
        for metric_name, result in results.items():
            category = result.category.value
            
            # 提取数值分数
            if isinstance(result.value, dict):
                # 对于复合指标，取主要分数或平均值
                if 'overall_resilience' in result.value:
                    score = result.value['overall_resilience']
                elif 'overall_efficiency' in result.value:
                    score = result.value['overall_efficiency']
                elif 'overall_recovery' in result.value:
                    score = result.value['overall_recovery']
                elif 'overall_adaptiveness' in result.value:
                    score = result.value['overall_adaptiveness']
                else:
                    # 计算数值字段的平均值
                    numeric_values = [v for v in result.value.values() 
                                    if isinstance(v, (int, float)) and not np.isnan(v)]
                    score = np.mean(numeric_values) if numeric_values else 0
            elif isinstance(result.value, (int, float)):
                score = result.value
            else:
                score = 0
            
            category_scores[category].append(max(0, min(1, score)))  # 限制在[0,1]范围内
        
        # 计算各类别平均分
        category_averages = {}
        for category, scores in category_scores.items():
            category_averages[category] = np.mean(scores) if scores else 0
        
        # 计算加权总分
        weights = {
            'resilience': 0.3,
            'network': 0.25,
            'resource': 0.25,
            'spatial': 0.15,
            'differential': 0.05
        }
        
        overall_score = sum(category_averages.get(category, 0) * weight 
                          for category, weight in weights.items())
        
        return {
            'overall_score': overall_score,
            **category_averages
        }
    
    def generate_evaluation_report(self, results: Dict[str, MetricResult], 
                                 output_path: Optional[str] = None) -> str:
        """生成评估报告"""
        report_lines = []
        report_lines.append("# 城市智能体系统评估报告")
        report_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"评估指标数量: {len(results)}")
        
        # 综合性能分数
        performance_scores = self.calculate_overall_performance_score(results)
        report_lines.append("\n## 综合性能分数")
        for category, score in performance_scores.items():
            report_lines.append(f"- {category}: {score:.3f}")
        
        # 按类别组织结果
        categories = defaultdict(list)
        for metric_name, result in results.items():
            categories[result.category.value].append((metric_name, result))
        
        # 详细指标报告
        for category, metrics in categories.items():
            report_lines.append(f"\n## {category.upper()}指标")
            
            for metric_name, result in metrics:
                report_lines.append(f"\n### {metric_name}")
                report_lines.append(f"- 描述: {result.description}")
                report_lines.append(f"- 层级: {result.level.value}")
                report_lines.append(f"- 时间: {result.timestamp.strftime('%H:%M:%S')}")
                
                if isinstance(result.value, dict):
                    report_lines.append("- 详细数值:")
                    for key, value in result.value.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  - {key}: {value:.4f}")
                        else:
                            report_lines.append(f"  - {key}: {value}")
                else:
                    report_lines.append(f"- 数值: {result.value}")
        
        # 关键发现和建议
        report_lines.append("\n## 关键发现")
        key_findings = self._extract_key_findings(results)
        for finding in key_findings:
            report_lines.append(f"- {finding}")
        
        report_lines.append("\n## 改进建议")
        recommendations = self._generate_recommendations(results)
        for recommendation in recommendations:
            report_lines.append(f"- {recommendation}")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"评估报告已保存到: {output_path}")
        
        return report_content
    
    def _extract_key_findings(self, results: Dict[str, MetricResult]) -> List[str]:
        """提取关键发现"""
        findings = []
        
        # 分析韧性表现
        resilience_results = [r for r in results.values() if r.category == MetricCategory.RESILIENCE]
        if resilience_results:
            avg_resilience = np.mean([self._extract_main_score(r) for r in resilience_results])
            if avg_resilience > 0.8:
                findings.append("系统展现出高韧性，能够有效应对灾害冲击")
            elif avg_resilience < 0.4:
                findings.append("系统韧性较低，需要加强抗灾能力建设")
        
        # 分析网络特征
        if 'differential_network' in results:
            diff_result = results['differential_network']
            if isinstance(diff_result.value, dict):
                diff_index = diff_result.value.get('differential_index', 0)
                if diff_index > 0.6:
                    findings.append("社会网络呈现明显的差序格局特征，亲密关系占主导")
                elif diff_index < 0.3:
                    findings.append("社会网络相对平等，差序格局特征不明显")
        
        # 分析资源分配
        if 'resource_distribution' in results:
            resource_result = results['resource_distribution']
            if isinstance(resource_result.value, dict):
                gini = resource_result.value.get('gini_coefficient', 0)
                if gini > 0.5:
                    findings.append("资源分配不平等程度较高")
                elif gini < 0.3:
                    findings.append("资源分配相对均匀")
        
        return findings
    
    def _generate_recommendations(self, results: Dict[str, MetricResult]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于韧性指标的建议
        if 'survival_resilience' in results:
            survival_result = results['survival_resilience']
            if isinstance(survival_result.value, dict):
                survival_rate = survival_result.value.get('survival_rate', 0)
                if survival_rate < 0.7:
                    recommendations.append("建议加强应急预案和生存技能培训")
        
        # 基于网络指标的建议
        if 'network_structure' in results:
            network_result = results['network_structure']
            if isinstance(network_result.value, dict):
                density = network_result.value.get('density', 0)
                if density < 0.1:
                    recommendations.append("建议增强社区连接，提高社会网络密度")
        
        # 基于空间指标的建议
        if 'evacuation_efficiency' in results:
            evac_result = results['evacuation_efficiency']
            if isinstance(evac_result.value, dict):
                success_rate = evac_result.value.get('success_rate', 0)
                if success_rate < 0.8:
                    recommendations.append("建议优化疏散路径规划和交通管理")
        
        return recommendations
    
    def _extract_main_score(self, result: MetricResult) -> float:
        """提取主要分数"""
        if isinstance(result.value, dict):
            # 寻找主要分数字段
            main_fields = ['overall_resilience', 'overall_efficiency', 'overall_recovery', 'overall_adaptiveness']
            for field in main_fields:
                if field in result.value:
                    return result.value[field]
            
            # 如果没有主要字段，计算数值字段平均值
            numeric_values = [v for v in result.value.values() 
                            if isinstance(v, (int, float)) and not np.isnan(v)]
            return np.mean(numeric_values) if numeric_values else 0
        elif isinstance(result.value, (int, float)):
            return result.value
        else:
            return 0
    
    def export_results_to_json(self, results: Dict[str, MetricResult], 
                              output_path: str) -> None:
        """导出结果到JSON文件"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        for metric_name, result in results.items():
            export_data['metrics'][metric_name] = {
                'category': result.category.value,
                'level': result.level.value,
                'value': result.value,
                'description': result.description,
                'unit': result.unit,
                'confidence': result.confidence,
                'metadata': result.metadata
            }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"评估结果已导出到: {output_path}")


def create_evaluation_system() -> ComprehensiveEvaluationSystem:
    """创建评估系统实例"""
    return ComprehensiveEvaluationSystem()


# 示例用法
if __name__ == "__main__":
    # 创建评估系统
    eval_system = create_evaluation_system()
    
    # 模拟数据
    agents = {
        'agent_1': {
            'status': 'alive',
            'resources': {'food': 10, 'water': 8},
            'location': (0, 0),
            'strategy_type': 'strong_differential'
        },
        'agent_2': {
            'status': 'alive', 
            'resources': {'food': 5, 'water': 12},
            'location': (1, 1),
            'strategy_type': 'weak_differential'
        }
    }
    
    network = nx.Graph()
    network.add_edge('agent_1', 'agent_2', relationship_type='family', strength=0.9)
    
    movement_history = [
        {
            'agent_id': 'agent_1',
            'start_location': (0, 0),
            'end_location': (1, 0),
            'distance': 1.0,
            'duration': 1.0,
            'timestamp': datetime.now()
        }
    ]
    
    resource_transactions = [
        {
            'giver': 'agent_1',
            'receiver': 'agent_2',
            'amount': 2,
            'relationship_type': 'family',
            'success': True
        }
    ]
    
    evacuation_data = [
        {
            'agent_id': 'agent_1',
            'evacuation_time': 30,
            'success': True,
            'actual_distance': 5.0,
            'optimal_distance': 4.5,
            'congestion_level': 0.3
        }
    ]
    
    initial_conditions = {
        'initial_population': 2,
        'initial_resources': {'food': 15, 'water': 20},
        'disaster_start_time': datetime.now() - timedelta(hours=2)
    }
    
    # 运行评估
    results = eval_system.run_comprehensive_evaluation(
        agents, network, movement_history, resource_transactions, 
        evacuation_data, initial_conditions
    )
    
    # 生成报告
    report = eval_system.generate_evaluation_report(results)
    print(report)
    
    # 导出结果
    eval_system.export_results_to_json(results, "evaluation_results.json")
    
    print("\n评估系统测试完成！")