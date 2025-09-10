"""社会关系网络系统模块

基于差序格局理论构建社会关系网络，实现亲疏远近的社交层次和互助机制。
支持不同家庭群体和社会关系层次的建模和分析。

作者: 城市智能体项目组
日期: 2024-12-30
"""

import logging
import random
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json

from .agent_attributes import SocialRelationship, RelationshipType
from .differential_order_strategies import HelpRequest, ResourceType

logger = logging.getLogger(__name__)


class NetworkType(Enum):
    """网络类型枚举"""
    FAMILY = "family"  # 家庭网络
    KINSHIP = "kinship"  # 亲属网络
    FRIENDSHIP = "friendship"  # 友谊网络
    NEIGHBORHOOD = "neighborhood"  # 邻里网络
    WORKPLACE = "workplace"  # 工作网络
    COMMUNITY = "community"  # 社区网络
    PROFESSIONAL = "professional"  # 专业网络


class InteractionType(Enum):
    """互动类型枚举"""
    HELP_REQUEST = "help_request"  # 求助
    HELP_PROVIDE = "help_provide"  # 提供帮助
    INFORMATION_SHARE = "information_share"  # 信息分享
    EMOTIONAL_SUPPORT = "emotional_support"  # 情感支持
    RESOURCE_EXCHANGE = "resource_exchange"  # 资源交换
    SOCIAL_VISIT = "social_visit"  # 社交拜访


@dataclass
class NetworkInteraction:
    """网络互动记录"""
    from_agent: str
    to_agent: str
    interaction_type: InteractionType
    resource_type: Optional[ResourceType] = None
    amount: float = 0.0
    success: bool = True
    timestamp: float = 0.0
    context: str = ""
    reciprocity_expected: bool = False


@dataclass
class NetworkMetrics:
    """网络指标"""
    density: float = 0.0  # 网络密度
    clustering_coefficient: float = 0.0  # 聚类系数
    average_path_length: float = 0.0  # 平均路径长度
    centrality_measures: Dict[str, float] = field(default_factory=dict)  # 中心性指标
    modularity: float = 0.0  # 模块化程度
    small_world_coefficient: float = 0.0  # 小世界系数


class SocialNetworkManager:
    """社会网络管理器"""
    
    def __init__(self):
        self.network = nx.Graph()  # 主网络图
        self.directed_network = nx.DiGraph()  # 有向网络（用于互助关系）
        self.relationships: Dict[Tuple[str, str], SocialRelationship] = {}
        self.interactions: List[NetworkInteraction] = []
        self.network_metrics: NetworkMetrics = NetworkMetrics()
        
        # 差序格局相关
        self.family_clusters: Dict[str, Set[str]] = {}  # 家族群体
        self.family_groups: Dict[str, Set[str]] = {}  # 家庭群体
        self.geographic_clusters: Dict[str, Set[str]] = {}  # 地理群体
        
    def add_agent(self, agent_id: str, attributes: Dict[str, Any] = None):
        """添加智能体到网络"""
        self.network.add_node(agent_id, **(attributes or {}))
        self.directed_network.add_node(agent_id, **(attributes or {}))
        
        # 提取家庭群体信息
        if attributes and 'family_group' in attributes:
            family_group = attributes['family_group']
            if family_group not in self.family_groups:
                self.family_groups[family_group] = set()
            self.family_groups[family_group].add(agent_id)
    
    def add_relationship(self, agent1: str, agent2: str, relationship: SocialRelationship):
        """添加社会关系"""
        # 添加到无向图
        self.network.add_edge(agent1, agent2, 
                            relationship_type=relationship.relationship_type,
                            strength=relationship.strength,
                            trust_level=relationship.trust_level)
        
        # 添加到有向图（双向）
        self.directed_network.add_edge(agent1, agent2, **relationship.__dict__)
        self.directed_network.add_edge(agent2, agent1, **relationship.__dict__)
        
        # 存储关系
        key = tuple(sorted([agent1, agent2]))
        self.relationships[key] = relationship
        
        # 更新家族群体
        if relationship.relationship_type == "family":
            self._update_family_clusters(agent1, agent2)
    
    def _update_family_clusters(self, agent1: str, agent2: str):
        """更新家族群体"""
        # 查找现有家族群体
        cluster1 = None
        cluster2 = None
        
        for family_id, members in self.family_clusters.items():
            if agent1 in members:
                cluster1 = family_id
            if agent2 in members:
                cluster2 = family_id
        
        if cluster1 and cluster2 and cluster1 != cluster2:
            # 合并两个家族群体
            self.family_clusters[cluster1].update(self.family_clusters[cluster2])
            del self.family_clusters[cluster2]
        elif cluster1:
            # 添加到现有群体
            self.family_clusters[cluster1].add(agent2)
        elif cluster2:
            # 添加到现有群体
            self.family_clusters[cluster2].add(agent1)
        else:
            # 创建新的家族群体
            family_id = f"family_{len(self.family_clusters)}"
            self.family_clusters[family_id] = {agent1, agent2}
    
    def get_relationship(self, agent1: str, agent2: str) -> Optional[SocialRelationship]:
        """获取两个智能体之间的关系"""
        key = tuple(sorted([agent1, agent2]))
        return self.relationships.get(key)
    
    def get_neighbors(self, agent_id: str, relationship_types: List[str] = None) -> List[str]:
        """获取邻居节点"""
        if agent_id not in self.network:
            return []
        
        neighbors = []
        for neighbor in self.network.neighbors(agent_id):
            if relationship_types:
                edge_data = self.network.get_edge_data(agent_id, neighbor)
                if edge_data and edge_data.get('relationship_type') in relationship_types:
                    neighbors.append(neighbor)
            else:
                neighbors.append(neighbor)
        
        return neighbors
    
    def get_differential_order_layers(self, agent_id: str) -> Dict[str, List[str]]:
        """获取差序格局的层次结构"""
        if agent_id not in self.network:
            return {}
        
        layers = {
            'inner_circle': [],  # 内圈：家人、密友
            'middle_circle': [],  # 中圈：邻居、同事
            'outer_circle': [],  # 外圈：熟人
            'strangers': []  # 陌生人
        }
        
        for neighbor in self.network.neighbors(agent_id):
            edge_data = self.network.get_edge_data(agent_id, neighbor)
            if not edge_data:
                continue
            
            rel_type = edge_data.get('relationship_type', '')
            strength = edge_data.get('strength', 0.0)
            
            if rel_type in ['family', 'close_friend'] or strength > 0.8:
                layers['inner_circle'].append(neighbor)
            elif rel_type in ['neighbor', 'colleague'] or strength > 0.5:
                layers['middle_circle'].append(neighbor)
            elif strength > 0.2:
                layers['outer_circle'].append(neighbor)
            else:
                layers['strangers'].append(neighbor)
        
        return layers
    
    def find_help_candidates(self, requester_id: str, resource_type: ResourceType, 
                           amount: float, urgency: int) -> List[Tuple[str, float]]:
        """基于差序格局原则寻找帮助候选人"""
        if requester_id not in self.network:
            return []
        
        candidates = []
        layers = self.get_differential_order_layers(requester_id)
        
        # 按差序格局层次搜索
        for layer_name, layer_members in layers.items():
            layer_priority = {
                'inner_circle': 1.0,
                'middle_circle': 0.7,
                'outer_circle': 0.4,
                'strangers': 0.1
            }.get(layer_name, 0.1)
            
            for candidate in layer_members:
                relationship = self.get_relationship(requester_id, candidate)
                if not relationship:
                    continue
                
                # 计算帮助概率
                help_probability = self._calculate_help_probability(
                    relationship, resource_type, amount, urgency, layer_priority
                )
                
                if help_probability > 0.1:  # 最低阈值
                    candidates.append((candidate, help_probability))
        
        # 按帮助概率排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates
    
    def _calculate_help_probability(self, relationship: SocialRelationship, 
                                  resource_type: ResourceType, amount: float, 
                                  urgency: int, layer_priority: float) -> float:
        """计算帮助概率"""
        # 基础概率（基于关系强度）
        base_prob = relationship.strength * relationship.trust_level
        
        # 层次调节
        layer_adjusted_prob = base_prob * layer_priority
        
        # 紧急程度调节
        urgency_factor = min(1.0, urgency / 10.0)
        
        # 互助历史调节
        mutual_support_factor = relationship.mutual_support
        
        # 地理距离调节
        distance_factor = max(0.1, 1.0 - relationship.geographic_distance / 50.0)
        
        # 综合概率
        final_prob = (layer_adjusted_prob * 0.4 + 
                     urgency_factor * 0.2 + 
                     mutual_support_factor * 0.2 + 
                     distance_factor * 0.2)
        
        return min(1.0, final_prob)
    
    def record_interaction(self, interaction: NetworkInteraction):
        """记录网络互动"""
        self.interactions.append(interaction)
        
        # 更新关系强度
        relationship = self.get_relationship(interaction.from_agent, interaction.to_agent)
        if relationship:
            if interaction.success:
                # 成功互动增强关系
                if interaction.interaction_type == InteractionType.HELP_PROVIDE:
                    relationship.mutual_support = min(1.0, relationship.mutual_support + 0.1)
                    relationship.trust_level = min(1.0, relationship.trust_level + 0.05)
                elif interaction.interaction_type == InteractionType.INFORMATION_SHARE:
                    relationship.contact_frequency = min(1.0, relationship.contact_frequency + 0.05)
            else:
                # 失败互动削弱关系
                relationship.trust_level = max(0.0, relationship.trust_level - 0.1)
    
    def simulate_help_network_activation(self, disaster_event: Dict[str, Any]) -> Dict[str, Any]:
        """模拟灾害时的互助网络激活"""
        affected_agents = disaster_event.get('affected_agents', [])
        severity = disaster_event.get('severity', 0.5)
        
        help_requests = []
        help_responses = []
        network_activation = defaultdict(int)
        
        # 生成求助请求
        for agent_id in affected_agents:
            if agent_id not in self.network:
                continue
            
            # 基于灾害严重程度决定求助需求
            if random.random() < severity:
                resource_needs = self._generate_resource_needs(agent_id, severity)
                
                for resource_type, amount in resource_needs.items():
                    # 寻找帮助候选人
                    candidates = self.find_help_candidates(
                        agent_id, resource_type, amount, 
                        int(severity * 10)
                    )
                    
                    help_request = {
                        'requester': agent_id,
                        'resource_type': resource_type,
                        'amount': amount,
                        'candidates': candidates[:5]  # 前5个候选人
                    }
                    help_requests.append(help_request)
                    
                    # 模拟帮助响应
                    for candidate, prob in candidates[:3]:  # 尝试前3个候选人
                        if random.random() < prob:
                            help_amount = min(amount, random.uniform(0.3, 1.0) * amount)
                            
                            help_response = {
                                'helper': candidate,
                                'requester': agent_id,
                                'resource_type': resource_type,
                                'amount_provided': help_amount,
                                'success': True
                            }
                            help_responses.append(help_response)
                            
                            # 记录网络激活
                            network_activation[candidate] += 1
                            
                            # 记录互动
                            interaction = NetworkInteraction(
                                from_agent=candidate,
                                to_agent=agent_id,
                                interaction_type=InteractionType.HELP_PROVIDE,
                                resource_type=resource_type,
                                amount=help_amount,
                                success=True,
                                timestamp=disaster_event.get('timestamp', 0.0)
                            )
                            self.record_interaction(interaction)
                            
                            break  # 找到帮助就停止
        
        return {
            'help_requests': help_requests,
            'help_responses': help_responses,
            'network_activation': dict(network_activation),
            'activation_rate': len(help_responses) / max(1, len(help_requests)),
            'most_active_helpers': sorted(network_activation.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
        }
    
    def _generate_resource_needs(self, agent_id: str, severity: float) -> Dict[ResourceType, float]:
        """生成资源需求"""
        needs = {}
        
        # 基于灾害严重程度生成不同资源需求
        if severity > 0.8:
            needs[ResourceType.SHELTER] = random.uniform(50, 100)
            needs[ResourceType.FOOD] = random.uniform(30, 60)
            needs[ResourceType.MEDICAL] = random.uniform(20, 50)
        elif severity > 0.5:
            needs[ResourceType.FOOD] = random.uniform(20, 40)
            needs[ResourceType.MONEY] = random.uniform(100, 500)
        else:
            needs[ResourceType.INFORMATION] = random.uniform(10, 20)
        
        return needs
    
    def calculate_network_metrics(self) -> NetworkMetrics:
        """计算网络指标"""
        if len(self.network) == 0:
            return NetworkMetrics()
        
        metrics = NetworkMetrics()
        
        # 网络密度
        metrics.density = nx.density(self.network)
        
        # 聚类系数
        metrics.clustering_coefficient = nx.average_clustering(self.network)
        
        # 平均路径长度
        if nx.is_connected(self.network):
            metrics.average_path_length = nx.average_shortest_path_length(self.network)
        else:
            # 对于非连通图，计算最大连通分量的平均路径长度
            largest_cc = max(nx.connected_components(self.network), key=len)
            subgraph = self.network.subgraph(largest_cc)
            metrics.average_path_length = nx.average_shortest_path_length(subgraph)
        
        # 中心性指标
        metrics.centrality_measures = {
            'degree': nx.degree_centrality(self.network),
            'betweenness': nx.betweenness_centrality(self.network),
            'closeness': nx.closeness_centrality(self.network),
            'eigenvector': nx.eigenvector_centrality(self.network, max_iter=1000)
        }
        
        # 模块化程度（使用社区检测）
        try:
            communities = nx.community.greedy_modularity_communities(self.network)
            metrics.modularity = nx.community.modularity(self.network, communities)
        except:
            metrics.modularity = 0.0
        
        # 小世界系数
        # 小世界系数 = (C/C_random) / (L/L_random)
        # 其中C是聚类系数，L是平均路径长度
        try:
            random_graph = nx.erdos_renyi_graph(len(self.network), metrics.density)
            random_clustering = nx.average_clustering(random_graph)
            random_path_length = nx.average_shortest_path_length(random_graph)
            
            if random_clustering > 0 and random_path_length > 0:
                clustering_ratio = metrics.clustering_coefficient / random_clustering
                path_length_ratio = metrics.average_path_length / random_path_length
                metrics.small_world_coefficient = clustering_ratio / path_length_ratio
        except:
            metrics.small_world_coefficient = 0.0
        
        self.network_metrics = metrics
        return metrics
    
    def analyze_differential_order_patterns(self) -> Dict[str, Any]:
        """分析差序格局模式"""
        analysis = {
            'family_homophily': {},  # 家庭群体同质性
            'family_cluster_sizes': [],  # 家族群体规模
            'cross_group_connections': 0,  # 跨群体连接
            'inner_circle_density': {},  # 内圈密度
            'help_flow_patterns': {}  # 互助流动模式
        }
        
        # 分析家庭群体同质性
        for family_group, members in self.family_groups.items():
            if len(members) < 2:
                continue
            
            internal_connections = 0
            total_possible = len(members) * (len(members) - 1) // 2
            
            for i, agent1 in enumerate(members):
                for agent2 in list(members)[i+1:]:
                    if self.network.has_edge(agent1, agent2):
                        internal_connections += 1
            
            analysis['family_homophily'][family_group] = {
                'size': len(members),
                'internal_density': internal_connections / max(1, total_possible),
                'members': list(members)
            }
        
        # 分析家族群体规模
        analysis['family_cluster_sizes'] = [len(cluster) for cluster in self.family_clusters.values()]
        
        # 分析跨群体连接
        cross_connections = 0
        total_connections = 0
        
        for edge in self.network.edges():
            agent1, agent2 = edge
            total_connections += 1
            
            # 检查是否为跨家庭群体连接
            family_group1 = self._get_agent_family_group(agent1)
            family_group2 = self._get_agent_family_group(agent2)
            
            if family_group1 and family_group2 and family_group1 != family_group2:
                cross_connections += 1
        
        analysis['cross_group_connections'] = cross_connections / max(1, total_connections)
        
        # 分析内圈密度
        for agent_id in self.network.nodes():
            layers = self.get_differential_order_layers(agent_id)
            inner_circle = layers['inner_circle']
            
            if len(inner_circle) > 1:
                inner_connections = 0
                total_inner_possible = len(inner_circle) * (len(inner_circle) - 1) // 2
                
                for i, agent1 in enumerate(inner_circle):
                    for agent2 in inner_circle[i+1:]:
                        if self.network.has_edge(agent1, agent2):
                            inner_connections += 1
                
                analysis['inner_circle_density'][agent_id] = {
                    'size': len(inner_circle),
                    'density': inner_connections / max(1, total_inner_possible)
                }
        
        # 分析互助流动模式
        help_interactions = [i for i in self.interactions 
                           if i.interaction_type == InteractionType.HELP_PROVIDE]
        
        if help_interactions:
            # 按关系类型统计互助
            help_by_relationship = defaultdict(list)
            
            for interaction in help_interactions:
                relationship = self.get_relationship(interaction.from_agent, interaction.to_agent)
                if relationship:
                    help_by_relationship[relationship.relationship_type].append(interaction.amount)
            
            for rel_type, amounts in help_by_relationship.items():
                analysis['help_flow_patterns'][rel_type] = {
                    'count': len(amounts),
                    'total_amount': sum(amounts),
                    'average_amount': sum(amounts) / len(amounts),
                    'max_amount': max(amounts)
                }
        
        return analysis
    
    def _get_agent_family_group(self, agent_id: str) -> Optional[str]:
        """获取智能体家庭群体"""
        node_data = self.network.nodes.get(agent_id, {})
        return node_data.get('family_group')
    
    def export_network_data(self, filepath: str):
        """导出网络数据"""
        data = {
            'nodes': [
                {'id': node, **self.network.nodes[node]} 
                for node in self.network.nodes()
            ],
            'edges': [
                {
                    'source': edge[0], 
                    'target': edge[1], 
                    **self.network.edges[edge]
                } 
                for edge in self.network.edges()
            ],
            'interactions': [
                {
                    'from_agent': i.from_agent,
                    'to_agent': i.to_agent,
                    'interaction_type': i.interaction_type.value,
                    'resource_type': i.resource_type.value if i.resource_type else None,
                    'amount': i.amount,
                    'success': i.success,
                    'timestamp': i.timestamp
                }
                for i in self.interactions
            ],
            'metrics': {
                'density': self.network_metrics.density,
                'clustering_coefficient': self.network_metrics.clustering_coefficient,
                'average_path_length': self.network_metrics.average_path_length,
                'modularity': self.network_metrics.modularity
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def generate_sample_network(self, num_agents: int = 100) -> 'SocialNetworkManager':
        """生成示例网络"""
        # 生成智能体
        family_groups = ['GroupA', 'GroupB', 'GroupC', 'GroupD', 'GroupE', 'GroupF', 'GroupG', 'GroupH', 'GroupI', 'GroupJ']
        
        for i in range(num_agents):
            agent_id = f"agent_{i:03d}"
            family_group = random.choice(family_groups)
            
            self.add_agent(agent_id, {
                'family_group': family_group,
                'age': random.randint(18, 80),
                'location': (random.uniform(39.8, 40.0), random.uniform(116.3, 116.5))
            })
        
        # 生成关系
        agents = list(self.network.nodes())
        
        # 家庭关系（同群体高概率）
        for agent in agents:
            agent_group = self.network.nodes[agent]['family_group']
            same_group_agents = [a for a in agents 
                                 if self.network.nodes[a]['family_group'] == agent_group and a != agent]
            
            # 每个智能体有1-3个家庭成员
            num_family = random.randint(1, min(3, len(same_group_agents)))
            family_members = random.sample(same_group_agents, num_family)
            
            for family_member in family_members:
                if not self.network.has_edge(agent, family_member):
                    relationship = SocialRelationship(
                        target_id=family_member,
                        relationship_type="family",
                        strength=random.uniform(0.7, 1.0),
                        trust_level=random.uniform(0.8, 1.0),
                        contact_frequency=random.uniform(0.6, 1.0),
                        mutual_support=random.uniform(0.7, 1.0),
                        geographic_distance=random.uniform(0.1, 5.0),
                        relationship_duration=random.randint(5, 30)
                    )
                    self.add_relationship(agent, family_member, relationship)
        
        # 邻居关系（地理位置相近）
        for agent in agents:
            agent_loc = self.network.nodes[agent]['location']
            
            # 寻找地理位置相近的智能体
            nearby_agents = []
            for other_agent in agents:
                if other_agent == agent:
                    continue
                
                other_loc = self.network.nodes[other_agent]['location']
                distance = ((agent_loc[0] - other_loc[0])**2 + (agent_loc[1] - other_loc[1])**2)**0.5
                
                if distance < 0.01:  # 相近阈值
                    nearby_agents.append((other_agent, distance))
            
            # 选择2-5个邻居
            nearby_agents.sort(key=lambda x: x[1])
            num_neighbors = random.randint(2, min(5, len(nearby_agents)))
            
            for neighbor, distance in nearby_agents[:num_neighbors]:
                if not self.network.has_edge(agent, neighbor):
                    relationship = SocialRelationship(
                        target_id=neighbor,
                        relationship_type="neighbor",
                        strength=random.uniform(0.3, 0.7),
                        trust_level=random.uniform(0.4, 0.8),
                        contact_frequency=random.uniform(0.3, 0.7),
                        mutual_support=random.uniform(0.2, 0.6),
                        geographic_distance=distance * 100,  # 转换为实际距离
                        relationship_duration=random.randint(1, 10)
                    )
                    self.add_relationship(agent, neighbor, relationship)
        
        # 随机友谊关系
        for agent in agents:
            num_friends = random.randint(3, 8)
            potential_friends = [a for a in agents if a != agent and not self.network.has_edge(agent, a)]
            
            if len(potential_friends) >= num_friends:
                friends = random.sample(potential_friends, num_friends)
                
                for friend in friends:
                    relationship = SocialRelationship(
                        target_id=friend,
                        relationship_type="friend",
                        strength=random.uniform(0.4, 0.9),
                        trust_level=random.uniform(0.5, 0.9),
                        contact_frequency=random.uniform(0.2, 0.8),
                        mutual_support=random.uniform(0.3, 0.8),
                        geographic_distance=random.uniform(1.0, 20.0),
                        relationship_duration=random.randint(1, 15)
                    )
                    self.add_relationship(agent, friend, relationship)
        
        return self


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建网络管理器
    network_manager = SocialNetworkManager()
    
    # 生成示例网络
    network_manager.generate_sample_network(50)
    
    print("=== 社会网络测试 ===")
    print(f"网络规模: {len(network_manager.network.nodes())} 个节点")
    print(f"关系数量: {len(network_manager.network.edges())} 条边")
    
    # 计算网络指标
    metrics = network_manager.calculate_network_metrics()
    print(f"\n=== 网络指标 ===")
    print(f"网络密度: {metrics.density:.3f}")
    print(f"聚类系数: {metrics.clustering_coefficient:.3f}")
    print(f"平均路径长度: {metrics.average_path_length:.3f}")
    print(f"小世界系数: {metrics.small_world_coefficient:.3f}")
    
    # 分析差序格局模式
    patterns = network_manager.analyze_differential_order_patterns()
    print(f"\n=== 差序格局分析 ===")
    print(f"跨群体连接比例: {patterns['cross_group_connections']:.3f}")
    print(f"家族群体数量: {len(patterns['family_cluster_sizes'])}")
    print(f"平均家族规模: {np.mean(patterns['family_cluster_sizes']):.1f}")
    
    # 模拟灾害互助
    disaster_event = {
        'affected_agents': random.sample(list(network_manager.network.nodes()), 20),
        'severity': 0.7,
        'timestamp': 1640995200.0
    }
    
    help_simulation = network_manager.simulate_help_network_activation(disaster_event)
    print(f"\n=== 互助网络模拟 ===")
    print(f"求助请求数: {len(help_simulation['help_requests'])}")
    print(f"帮助响应数: {len(help_simulation['help_responses'])}")
    print(f"网络激活率: {help_simulation['activation_rate']:.3f}")
    print(f"最活跃帮助者: {help_simulation['most_active_helpers'][:3]}")