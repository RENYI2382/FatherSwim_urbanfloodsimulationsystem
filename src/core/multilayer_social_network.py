#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层社会网络构建器
基于差序格局理论的多重关系网络建模

核心功能：
1. 血缘网络层：基于家族谱系的稳定关系
2. 地缘网络层：基于地理距离的邻里关系
3. 业缘网络层：基于职业合作的工作关系
4. 学缘网络层：基于教育经历的同学关系
5. 社缘网络层：基于兴趣爱好的社交关系
6. 圈子动态演化：网络结构的时间演变
7. 差序格局计算：综合关系强度和影响力
"""

import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from collections import defaultdict, Counter
from scipy.spatial.distance import euclidean
from scipy.stats import powerlaw, norm


class NetworkLayerType(Enum):
    """网络层类型"""
    KINSHIP = "血缘网络"      # 家族关系网络
    GEOGRAPHIC = "地缘网络"   # 地理邻里网络
    PROFESSIONAL = "业缘网络" # 职业工作网络
    EDUCATIONAL = "学缘网络"  # 教育同学网络
    SOCIAL = "社缘网络"       # 社交兴趣网络


class RelationshipStrength(Enum):
    """关系强度等级"""
    VERY_WEAK = 0.1    # 很弱
    WEAK = 0.3         # 弱
    MODERATE = 0.5     # 中等
    STRONG = 0.7       # 强
    VERY_STRONG = 0.9  # 很强


@dataclass
class AgentProfile:
    """智能体档案信息"""
    agent_id: int
    age: int
    gender: str
    education_level: str
    occupation: str
    income_level: str
    location: Tuple[float, float]
    family_size: int
    social_activity_level: float  # 社交活跃度 [0, 1]
    cultural_background: str
    personality_traits: Dict[str, float] = field(default_factory=dict)
    life_stage: str = "adult"  # child, youth, adult, elderly


@dataclass
class RelationshipEdge:
    """关系边详细信息"""
    source_id: int
    target_id: int
    layer_type: NetworkLayerType
    strength: float  # 关系强度 [0, 1]
    intimacy: float  # 亲密程度 [0, 1]
    trust: float     # 信任程度 [0, 1]
    reciprocity: float  # 互惠程度 [-1, 1]
    interaction_frequency: float  # 交互频率 [0, 1]
    relationship_duration: int    # 关系持续时间（天）
    last_interaction: int        # 最后交互时间
    relationship_quality: float  # 关系质量 [0, 1]
    influence_asymmetry: float   # 影响力不对称性 [-1, 1]
    
    # 特定层的属性
    kinship_degree: Optional[int] = None      # 血缘关系度数（几代内亲属）
    geographic_distance: Optional[float] = None  # 地理距离
    professional_hierarchy: Optional[int] = None # 职业层级差异
    educational_similarity: Optional[float] = None # 教育背景相似度
    interest_overlap: Optional[float] = None     # 兴趣重叠度


@dataclass
class SocialCircle:
    """社会圈子结构"""
    circle_id: str
    circle_name: str
    circle_type: str  # family, extended_family, neighborhood, workplace, school, hobby
    leader_id: Optional[int]
    core_members: Set[int]      # 核心成员
    peripheral_members: Set[int] # 边缘成员
    cohesion_level: float       # 凝聚力 [0, 1]
    influence_radius: float     # 影响半径
    activity_level: float       # 活跃程度 [0, 1]
    resource_sharing_level: float # 资源共享水平 [0, 1]
    decision_making_style: str  # democratic, hierarchical, consensus
    collective_memory: Dict     # 集体记忆和经验
    formation_time: int         # 圈子形成时间
    stability_score: float      # 稳定性评分 [0, 1]


class MultilayerSocialNetwork:
    """多层社会网络构建器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # 多层网络图
        self.layers = {
            NetworkLayerType.KINSHIP: nx.Graph(),
            NetworkLayerType.GEOGRAPHIC: nx.Graph(),
            NetworkLayerType.PROFESSIONAL: nx.Graph(),
            NetworkLayerType.EDUCATIONAL: nx.Graph(),
            NetworkLayerType.SOCIAL: nx.Graph()
        }
        
        # 综合网络（所有层的叠加）
        self.composite_network = nx.Graph()
        
        # 智能体档案
        self.agent_profiles: Dict[int, AgentProfile] = {}
        
        # 关系边详细信息
        self.relationship_edges: Dict[Tuple[int, int, NetworkLayerType], RelationshipEdge] = {}
        
        # 社会圈子
        self.social_circles: Dict[str, SocialCircle] = {}
        self.agent_circles: Dict[int, Set[str]] = defaultdict(set)
        
        # 网络统计信息
        self.network_stats = {
            'total_agents': 0,
            'total_relationships': 0,
            'layer_densities': {},
            'clustering_coefficients': {},
            'average_path_lengths': {},
            'degree_distributions': {}
        }
        
        # 差序格局参数
        self.differential_weights = {
            NetworkLayerType.KINSHIP: 0.4,      # 血缘关系权重最高
            NetworkLayerType.GEOGRAPHIC: 0.25,   # 地缘关系
            NetworkLayerType.PROFESSIONAL: 0.2,  # 业缘关系
            NetworkLayerType.EDUCATIONAL: 0.1,   # 学缘关系
            NetworkLayerType.SOCIAL: 0.05        # 社缘关系权重最低
        }
    
    def _default_config(self) -> Dict:
        """默认配置参数"""
        return {
            'kinship_network': {
                'max_family_size': 8,
                'extended_family_probability': 0.3,
                'kinship_strength_decay': 0.7,  # 每代关系强度衰减
                'clan_formation_probability': 0.1
            },
            'geographic_network': {
                'neighborhood_radius': 5.0,
                'distance_decay_factor': 0.1,
                'community_formation_probability': 0.4,
                'mobility_rate': 0.05  # 每时间步的迁移概率
            },
            'professional_network': {
                'workplace_size_distribution': 'lognormal',
                'hierarchy_levels': 5,
                'cross_department_probability': 0.2,
                'professional_mobility': 0.02
            },
            'educational_network': {
                'class_size_range': (20, 40),
                'alumni_connection_probability': 0.1,
                'education_homophily': 0.6
            },
            'social_network': {
                'interest_categories': 10,
                'social_activity_frequency': 0.3,
                'friendship_formation_rate': 0.05,
                'social_influence_radius': 3.0
            },
            'circle_formation': {
                'min_circle_size': 3,
                'max_circle_size': 15,
                'leadership_emergence_probability': 0.3,
                'circle_stability_threshold': 0.6
            }
        }
    
    def build_network(self, agent_profiles: List[AgentProfile]) -> None:
        """构建完整的多层社会网络"""
        print(f"开始构建多层社会网络，智能体数量: {len(agent_profiles)}")
        
        # 存储智能体档案
        for profile in agent_profiles:
            self.agent_profiles[profile.agent_id] = profile
            # 在所有层中添加节点
            for layer in self.layers.values():
                layer.add_node(profile.agent_id, **profile.__dict__)
            self.composite_network.add_node(profile.agent_id, **profile.__dict__)
        
        self.network_stats['total_agents'] = len(agent_profiles)
        
        # 逐层构建网络
        print("构建血缘网络层...")
        self._build_kinship_layer()
        
        print("构建地缘网络层...")
        self._build_geographic_layer()
        
        print("构建业缘网络层...")
        self._build_professional_layer()
        
        print("构建学缘网络层...")
        self._build_educational_layer()
        
        print("构建社缘网络层...")
        self._build_social_layer()
        
        # 构建综合网络
        print("整合多层网络...")
        self._build_composite_network()
        
        # 识别和构建社会圈子
        print("识别社会圈子...")
        self._identify_social_circles()
        
        # 计算网络统计
        print("计算网络统计信息...")
        self._calculate_network_statistics()
        
        print("多层社会网络构建完成！")
    
    def _build_kinship_layer(self) -> None:
        """构建血缘网络层"""
        kinship_config = self.config['kinship_network']
        agents = list(self.agent_profiles.values())
        
        # 按家庭分组
        families = self._group_agents_by_family(agents)
        
        for family_id, family_members in families.items():
            # 构建核心家庭网络（全连接）
            for i, agent1 in enumerate(family_members):
                for agent2 in family_members[i+1:]:
                    strength = self._calculate_kinship_strength(agent1, agent2)
                    self._add_relationship_edge(
                        agent1.agent_id, agent2.agent_id,
                        NetworkLayerType.KINSHIP, strength,
                        kinship_degree=1  # 直系亲属
                    )
        
        # 构建扩展家族网络
        self._build_extended_kinship_network(families, kinship_config)
    
    def _build_geographic_layer(self) -> None:
        """构建地缘网络层"""
        geo_config = self.config['geographic_network']
        agents = list(self.agent_profiles.values())
        
        # 基于地理距离构建邻里关系
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                distance = euclidean(agent1.location, agent2.location)
                
                if distance <= geo_config['neighborhood_radius']:
                    # 距离越近，关系越强
                    strength = math.exp(-distance * geo_config['distance_decay_factor'])
                    
                    # 考虑社会经济地位相似性
                    similarity_bonus = self._calculate_socioeconomic_similarity(agent1, agent2) * 0.2
                    final_strength = min(1.0, strength + similarity_bonus)
                    
                    if final_strength > 0.1:  # 只保留有意义的关系
                        self._add_relationship_edge(
                            agent1.agent_id, agent2.agent_id,
                            NetworkLayerType.GEOGRAPHIC, final_strength,
                            geographic_distance=distance
                        )
    
    def _build_professional_layer(self) -> None:
        """构建业缘网络层"""
        prof_config = self.config['professional_network']
        agents = list(self.agent_profiles.values())
        
        # 按职业和工作场所分组
        workplaces = self._group_agents_by_workplace(agents)
        
        for workplace_id, colleagues in workplaces.items():
            # 同事关系网络
            for i, agent1 in enumerate(colleagues):
                for agent2 in colleagues[i+1:]:
                    # 基于职业层级和部门相似性计算关系强度
                    strength = self._calculate_professional_strength(agent1, agent2)
                    
                    if strength > 0.1:
                        hierarchy_diff = abs(
                            self._get_hierarchy_level(agent1.occupation) - 
                            self._get_hierarchy_level(agent2.occupation)
                        )
                        
                        self._add_relationship_edge(
                            agent1.agent_id, agent2.agent_id,
                            NetworkLayerType.PROFESSIONAL, strength,
                            professional_hierarchy=hierarchy_diff
                        )
        
        # 跨工作场所的职业网络（行业协会、专业组织等）
        self._build_cross_workplace_networks(agents, prof_config)
    
    def _build_educational_layer(self) -> None:
        """构建学缘网络层"""
        edu_config = self.config['educational_network']
        agents = list(self.agent_profiles.values())
        
        # 按教育背景分组
        education_groups = self._group_agents_by_education(agents)
        
        for edu_level, classmates in education_groups.items():
            # 同学关系网络
            for i, agent1 in enumerate(classmates):
                for agent2 in classmates[i+1:]:
                    # 基于年龄相近程度和教育相似性
                    age_similarity = 1.0 - abs(agent1.age - agent2.age) / 50.0
                    edu_similarity = self._calculate_education_similarity(agent1, agent2)
                    
                    strength = (age_similarity * 0.4 + edu_similarity * 0.6) * \
                               edu_config['alumni_connection_probability']
                    
                    if strength > 0.05:
                        self._add_relationship_edge(
                            agent1.agent_id, agent2.agent_id,
                            NetworkLayerType.EDUCATIONAL, strength,
                            educational_similarity=edu_similarity
                        )
    
    def _build_social_layer(self) -> None:
        """构建社缘网络层"""
        social_config = self.config['social_network']
        agents = list(self.agent_profiles.values())
        
        # 基于兴趣爱好和社交活动构建关系
        interest_groups = self._group_agents_by_interests(agents)
        
        for interest, participants in interest_groups.items():
            # 兴趣小组内的社交关系
            for i, agent1 in enumerate(participants):
                for agent2 in participants[i+1:]:
                    # 基于社交活跃度和兴趣重叠度
                    activity_compatibility = min(agent1.social_activity_level, agent2.social_activity_level)
                    interest_overlap = self._calculate_interest_overlap(agent1, agent2)
                    
                    strength = (activity_compatibility * 0.5 + interest_overlap * 0.5) * \
                               social_config['friendship_formation_rate']
                    
                    if strength > 0.03:
                        self._add_relationship_edge(
                            agent1.agent_id, agent2.agent_id,
                            NetworkLayerType.SOCIAL, strength,
                            interest_overlap=interest_overlap
                        )
    
    def _build_composite_network(self) -> None:
        """构建综合网络"""
        # 整合所有层的关系
        for layer_type, layer_graph in self.layers.items():
            for edge in layer_graph.edges(data=True):
                source, target, data = edge
                
                # 如果综合网络中已存在该边，则累加权重
                if self.composite_network.has_edge(source, target):
                    current_weight = self.composite_network[source][target].get('weight', 0)
                    layer_weight = data.get('weight', 0) * self.differential_weights[layer_type]
                    new_weight = current_weight + layer_weight
                    
                    self.composite_network[source][target]['weight'] = min(1.0, new_weight)
                    self.composite_network[source][target]['layers'].append(layer_type.value)
                else:
                    # 新建边
                    weight = data.get('weight', 0) * self.differential_weights[layer_type]
                    self.composite_network.add_edge(
                        source, target,
                        weight=weight,
                        layers=[layer_type.value]
                    )
    
    def _identify_social_circles(self) -> None:
        """识别社会圈子"""
        circle_config = self.config['circle_formation']
        
        # 基于不同层识别不同类型的圈子
        self._identify_family_circles()
        self._identify_neighborhood_circles()
        self._identify_workplace_circles()
        self._identify_friend_circles()
        
        # 识别跨层的复合圈子
        self._identify_composite_circles()
    
    def _identify_family_circles(self) -> None:
        """识别家庭圈子"""
        kinship_layer = self.layers[NetworkLayerType.KINSHIP]
        
        # 使用社区检测算法识别家族群体
        communities = nx.community.greedy_modularity_communities(kinship_layer)
        
        for i, community in enumerate(communities):
            if len(community) >= 3:  # 最小圈子规模
                circle_id = f"family_circle_{i}"
                
                # 选择圈子领导者（通常是年龄最大或度中心性最高的）
                leader_id = self._select_circle_leader(community, kinship_layer, 'age')
                
                circle = SocialCircle(
                    circle_id=circle_id,
                    circle_name=f"家族圈{i+1}",
                    circle_type="family",
                    leader_id=leader_id,
                    core_members=set(community),
                    peripheral_members=set(),
                    cohesion_level=self._calculate_circle_cohesion(community, kinship_layer),
                    influence_radius=2.0,  # 家庭圈影响力较强但范围有限
                    activity_level=0.8,    # 家庭圈通常很活跃
                    resource_sharing_level=0.9,  # 家庭内资源共享程度高
                    decision_making_style="hierarchical",
                    collective_memory={},
                    formation_time=0,
                    stability_score=0.9     # 家庭圈稳定性很高
                )
                
                self.social_circles[circle_id] = circle
                
                # 更新智能体的圈子归属
                for agent_id in community:
                    self.agent_circles[agent_id].add(circle_id)
    
    def _identify_neighborhood_circles(self) -> None:
        """识别邻里圈子"""
        geo_layer = self.layers[NetworkLayerType.GEOGRAPHIC]
        
        communities = nx.community.greedy_modularity_communities(geo_layer)
        
        for i, community in enumerate(communities):
            if len(community) >= 4:
                circle_id = f"neighborhood_circle_{i}"
                leader_id = self._select_circle_leader(community, geo_layer, 'social_activity')
                
                circle = SocialCircle(
                    circle_id=circle_id,
                    circle_name=f"邻里圈{i+1}",
                    circle_type="neighborhood",
                    leader_id=leader_id,
                    core_members=set(list(community)[:len(community)//2]),
                    peripheral_members=set(list(community)[len(community)//2:]),
                    cohesion_level=self._calculate_circle_cohesion(community, geo_layer),
                    influence_radius=1.5,
                    activity_level=0.6,
                    resource_sharing_level=0.5,
                    decision_making_style="consensus",
                    collective_memory={},
                    formation_time=0,
                    stability_score=0.6
                )
                
                self.social_circles[circle_id] = circle
                
                for agent_id in community:
                    self.agent_circles[agent_id].add(circle_id)
    
    def _identify_workplace_circles(self) -> None:
        """识别工作圈子"""
        prof_layer = self.layers[NetworkLayerType.PROFESSIONAL]
        
        communities = nx.community.greedy_modularity_communities(prof_layer)
        
        for i, community in enumerate(communities):
            if len(community) >= 3:
                circle_id = f"workplace_circle_{i}"
                leader_id = self._select_circle_leader(community, prof_layer, 'hierarchy')
                
                circle = SocialCircle(
                    circle_id=circle_id,
                    circle_name=f"工作圈{i+1}",
                    circle_type="workplace",
                    leader_id=leader_id,
                    core_members=set(list(community)[:max(1, len(community)//3)]),
                    peripheral_members=set(list(community)[len(community)//3:]),
                    cohesion_level=self._calculate_circle_cohesion(community, prof_layer),
                    influence_radius=1.2,
                    activity_level=0.7,
                    resource_sharing_level=0.4,
                    decision_making_style="hierarchical",
                    collective_memory={},
                    formation_time=0,
                    stability_score=0.5
                )
                
                self.social_circles[circle_id] = circle
                
                for agent_id in community:
                    self.agent_circles[agent_id].add(circle_id)
    
    def _identify_friend_circles(self) -> None:
        """识别朋友圈子"""
        social_layer = self.layers[NetworkLayerType.SOCIAL]
        
        communities = nx.community.greedy_modularity_communities(social_layer)
        
        for i, community in enumerate(communities):
            if len(community) >= 3:
                circle_id = f"friend_circle_{i}"
                leader_id = self._select_circle_leader(community, social_layer, 'social_activity')
                
                circle = SocialCircle(
                    circle_id=circle_id,
                    circle_name=f"朋友圈{i+1}",
                    circle_type="friend",
                    leader_id=leader_id,
                    core_members=set(community),
                    peripheral_members=set(),
                    cohesion_level=self._calculate_circle_cohesion(community, social_layer),
                    influence_radius=1.0,
                    activity_level=0.8,
                    resource_sharing_level=0.3,
                    decision_making_style="democratic",
                    collective_memory={},
                    formation_time=0,
                    stability_score=0.4
                )
                
                self.social_circles[circle_id] = circle
                
                for agent_id in community:
                    self.agent_circles[agent_id].add(circle_id)
    
    def _identify_composite_circles(self) -> None:
        """识别跨层复合圈子"""
        # 寻找在多个层中都有强连接的群体
        composite_communities = nx.community.greedy_modularity_communities(self.composite_network)
        
        for i, community in enumerate(composite_communities):
            if len(community) >= 5:  # 复合圈子规模要求更大
                # 检查是否跨越多个层
                layer_coverage = self._check_layer_coverage(community)
                
                if len(layer_coverage) >= 2:  # 至少跨越两个层
                    circle_id = f"composite_circle_{i}"
                    leader_id = self._select_circle_leader(community, self.composite_network, 'composite')
                    
                    circle = SocialCircle(
                        circle_id=circle_id,
                        circle_name=f"复合圈{i+1}",
                        circle_type="composite",
                        leader_id=leader_id,
                        core_members=set(list(community)[:len(community)//2]),
                        peripheral_members=set(list(community)[len(community)//2:]),
                        cohesion_level=self._calculate_circle_cohesion(community, self.composite_network),
                        influence_radius=2.5,  # 复合圈影响力更大
                        activity_level=0.7,
                        resource_sharing_level=0.6,
                        decision_making_style="consensus",
                        collective_memory={},
                        formation_time=0,
                        stability_score=0.7
                    )
                    
                    self.social_circles[circle_id] = circle
                    
                    for agent_id in community:
                        self.agent_circles[agent_id].add(circle_id)
    
    def get_differential_influence(self, source_id: int, target_id: int) -> float:
        """计算差序格局影响力"""
        total_influence = 0.0
        
        # 累加各层的影响力
        for layer_type, layer_graph in self.layers.items():
            if layer_graph.has_edge(source_id, target_id):
                edge_weight = layer_graph[source_id][target_id].get('weight', 0)
                layer_weight = self.differential_weights[layer_type]
                total_influence += edge_weight * layer_weight
        
        return min(1.0, total_influence)
    
    def get_circle_influence_network(self, agent_id: int) -> Dict[str, float]:
        """获取智能体在各圈子中的影响力"""
        influences = {}
        
        for circle_id in self.agent_circles[agent_id]:
            circle = self.social_circles[circle_id]
            
            if circle.leader_id == agent_id:
                influences[circle_id] = 1.0  # 领导者影响力最大
            elif agent_id in circle.core_members:
                influences[circle_id] = 0.7  # 核心成员影响力较大
            else:
                influences[circle_id] = 0.3  # 边缘成员影响力较小
        
        return influences
    
    def simulate_network_evolution(self, time_steps: int) -> None:
        """模拟网络演化"""
        for step in range(time_steps):
            # 关系强度衰减
            self._decay_relationships()
            
            # 新关系形成
            self._form_new_relationships()
            
            # 圈子演化
            self._evolve_circles()
            
            # 更新网络统计
            if step % 10 == 0:
                self._calculate_network_statistics()
    
    def _calculate_network_statistics(self) -> None:
        """计算网络统计信息"""
        for layer_type, layer_graph in self.layers.items():
            layer_name = layer_type.value
            
            # 网络密度
            self.network_stats['layer_densities'][layer_name] = nx.density(layer_graph)
            
            # 聚类系数
            if len(layer_graph.nodes()) > 0:
                self.network_stats['clustering_coefficients'][layer_name] = \
                    nx.average_clustering(layer_graph)
            
            # 平均路径长度
            if nx.is_connected(layer_graph):
                self.network_stats['average_path_lengths'][layer_name] = \
                    nx.average_shortest_path_length(layer_graph)
            
            # 度分布
            degrees = [d for n, d in layer_graph.degree()]
            self.network_stats['degree_distributions'][layer_name] = {
                'mean': np.mean(degrees) if degrees else 0,
                'std': np.std(degrees) if degrees else 0,
                'max': max(degrees) if degrees else 0
            }
        
        # 总关系数
        self.network_stats['total_relationships'] = sum(
            len(layer.edges()) for layer in self.layers.values()
        )
    
    def get_network_summary(self) -> Dict:
        """获取网络摘要信息"""
        return {
            'basic_stats': self.network_stats,
            'circle_summary': {
                'total_circles': len(self.social_circles),
                'circle_types': Counter(circle.circle_type for circle in self.social_circles.values()),
                'average_circle_size': np.mean([
                    len(circle.core_members) + len(circle.peripheral_members)
                    for circle in self.social_circles.values()
                ]) if self.social_circles else 0
            },
            'differential_pattern': {
                'layer_weights': {k.value: v for k, v in self.differential_weights.items()},
                'cross_layer_connections': self._count_cross_layer_connections()
            }
        }
    
    # 辅助方法
    def _add_relationship_edge(self, source_id: int, target_id: int, 
                              layer_type: NetworkLayerType, strength: float, **kwargs):
        """添加关系边"""
        layer_graph = self.layers[layer_type]
        layer_graph.add_edge(source_id, target_id, weight=strength)
        
        # 存储详细的关系信息
        edge_key = (min(source_id, target_id), max(source_id, target_id), layer_type)
        
        relationship_edge = RelationshipEdge(
            source_id=source_id,
            target_id=target_id,
            layer_type=layer_type,
            strength=strength,
            intimacy=strength * random.uniform(0.8, 1.2),
            trust=strength * random.uniform(0.7, 1.1),
            reciprocity=random.uniform(-0.2, 0.2),
            interaction_frequency=strength * random.uniform(0.5, 1.0),
            relationship_duration=random.randint(30, 3650),  # 30天到10年
            last_interaction=0,
            relationship_quality=strength * random.uniform(0.8, 1.0),
            influence_asymmetry=random.uniform(-0.3, 0.3),
            **kwargs
        )
        
        self.relationship_edges[edge_key] = relationship_edge
    
    def _group_agents_by_family(self, agents: List[AgentProfile]) -> Dict[int, List[AgentProfile]]:
        """按家庭分组智能体"""
        families = defaultdict(list)
        
        # 简化的家庭分组逻辑
        family_id = 0
        for i in range(0, len(agents), random.randint(2, 6)):
            family_members = agents[i:i+random.randint(2, 6)]
            families[family_id] = family_members
            family_id += 1
        
        return dict(families)
    
    def _calculate_kinship_strength(self, agent1: AgentProfile, agent2: AgentProfile) -> float:
        """计算血缘关系强度"""
        # 基于年龄差异和家庭规模
        age_factor = 1.0 - abs(agent1.age - agent2.age) / 100.0
        family_factor = 1.0 / max(agent1.family_size, agent2.family_size)
        
        return max(0.3, min(1.0, age_factor * 0.6 + family_factor * 0.4))
    
    def _calculate_socioeconomic_similarity(self, agent1: AgentProfile, agent2: AgentProfile) -> float:
        """计算社会经济地位相似性"""
        # 简化的相似性计算
        education_sim = 1.0 if agent1.education_level == agent2.education_level else 0.5
        income_sim = 1.0 if agent1.income_level == agent2.income_level else 0.3
        
        return (education_sim + income_sim) / 2.0
    
    def _group_agents_by_workplace(self, agents: List[AgentProfile]) -> Dict[str, List[AgentProfile]]:
        """按工作场所分组"""
        workplaces = defaultdict(list)
        
        for agent in agents:
            # 基于职业类型分组
            workplace_key = f"{agent.occupation}_{hash(agent.agent_id) % 10}"
            workplaces[workplace_key].append(agent)
        
        return dict(workplaces)
    
    def _calculate_professional_strength(self, agent1: AgentProfile, agent2: AgentProfile) -> float:
        """计算职业关系强度"""
        if agent1.occupation == agent2.occupation:
            return random.uniform(0.4, 0.8)
        else:
            return random.uniform(0.1, 0.3)
    
    def _get_hierarchy_level(self, occupation: str) -> int:
        """获取职业层级"""
        # 简化的职业层级映射
        hierarchy_map = {
            'manager': 4, 'engineer': 3, 'technician': 2, 'worker': 1, 'student': 0
        }
        return hierarchy_map.get(occupation.lower(), 2)
    
    def _build_extended_kinship_network(self, families: Dict, config: Dict):
        """构建扩展血缘网络"""
        # 在家族间建立扩展亲属关系
        family_list = list(families.values())
        
        for i, family1 in enumerate(family_list):
            for family2 in family_list[i+1:]:
                if random.random() < config['extended_family_probability']:
                    # 随机选择两个家庭的成员建立亲属关系
                    member1 = random.choice(family1)
                    member2 = random.choice(family2)
                    
                    strength = config['kinship_strength_decay'] * random.uniform(0.3, 0.7)
                    self._add_relationship_edge(
                        member1.agent_id, member2.agent_id,
                        NetworkLayerType.KINSHIP, strength,
                        kinship_degree=2  # 二代亲属
                    )
    
    def _build_cross_workplace_networks(self, agents: List[AgentProfile], config: Dict):
        """构建跨工作场所网络"""
        # 基于相同职业建立跨工作场所联系
        occupation_groups = defaultdict(list)
        
        for agent in agents:
            occupation_groups[agent.occupation].append(agent)
        
        for occupation, members in occupation_groups.items():
            if len(members) > 1:
                for i, agent1 in enumerate(members):
                    for agent2 in members[i+1:]:
                        if random.random() < config['cross_department_probability']:
                            strength = random.uniform(0.2, 0.5)
                            self._add_relationship_edge(
                                agent1.agent_id, agent2.agent_id,
                                NetworkLayerType.PROFESSIONAL, strength
                            )
    
    def _group_agents_by_education(self, agents: List[AgentProfile]) -> Dict[str, List[AgentProfile]]:
        """按教育背景分组"""
        education_groups = defaultdict(list)
        
        for agent in agents:
            education_groups[agent.education_level].append(agent)
        
        return dict(education_groups)
    
    def _calculate_education_similarity(self, agent1: AgentProfile, agent2: AgentProfile) -> float:
        """计算教育背景相似性"""
        if agent1.education_level == agent2.education_level:
            return random.uniform(0.6, 0.9)
        else:
            return random.uniform(0.1, 0.4)
    
    def _group_agents_by_interests(self, agents: List[AgentProfile]) -> Dict[str, List[AgentProfile]]:
        """按兴趣爱好分组"""
        interest_groups = defaultdict(list)
        
        interests = ['sports', 'music', 'reading', 'travel', 'cooking', 'technology']
        
        for agent in agents:
            # 每个智能体随机分配1-3个兴趣
            agent_interests = random.sample(interests, random.randint(1, 3))
            for interest in agent_interests:
                interest_groups[interest].append(agent)
        
        return dict(interest_groups)
    
    def _calculate_interest_overlap(self, agent1: AgentProfile, agent2: AgentProfile) -> float:
        """计算兴趣重叠度"""
        # 简化的兴趣重叠计算
        return random.uniform(0.2, 0.8)
    
    def _select_circle_leader(self, community: Set[int], graph: nx.Graph, criteria: str) -> Optional[int]:
        """选择圈子领导者"""
        if not community:
            return None
        
        if criteria == 'age':
            # 选择年龄最大的
            return max(community, key=lambda x: self.agent_profiles[x].age)
        elif criteria == 'social_activity':
            # 选择社交活跃度最高的
            return max(community, key=lambda x: self.agent_profiles[x].social_activity_level)
        elif criteria == 'hierarchy':
            # 选择职业层级最高的
            return max(community, key=lambda x: self._get_hierarchy_level(self.agent_profiles[x].occupation))
        else:
            # 选择度中心性最高的
            centralities = nx.degree_centrality(graph.subgraph(community))
            return max(centralities, key=centralities.get)
    
    def _calculate_circle_cohesion(self, community: Set[int], graph: nx.Graph) -> float:
        """计算圈子凝聚力"""
        if len(community) < 2:
            return 0.0
        
        subgraph = graph.subgraph(community)
        return nx.density(subgraph)
    
    def _check_layer_coverage(self, community: Set[int]) -> Set[str]:
        """检查社区跨越的网络层"""
        layers_covered = set()
        
        for layer_type, layer_graph in self.layers.items():
            subgraph = layer_graph.subgraph(community)
            if len(subgraph.edges()) > 0:
                layers_covered.add(layer_type.value)
        
        return layers_covered
    
    def _count_cross_layer_connections(self) -> Dict[str, int]:
        """统计跨层连接"""
        cross_connections = defaultdict(int)
        
        for agent_id in self.agent_profiles:
            layers_connected = []
            for layer_type, layer_graph in self.layers.items():
                if agent_id in layer_graph and len(list(layer_graph.neighbors(agent_id))) > 0:
                    layers_connected.append(layer_type.value)
            
            if len(layers_connected) > 1:
                cross_connections['-'.join(sorted(layers_connected))] += 1
        
        return dict(cross_connections)
    
    def _decay_relationships(self):
        """关系强度衰减"""
        decay_rate = 0.01
        
        for edge_key, edge in list(self.relationship_edges.items()):
            # 基于交互频率调整衰减率
            actual_decay = decay_rate * (1 - edge.interaction_frequency)
            edge.strength = max(0.1, edge.strength - actual_decay)
            
            # 如果关系强度过低，移除关系
            if edge.strength < 0.1:
                source, target, layer_type = edge_key
                self.layers[layer_type].remove_edge(source, target)
                del self.relationship_edges[edge_key]
    
    def _form_new_relationships(self):
        """形成新关系"""
        formation_rate = 0.005
        
        agents = list(self.agent_profiles.keys())
        
        for _ in range(int(len(agents) * formation_rate)):
            agent1, agent2 = random.sample(agents, 2)
            
            # 基于地理距离和社会相似性决定是否形成新关系
            profile1 = self.agent_profiles[agent1]
            profile2 = self.agent_profiles[agent2]
            
            distance = euclidean(profile1.location, profile2.location)
            similarity = self._calculate_socioeconomic_similarity(profile1, profile2)
            
            if distance < 10.0 and similarity > 0.5 and random.random() < 0.1:
                strength = random.uniform(0.2, 0.5)
                self._add_relationship_edge(
                    agent1, agent2, NetworkLayerType.SOCIAL, strength
                )
    
    def _evolve_circles(self):
        """圈子演化"""
        # 圈子成员流动
        for circle_id, circle in list(self.social_circles.items()):
            # 成员离开
            if random.random() < 0.02:  # 2%的概率有成员离开
                if len(circle.peripheral_members) > 0:
                    leaving_member = random.choice(list(circle.peripheral_members))
                    circle.peripheral_members.remove(leaving_member)
                    self.agent_circles[leaving_member].discard(circle_id)
            
            # 新成员加入
            if random.random() < 0.03:  # 3%的概率有新成员加入
                potential_members = [
                    agent_id for agent_id in self.agent_profiles
                    if agent_id not in circle.core_members and 
                       agent_id not in circle.peripheral_members
                ]
                
                if potential_members:
                    new_member = random.choice(potential_members)
                    circle.peripheral_members.add(new_member)
                    self.agent_circles[new_member].add(circle_id)
            
            # 更新圈子凝聚力
            all_members = circle.core_members.union(circle.peripheral_members)
            if len(all_members) >= 2:
                # 基于成员间的平均关系强度更新凝聚力
                total_strength = 0
                pair_count = 0
                
                for member1 in all_members:
                    for member2 in all_members:
                        if member1 < member2:
                            influence = self.get_differential_influence(member1, member2)
                            total_strength += influence
                            pair_count += 1
                
                if pair_count > 0:
                    circle.cohesion_level = total_strength / pair_count
            
            # 如果圈子太小或凝聚力太低，解散圈子
            if len(all_members) < 2 or circle.cohesion_level < 0.1:
                for member_id in all_members:
                    self.agent_circles[member_id].discard(circle_id)
                del self.social_circles[circle_id]


def create_agent_profiles(num_agents: int) -> List[AgentProfile]:
    """创建智能体档案"""
    profiles = []
    
    occupations = ['manager', 'engineer', 'technician', 'worker', 'teacher', 'doctor', 'student']
    education_levels = ['primary', 'secondary', 'bachelor', 'master', 'phd']
    income_levels = ['low', 'medium', 'high']
    genders = ['male', 'female']
    cultures = ['han', 'minority']
    
    for i in range(num_agents):
        profile = AgentProfile(
            agent_id=i,
            age=random.randint(18, 80),
            gender=random.choice(genders),
            education_level=random.choice(education_levels),
            occupation=random.choice(occupations),
            income_level=random.choice(income_levels),
            location=(random.uniform(0, 100), random.uniform(0, 100)),
            family_size=random.randint(1, 6),
            social_activity_level=random.uniform(0.1, 1.0),
            cultural_background=random.choice(cultures),
            personality_traits={
                'openness': random.uniform(0, 1),
                'conscientiousness': random.uniform(0, 1),
                'extraversion': random.uniform(0, 1),
                'agreeableness': random.uniform(0, 1),
                'neuroticism': random.uniform(0, 1)
            }
        )
        profiles.append(profile)
    
    return profiles


if __name__ == "__main__":
    # 测试多层社会网络构建
    print("创建智能体档案...")
    profiles = create_agent_profiles(50)
    
    print("构建多层社会网络...")
    network = MultilayerSocialNetwork()
    network.build_network(profiles)
    
    print("\n网络摘要:")
    summary = network.get_network_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("\n测试差序格局影响力计算:")
    if len(profiles) >= 2:
        agent1_id = profiles[0].agent_id
        agent2_id = profiles[1].agent_id
        influence = network.get_differential_influence(agent1_id, agent2_id)
        print(f"智能体 {agent1_id} 对智能体 {agent2_id} 的差序格局影响力: {influence:.3f}")
    
    print("\n测试圈子影响力网络:")
    if profiles:
        agent_id = profiles[0].agent_id
        circle_influences = network.get_circle_influence_network(agent_id)
        print(f"智能体 {agent_id} 的圈子影响力:")
        for circle_id, influence in circle_influences.items():
            circle_name = network.social_circles[circle_id].circle_name
            print(f"  {circle_name}: {influence:.3f}")
    
    print("\n多层社会网络构建完成！")