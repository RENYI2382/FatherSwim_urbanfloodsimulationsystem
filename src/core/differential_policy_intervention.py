#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
差序化政策干预机制
基于差序格局和圈子理论的政策传播与干预系统

核心功能：
1. 政策传导路径分析
2. 圈子影响力评估
3. 差序化投放策略
4. 政策效果评估
5. 动态调控机制
6. 集体行动预测
7. 临界点识别
"""

import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from collections import defaultdict, Counter, deque
from scipy.stats import norm, beta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class PolicyType(Enum):
    """政策类型枚举"""
    EVACUATION_ORDER = "疏散命令"        # 强制疏散政策
    SAFETY_GUIDANCE = "安全指导"         # 安全行为指导
    RESOURCE_ALLOCATION = "资源分配"     # 资源配置政策
    INFORMATION_DISCLOSURE = "信息公开"  # 信息透明化政策
    INCENTIVE_MECHANISM = "激励机制"     # 行为激励政策
    SOCIAL_MOBILIZATION = "社会动员"    # 集体行动动员
    EMERGENCY_RESPONSE = "应急响应"      # 紧急响应措施


class PropagationChannel(Enum):
    """政策传播渠道"""
    OFFICIAL_MEDIA = "官方媒体"          # 政府官方渠道
    SOCIAL_MEDIA = "社交媒体"           # 社交网络平台
    COMMUNITY_LEADER = "社区领袖"       # 社区意见领袖
    FAMILY_NETWORK = "家庭网络"         # 家族关系网络
    WORKPLACE_NETWORK = "工作网络"      # 职场关系网络
    PEER_INFLUENCE = "同伴影响"         # 同龄群体影响
    GRASSROOTS_ORGANIZATION = "基层组织" # 基层组织传播


class EffectivenessLevel(Enum):
    """政策效果等级"""
    VERY_LOW = 0.1     # 很低
    LOW = 0.3          # 低
    MODERATE = 0.5     # 中等
    HIGH = 0.7         # 高
    VERY_HIGH = 0.9    # 很高


@dataclass
class PolicyIntervention:
    """政策干预措施"""
    policy_id: str
    policy_type: PolicyType
    target_population: Set[str]
    policy_content: Dict[str, Any]
    implementation_channels: List[PropagationChannel]
    priority_level: float = 0.5
    urgency_level: float = 0.5
    
    # 动态状态
    current_adopters: Set[str] = field(default_factory=set)
    resistance_groups: Set[str] = field(default_factory=set)
    diffusion_network: nx.Graph = field(default_factory=nx.Graph)
    current_time_step: int = 0


@dataclass
class PolicyAgent:
    """政策智能体"""
    agent_id: str
    
    # 个体特征
    risk_perception: float = 0.5
    social_influence_sensitivity: float = 0.5
    opinion_leadership: float = 0.5
    trust_in_authority: float = 0.5
    
    # 政策相关状态
    current_policy_exposure: Dict[str, float] = field(default_factory=dict)
    policy_adoption_history: Dict[str, Dict] = field(default_factory=dict)
    adoption_threshold: float = 0.5


@dataclass
class PolicyDiffusionState:
    """政策扩散状态"""
    time_step: int
    policy_id: str
    total_exposed: int
    total_adopted: int
    total_resistant: int
    adoption_rate: float
    diffusion_speed: float
    network_density: float
    clustering_effect: float


class DifferentialPolicyInterventionSystem:
    """差序化政策干预系统"""
    
    def __init__(self, social_network, config: Dict = None):
        self.social_network = social_network
        self.config = config or self._default_config()
        self.policy_agents = {}
        self.active_policies = {}
        self.policy_history = []
        
        self._initialize_policy_agents()
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'diffusion_parameters': {
                'base_transmission_rate': 0.1,
                'social_influence_weight': 0.3,
                'authority_influence_weight': 0.4,
                'peer_influence_weight': 0.3
            },
            'agent_parameters': {
                'risk_perception_range': (0.2, 0.8),
                'trust_range': (0.3, 0.9),
                'influence_sensitivity_range': (0.1, 0.7)
            },
            'policy_parameters': {
                'adoption_threshold_range': (0.3, 0.7),
                'resistance_threshold': 0.8
            }
        }
    
    def _initialize_policy_agents(self):
        """初始化政策智能体"""
        for agent_id in self.social_network.composite_network.nodes():
            self.policy_agents[agent_id] = PolicyAgent(
                agent_id=str(agent_id),
                risk_perception=np.random.uniform(0.2, 0.8),
                social_influence_sensitivity=np.random.uniform(0.1, 0.7),
                opinion_leadership=np.random.uniform(0.1, 0.9),
                trust_in_authority=np.random.uniform(0.3, 0.9),
                adoption_threshold=np.random.uniform(0.3, 0.7)
            )
    
    def implement_policy(self, policy: PolicyIntervention) -> str:
        """实施政策干预"""
        policy_id = policy.policy_id
        self.active_policies[policy_id] = policy
        
        # 初始化政策扩散网络
        policy.diffusion_network = nx.Graph()
        policy.diffusion_network.add_nodes_from(policy.target_population)
        
        # 识别初始采纳者
        self._identify_initial_adopters(policy)
        
        return policy_id
    
    def simulate_policy_diffusion(self, policy_id: str, time_steps: int) -> List[PolicyDiffusionState]:
        """模拟政策扩散过程"""
        if policy_id not in self.active_policies:
            raise ValueError(f"Policy {policy_id} not found")
        
        policy = self.active_policies[policy_id]
        diffusion_states = []
        
        for step in range(time_steps):
            policy.current_time_step = step
            
            # 政策传播
            self._propagate_policy(policy)
            
            # 社会影响
            self._apply_social_influence(policy)
            
            # 更新采纳状态
            self._update_adoption_states(policy)
            
            # 记录扩散状态
            state = self._calculate_diffusion_state(policy, step)
            diffusion_states.append(state)
            
            # 检查临界点
            if self._detect_tipping_point(policy, state):
                break
        
        return diffusion_states
    
    def evaluate_policy_effectiveness(self, policy_id: str) -> Dict[str, float]:
        """评估政策效果"""
        if policy_id not in self.active_policies:
            return {}
        
        policy = self.active_policies[policy_id]
        
        # 基本效果指标
        adoption_rate = len(policy.current_adopters) / len(policy.target_population)
        resistance_rate = len(policy.resistance_groups) / len(policy.target_population)
        coverage_rate = len(policy.current_adopters.union(policy.resistance_groups)) / len(policy.target_population)
        
        # 差序公平性
        equity_score = self._calculate_differential_equity(policy)
        
        return {
            'adoption_rate': adoption_rate,
            'resistance_rate': resistance_rate,
            'coverage_rate': coverage_rate,
            'equity_score': equity_score,
            'overall_effectiveness': (adoption_rate * 0.4 + (1-resistance_rate) * 0.3 + equity_score * 0.3)
        }
    
    def analyze_collective_action_potential(self, policy_id: str) -> Dict:
        """分析集体行动潜力"""
        if policy_id not in self.active_policies:
            return {}
        
        policy = self.active_policies[policy_id]
        
        # 网络结构分析
        network_analysis = self._analyze_network_structure_for_collective_action()
        
        # 识别潜在行动群体
        action_groups = self._identify_collective_action_groups(policy)
        
        # 计算集体行动概率
        action_probability = self._calculate_collective_action_probability(action_groups)
        
        # 预测影响
        impact_prediction = self._predict_collective_action_impact(action_groups, policy)
        
        # 风险评估
        risks = self._assess_collective_action_risks(action_groups, impact_prediction)
        
        return {
            'network_analysis': network_analysis,
            'action_groups': action_groups,
            'probability': action_probability,
            'impact_prediction': impact_prediction,
            'risks': risks
        }
    
    def generate_policy_recommendations(self, policy_id: str) -> Dict[str, List[str]]:
        """生成政策建议"""
        if policy_id not in self.active_policies:
            return {}
        
        policy = self.active_policies[policy_id]
        collective_action_analysis = self.analyze_collective_action_potential(policy_id)
        network_analysis = collective_action_analysis.get('network_analysis', {})
        action_groups = collective_action_analysis.get('action_groups', [])
        risks = collective_action_analysis.get('risks', [])
        
        recommendations = {
            'risk_mitigation': self._generate_risk_mitigation_recommendations(risks, action_groups),
            'resistance_management': self._generate_resistance_management_recommendations(action_groups),
            'network_integration': self._generate_network_integration_recommendations(network_analysis),
            'differential_strategy': self._generate_differential_strategy_recommendations(policy),
            'critical_alerts': self._generate_critical_alerts(policy, collective_action_analysis)
        }
        
        return recommendations
    
    def _identify_initial_adopters(self, policy: PolicyIntervention):
        """识别初始采纳者"""
        # 基于智能体特征识别早期采纳者
        for agent_id in policy.target_population:
            agent = self.policy_agents[agent_id]
            
            # 计算初始采纳概率
            adoption_score = (
                agent.trust_in_authority * 0.4 +
                agent.risk_perception * 0.3 +
                agent.opinion_leadership * 0.2 +
                (1 - agent.social_influence_sensitivity) * 0.1
            )
            
            if adoption_score > 0.7 and np.random.random() < 0.3:
                policy.current_adopters.add(agent_id)
                policy.diffusion_network.add_node(agent_id)
    
    def _propagate_policy(self, policy: PolicyIntervention):
        """政策传播"""
        new_exposures = defaultdict(float)
        
        # 通过社会网络传播
        for adopter in policy.current_adopters:
            neighbors = list(self.social_network.composite_network.neighbors(adopter))
            for neighbor in neighbors:
                if neighbor in policy.target_population:
                    # 计算传播强度
                    transmission_strength = self._calculate_transmission_strength(adopter, neighbor, policy)
                    new_exposures[neighbor] += transmission_strength
        
        # 通过官方渠道传播
        if PropagationChannel.OFFICIAL_MEDIA in policy.implementation_channels:
            for agent_id in policy.target_population:
                agent = self.policy_agents[agent_id]
                official_exposure = agent.trust_in_authority * policy.priority_level * 0.2
                new_exposures[agent_id] += official_exposure
        
        # 更新暴露度
        for agent_id, exposure in new_exposures.items():
            current_exposure = self.policy_agents[agent_id].current_policy_exposure.get(policy.policy_id, 0.0)
            self.policy_agents[agent_id].current_policy_exposure[policy.policy_id] = min(1.0, current_exposure + exposure)
    
    def _calculate_transmission_strength(self, source_id: str, target_id: str, policy: PolicyIntervention) -> float:
        """计算传播强度"""
        source_agent = self.policy_agents[source_id]
        target_agent = self.policy_agents[target_id]
        
        # 获取关系强度
        edge_data = self.social_network.composite_network.get_edge_data(source_id, target_id, {})
        relationship_strength = edge_data.get('weight', 0.5)
        
        # 计算传播强度
        transmission_strength = (
            source_agent.opinion_leadership * 0.4 +
            relationship_strength * 0.3 +
            target_agent.social_influence_sensitivity * 0.2 +
            policy.urgency_level * 0.1
        )
        
        return min(1.0, transmission_strength * 0.1)  # 基础传播率
    
    def _apply_social_influence(self, policy: PolicyIntervention):
        """应用社会影响"""
        for agent_id in policy.target_population:
            if agent_id not in policy.current_adopters and agent_id not in policy.resistance_groups:
                social_pressure = self._calculate_social_pressure(agent_id, policy)
                
                # 更新暴露度
                current_exposure = self.policy_agents[agent_id].current_policy_exposure.get(policy.policy_id, 0.0)
                self.policy_agents[agent_id].current_policy_exposure[policy.policy_id] = min(1.0, current_exposure + social_pressure * 0.1)
    
    def _calculate_social_pressure(self, agent_id: str, policy: PolicyIntervention) -> float:
        """计算社会压力"""
        if agent_id not in self.social_network.composite_network:
            return 0.0
        
        neighbors = list(self.social_network.composite_network.neighbors(agent_id))
        if not neighbors:
            return 0.0
        
        total_pressure = 0.0
        neighbor_count = 0
        
        for neighbor in neighbors:
            if neighbor in policy.target_population:
                edge_data = self.social_network.composite_network.get_edge_data(agent_id, neighbor, {})
                relationship_strength = edge_data.get('weight', 0.5)
                
                if neighbor in policy.current_adopters:
                    total_pressure += relationship_strength
                elif neighbor in policy.resistance_groups:
                    total_pressure -= relationship_strength * 0.5
                
                neighbor_count += 1
        
        return total_pressure / max(1, neighbor_count)
    
    def _update_adoption_states(self, policy: PolicyIntervention):
        """更新采纳状态"""
        new_adopters = set()
        new_resisters = set()
        
        for agent_id, agent in self.policy_agents.items():
            if agent_id in policy.target_population:
                if agent_id not in policy.current_adopters and agent_id not in policy.resistance_groups:
                    # 计算采纳概率
                    adoption_prob = self._calculate_adoption_probability(agent, policy)
                    
                    if np.random.random() < adoption_prob:
                        new_adopters.add(agent_id)
                        agent.policy_adoption_history[policy.policy_id] = {
                            'adopted': True,
                            'adoption_time': policy.current_time_step,
                            'adoption_probability': adoption_prob
                        }
                    elif adoption_prob < 0.2:  # 低采纳概率可能导致抗拒
                        resistance_prob = (0.2 - adoption_prob) * 2.0
                        if np.random.random() < resistance_prob:
                            new_resisters.add(agent_id)
                            agent.policy_adoption_history[policy.policy_id] = {
                                'adopted': False,
                                'resistance': True,
                                'resistance_time': policy.current_time_step
                            }
        
        # 更新政策状态
        policy.current_adopters.update(new_adopters)
        policy.resistance_groups.update(new_resisters)
    
    def _calculate_adoption_probability(self, agent: PolicyAgent, policy: PolicyIntervention) -> float:
        """计算采纳概率"""
        base_probability = 0.3
        
        # 智能体特征影响
        risk_factor = agent.risk_perception * 0.3
        social_factor = agent.social_influence_sensitivity * 0.2
        leadership_factor = agent.opinion_leadership * 0.1
        trust_factor = agent.trust_in_authority * 0.2
        
        # 政策特征影响
        priority_factor = policy.priority_level * 0.1
        urgency_factor = policy.urgency_level * 0.1
        
        # 暴露度影响
        exposure_level = agent.current_policy_exposure.get(policy.policy_id, 0.0)
        exposure_factor = min(0.3, exposure_level * 0.3)
        
        # 网络影响
        network_influence = self._calculate_network_influence(agent.agent_id, policy)
        network_factor = network_influence * 0.2
        
        total_probability = (
            base_probability + risk_factor + social_factor + leadership_factor +
            trust_factor + priority_factor + urgency_factor + exposure_factor + network_factor
        )
        
        return max(0.0, min(1.0, total_probability))
    
    def _calculate_network_influence(self, agent_id: str, policy: PolicyIntervention) -> float:
        """计算网络影响力"""
        if agent_id not in self.social_network.composite_network:
            return 0.0
        
        neighbors = list(self.social_network.composite_network.neighbors(agent_id))
        if not neighbors:
            return 0.0
        
        weighted_influence = 0.0
        total_weight = 0.0
        
        for neighbor in neighbors:
            edge_data = self.social_network.composite_network.get_edge_data(agent_id, neighbor, {})
            relationship_strength = edge_data.get('weight', 0.5)
            
            if neighbor in policy.current_adopters:
                weighted_influence += relationship_strength
            elif neighbor in policy.resistance_groups:
                weighted_influence -= relationship_strength * 0.5
            
            total_weight += relationship_strength
        
        if total_weight > 0:
            return weighted_influence / total_weight
        else:
            return 0.0
    
    def _calculate_diffusion_state(self, policy: PolicyIntervention, time_step: int) -> PolicyDiffusionState:
        """计算扩散状态"""
        total_exposed = sum(1 for agent_id in policy.target_population 
                          if self.policy_agents[agent_id].current_policy_exposure.get(policy.policy_id, 0) > 0)
        total_adopted = len(policy.current_adopters)
        total_resistant = len(policy.resistance_groups)
        
        adoption_rate = total_adopted / len(policy.target_population) if policy.target_population else 0
        
        # 计算扩散速度（相对于上一时间步）
        if time_step > 0 and hasattr(policy, 'previous_adopted_count'):
            diffusion_speed = (total_adopted - policy.previous_adopted_count) / len(policy.target_population)
        else:
            diffusion_speed = adoption_rate
        
        policy.previous_adopted_count = total_adopted
        
        return PolicyDiffusionState(
            time_step=time_step,
            policy_id=policy.policy_id,
            total_exposed=total_exposed,
            total_adopted=total_adopted,
            total_resistant=total_resistant,
            adoption_rate=adoption_rate,
            diffusion_speed=diffusion_speed,
            network_density=nx.density(self.social_network.composite_network),
            clustering_effect=nx.average_clustering(self.social_network.composite_network)
        )
    
    def _detect_tipping_point(self, policy: PolicyIntervention, state: PolicyDiffusionState) -> bool:
        """检测临界点"""
        # 如果采纳率超过80%或扩散速度接近0，认为达到临界点
        return state.adoption_rate > 0.8 or (state.time_step > 5 and state.diffusion_speed < 0.001)
    
    # 辅助方法实现
    def _calculate_differential_equity(self, policy: PolicyIntervention) -> float:
        """计算差序公平性"""
        layer_adoption_rates = {}
        
        for layer_type, layer_graph in self.social_network.layers.items():
            layer_agents = set(layer_graph.nodes()).intersection(policy.target_population)
            if layer_agents:
                adopted_in_layer = layer_agents.intersection(policy.current_adopters)
                layer_adoption_rates[layer_type.value] = len(adopted_in_layer) / len(layer_agents)
        
        if not layer_adoption_rates:
            return 1.0
        
        rates = list(layer_adoption_rates.values())
        rate_variance = np.var(rates)
        equity_score = max(0.0, 1.0 - rate_variance * 5.0)
        return equity_score
    
    def _analyze_network_structure_for_collective_action(self) -> Dict:
        """分析网络结构对集体行动的影响"""
        network = self.social_network.composite_network
        
        density = nx.density(network)
        clustering = nx.average_clustering(network)
        
        components = list(nx.connected_components(network))
        largest_component_size = len(max(components, key=len)) if components else 0
        fragmentation = 1.0 - (largest_component_size / network.number_of_nodes())
        
        degree_centrality = nx.degree_centrality(network)
        key_nodes = [node for node, centrality in degree_centrality.items()
                    if centrality > np.percentile(list(degree_centrality.values()), 90)]
        
        return {
            'density': density,
            'clustering': clustering,
            'fragmentation': fragmentation,
            'largest_component_ratio': largest_component_size / network.number_of_nodes(),
            'key_nodes_count': len(key_nodes),
            'key_nodes': key_nodes[:10],
            'collective_action_potential': self._assess_collective_action_potential(density, clustering, fragmentation)
        }
    
    def _assess_collective_action_potential(self, density: float, clustering: float, fragmentation: float) -> float:
        """评估集体行动潜力"""
        density_factor = density
        clustering_factor = clustering
        cohesion_factor = 1.0 - fragmentation
        
        potential = (density_factor * 0.4 + clustering_factor * 0.3 + cohesion_factor * 0.3)
        return min(1.0, potential)
    
    def _identify_collective_action_groups(self, policy: PolicyIntervention) -> List[Dict]:
        """识别潜在的集体行动群体"""
        action_groups = []
        
        if policy.resistance_groups:
            resisters_subgraph = self.social_network.composite_network.subgraph(policy.resistance_groups)
            
            try:
                communities = nx.community.greedy_modularity_communities(resisters_subgraph)
                
                for i, community in enumerate(communities):
                    if len(community) >= 3:
                        group_centrality = np.mean([
                            nx.degree_centrality(self.social_network.composite_network)[node]
                            for node in community
                        ])
                        
                        group_cohesion = nx.density(resisters_subgraph.subgraph(community))
                        
                        leader_candidates = [
                            node for node in community
                            if self.policy_agents[node].opinion_leadership > 0.7
                        ]
                        
                        action_groups.append({
                            'group_id': f'resistance_group_{i}',
                            'members': list(community),
                            'size': len(community),
                            'centrality': group_centrality,
                            'cohesion': group_cohesion,
                            'potential_leaders': leader_candidates,
                            'action_type': 'resistance'
                        })
            except:
                components = list(nx.connected_components(resisters_subgraph))
                for i, component in enumerate(components):
                    if len(component) >= 3:
                        action_groups.append({
                            'group_id': f'resistance_component_{i}',
                            'members': list(component),
                            'size': len(component),
                            'action_type': 'resistance'
                        })
        
        return action_groups
    
    def _calculate_collective_action_probability(self, action_groups: List[Dict]) -> float:
        """计算集体行动发生概率"""
        if not action_groups:
            return 0.0
        
        total_probability = 0.0
        for group in action_groups:
            size_factor = min(1.0, group['size'] / 50.0)
            cohesion_factor = group.get('cohesion', 0.5)
            leadership_factor = min(1.0, len(group.get('potential_leaders', [])) / 3.0)
            
            group_probability = (size_factor * 0.4 + cohesion_factor * 0.4 + leadership_factor * 0.2)
            total_probability += group_probability
        
        return min(1.0, total_probability / len(action_groups))
    
    def _predict_collective_action_impact(self, action_groups: List[Dict], policy: PolicyIntervention) -> Dict:
        """预测集体行动的影响"""
        if not action_groups:
            return {'impact_level': 'none', 'affected_population': 0, 'policy_disruption': 0.0}
        
        total_participants = sum(group['size'] for group in action_groups)
        total_population = len(policy.target_population)
        
        direct_impact_ratio = total_participants / total_population
        
        indirect_multiplier = 1.0
        for group in action_groups:
            group_centrality = group.get('centrality', 0.5)
            indirect_multiplier += group_centrality * group['size'] / 100.0
        
        total_impact_ratio = min(1.0, direct_impact_ratio * indirect_multiplier)
        policy_disruption = min(1.0, total_impact_ratio * 2.0)
        
        if total_impact_ratio < 0.1:
            impact_level = 'low'
        elif total_impact_ratio < 0.3:
            impact_level = 'medium'
        else:
            impact_level = 'high'
        
        return {
            'impact_level': impact_level,
            'affected_population': int(total_impact_ratio * total_population),
            'policy_disruption': policy_disruption,
            'direct_participants': total_participants,
            'indirect_multiplier': indirect_multiplier
        }
    
    def _assess_collective_action_risks(self, action_groups: List[Dict], impact_prediction: Dict) -> List[str]:
        """评估集体行动风险"""
        risks = []
        
        impact_level = impact_prediction['impact_level']
        if impact_level == 'high':
            risks.extend([
                '政策实施可能面临严重阻力',
                '可能引发大规模抗议或抵制行为',
                '政策目标达成率可能显著下降'
            ])
        elif impact_level == 'medium':
            risks.extend([
                '政策推进速度可能放缓',
                '部分地区或群体可能出现抵制情绪'
            ])
        
        for group in action_groups:
            if group['size'] > 20:
                risks.append(f'发现大规模抵制群体（{group["size"]}人），需重点关注')
            
            if len(group.get('potential_leaders', [])) > 2:
                risks.append('存在多个意见领袖，可能形成有组织的抵制行动')
        
        if len(action_groups) > 3:
            risks.append('多个抵制群体并存，可能形成联合行动')
        
        return risks
    
    def _generate_risk_mitigation_recommendations(self, risks: List[str], action_groups: List[Dict]) -> List[str]:
        """生成风险缓解建议"""
        recommendations = []
        
        if '政策实施可能面临严重阻力' in risks:
            recommendations.extend([
                '建议分阶段实施政策，降低一次性冲击',
                '加强政策解释和沟通工作',
                '考虑设立过渡期和缓冲机制'
            ])
        
        if '可能引发大规模抗议或抵制行为' in risks:
            recommendations.extend([
                '提前制定应急预案和危机管理方案',
                '建立多渠道沟通机制，及时回应关切'
            ])
        
        has_leaders = any(len(group.get('potential_leaders', [])) > 0 for group in action_groups)
        if has_leaders:
            recommendations.extend([
                '重点关注意见领袖，考虑个别沟通和协商',
                '可考虑邀请关键意见领袖参与政策制定过程'
            ])
        
        large_groups = [group for group in action_groups if group['size'] > 15]
        if large_groups:
            recommendations.extend([
                '对大规模抵制群体采用差异化沟通策略',
                '考虑在这些群体集中的区域增加政策支持措施'
            ])
        
        return recommendations
    
    def _generate_resistance_management_recommendations(self, action_groups: List[Dict]) -> List[str]:
        """生成抗拒管理建议"""
        if not action_groups:
            return ['当前无明显抗拒群体，建议保持现有政策推进节奏']
        
        recommendations = []
        
        if len(action_groups) == 1:
            recommendations.append('针对单一抗拒群体，建议采用精准化沟通策略')
        elif len(action_groups) <= 3:
            recommendations.append('存在多个抗拒群体，建议分别制定应对策略')
        else:
            recommendations.append('抗拒群体较多，建议重新评估政策设计的合理性')
        
        total_resisters = sum(group['size'] for group in action_groups)
        if total_resisters > 100:
            recommendations.append('抗拒人数较多，建议暂缓政策实施，先进行充分沟通')
        elif total_resisters > 50:
            recommendations.append('建议增加政策宣传和解释工作的投入')
        
        high_cohesion_groups = [group for group in action_groups if group.get('cohesion', 0) > 0.7]
        if high_cohesion_groups:
            recommendations.extend([
                '发现高凝聚力抗拒群体，需要特别关注其动向',
                '建议通过群体内部关键人物进行间接影响'
            ])
        
        return recommendations
    
    def _generate_network_integration_recommendations(self, network_analysis: Dict) -> List[str]:
        """生成网络整合建议"""
        recommendations = []
        
        density = network_analysis['density']
        clustering = network_analysis['clustering']
        fragmentation = network_analysis['fragmentation']
        
        if density < 0.1:
            recommendations.extend([
                '网络密度较低，建议通过社区活动增强社会联系',
                '可考虑建立更多的沟通桥梁和中介机构'
            ])
        elif density > 0.5:
            recommendations.append('网络密度较高，信息传播效率好，可加快政策推进')
        
        if clustering < 0.3:
            recommendations.append('社会聚类程度较低，建议加强基层组织建设')
        elif clustering > 0.7:
            recommendations.append('存在明显的社会群体分化，需要跨群体的整合措施')
        
        if fragmentation > 0.3:
            recommendations.extend([
                '网络碎片化程度较高，建议建立跨群体的协调机制',
                '可考虑通过关键节点进行网络整合'
            ])
        
        key_nodes_count = network_analysis['key_nodes_count']
        if key_nodes_count < 5:
            recommendations.append('关键节点较少，建议培养更多的社区领袖和意见领袖')
        elif key_nodes_count > 20:
            recommendations.append('关键节点较多，可以充分利用这些节点进行政策传播')
        
        return recommendations
    
    def _generate_differential_strategy_recommendations(self, policy: PolicyIntervention) -> List[str]:
        """生成差序化策略建议"""
        recommendations = []
        
        layer_adoption = {}
        for layer_type, layer_graph in self.social_network.layers.items():
            layer_agents = set(layer_graph.nodes()).intersection(policy.target_population)
            if layer_agents:
                adopted_in_layer = layer_agents.intersection(policy.current_adopters)
                layer_adoption[layer_type.value] = len(adopted_in_layer) / len(layer_agents)
        
        if layer_adoption:
            sorted_layers = sorted(layer_adoption.items(), key=lambda x: x[1])
            
            lowest_layer, lowest_rate = sorted_layers[0]
            if lowest_rate < 0.3:
                recommendations.append(f'{lowest_layer}关系网络中的采纳率较低（{lowest_rate:.1%}），需要加强针对性措施')
                
                layer_strategies = {
                    'kinship': '建议通过家庭和亲属网络进行政策宣传',
                    'geographic': '建议加强社区和邻里层面的政策推广',
                    'professional': '建议通过行业协会和职业网络进行政策传播',
                    'educational': '建议利用校友网络和教育机构进行政策推广',
                    'social': '建议通过社交活动和兴趣群体进行政策传播'
                }
                
                if lowest_layer in layer_strategies:
                    recommendations.append(layer_strategies[lowest_layer])
            
            highest_layer, highest_rate = sorted_layers[-1]
            if highest_rate > 0.7:
                recommendations.append(f'{highest_layer}关系网络中的采纳率较高（{highest_rate:.1%}），可以作为政策推广的重点渠道')
        
        recommendations.extend([
            '建议采用"由近及远"的差序化推广策略',
            '优先通过强关系网络建立政策信任，再向弱关系网络扩散',
            '考虑设立关系网络中的"政策大使"角色'
        ])
        
        return recommendations
    
    def _generate_critical_alerts(self, policy: PolicyIntervention, collective_action_analysis: Dict) -> List[str]:
        """生成关键警报"""
        alerts = []
        
        action_probability = collective_action_analysis.get('probability', 0)
        if action_probability > 0.7:
            alerts.append('🚨 高风险警报：集体抵制行动发生概率很高，建议立即采取预防措施')
        elif action_probability > 0.5:
            alerts.append('⚠️ 中风险警报：存在集体抵制行动风险，建议加强监控和沟通')
        
        impact_prediction = collective_action_analysis.get('impact_prediction', {})
        if impact_prediction.get('impact_level') == 'high':
            alerts.append('🚨 严重影响警报：预计集体行动将严重影响政策实施')
            affected_pop = impact_prediction.get('affected_population', 0)
            alerts.append(f'📊 影响范围：预计影响人口 {affected_pop} 人')
        
        if hasattr(policy, 'current_adopters') and hasattr(policy, 'target_population'):
            adoption_rate = len(policy.current_adopters) / len(policy.target_population)
            if adoption_rate < 0.2:
                alerts.append('📉 低采纳率警报：政策采纳率过低，需要调整推广策略')
        
        if hasattr(policy, 'resistance_groups') and policy.resistance_groups:
            resistance_rate = len(policy.resistance_groups) / len(policy.target_population)
            if resistance_rate > 0.3:
                alerts.append('🛑 高抗拒率警报：抗拒人群比例过高，建议重新评估政策设计')
        
        return alerts
    
    def create_example_policy_intervention(self) -> PolicyIntervention:
        """创建示例政策干预"""
        policy = PolicyIntervention(
            policy_id="flood_evacuation_policy_001",
            policy_type=PolicyType.EMERGENCY_RESPONSE,
            target_population=set(self.policy_agents.keys()),
            policy_content={
                'title': '洪灾紧急疏散政策',
                'description': '在洪灾预警发布后，居民应立即按照指定路线疏散至安全区域',
                'evacuation_routes': ['route_A', 'route_B', 'route_C'],
                'safe_zones': ['zone_1', 'zone_2', 'zone_3'],
                'emergency_contacts': ['110', '119', '120'],
                'required_actions': [
                    '收听官方通知',
                    '准备应急物品',
                    '按指定路线疏散',
                    '到达安全区域后报告'
                ]
            },
            implementation_channels=[
                PropagationChannel.OFFICIAL_MEDIA,
                PropagationChannel.SOCIAL_NETWORK,
                PropagationChannel.COMMUNITY_LEADER
            ],
            priority_level=0.9,
            urgency_level=0.8
        )
        
        return policy