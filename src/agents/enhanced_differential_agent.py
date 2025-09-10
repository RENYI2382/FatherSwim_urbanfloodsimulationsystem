#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的差序格局智能体模型
基于文献理论整合：差序格局理论 + 圈子理论 + ABM科技人文融合框架

主要特性：
1. 多层次关系网络建模（血缘、地缘、业缘、社缘）
2. C型决策机制（圈内人情化，圈外制度化）
3. 三重动机结构（成就、权力、亲和需要）
4. 情境化行为切换（平时、预警、危机、恢复）
5. 自适应学习与记忆机制
"""

import random
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import networkx as nx
from collections import defaultdict
import json


class RelationshipType(Enum):
    """关系类型枚举 - 基于差序格局理论"""
    KINSHIP = "血缘"      # 家族关系 - 最稳定，影响力最强
    GEOGRAPHIC = "地缘"   # 邻里关系 - 基于居住距离
    PROFESSIONAL = "业缘" # 工作关系 - 基于职业合作
    EDUCATIONAL = "学缘"  # 同学关系 - 基于教育经历
    SOCIAL = "社缘"       # 社交关系 - 基于兴趣爱好


class DisasterPhase(Enum):
    """灾害阶段枚举 - 情境化行为切换"""
    NORMAL = "平时"       # 日常状态
    WARNING = "预警"      # 预警状态
    CRISIS = "危机"       # 危机状态
    RECOVERY = "恢复"     # 恢复状态


class CircleType(Enum):
    """圈子类型枚举 - 基于圈子理论"""
    FAMILY = "家庭圈"     # 家庭核心圈
    KINSHIP = "亲族圈"    # 扩展亲属圈
    NEIGHBOR = "邻里圈"   # 邻里社区圈
    WORK = "工作圈"       # 职业工作圈
    FRIEND = "朋友圈"     # 社交朋友圈
    INTEREST = "兴趣圈"   # 兴趣爱好圈


@dataclass
class RelationshipEdge:
    """关系边数据结构"""
    source_id: int
    target_id: int
    relationship_type: RelationshipType
    intimacy_score: float  # 亲疏程度 [0, 1]
    influence_weight: float  # 影响力权重 [0, 1]
    interaction_frequency: float  # 交互频率 [0, 1]
    trust_level: float  # 信任程度 [0, 1]
    reciprocity_balance: float  # 互惠平衡 [-1, 1]
    last_interaction_time: int  # 最后交互时间
    relationship_duration: int  # 关系持续时间


@dataclass
class SocialCircle:
    """社会圈子数据结构"""
    circle_id: str
    circle_type: CircleType
    leader_id: Optional[int]  # 圈子领导者
    members: Set[int]  # 圈子成员
    cohesion_level: float  # 凝聚力水平 [0, 1]
    influence_radius: float  # 影响半径
    decision_autonomy: float  # 决策自主性 [0, 1]
    resource_sharing_level: float  # 资源共享水平 [0, 1]
    collective_memory: Dict  # 集体记忆


class DifferentialNetwork:
    """差序格局网络结构"""
    
    def __init__(self):
        self.relationships: Dict[Tuple[int, int], RelationshipEdge] = {}
        self.circles: Dict[str, SocialCircle] = {}
        self.agent_circles: Dict[int, Set[str]] = defaultdict(set)
        
        # 多层网络图
        self.kinship_layer = nx.Graph()
        self.geographic_layer = nx.Graph()
        self.professional_layer = nx.Graph()
        self.educational_layer = nx.Graph()
        self.social_layer = nx.Graph()
        
        self.layer_weights = {
            RelationshipType.KINSHIP: 0.4,      # 血缘关系权重最高
            RelationshipType.GEOGRAPHIC: 0.25,   # 地缘关系次之
            RelationshipType.PROFESSIONAL: 0.2,  # 业缘关系
            RelationshipType.EDUCATIONAL: 0.1,   # 学缘关系
            RelationshipType.SOCIAL: 0.05        # 社缘关系权重最低
        }
    
    def add_relationship(self, edge: RelationshipEdge):
        """添加关系边"""
        key = (min(edge.source_id, edge.target_id), max(edge.source_id, edge.target_id))
        self.relationships[key] = edge
        
        # 添加到对应的网络层
        if edge.relationship_type == RelationshipType.KINSHIP:
            self.kinship_layer.add_edge(edge.source_id, edge.target_id, weight=edge.influence_weight)
        elif edge.relationship_type == RelationshipType.GEOGRAPHIC:
            self.geographic_layer.add_edge(edge.source_id, edge.target_id, weight=edge.influence_weight)
        elif edge.relationship_type == RelationshipType.PROFESSIONAL:
            self.professional_layer.add_edge(edge.source_id, edge.target_id, weight=edge.influence_weight)
        elif edge.relationship_type == RelationshipType.EDUCATIONAL:
            self.educational_layer.add_edge(edge.source_id, edge.target_id, weight=edge.influence_weight)
        elif edge.relationship_type == RelationshipType.SOCIAL:
            self.social_layer.add_edge(edge.source_id, edge.target_id, weight=edge.influence_weight)
    
    def get_relationship_strength(self, agent1_id: int, agent2_id: int) -> float:
        """计算两个智能体之间的综合关系强度"""
        key = (min(agent1_id, agent2_id), max(agent1_id, agent2_id))
        if key not in self.relationships:
            return 0.0
        
        edge = self.relationships[key]
        base_strength = edge.intimacy_score * edge.influence_weight * edge.trust_level
        
        # 基于交互频率的动态调整
        frequency_bonus = edge.interaction_frequency * 0.2
        
        # 基于互惠平衡的调整
        reciprocity_adjustment = abs(edge.reciprocity_balance) * 0.1
        
        return min(1.0, base_strength + frequency_bonus - reciprocity_adjustment)
    
    def get_circle_influence(self, agent_id: int, circle_id: str) -> float:
        """计算智能体在特定圈子中的影响力"""
        if circle_id not in self.circles or agent_id not in self.circles[circle_id].members:
            return 0.0
        
        circle = self.circles[circle_id]
        
        # 领导者影响力最高
        if circle.leader_id == agent_id:
            return 1.0
        
        # 基于与圈子成员的关系强度计算影响力
        total_strength = 0.0
        member_count = 0
        
        for member_id in circle.members:
            if member_id != agent_id:
                strength = self.get_relationship_strength(agent_id, member_id)
                total_strength += strength
                member_count += 1
        
        if member_count == 0:
            return 0.0
        
        average_strength = total_strength / member_count
        return average_strength * circle.cohesion_level


class ChineseDecisionModel:
    """C型决策模型 - 基于圈子理论"""
    
    def __init__(self):
        self.decision_history = []
        self.consensus_threshold = 0.6  # 圈内共识阈值
        self.leader_weight = 0.4        # 领导者决策权重
        self.emotion_weight = 0.6       # 情感决策权重（圈内）
        self.rule_weight = 0.8          # 规则决策权重（圈外）
    
    def make_evacuation_decision(self, agent, situation: Dict, network: DifferentialNetwork) -> Dict:
        """制定疏散决策"""
        decision_factors = {
            'inner_circle_consensus': 0.0,
            'leader_influence': 0.0,
            'rule_based_logic': 0.0,
            'emotion_based_logic': 0.0,
            'final_decision': False,
            'confidence_level': 0.0
        }
        
        # 第一步：圈内共识寻求
        inner_circle_opinion = self._consult_inner_circle(agent, network, situation)
        decision_factors['inner_circle_consensus'] = inner_circle_opinion
        
        # 第二步：领导决策权重
        leader_influence = self._get_leader_influence(agent, network, situation)
        decision_factors['leader_influence'] = leader_influence
        
        # 第三步：情境化控制
        if self._is_inner_circle_context(agent, situation):
            # 圈内：情感化决策
            emotion_score = self._emotion_based_decision(inner_circle_opinion, leader_influence)
            decision_factors['emotion_based_logic'] = emotion_score
            
            final_score = (
                inner_circle_opinion * 0.4 +
                leader_influence * 0.3 +
                emotion_score * 0.3
            )
            decision_factors['confidence_level'] = 0.8  # 圈内决策信心较高
            
        else:
            # 圈外：规则化决策
            rule_score = self._rule_based_decision(situation)
            decision_factors['rule_based_logic'] = rule_score
            
            final_score = (
                rule_score * 0.6 +
                leader_influence * 0.2 +
                inner_circle_opinion * 0.2
            )
            decision_factors['confidence_level'] = 0.6  # 圈外决策信心中等
        
        decision_factors['final_decision'] = final_score > 0.5
        
        # 记录决策历史
        self.decision_history.append({
            'timestamp': situation.get('current_time', 0),
            'decision': decision_factors['final_decision'],
            'factors': decision_factors.copy()
        })
        
        return decision_factors
    
    def _consult_inner_circle(self, agent, network: DifferentialNetwork, situation: Dict) -> float:
        """咨询内圈意见"""
        inner_circles = [cid for cid in agent.social_circles 
                        if network.circles[cid].circle_type in [CircleType.FAMILY, CircleType.KINSHIP]]
        
        if not inner_circles:
            return 0.5  # 默认中性意见
        
        total_opinion = 0.0
        total_weight = 0.0
        
        for circle_id in inner_circles:
            circle = network.circles[circle_id]
            circle_opinion = self._get_circle_opinion(circle, situation)
            circle_weight = circle.cohesion_level * len(circle.members)
            
            total_opinion += circle_opinion * circle_weight
            total_weight += circle_weight
        
        return total_opinion / total_weight if total_weight > 0 else 0.5
    
    def _get_leader_influence(self, agent, network: DifferentialNetwork, situation: Dict) -> float:
        """获取领导者影响"""
        leader_influences = []
        
        for circle_id in agent.social_circles:
            circle = network.circles[circle_id]
            if circle.leader_id and circle.leader_id != agent.agent_id:
                # 模拟领导者的决策倾向
                leader_decision_tendency = random.uniform(0.3, 0.9)  # 领导者通常更倾向于行动
                influence_strength = network.get_circle_influence(circle.leader_id, circle_id)
                
                leader_influences.append(leader_decision_tendency * influence_strength)
        
        return np.mean(leader_influences) if leader_influences else 0.5
    
    def _is_inner_circle_context(self, agent, situation: Dict) -> bool:
        """判断是否为圈内情境"""
        # 基于情境特征判断
        family_nearby = situation.get('family_members_nearby', 0) > 0
        close_friends_nearby = situation.get('close_friends_nearby', 0) > 0
        familiar_environment = situation.get('familiarity_level', 0) > 0.7
        
        return family_nearby or close_friends_nearby or familiar_environment
    
    def _emotion_based_decision(self, inner_opinion: float, leader_influence: float) -> float:
        """情感化决策逻辑"""
        # 情感决策更重视人际关系和群体和谐
        emotion_factors = {
            'group_harmony': inner_opinion,
            'authority_respect': leader_influence,
            'risk_aversion': random.uniform(0.2, 0.8),  # 情感决策风险厌恶程度变化大
            'collective_benefit': random.uniform(0.4, 0.9)  # 更关注集体利益
        }
        
        return np.mean(list(emotion_factors.values()))
    
    def _rule_based_decision(self, situation: Dict) -> float:
        """规则化决策逻辑"""
        # 规则决策更重视客观条件和制度规范
        rule_factors = {
            'official_warning': situation.get('official_warning_level', 0),
            'objective_risk': situation.get('flood_risk_level', 0),
            'evacuation_order': situation.get('evacuation_order', 0),
            'resource_availability': situation.get('evacuation_resources', 0.5),
            'legal_compliance': 0.8  # 规则决策更重视法律合规
        }
        
        return np.mean(list(rule_factors.values()))
    
    def _get_circle_opinion(self, circle: SocialCircle, situation: Dict) -> float:
        """获取圈子整体意见"""
        # 基于圈子类型和情境生成意见
        base_opinion = 0.5
        
        if circle.circle_type == CircleType.FAMILY:
            # 家庭圈更关注安全，倾向于保守决策
            base_opinion = 0.7
        elif circle.circle_type == CircleType.WORK:
            # 工作圈更理性，基于客观条件
            base_opinion = situation.get('objective_risk', 0.5)
        elif circle.circle_type == CircleType.NEIGHBOR:
            # 邻里圈受环境影响大
            base_opinion = situation.get('neighborhood_evacuation_rate', 0.5)
        
        # 圈子凝聚力影响意见一致性
        opinion_variance = (1 - circle.cohesion_level) * 0.3
        final_opinion = np.clip(
            base_opinion + random.uniform(-opinion_variance, opinion_variance),
            0.0, 1.0
        )
        
        return final_opinion


class EnhancedDifferentialAgent:
    """增强的差序格局智能体"""
    
    def __init__(self, agent_id: int, **kwargs):
        self.agent_id = agent_id
        
        # 基于圈子理论的三重动机结构
        self.achievement_need = kwargs.get('achievement_need', random.uniform(0.3, 0.9))
        self.power_need = kwargs.get('power_need', random.uniform(0.2, 0.8))
        self.affiliation_need = kwargs.get('affiliation_need', random.uniform(0.4, 0.9))
        
        # 基于ABM框架的适应性参数
        self.learning_rate = kwargs.get('learning_rate', random.uniform(0.01, 0.1))
        self.risk_tolerance = kwargs.get('risk_tolerance', random.uniform(0.2, 0.8))
        self.social_influence_sensitivity = kwargs.get('social_influence_sensitivity', random.uniform(0.3, 0.9))
        
        # 差序格局网络
        self.social_circles: Set[str] = set()
        self.relationship_network = DifferentialNetwork()
        
        # C型决策模型
        self.decision_model = ChineseDecisionModel()
        
        # 状态变量
        self.current_phase = DisasterPhase.NORMAL
        self.location = kwargs.get('location', (0, 0))
        self.health_status = kwargs.get('health_status', 1.0)
        self.resource_level = kwargs.get('resource_level', 0.5)
        self.stress_level = 0.0
        
        # 记忆与学习
        self.experience_memory = []
        self.collective_memory = {}  # 与圈子共享的集体记忆
        self.adaptation_history = []
        
        # 行为统计
        self.behavior_stats = {
            'evacuation_decisions': 0,
            'help_given': 0,
            'help_received': 0,
            'circle_interactions': 0,
            'leadership_actions': 0
        }
    
    def update_phase(self, new_phase: DisasterPhase, situation: Dict):
        """更新灾害阶段并调整行为模式"""
        old_phase = self.current_phase
        self.current_phase = new_phase
        
        # 阶段转换时的行为调整
        if old_phase != new_phase:
            self._adapt_to_phase_change(old_phase, new_phase, situation)
    
    def make_decision(self, situation: Dict) -> Dict:
        """制定决策 - 整合多种理论框架"""
        # 基于当前阶段选择决策模式
        if self.current_phase == DisasterPhase.NORMAL:
            return self._normal_phase_decision(situation)
        elif self.current_phase == DisasterPhase.WARNING:
            return self._warning_phase_decision(situation)
        elif self.current_phase == DisasterPhase.CRISIS:
            return self._crisis_phase_decision(situation)
        elif self.current_phase == DisasterPhase.RECOVERY:
            return self._recovery_phase_decision(situation)
    
    def _crisis_phase_decision(self, situation: Dict) -> Dict:
        """危机阶段决策 - 基于差序格局和圈子理论"""
        # 使用C型决策模型
        decision_result = self.decision_model.make_evacuation_decision(
            self, situation, self.relationship_network
        )
        
        # 基于三重动机调整决策
        motivation_adjustment = self._calculate_motivation_adjustment(situation)
        
        # 社会影响调整
        social_influence = self._calculate_social_influence(situation)
        
        # 最终决策整合
        final_decision = {
            'action_type': 'evacuation' if decision_result['final_decision'] else 'stay',
            'confidence': decision_result['confidence_level'],
            'motivation_factors': motivation_adjustment,
            'social_factors': social_influence,
            'decision_rationale': self._generate_decision_rationale(decision_result)
        }
        
        # 更新行为统计
        self.behavior_stats['evacuation_decisions'] += 1
        
        # 学习与适应
        self._update_learning(situation, final_decision)
        
        return final_decision
    
    def _calculate_motivation_adjustment(self, situation: Dict) -> Dict:
        """计算三重动机调整因子"""
        # 成就需要：追求个人成功和卓越
        achievement_factor = self.achievement_need * situation.get('success_opportunity', 0.5)
        
        # 权力需要：追求影响力和控制力
        power_factor = self.power_need * situation.get('leadership_opportunity', 0.5)
        
        # 亲和需要：追求人际和谐和归属感
        affiliation_factor = self.affiliation_need * situation.get('group_cohesion', 0.5)
        
        return {
            'achievement': achievement_factor,
            'power': power_factor,
            'affiliation': affiliation_factor,
            'dominant_motivation': max(
                ('achievement', achievement_factor),
                ('power', power_factor),
                ('affiliation', affiliation_factor),
                key=lambda x: x[1]
            )[0]
        }
    
    def _calculate_social_influence(self, situation: Dict) -> Dict:
        """计算社会影响因子"""
        influences = {
            'kinship_influence': 0.0,
            'geographic_influence': 0.0,
            'professional_influence': 0.0,
            'circle_pressure': 0.0,
            'leader_guidance': 0.0
        }
        
        # 计算各类关系的影响
        for circle_id in self.social_circles:
            circle = self.relationship_network.circles.get(circle_id)
            if not circle:
                continue
            
            circle_influence = self.relationship_network.get_circle_influence(self.agent_id, circle_id)
            
            if circle.circle_type == CircleType.FAMILY or circle.circle_type == CircleType.KINSHIP:
                influences['kinship_influence'] += circle_influence
            elif circle.circle_type == CircleType.NEIGHBOR:
                influences['geographic_influence'] += circle_influence
            elif circle.circle_type == CircleType.WORK:
                influences['professional_influence'] += circle_influence
            
            # 圈子压力
            if circle.cohesion_level > 0.7:
                influences['circle_pressure'] += circle_influence * circle.cohesion_level
            
            # 领导者指导
            if circle.leader_id and circle.leader_id != self.agent_id:
                influences['leader_guidance'] += circle_influence * 0.8
        
        # 归一化处理
        max_influence = max(influences.values()) if influences.values() else 1.0
        if max_influence > 0:
            influences = {k: v/max_influence for k, v in influences.items()}
        
        return influences
    
    def _generate_decision_rationale(self, decision_result: Dict) -> str:
        """生成决策理由"""
        rationales = []
        
        if decision_result['inner_circle_consensus'] > 0.6:
            rationales.append("家人朋友都建议这样做")
        
        if decision_result['leader_influence'] > 0.7:
            rationales.append("圈子里有威望的人这样建议")
        
        if decision_result['emotion_based_logic'] > 0.6:
            rationales.append("从情理上考虑这是对的")
        
        if decision_result['rule_based_logic'] > 0.6:
            rationales.append("按照规定应该这样做")
        
        if not rationales:
            rationales.append("综合考虑各种因素")
        
        return "；".join(rationales)
    
    def _update_learning(self, situation: Dict, decision: Dict):
        """更新学习和适应"""
        # 记录经验
        experience = {
            'timestamp': situation.get('current_time', 0),
            'situation_features': situation.copy(),
            'decision': decision.copy(),
            'outcome': None  # 将在后续更新
        }
        
        self.experience_memory.append(experience)
        
        # 限制记忆长度
        if len(self.experience_memory) > 100:
            self.experience_memory.pop(0)
        
        # 适应性调整
        if len(self.experience_memory) > 10:
            self._adaptive_parameter_adjustment()
    
    def _adaptive_parameter_adjustment(self):
        """自适应参数调整"""
        # 基于历史经验调整参数
        recent_experiences = self.experience_memory[-10:]
        
        # 分析决策成功率
        successful_decisions = sum(1 for exp in recent_experiences 
                                 if exp.get('outcome', {}).get('success', False))
        success_rate = successful_decisions / len(recent_experiences)
        
        # 调整风险容忍度
        if success_rate > 0.7:
            self.risk_tolerance = min(1.0, self.risk_tolerance + self.learning_rate)
        elif success_rate < 0.3:
            self.risk_tolerance = max(0.0, self.risk_tolerance - self.learning_rate)
        
        # 调整社会影响敏感性
        social_decisions = sum(1 for exp in recent_experiences 
                             if exp.get('decision', {}).get('social_factors', {}))
        if social_decisions > 7:  # 过度依赖社会影响
            self.social_influence_sensitivity = max(0.1, self.social_influence_sensitivity - self.learning_rate)
        
        # 记录适应历史
        self.adaptation_history.append({
            'timestamp': len(self.experience_memory),
            'risk_tolerance': self.risk_tolerance,
            'social_influence_sensitivity': self.social_influence_sensitivity,
            'success_rate': success_rate
        })
    
    def _normal_phase_decision(self, situation: Dict) -> Dict:
        """平时阶段决策"""
        return {
            'action_type': 'routine',
            'confidence': 0.8,
            'motivation_factors': self._calculate_motivation_adjustment(situation),
            'social_factors': {'routine_social_interaction': 0.5},
            'decision_rationale': '日常生活状态'
        }
    
    def _warning_phase_decision(self, situation: Dict) -> Dict:
        """预警阶段决策"""
        # 预警阶段更多依赖圈子共识
        circle_consensus = self._get_circle_consensus_on_preparation(situation)
        
        return {
            'action_type': 'prepare' if circle_consensus > 0.5 else 'monitor',
            'confidence': 0.6,
            'motivation_factors': self._calculate_motivation_adjustment(situation),
            'social_factors': {'circle_consensus': circle_consensus},
            'decision_rationale': '根据圈子里的讨论和建议'
        }
    
    def _recovery_phase_decision(self, situation: Dict) -> Dict:
        """恢复阶段决策"""
        # 恢复阶段重视集体重建和互助
        mutual_aid_opportunity = situation.get('mutual_aid_opportunity', 0.5)
        
        return {
            'action_type': 'rebuild' if mutual_aid_opportunity > 0.4 else 'recover',
            'confidence': 0.7,
            'motivation_factors': self._calculate_motivation_adjustment(situation),
            'social_factors': {'mutual_aid': mutual_aid_opportunity},
            'decision_rationale': '大家一起重建家园'
        }
    
    def _get_circle_consensus_on_preparation(self, situation: Dict) -> float:
        """获取圈子对准备工作的共识"""
        consensus_scores = []
        
        for circle_id in self.social_circles:
            circle = self.relationship_network.circles.get(circle_id)
            if circle:
                # 模拟圈子对准备工作的态度
                if circle.circle_type == CircleType.FAMILY:
                    consensus_scores.append(0.8)  # 家庭圈通常支持准备
                elif circle.circle_type == CircleType.NEIGHBOR:
                    consensus_scores.append(situation.get('neighborhood_preparation_level', 0.5))
                else:
                    consensus_scores.append(0.6)  # 其他圈子中等支持
        
        return np.mean(consensus_scores) if consensus_scores else 0.5
    
    def _adapt_to_phase_change(self, old_phase: DisasterPhase, new_phase: DisasterPhase, situation: Dict):
        """适应阶段变化"""
        # 压力水平调整
        if new_phase == DisasterPhase.CRISIS:
            self.stress_level = min(1.0, self.stress_level + 0.3)
        elif new_phase == DisasterPhase.RECOVERY:
            self.stress_level = max(0.0, self.stress_level - 0.2)
        
        # 社会影响敏感性调整
        if new_phase == DisasterPhase.CRISIS:
            # 危机时更依赖社会网络
            self.social_influence_sensitivity = min(1.0, self.social_influence_sensitivity + 0.1)
        
        # 记录阶段转换
        self.experience_memory.append({
            'type': 'phase_transition',
            'from_phase': old_phase.value,
            'to_phase': new_phase.value,
            'timestamp': situation.get('current_time', 0),
            'stress_adjustment': self.stress_level
        })
    
    def interact_with_agent(self, other_agent, interaction_type: str, situation: Dict) -> Dict:
        """与其他智能体交互"""
        # 基于关系强度决定交互效果
        relationship_strength = self.relationship_network.get_relationship_strength(
            self.agent_id, other_agent.agent_id
        )
        
        interaction_result = {
            'success': relationship_strength > 0.3,
            'influence_exchanged': relationship_strength * 0.5,
            'trust_change': 0.0,
            'reciprocity_update': 0.0
        }
        
        # 更新交互统计
        self.behavior_stats['circle_interactions'] += 1
        
        if interaction_type == 'help_request':
            if interaction_result['success']:
                self.behavior_stats['help_received'] += 1
                interaction_result['trust_change'] = 0.1
                interaction_result['reciprocity_update'] = -0.1  # 欠人情
        
        elif interaction_type == 'help_offer':
            if interaction_result['success']:
                self.behavior_stats['help_given'] += 1
                interaction_result['trust_change'] = 0.1
                interaction_result['reciprocity_update'] = 0.1   # 积累人情
        
        return interaction_result
    
    def get_agent_summary(self) -> Dict:
        """获取智能体摘要信息"""
        return {
            'agent_id': self.agent_id,
            'current_phase': self.current_phase.value,
            'motivation_profile': {
                'achievement_need': self.achievement_need,
                'power_need': self.power_need,
                'affiliation_need': self.affiliation_need
            },
            'adaptation_profile': {
                'learning_rate': self.learning_rate,
                'risk_tolerance': self.risk_tolerance,
                'social_influence_sensitivity': self.social_influence_sensitivity
            },
            'social_network': {
                'circles_count': len(self.social_circles),
                'total_relationships': len(self.relationship_network.relationships)
            },
            'behavior_stats': self.behavior_stats.copy(),
            'current_state': {
                'location': self.location,
                'health_status': self.health_status,
                'stress_level': self.stress_level,
                'resource_level': self.resource_level
            }
        }


def create_enhanced_agent_population(size: int, network_config: Dict = None) -> List[EnhancedDifferentialAgent]:
    """创建增强智能体群体"""
    agents = []
    
    for i in range(size):
        # 生成多样化的个性参数
        agent_params = {
            'achievement_need': np.random.beta(2, 2),  # 偏向中等成就需要
            'power_need': np.random.beta(1.5, 3),      # 偏向较低权力需要
            'affiliation_need': np.random.beta(3, 1.5), # 偏向较高亲和需要（符合中国文化）
            'learning_rate': np.random.uniform(0.01, 0.1),
            'risk_tolerance': np.random.beta(2, 3),     # 偏向风险厌恶
            'social_influence_sensitivity': np.random.beta(3, 2), # 偏向高社会影响敏感性
            'location': (random.uniform(0, 100), random.uniform(0, 100)),
            'health_status': np.random.beta(8, 2),      # 大多数人健康状况良好
            'resource_level': np.random.beta(3, 3)      # 资源水平正态分布
        }
        
        agent = EnhancedDifferentialAgent(i, **agent_params)
        agents.append(agent)
    
    # 构建社会网络
    if network_config:
        _build_social_network(agents, network_config)
    
    return agents


def _build_social_network(agents: List[EnhancedDifferentialAgent], config: Dict):
    """构建社会网络结构"""
    # 这里可以实现复杂的网络构建逻辑
    # 基于差序格局理论构建多层次关系网络
    pass


if __name__ == "__main__":
    # 测试代码
    print("创建增强的差序格局智能体群体...")
    agents = create_enhanced_agent_population(10)
    
    print(f"成功创建 {len(agents)} 个智能体")
    
    # 测试智能体决策
    test_situation = {
        'current_time': 100,
        'flood_risk_level': 0.7,
        'official_warning_level': 0.8,
        'family_members_nearby': 2,
        'evacuation_resources': 0.6
    }
    
    agent = agents[0]
    agent.update_phase(DisasterPhase.CRISIS, test_situation)
    decision = agent.make_decision(test_situation)
    
    print("\n智能体决策测试:")
    print(f"决策类型: {decision['action_type']}")
    print(f"决策信心: {decision['confidence']:.2f}")
    print(f"决策理由: {decision['decision_rationale']}")
    
    print("\n智能体摘要:")
    summary = agent.get_agent_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))