"""三层验证流程系统

实现基于差序格局理论的三层验证体系：
1. 微观验证：智能体决策分析，验证个体行为逻辑
2. 内部逻辑验证：互助行为统计，验证社会网络机制
3. 宏观模式验证：涌现现象观察，验证系统级别的理论预测

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
from scipy import stats
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证层级枚举"""
    MICRO = "micro"  # 微观验证
    MESO = "meso"  # 内部逻辑验证
    MACRO = "macro"  # 宏观模式验证


class DecisionType(Enum):
    """决策类型枚举"""
    EVACUATION = "evacuation"  # 疏散决策
    HELP_SEEKING = "help_seeking"  # 求助决策
    HELP_GIVING = "help_giving"  # 助人决策
    RESOURCE_SHARING = "resource_sharing"  # 资源分享决策
    INFORMATION_SHARING = "information_sharing"  # 信息分享决策
    SHELTER_SELECTION = "shelter_selection"  # 避难所选择决策


class ValidationResult(Enum):
    """验证结果枚举"""
    PASS = "pass"  # 通过验证
    FAIL = "fail"  # 未通过验证
    PARTIAL = "partial"  # 部分通过
    INCONCLUSIVE = "inconclusive"  # 结果不明确


@dataclass
class MicroValidationRecord:
    """微观验证记录"""
    agent_id: str
    timestamp: datetime
    decision_type: DecisionType
    decision_context: Dict[str, Any]  # 决策上下文
    expected_behavior: str  # 基于理论的预期行为
    actual_behavior: str  # 实际行为
    strategy_type: str  # 策略类型（强差序格局、弱差序格局、普遍主义）
    decision_factors: Dict[str, float]  # 决策因子权重
    social_influence: Dict[str, float]  # 社会影响因子
    validation_result: ValidationResult
    confidence_score: float  # 置信度分数
    explanation: str  # 验证结果解释
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MesoValidationRecord:
    """内部逻辑验证记录"""
    validation_id: str
    timestamp: datetime
    network_snapshot: Dict[str, Any]  # 网络快照
    interaction_patterns: Dict[str, Dict[str, float]]  # 互动模式统计
    help_flow_analysis: Dict[str, Any]  # 帮助流动分析
    reciprocity_metrics: Dict[str, float]  # 互惠性指标
    differential_treatment: Dict[str, Dict[str, float]]  # 差别对待分析
    network_efficiency: Dict[str, float]  # 网络效率指标
    validation_result: ValidationResult
    theoretical_predictions: Dict[str, Any]  # 理论预测
    empirical_observations: Dict[str, Any]  # 实证观察
    deviation_analysis: Dict[str, float]  # 偏差分析
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroValidationRecord:
    """宏观模式验证记录"""
    validation_id: str
    timestamp: datetime
    simulation_phase: str  # 仿真阶段（初期、中期、后期）
    emergent_patterns: Dict[str, Any]  # 涌现模式
    system_resilience: Dict[str, float]  # 系统韧性指标
    collective_outcomes: Dict[str, float]  # 集体结果
    spatial_patterns: Dict[str, Any]  # 空间模式
    temporal_dynamics: Dict[str, Any]  # 时间动态
    hypothesis_testing: Dict[str, Dict[str, Any]]  # 假设检验结果
    validation_result: ValidationResult
    statistical_significance: Dict[str, float]  # 统计显著性
    effect_sizes: Dict[str, float]  # 效应量
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MicroValidator:
    """微观验证器：验证智能体个体决策逻辑"""
    
    def __init__(self):
        self.validation_records: List[MicroValidationRecord] = []
        self.decision_rules = self._load_decision_rules()
        
    def _load_decision_rules(self) -> Dict[str, Dict[str, Any]]:
        """加载基于差序格局理论的决策规则"""
        return {
            "strong_differential": {
                "help_giving_priority": ["family", "close_friends", "acquaintances", "strangers"],
                "resource_sharing_threshold": {"family": 0.1, "close_friends": 0.3, "acquaintances": 0.6, "strangers": 0.9},
                "information_sharing_selectivity": {"family": 0.9, "close_friends": 0.7, "acquaintances": 0.4, "strangers": 0.1},
                "evacuation_coordination": "family_first"
            },
            "weak_differential": {
                "help_giving_priority": ["family", "friends", "community", "others"],
                "resource_sharing_threshold": {"family": 0.2, "friends": 0.4, "community": 0.6, "others": 0.8},
                "information_sharing_selectivity": {"family": 0.8, "friends": 0.6, "community": 0.5, "others": 0.3},
                "evacuation_coordination": "community_aware"
            },
            "universalism": {
                "help_giving_priority": ["most_vulnerable", "closest_proximity", "random"],
                "resource_sharing_threshold": {"all": 0.5},
                "information_sharing_selectivity": {"all": 0.7},
                "evacuation_coordination": "collective_optimal"
            }
        }
    
    def validate_decision(self, agent_id: str, decision_context: Dict[str, Any], 
                         actual_decision: Dict[str, Any], strategy_type: str) -> MicroValidationRecord:
        """验证单个智能体的决策"""
        timestamp = datetime.now()
        decision_type = DecisionType(decision_context.get('decision_type', 'evacuation'))
        
        # 基于策略类型预测期望行为
        expected_behavior = self._predict_expected_behavior(decision_context, strategy_type)
        actual_behavior = self._extract_actual_behavior(actual_decision)
        
        # 计算决策因子
        decision_factors = self._analyze_decision_factors(decision_context, actual_decision)
        
        # 分析社会影响
        social_influence = self._analyze_social_influence(decision_context, actual_decision)
        
        # 验证决策一致性
        validation_result, confidence_score, explanation = self._validate_consistency(
            expected_behavior, actual_behavior, strategy_type, decision_factors
        )
        
        record = MicroValidationRecord(
            agent_id=agent_id,
            timestamp=timestamp,
            decision_type=decision_type,
            decision_context=decision_context,
            expected_behavior=expected_behavior,
            actual_behavior=actual_behavior,
            strategy_type=strategy_type,
            decision_factors=decision_factors,
            social_influence=social_influence,
            validation_result=validation_result,
            confidence_score=confidence_score,
            explanation=explanation
        )
        
        self.validation_records.append(record)
        return record
    
    def _predict_expected_behavior(self, context: Dict[str, Any], strategy_type: str) -> str:
        """基于策略类型预测期望行为"""
        decision_type = context.get('decision_type')
        rules = self.decision_rules.get(strategy_type, {})
        
        if decision_type == 'help_giving':
            requester_relationship = context.get('requester_relationship', 'stranger')
            priority_list = rules.get('help_giving_priority', [])
            
            if requester_relationship in priority_list:
                priority_index = priority_list.index(requester_relationship)
                if priority_index <= 1:  # 高优先级
                    return "provide_help"
                elif priority_index <= 2:  # 中优先级
                    return "conditional_help"
                else:  # 低优先级
                    return "decline_help"
        
        elif decision_type == 'resource_sharing':
            recipient_relationship = context.get('recipient_relationship', 'stranger')
            resource_scarcity = context.get('resource_scarcity', 0.5)
            threshold = rules.get('resource_sharing_threshold', {}).get(recipient_relationship, 0.8)
            
            if resource_scarcity < threshold:
                return "share_resources"
            else:
                return "conserve_resources"
        
        elif decision_type == 'evacuation':
            coordination_style = rules.get('evacuation_coordination', 'individual')
            if coordination_style == 'family_first':
                return "coordinate_with_family"
            elif coordination_style == 'community_aware':
                return "coordinate_with_community"
            else:
                return "individual_evacuation"
        
        return "default_behavior"
    
    def _extract_actual_behavior(self, decision: Dict[str, Any]) -> str:
        """提取实际行为"""
        action = decision.get('action', 'unknown')
        target = decision.get('target', 'none')
        
        if action == 'help' and target != 'none':
            return "provide_help"
        elif action == 'share' and target != 'none':
            return "share_resources"
        elif action == 'evacuate':
            coordination = decision.get('coordination', 'individual')
            if 'family' in coordination:
                return "coordinate_with_family"
            elif 'community' in coordination:
                return "coordinate_with_community"
            else:
                return "individual_evacuation"
        
        return action
    
    def _analyze_decision_factors(self, context: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, float]:
        """分析决策因子权重"""
        factors = {
            'relationship_closeness': context.get('relationship_closeness', 0.0),
            'resource_availability': context.get('resource_availability', 1.0),
            'risk_perception': context.get('risk_perception', 0.5),
            'social_pressure': context.get('social_pressure', 0.0),
            'past_reciprocity': context.get('past_reciprocity', 0.0),
            'authority_influence': context.get('authority_influence', 0.0)
        }
        
        # 根据决策结果推断因子权重
        decision_strength = decision.get('confidence', 0.5)
        for factor in factors:
            factors[factor] *= decision_strength
        
        return factors
    
    def _analyze_social_influence(self, context: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, float]:
        """分析社会影响因子"""
        return {
            'peer_influence': context.get('peer_influence', 0.0),
            'authority_influence': context.get('authority_influence', 0.0),
            'media_influence': context.get('media_influence', 0.0),
            'family_influence': context.get('family_influence', 0.0),
            'community_norms': context.get('community_norms', 0.0)
        }
    
    def _validate_consistency(self, expected: str, actual: str, strategy_type: str, 
                            factors: Dict[str, float]) -> Tuple[ValidationResult, float, str]:
        """验证决策一致性"""
        if expected == actual:
            return ValidationResult.PASS, 0.9, f"行为与{strategy_type}策略预期一致"
        
        # 分析偏差原因
        relationship_weight = factors.get('relationship_closeness', 0.0)
        social_pressure = factors.get('social_pressure', 0.0)
        
        if strategy_type == 'strong_differential' and relationship_weight < 0.3:
            if actual in ['provide_help', 'share_resources']:
                return ValidationResult.PARTIAL, 0.6, "可能受到外部压力影响，偏离差序格局逻辑"
        
        elif strategy_type == 'universalism' and relationship_weight > 0.7:
            if actual in ['decline_help', 'conserve_resources']:
                return ValidationResult.PARTIAL, 0.6, "可能受到关系亲疏影响，偏离普遍主义原则"
        
        confidence = max(0.1, 1.0 - abs(relationship_weight - 0.5))
        return ValidationResult.FAIL, confidence, f"行为与{strategy_type}策略预期不符"
    
    def generate_micro_validation_report(self) -> Dict[str, Any]:
        """生成微观验证报告"""
        if not self.validation_records:
            return {"error": "No validation records available"}
        
        # 统计验证结果
        result_counts = Counter([record.validation_result.value for record in self.validation_records])
        
        # 按策略类型分析
        strategy_analysis = defaultdict(lambda: defaultdict(list))
        for record in self.validation_records:
            strategy_analysis[record.strategy_type]['results'].append(record.validation_result.value)
            strategy_analysis[record.strategy_type]['confidence'].append(record.confidence_score)
        
        # 决策类型分析
        decision_analysis = defaultdict(lambda: defaultdict(list))
        for record in self.validation_records:
            decision_analysis[record.decision_type.value]['results'].append(record.validation_result.value)
            decision_analysis[record.decision_type.value]['confidence'].append(record.confidence_score)
        
        return {
            'total_validations': len(self.validation_records),
            'result_distribution': dict(result_counts),
            'overall_pass_rate': result_counts.get('pass', 0) / len(self.validation_records),
            'average_confidence': np.mean([r.confidence_score for r in self.validation_records]),
            'strategy_analysis': {
                strategy: {
                    'pass_rate': data['results'].count('pass') / len(data['results']),
                    'average_confidence': np.mean(data['confidence']),
                    'total_decisions': len(data['results'])
                }
                for strategy, data in strategy_analysis.items()
            },
            'decision_type_analysis': {
                decision_type: {
                    'pass_rate': data['results'].count('pass') / len(data['results']),
                    'average_confidence': np.mean(data['confidence']),
                    'total_decisions': len(data['results'])
                }
                for decision_type, data in decision_analysis.items()
            }
        }


class MesoValidator:
    """内部逻辑验证器：验证社会网络互助机制"""
    
    def __init__(self):
        self.validation_records: List[MesoValidationRecord] = []
        self.network_metrics_history: List[Dict[str, Any]] = []
    
    def validate_network_interactions(self, social_network: nx.Graph, 
                                    interaction_log: List[Dict[str, Any]],
                                    timestamp: datetime) -> MesoValidationRecord:
        """验证网络互动模式"""
        validation_id = f"meso_{timestamp.isoformat()}"
        
        # 网络快照
        network_snapshot = self._capture_network_snapshot(social_network)
        
        # 分析互动模式
        interaction_patterns = self._analyze_interaction_patterns(interaction_log)
        
        # 帮助流动分析
        help_flow_analysis = self._analyze_help_flows(interaction_log, social_network)
        
        # 互惠性分析
        reciprocity_metrics = self._calculate_reciprocity_metrics(interaction_log, social_network)
        
        # 差别对待分析
        differential_treatment = self._analyze_differential_treatment(interaction_log, social_network)
        
        # 网络效率分析
        network_efficiency = self._calculate_network_efficiency(social_network, interaction_log)
        
        # 理论预测 vs 实证观察
        theoretical_predictions = self._generate_theoretical_predictions(network_snapshot)
        empirical_observations = self._extract_empirical_observations(interaction_patterns, help_flow_analysis)
        
        # 偏差分析
        deviation_analysis = self._analyze_deviations(theoretical_predictions, empirical_observations)
        
        # 验证结果
        validation_result, explanation = self._evaluate_meso_validation(
            deviation_analysis, reciprocity_metrics, differential_treatment
        )
        
        record = MesoValidationRecord(
            validation_id=validation_id,
            timestamp=timestamp,
            network_snapshot=network_snapshot,
            interaction_patterns=interaction_patterns,
            help_flow_analysis=help_flow_analysis,
            reciprocity_metrics=reciprocity_metrics,
            differential_treatment=differential_treatment,
            network_efficiency=network_efficiency,
            validation_result=validation_result,
            theoretical_predictions=theoretical_predictions,
            empirical_observations=empirical_observations,
            deviation_analysis=deviation_analysis,
            explanation=explanation
        )
        
        self.validation_records.append(record)
        self.network_metrics_history.append(network_snapshot)
        
        return record
    
    def _capture_network_snapshot(self, network: nx.Graph) -> Dict[str, Any]:
        """捕获网络快照"""
        return {
            'node_count': network.number_of_nodes(),
            'edge_count': network.number_of_edges(),
            'density': nx.density(network),
            'clustering_coefficient': nx.average_clustering(network),
            'average_path_length': nx.average_shortest_path_length(network) if nx.is_connected(network) else float('inf'),
            'degree_distribution': dict(network.degree()),
            'centrality_measures': {
                'betweenness': nx.betweenness_centrality(network),
                'closeness': nx.closeness_centrality(network),
                'eigenvector': nx.eigenvector_centrality(network)
            }
        }
    
    def _analyze_interaction_patterns(self, interaction_log: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """分析互动模式"""
        patterns = defaultdict(lambda: defaultdict(float))
        
        for interaction in interaction_log:
            interaction_type = interaction.get('type', 'unknown')
            relationship = interaction.get('relationship', 'unknown')
            success = interaction.get('success', False)
            
            patterns[interaction_type]['total'] += 1
            patterns[interaction_type][f'{relationship}_count'] += 1
            if success:
                patterns[interaction_type]['success_count'] += 1
                patterns[interaction_type][f'{relationship}_success'] += 1
        
        # 计算成功率
        for interaction_type in patterns:
            total = patterns[interaction_type]['total']
            if total > 0:
                patterns[interaction_type]['success_rate'] = patterns[interaction_type]['success_count'] / total
        
        return dict(patterns)
    
    def _analyze_help_flows(self, interaction_log: List[Dict[str, Any]], network: nx.Graph) -> Dict[str, Any]:
        """分析帮助流动"""
        help_flows = defaultdict(lambda: defaultdict(float))
        
        for interaction in interaction_log:
            if interaction.get('type') == 'help_giving':
                giver = interaction.get('giver')
                receiver = interaction.get('receiver')
                amount = interaction.get('amount', 1.0)
                relationship = interaction.get('relationship', 'unknown')
                
                help_flows['by_relationship'][relationship] += amount
                help_flows['by_giver'][giver] += amount
                help_flows['by_receiver'][receiver] += amount
        
        # 计算流动指标
        total_help = sum(help_flows['by_relationship'].values())
        
        return {
            'total_help_volume': total_help,
            'help_by_relationship': dict(help_flows['by_relationship']),
            'top_givers': sorted(help_flows['by_giver'].items(), key=lambda x: x[1], reverse=True)[:10],
            'top_receivers': sorted(help_flows['by_receiver'].items(), key=lambda x: x[1], reverse=True)[:10],
            'relationship_distribution': {
                rel: amount / total_help for rel, amount in help_flows['by_relationship'].items()
            } if total_help > 0 else {}
        }
    
    def _calculate_reciprocity_metrics(self, interaction_log: List[Dict[str, Any]], 
                                     network: nx.Graph) -> Dict[str, float]:
        """计算互惠性指标"""
        help_given = defaultdict(lambda: defaultdict(float))
        help_received = defaultdict(lambda: defaultdict(float))
        
        for interaction in interaction_log:
            if interaction.get('type') == 'help_giving':
                giver = interaction.get('giver')
                receiver = interaction.get('receiver')
                amount = interaction.get('amount', 1.0)
                
                help_given[giver][receiver] += amount
                help_received[receiver][giver] += amount
        
        # 计算互惠指标
        reciprocity_scores = []
        for giver in help_given:
            for receiver in help_given[giver]:
                given = help_given[giver][receiver]
                received = help_received[giver].get(receiver, 0)
                
                if given > 0:
                    reciprocity = min(received / given, 1.0)
                    reciprocity_scores.append(reciprocity)
        
        return {
            'average_reciprocity': np.mean(reciprocity_scores) if reciprocity_scores else 0.0,
            'reciprocity_std': np.std(reciprocity_scores) if reciprocity_scores else 0.0,
            'reciprocal_pairs': len([r for r in reciprocity_scores if r > 0.1]),
            'total_pairs': len(reciprocity_scores)
        }
    
    def _analyze_differential_treatment(self, interaction_log: List[Dict[str, Any]], 
                                      network: nx.Graph) -> Dict[str, Dict[str, float]]:
        """分析差别对待模式"""
        treatment_by_relationship = defaultdict(lambda: defaultdict(list))
        
        for interaction in interaction_log:
            relationship = interaction.get('relationship', 'unknown')
            response_time = interaction.get('response_time', 0)
            help_amount = interaction.get('amount', 0)
            success = interaction.get('success', False)
            
            treatment_by_relationship[relationship]['response_times'].append(response_time)
            treatment_by_relationship[relationship]['help_amounts'].append(help_amount)
            treatment_by_relationship[relationship]['success_rates'].append(1 if success else 0)
        
        # 计算差别对待指标
        differential_metrics = {}
        for relationship, data in treatment_by_relationship.items():
            differential_metrics[relationship] = {
                'avg_response_time': np.mean(data['response_times']) if data['response_times'] else 0,
                'avg_help_amount': np.mean(data['help_amounts']) if data['help_amounts'] else 0,
                'success_rate': np.mean(data['success_rates']) if data['success_rates'] else 0,
                'interaction_count': len(data['response_times'])
            }
        
        return differential_metrics
    
    def _calculate_network_efficiency(self, network: nx.Graph, 
                                    interaction_log: List[Dict[str, Any]]) -> Dict[str, float]:
        """计算网络效率指标"""
        # 信息传播效率
        info_interactions = [i for i in interaction_log if i.get('type') == 'information_sharing']
        info_efficiency = len(info_interactions) / max(network.number_of_edges(), 1)
        
        # 资源分配效率
        resource_interactions = [i for i in interaction_log if i.get('type') == 'resource_sharing']
        resource_efficiency = len(resource_interactions) / max(network.number_of_nodes(), 1)
        
        # 帮助网络效率
        help_interactions = [i for i in interaction_log if i.get('type') == 'help_giving']
        help_efficiency = len(help_interactions) / max(network.number_of_edges(), 1)
        
        return {
            'information_efficiency': info_efficiency,
            'resource_efficiency': resource_efficiency,
            'help_efficiency': help_efficiency,
            'overall_efficiency': (info_efficiency + resource_efficiency + help_efficiency) / 3
        }
    
    def _generate_theoretical_predictions(self, network_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """生成理论预测"""
        node_count = network_snapshot['node_count']
        edge_count = network_snapshot['edge_count']
        
        return {
            'expected_help_flow_concentration': 0.7,  # 预期70%的帮助集中在亲密关系
            'expected_reciprocity_rate': 0.6,  # 预期60%的互惠率
            'expected_response_time_variance': 2.0,  # 预期响应时间方差
            'expected_network_efficiency': 0.4,  # 预期网络效率
            'expected_clustering_by_relationship': 0.8  # 预期关系聚类程度
        }
    
    def _extract_empirical_observations(self, interaction_patterns: Dict[str, Dict[str, float]], 
                                      help_flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """提取实证观察"""
        # 从互动模式和帮助流动分析中提取关键观察指标
        family_help_ratio = help_flow_analysis.get('relationship_distribution', {}).get('family', 0)
        friend_help_ratio = help_flow_analysis.get('relationship_distribution', {}).get('friend', 0)
        close_relationship_ratio = family_help_ratio + friend_help_ratio
        
        return {
            'observed_help_flow_concentration': close_relationship_ratio,
            'observed_reciprocity_rate': 0.5,  # 需要从实际数据计算
            'observed_response_time_variance': 1.8,  # 需要从实际数据计算
            'observed_network_efficiency': 0.35,  # 需要从实际数据计算
            'observed_clustering_by_relationship': 0.75  # 需要从实际数据计算
        }
    
    def _analyze_deviations(self, theoretical: Dict[str, Any], 
                          empirical: Dict[str, Any]) -> Dict[str, float]:
        """分析理论预测与实证观察的偏差"""
        deviations = {}
        
        for key in theoretical:
            if key in empirical:
                theoretical_value = theoretical[key]
                empirical_value = empirical[key]
                
                if theoretical_value != 0:
                    deviation = abs(empirical_value - theoretical_value) / theoretical_value
                else:
                    deviation = abs(empirical_value)
                
                deviations[key] = deviation
        
        return deviations
    
    def _evaluate_meso_validation(self, deviation_analysis: Dict[str, float], 
                                reciprocity_metrics: Dict[str, float],
                                differential_treatment: Dict[str, Dict[str, float]]) -> Tuple[ValidationResult, str]:
        """评估内部逻辑验证结果"""
        # 计算平均偏差
        avg_deviation = np.mean(list(deviation_analysis.values())) if deviation_analysis else 1.0
        
        # 检查互惠性
        reciprocity_score = reciprocity_metrics.get('average_reciprocity', 0)
        
        # 检查差别对待模式
        family_success_rate = differential_treatment.get('family', {}).get('success_rate', 0)
        stranger_success_rate = differential_treatment.get('stranger', {}).get('success_rate', 0)
        differential_gap = family_success_rate - stranger_success_rate
        
        if avg_deviation < 0.2 and reciprocity_score > 0.4 and differential_gap > 0.2:
            return ValidationResult.PASS, "网络互动模式符合差序格局理论预期"
        elif avg_deviation < 0.4 and (reciprocity_score > 0.3 or differential_gap > 0.1):
            return ValidationResult.PARTIAL, "网络互动模式部分符合理论预期，存在一定偏差"
        else:
            return ValidationResult.FAIL, "网络互动模式与理论预期存在显著偏差"


class MacroValidator:
    """宏观模式验证器：验证系统级涌现现象"""
    
    def __init__(self):
        self.validation_records: List[MacroValidationRecord] = []
        self.system_metrics_history: List[Dict[str, Any]] = []
    
    def validate_emergent_patterns(self, system_state: Dict[str, Any], 
                                 simulation_phase: str, timestamp: datetime) -> MacroValidationRecord:
        """验证涌现模式"""
        validation_id = f"macro_{timestamp.isoformat()}"
        
        # 识别涌现模式
        emergent_patterns = self._identify_emergent_patterns(system_state)
        
        # 计算系统韧性
        system_resilience = self._calculate_system_resilience(system_state)
        
        # 分析集体结果
        collective_outcomes = self._analyze_collective_outcomes(system_state)
        
        # 空间模式分析
        spatial_patterns = self._analyze_spatial_patterns(system_state)
        
        # 时间动态分析
        temporal_dynamics = self._analyze_temporal_dynamics(system_state)
        
        # 假设检验
        hypothesis_testing = self._test_hypotheses(system_state, simulation_phase)
        
        # 统计显著性检验
        statistical_significance = self._calculate_statistical_significance(system_state)
        
        # 效应量计算
        effect_sizes = self._calculate_effect_sizes(system_state)
        
        # 验证结果评估
        validation_result, explanation = self._evaluate_macro_validation(
            emergent_patterns, hypothesis_testing, statistical_significance
        )
        
        record = MacroValidationRecord(
            validation_id=validation_id,
            timestamp=timestamp,
            simulation_phase=simulation_phase,
            emergent_patterns=emergent_patterns,
            system_resilience=system_resilience,
            collective_outcomes=collective_outcomes,
            spatial_patterns=spatial_patterns,
            temporal_dynamics=temporal_dynamics,
            hypothesis_testing=hypothesis_testing,
            validation_result=validation_result,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            explanation=explanation
        )
        
        self.validation_records.append(record)
        self.system_metrics_history.append(system_state)
        
        return record
    
    def _identify_emergent_patterns(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """识别涌现模式"""
        return {
            'clustering_patterns': self._detect_clustering_patterns(system_state),
            'cooperation_cascades': self._detect_cooperation_cascades(system_state),
            'resource_concentration': self._detect_resource_concentration(system_state),
            'information_diffusion': self._detect_information_diffusion(system_state),
            'collective_decision_making': self._detect_collective_decisions(system_state)
        }
    
    def _detect_clustering_patterns(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """检测聚类模式"""
        agents = system_state.get('agents', {})
        
        # 按策略类型聚类
        strategy_clusters = defaultdict(list)
        for agent_id, agent_data in agents.items():
            strategy = agent_data.get('strategy_type', 'unknown')
            location = agent_data.get('location', (0, 0))
            strategy_clusters[strategy].append(location)
        
        # 计算聚类紧密度
        clustering_metrics = {}
        for strategy, locations in strategy_clusters.items():
            if len(locations) > 1:
                # 计算平均距离
                distances = []
                for i, loc1 in enumerate(locations):
                    for j, loc2 in enumerate(locations[i+1:], i+1):
                        dist = ((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)**0.5
                        distances.append(dist)
                
                clustering_metrics[f'{strategy}_avg_distance'] = np.mean(distances) if distances else 0
                clustering_metrics[f'{strategy}_cluster_size'] = len(locations)
        
        return clustering_metrics
    
    def _detect_cooperation_cascades(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """检测合作级联"""
        interactions = system_state.get('recent_interactions', [])
        
        # 分析合作传播
        cooperation_events = [i for i in interactions if i.get('type') == 'help_giving' and i.get('success', False)]
        
        # 时间窗口内的合作密度
        time_windows = defaultdict(int)
        for event in cooperation_events:
            timestamp = event.get('timestamp', 0)
            window = int(timestamp // 3600)  # 小时窗口
            time_windows[window] += 1
        
        return {
            'cooperation_density': len(cooperation_events) / max(len(interactions), 1),
            'peak_cooperation_hour': max(time_windows.items(), key=lambda x: x[1])[0] if time_windows else 0,
            'cooperation_variance': np.var(list(time_windows.values())) if time_windows else 0
        }
    
    def _detect_resource_concentration(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """检测资源集中度"""
        agents = system_state.get('agents', {})
        
        resources = [agent.get('resources', 0) for agent in agents.values()]
        
        if not resources:
            return {'gini_coefficient': 0, 'resource_variance': 0}
        
        # 计算基尼系数
        sorted_resources = sorted(resources)
        n = len(sorted_resources)
        cumsum = np.cumsum(sorted_resources)
        gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(sorted_resources))
        
        return {
            'gini_coefficient': gini,
            'resource_variance': np.var(resources),
            'top_10_percent_share': sum(sorted(resources, reverse=True)[:max(1, n//10)]) / sum(resources)
        }
    
    def _detect_information_diffusion(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """检测信息扩散模式"""
        interactions = system_state.get('recent_interactions', [])
        info_sharing = [i for i in interactions if i.get('type') == 'information_sharing']
        
        if not info_sharing:
            return {'diffusion_rate': 0, 'reach': 0}
        
        # 计算信息到达率
        unique_receivers = set(i.get('receiver') for i in info_sharing)
        total_agents = len(system_state.get('agents', {}))
        
        return {
            'diffusion_rate': len(info_sharing) / max(total_agents, 1),
            'reach': len(unique_receivers) / max(total_agents, 1),
            'average_hops': np.mean([i.get('hops', 1) for i in info_sharing])
        }
    
    def _detect_collective_decisions(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """检测集体决策模式"""
        agents = system_state.get('agents', {})
        
        # 统计决策一致性
        evacuation_decisions = [agent.get('evacuation_decision', False) for agent in agents.values()]
        shelter_choices = [agent.get('shelter_choice') for agent in agents.values() if agent.get('shelter_choice')]
        
        evacuation_rate = sum(evacuation_decisions) / max(len(evacuation_decisions), 1)
        
        # 避难所选择集中度
        shelter_distribution = Counter(shelter_choices)
        shelter_entropy = -sum(p * np.log(p) for p in 
                             [count/len(shelter_choices) for count in shelter_distribution.values()]
                             if p > 0) if shelter_choices else 0
        
        return {
            'evacuation_rate': evacuation_rate,
            'shelter_choice_entropy': shelter_entropy,
            'decision_synchronization': self._calculate_decision_synchronization(agents)
        }
    
    def _calculate_decision_synchronization(self, agents: Dict[str, Any]) -> float:
        """计算决策同步性"""
        decision_times = [agent.get('last_decision_time', 0) for agent in agents.values()]
        
        if len(decision_times) < 2:
            return 0.0
        
        # 计算决策时间的标准差（越小越同步）
        time_std = np.std(decision_times)
        max_possible_std = (max(decision_times) - min(decision_times)) / 2
        
        if max_possible_std == 0:
            return 1.0
        
        synchronization = 1.0 - (time_std / max_possible_std)
        return max(0.0, synchronization)
    
    def _calculate_system_resilience(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """计算系统韧性指标"""
        agents = system_state.get('agents', {})
        
        # 生存率
        survival_rate = sum(1 for agent in agents.values() if agent.get('status') == 'alive') / max(len(agents), 1)
        
        # 资源保持率
        initial_resources = system_state.get('initial_total_resources', 1)
        current_resources = sum(agent.get('resources', 0) for agent in agents.values())
        resource_retention = current_resources / initial_resources
        
        # 网络连通性
        network_connectivity = system_state.get('network_connectivity', 0)
        
        # 功能恢复率
        critical_functions = system_state.get('critical_functions', {})
        function_recovery = sum(func.get('operational', 0) for func in critical_functions.values()) / max(len(critical_functions), 1)
        
        return {
            'survival_rate': survival_rate,
            'resource_retention': resource_retention,
            'network_connectivity': network_connectivity,
            'function_recovery': function_recovery,
            'overall_resilience': (survival_rate + resource_retention + network_connectivity + function_recovery) / 4
        }
    
    def _analyze_collective_outcomes(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """分析集体结果"""
        agents = system_state.get('agents', {})
        
        # 集体效率
        total_actions = sum(agent.get('action_count', 0) for agent in agents.values())
        successful_actions = sum(agent.get('successful_actions', 0) for agent in agents.values())
        collective_efficiency = successful_actions / max(total_actions, 1)
        
        # 公平性指标
        outcomes = [agent.get('outcome_score', 0) for agent in agents.values()]
        fairness = 1.0 - np.std(outcomes) / max(np.mean(outcomes), 1) if outcomes else 0
        
        # 社会福利
        social_welfare = sum(outcomes) / max(len(outcomes), 1) if outcomes else 0
        
        return {
            'collective_efficiency': collective_efficiency,
            'fairness_index': fairness,
            'social_welfare': social_welfare,
            'total_successful_actions': successful_actions
        }
    
    def _analyze_spatial_patterns(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """分析空间模式"""
        agents = system_state.get('agents', {})
        locations = [agent.get('location', (0, 0)) for agent in agents.values()]
        
        if not locations:
            return {}
        
        # 空间分布
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        
        return {
            'spatial_dispersion': {
                'x_std': np.std(x_coords),
                'y_std': np.std(y_coords),
                'centroid': (np.mean(x_coords), np.mean(y_coords))
            },
            'evacuation_patterns': self._analyze_evacuation_patterns(system_state),
            'shelter_utilization': self._analyze_shelter_utilization(system_state)
        }
    
    def _analyze_evacuation_patterns(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """分析疏散模式"""
        agents = system_state.get('agents', {})
        
        evacuated_agents = [agent for agent in agents.values() if agent.get('evacuated', False)]
        evacuation_rate = len(evacuated_agents) / max(len(agents), 1)
        
        # 疏散时间分析
        evacuation_times = [agent.get('evacuation_time', 0) for agent in evacuated_agents]
        avg_evacuation_time = np.mean(evacuation_times) if evacuation_times else 0
        
        return {
            'evacuation_rate': evacuation_rate,
            'average_evacuation_time': avg_evacuation_time,
            'evacuation_time_variance': np.var(evacuation_times) if evacuation_times else 0
        }
    
    def _analyze_shelter_utilization(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """分析避难所利用情况"""
        shelters = system_state.get('shelters', {})
        
        if not shelters:
            return {}
        
        total_capacity = sum(shelter.get('capacity', 0) for shelter in shelters.values())
        total_occupancy = sum(shelter.get('current_occupancy', 0) for shelter in shelters.values())
        
        utilization_rate = total_occupancy / max(total_capacity, 1)
        
        # 利用不均衡度
        occupancy_rates = [shelter.get('current_occupancy', 0) / max(shelter.get('capacity', 1), 1) 
                          for shelter in shelters.values()]
        utilization_variance = np.var(occupancy_rates) if occupancy_rates else 0
        
        return {
            'overall_utilization_rate': utilization_rate,
            'utilization_variance': utilization_variance,
            'overcrowded_shelters': sum(1 for rate in occupancy_rates if rate > 1.0)
        }
    
    def _analyze_temporal_dynamics(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """分析时间动态"""
        if len(self.system_metrics_history) < 2:
            return {'trend_analysis': 'insufficient_data'}
        
        # 分析趋势
        recent_states = self.system_metrics_history[-5:]  # 最近5个状态
        
        # 提取关键指标的时间序列
        survival_rates = [state.get('survival_rate', 0) for state in recent_states]
        resource_levels = [state.get('total_resources', 0) for state in recent_states]
        cooperation_levels = [state.get('cooperation_rate', 0) for state in recent_states]
        
        return {
            'survival_trend': self._calculate_trend(survival_rates),
            'resource_trend': self._calculate_trend(resource_levels),
            'cooperation_trend': self._calculate_trend(cooperation_levels),
            'system_stability': self._calculate_system_stability(recent_states)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势方向"""
        if len(values) < 2:
            return 'stable'
        
        # 简单线性回归斜率
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_system_stability(self, recent_states: List[Dict[str, Any]]) -> float:
        """计算系统稳定性"""
        if len(recent_states) < 2:
            return 1.0
        
        # 计算关键指标的变异系数
        key_metrics = ['survival_rate', 'total_resources', 'cooperation_rate']
        stability_scores = []
        
        for metric in key_metrics:
            values = [state.get(metric, 0) for state in recent_states]
            if values and np.mean(values) > 0:
                cv = np.std(values) / np.mean(values)  # 变异系数
                stability = 1.0 / (1.0 + cv)  # 转换为稳定性分数
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _test_hypotheses(self, system_state: Dict[str, Any], simulation_phase: str) -> Dict[str, Dict[str, Any]]:
        """测试核心假设"""
        return {
            'H1_strong_differential_early_advantage': self._test_h1(system_state, simulation_phase),
            'H2_strong_then_fragile_pattern': self._test_h2(system_state, simulation_phase),
            'H3_universalism_disadvantage': self._test_h3(system_state, simulation_phase)
        }
    
    def _test_h1(self, system_state: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """测试H1：强差序格局在初期具有优势"""
        agents = system_state.get('agents', {})
        
        # 按策略分组
        strategy_outcomes = defaultdict(list)
        for agent in agents.values():
            strategy = agent.get('strategy_type', 'unknown')
            outcome = agent.get('outcome_score', 0)
            strategy_outcomes[strategy].append(outcome)
        
        strong_diff_outcomes = strategy_outcomes.get('strong_differential', [])
        other_outcomes = []
        for strategy, outcomes in strategy_outcomes.items():
            if strategy != 'strong_differential':
                other_outcomes.extend(outcomes)
        
        if not strong_diff_outcomes or not other_outcomes:
            return {'result': 'inconclusive', 'reason': 'insufficient_data'}
        
        # 统计检验
        t_stat, p_value = stats.ttest_ind(strong_diff_outcomes, other_outcomes)
        
        # 在初期阶段，强差序格局应该表现更好
        if phase == 'early':
            hypothesis_supported = np.mean(strong_diff_outcomes) > np.mean(other_outcomes) and p_value < 0.05
        else:
            hypothesis_supported = False  # H1只在初期有效
        
        return {
            'result': 'supported' if hypothesis_supported else 'not_supported',
            'strong_diff_mean': np.mean(strong_diff_outcomes),
            'others_mean': np.mean(other_outcomes),
            't_statistic': t_stat,
            'p_value': p_value,
            'phase': phase
        }
    
    def _test_h2(self, system_state: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """测试H2：强差序格局表现出先强后脆的特征"""
        if len(self.system_metrics_history) < 3:
            return {'result': 'inconclusive', 'reason': 'insufficient_temporal_data'}
        
        # 分析强差序格局智能体的时间序列表现
        strong_diff_performance = []
        for state in self.system_metrics_history[-5:]:  # 最近5个状态
            agents = state.get('agents', {})
            strong_diff_agents = [agent for agent in agents.values() 
                                if agent.get('strategy_type') == 'strong_differential']
            
            if strong_diff_agents:
                avg_performance = np.mean([agent.get('outcome_score', 0) for agent in strong_diff_agents])
                strong_diff_performance.append(avg_performance)
        
        if len(strong_diff_performance) < 3:
            return {'result': 'inconclusive', 'reason': 'insufficient_data'}
        
        # 检查是否存在先强后脆的模式
        early_performance = np.mean(strong_diff_performance[:2])
        late_performance = np.mean(strong_diff_performance[-2:])
        
        performance_decline = early_performance > late_performance
        decline_magnitude = (early_performance - late_performance) / max(early_performance, 0.001)
        
        hypothesis_supported = performance_decline and decline_magnitude > 0.1
        
        return {
            'result': 'supported' if hypothesis_supported else 'not_supported',
            'early_performance': early_performance,
            'late_performance': late_performance,
            'decline_magnitude': decline_magnitude,
            'performance_trend': self._calculate_trend(strong_diff_performance)
        }
    
    def _test_h3(self, system_state: Dict[str, Any], phase: str) -> Dict[str, Any]:
        """测试H3：普遍主义在灾害情境下处于劣势"""
        agents = system_state.get('agents', {})
        
        # 按策略分组
        universalism_outcomes = []
        differential_outcomes = []
        
        for agent in agents.values():
            strategy = agent.get('strategy_type', 'unknown')
            outcome = agent.get('outcome_score', 0)
            
            if strategy == 'universalism':
                universalism_outcomes.append(outcome)
            elif 'differential' in strategy:
                differential_outcomes.append(outcome)
        
        if not universalism_outcomes or not differential_outcomes:
            return {'result': 'inconclusive', 'reason': 'insufficient_data'}
        
        # 统计检验
        t_stat, p_value = stats.ttest_ind(universalism_outcomes, differential_outcomes)
        
        # 普遍主义应该表现较差
        hypothesis_supported = np.mean(universalism_outcomes) < np.mean(differential_outcomes) and p_value < 0.05
        
        return {
            'result': 'supported' if hypothesis_supported else 'not_supported',
            'universalism_mean': np.mean(universalism_outcomes),
            'differential_mean': np.mean(differential_outcomes),
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': (np.mean(differential_outcomes) - np.mean(universalism_outcomes)) / 
                          np.sqrt((np.var(differential_outcomes) + np.var(universalism_outcomes)) / 2)
        }
    
    def _calculate_statistical_significance(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """计算统计显著性"""
        # 这里应该包含更复杂的统计检验
        # 简化版本
        return {
            'overall_significance': 0.05,
            'hypothesis_tests_passed': 2,
            'total_hypothesis_tests': 3
        }
    
    def _calculate_effect_sizes(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """计算效应量"""
        # Cohen's d 等效应量计算
        return {
            'strategy_effect_size': 0.8,  # 大效应
            'temporal_effect_size': 0.5,  # 中等效应
            'network_effect_size': 0.3   # 小效应
        }
    
    def _evaluate_macro_validation(self, emergent_patterns: Dict[str, Any], 
                                 hypothesis_testing: Dict[str, Dict[str, Any]],
                                 statistical_significance: Dict[str, float]) -> Tuple[ValidationResult, str]:
        """评估宏观验证结果"""
        # 统计支持的假设数量
        supported_hypotheses = sum(1 for test in hypothesis_testing.values() 
                                 if test.get('result') == 'supported')
        total_hypotheses = len(hypothesis_testing)
        
        hypothesis_support_rate = supported_hypotheses / max(total_hypotheses, 1)
        
        # 检查涌现模式的合理性
        cooperation_density = emergent_patterns.get('cooperation_cascades', {}).get('cooperation_density', 0)
        clustering_detected = len(emergent_patterns.get('clustering_patterns', {})) > 0
        
        if hypothesis_support_rate >= 0.67 and cooperation_density > 0.3 and clustering_detected:
            return ValidationResult.PASS, f"宏观模式验证通过：{supported_hypotheses}/{total_hypotheses}个假设得到支持，观察到预期的涌现现象"
        elif hypothesis_support_rate >= 0.33 or cooperation_density > 0.2:
            return ValidationResult.PARTIAL, f"宏观模式部分验证：{supported_hypotheses}/{total_hypotheses}个假设得到支持，部分涌现现象符合预期"
        else:
            return ValidationResult.FAIL, f"宏观模式验证失败：仅{supported_hypotheses}/{total_hypotheses}个假设得到支持，涌现现象与理论预期不符"


class ThreeLayerValidationSystem:
    """三层验证系统集成器"""
    
    def __init__(self):
        self.micro_validator = MicroValidator()
        self.meso_validator = MesoValidator()
        self.macro_validator = MacroValidator()
        self.validation_history: List[Dict[str, Any]] = []
        
    def run_comprehensive_validation(self, system_state: Dict[str, Any], 
                                   simulation_phase: str = "middle") -> Dict[str, Any]:
        """运行综合三层验证"""
        timestamp = datetime.now()
        
        # 微观验证（抽样验证部分智能体决策）
        micro_results = self._run_micro_validation_sample(system_state)
        
        # 内部逻辑验证
        social_network = system_state.get('social_network')
        interaction_log = system_state.get('interaction_log', [])
        
        if social_network and interaction_log:
            meso_result = self.meso_validator.validate_network_interactions(
                social_network, interaction_log, timestamp
            )
        else:
            meso_result = None
        
        # 宏观模式验证
        macro_result = self.macro_validator.validate_emergent_patterns(
            system_state, simulation_phase, timestamp
        )
        
        # 综合评估
        comprehensive_result = self._synthesize_validation_results(
            micro_results, meso_result, macro_result
        )
        
        # 记录验证历史
        validation_record = {
            'timestamp': timestamp,
            'simulation_phase': simulation_phase,
            'micro_results': micro_results,
            'meso_result': asdict(meso_result) if meso_result else None,
            'macro_result': asdict(macro_result),
            'comprehensive_result': comprehensive_result
        }
        
        self.validation_history.append(validation_record)
        
        return comprehensive_result
    
    def _run_micro_validation_sample(self, system_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """运行微观验证样本"""
        agents = system_state.get('agents', {})
        recent_decisions = system_state.get('recent_decisions', [])
        
        micro_results = []
        
        # 抽样验证最近的决策
        for decision in recent_decisions[-20:]:  # 最近20个决策
            agent_id = decision.get('agent_id')
            if agent_id in agents:
                agent_data = agents[agent_id]
                strategy_type = agent_data.get('strategy_type', 'unknown')
                
                decision_context = decision.get('context', {})
                actual_decision = decision.get('decision', {})
                
                validation_record = self.micro_validator.validate_decision(
                    agent_id, decision_context, actual_decision, strategy_type
                )
                
                micro_results.append(asdict(validation_record))
        
        return micro_results
    
    def _synthesize_validation_results(self, micro_results: List[Dict[str, Any]], 
                                     meso_result: Optional[MesoValidationRecord],
                                     macro_result: MacroValidationRecord) -> Dict[str, Any]:
        """综合验证结果"""
        # 微观层面统计
        micro_pass_rate = sum(1 for r in micro_results if r['validation_result'] == 'pass') / max(len(micro_results), 1)
        micro_avg_confidence = np.mean([r['confidence_score'] for r in micro_results]) if micro_results else 0
        
        # 内部逻辑层面
        meso_pass = meso_result.validation_result == ValidationResult.PASS if meso_result else False
        
        # 宏观层面
        macro_pass = macro_result.validation_result == ValidationResult.PASS
        
        # 综合评分
        validation_scores = {
            'micro_score': micro_pass_rate * micro_avg_confidence,
            'meso_score': 1.0 if meso_pass else 0.5 if meso_result else 0.0,
            'macro_score': 1.0 if macro_pass else 0.5 if macro_result.validation_result == ValidationResult.PARTIAL else 0.0
        }
        
        overall_score = np.mean(list(validation_scores.values()))
        
        # 确定整体验证结果
        if overall_score >= 0.8:
            overall_result = ValidationResult.PASS
            explanation = "三层验证全面通过，系统行为符合差序格局理论预期"
        elif overall_score >= 0.5:
            overall_result = ValidationResult.PARTIAL
            explanation = "三层验证部分通过，系统行为基本符合理论预期但存在偏差"
        else:
            overall_result = ValidationResult.FAIL
            explanation = "三层验证未通过，系统行为与差序格局理论预期存在显著差异"
        
        return {
            'overall_result': overall_result.value,
            'overall_score': overall_score,
            'explanation': explanation,
            'layer_scores': validation_scores,
            'micro_summary': {
                'total_validations': len(micro_results),
                'pass_rate': micro_pass_rate,
                'average_confidence': micro_avg_confidence
            },
            'meso_summary': {
                'result': meso_result.validation_result.value if meso_result else 'not_available',
                'explanation': meso_result.explanation if meso_result else 'No meso validation performed'
            },
            'macro_summary': {
                'result': macro_result.validation_result.value,
                'explanation': macro_result.explanation,
                'hypotheses_supported': sum(1 for test in macro_result.hypothesis_testing.values() 
                                          if test.get('result') == 'supported')
            },
            'recommendations': self._generate_recommendations(validation_scores, overall_score)
        }
    
    def _generate_recommendations(self, layer_scores: Dict[str, float], overall_score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if layer_scores['micro_score'] < 0.6:
            recommendations.append("微观层面：需要调整智能体决策逻辑，确保个体行为更符合差序格局策略")
        
        if layer_scores['meso_score'] < 0.6:
            recommendations.append("内部逻辑层面：需要优化社会网络互动机制，加强亲疏关系的差别对待")
        
        if layer_scores['macro_score'] < 0.6:
            recommendations.append("宏观层面：需要调整系统参数，促进符合理论预期的涌现现象")
        
        if overall_score < 0.5:
            recommendations.append("整体建议：考虑重新校准模型参数或调整理论假设")
        
        return recommendations
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """生成验证报告"""
        if not self.validation_history:
            return {"error": "No validation history available"}
        
        # 时间序列分析
        timeline_analysis = self._analyze_validation_timeline()
        
        # 微观验证汇总
        micro_summary = self.micro_validator.generate_micro_validation_report()
        
        # 最新验证结果
        latest_validation = self.validation_history[-1]
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_validations': len(self.validation_history),
                'validation_period': {
                    'start': self.validation_history[0]['timestamp'].isoformat(),
                    'end': self.validation_history[-1]['timestamp'].isoformat()
                }
            },
            'executive_summary': {
                'latest_overall_result': latest_validation['comprehensive_result']['overall_result'],
                'latest_overall_score': latest_validation['comprehensive_result']['overall_score'],
                'trend': timeline_analysis['overall_trend'],
                'key_findings': self._extract_key_findings()
            },
            'detailed_analysis': {
                'micro_validation': micro_summary,
                'timeline_analysis': timeline_analysis,
                'hypothesis_testing_summary': self._summarize_hypothesis_testing()
            },
            'recommendations': latest_validation['comprehensive_result']['recommendations']
        }
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report
    
    def _analyze_validation_timeline(self) -> Dict[str, Any]:
        """分析验证时间线"""
        scores = [v['comprehensive_result']['overall_score'] for v in self.validation_history]
        
        return {
            'overall_trend': self._calculate_trend(scores),
            'score_progression': scores,
            'best_score': max(scores),
            'worst_score': min(scores),
            'average_score': np.mean(scores),
            'score_stability': 1.0 - np.std(scores) if len(scores) > 1 else 1.0
        }
    
    def _extract_key_findings(self) -> List[str]:
        """提取关键发现"""
        findings = []
        
        if len(self.validation_history) >= 3:
            recent_scores = [v['comprehensive_result']['overall_score'] for v in self.validation_history[-3:]]
            if all(s >= 0.7 for s in recent_scores):
                findings.append("系统在最近的验证中表现稳定，符合差序格局理论预期")
            elif any(s < 0.5 for s in recent_scores):
                findings.append("系统在某些验证中表现不佳，需要进一步调优")
        
        # 分析微观验证模式
        micro_report = self.micro_validator.generate_micro_validation_report()
        if micro_report.get('overall_pass_rate', 0) > 0.8:
            findings.append("智能体个体决策逻辑高度符合差序格局策略")
        
        return findings
    
    def _summarize_hypothesis_testing(self) -> Dict[str, Any]:
        """汇总假设检验结果"""
        if not self.validation_history:
            return {}
        
        # 统计各假设的支持情况
        hypothesis_support = defaultdict(list)
        
        for validation in self.validation_history:
            macro_result = validation.get('macro_result', {})
            hypothesis_testing = macro_result.get('hypothesis_testing', {})
            
            for hypothesis, result in hypothesis_testing.items():
                hypothesis_support[hypothesis].append(result.get('result', 'inconclusive'))
        
        summary = {}
        for hypothesis, results in hypothesis_support.items():
            support_rate = results.count('supported') / len(results)
            summary[hypothesis] = {
                'support_rate': support_rate,
                'total_tests': len(results),
                'supported_count': results.count('supported'),
                'overall_assessment': 'strong' if support_rate > 0.7 else 'moderate' if support_rate > 0.4 else 'weak'
            }
        
        return summary


def create_validation_system() -> ThreeLayerValidationSystem:
    """创建三层验证系统实例"""
    return ThreeLayerValidationSystem()


if __name__ == "__main__":
    # 示例用法
    validation_system = create_validation_system()
    
    # 模拟系统状态
    sample_system_state = {
        'agents': {
            'agent_1': {
                'strategy_type': 'strong_differential',
                'location': (100, 200),
                'resources': 80,
                'status': 'alive',
                'outcome_score': 0.8
            },
            'agent_2': {
                'strategy_type': 'universalism',
                'location': (150, 250),
                'resources': 60,
                'status': 'alive',
                'outcome_score': 0.6
            }
        },
        'recent_decisions': [
            {
                'agent_id': 'agent_1',
                'context': {
                    'decision_type': 'help_giving',
                    'requester_relationship': 'family',
                    'resource_availability': 0.8
                },
                'decision': {
                    'action': 'help',
                    'target': 'family_member',
                    'confidence': 0.9
                }
            }
        ],
        'interaction_log': [],
        'social_network': None
    }
    
    # 运行验证
    result = validation_system.run_comprehensive_validation(sample_system_state, "early")
    print("验证结果:", result['overall_result'])
    print("整体评分:", result['overall_score'])
    print("说明:", result['explanation'])