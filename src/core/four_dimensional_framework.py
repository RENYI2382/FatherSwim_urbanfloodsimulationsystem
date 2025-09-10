"""
四维决策标准集成框架
基于DDABM论文的完整四维决策标准实现
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# 导入新的评估模块
from .evacuation_cost_assessment import EvacuationCostAssessment, CostComponents
from .property_attachment_assessment import PropertyAttachmentAssessment, AttachmentComponents
from .information_trust_assessment import InformationTrustAssessment, TrustComponents, InformationSource

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DecisionType(Enum):
    """决策类型枚举"""
    STAY = "stay"
    PREPARE = "prepare"
    EVACUATE = "evacuate"
    EMERGENCY_EVACUATE = "emergency_evacuate"

@dataclass
class FourDimensionalCriteria:
    """四维决策标准数据类"""
    safety_concern: float = 0.0           # w₁: 安全性关注
    information_trust: float = 0.0        # w₂: 信息信任
    evacuation_cost: float = 0.0          # w₃: 疏散成本
    property_attachment: float = 0.0      # w₄: 财产依恋
    
    # 组件详情
    trust_components: Optional[TrustComponents] = None
    cost_components: Optional[CostComponents] = None
    attachment_components: Optional[AttachmentComponents] = None

@dataclass
class DecisionContext:
    """决策上下文"""
    weather_data: Dict[str, Any] = field(default_factory=dict)
    agent_profile: Dict[str, Any] = field(default_factory=dict)
    social_context: Dict[str, Any] = field(default_factory=dict)
    time_context: Dict[str, Any] = field(default_factory=dict)
    location_data: Dict[str, Any] = field(default_factory=dict)
    information_sources: List[InformationSource] = field(default_factory=list)
    information_content: Dict[str, Any] = field(default_factory=dict)

class FourDimensionalDecisionFramework:
    """
    四维决策标准框架
    基于DDABM论文的完整实现
    """
    
    def __init__(self, use_dynamic_weights: bool = True):
        self.use_dynamic_weights = use_dynamic_weights
        
        # 初始化评估模块
        self.cost_assessor = EvacuationCostAssessment()
        self.attachment_assessor = PropertyAttachmentAssessment()
        self.trust_assessor = InformationTrustAssessment()
        
        # 基础权重配置（基于DDABM论文）
        self.base_weights = {
            'safety_concern': 0.40,      # w₁: 安全性权重最高
            'information_trust': 0.25,   # w₂: 信息信任权重
            'evacuation_cost': 0.20,     # w₃: 疏散成本权重
            'property_attachment': 0.15  # w₄: 财产依恋权重
        }
        
        # 动态权重调整因子
        self.dynamic_factors = {
            'time_pressure': 0.0,        # 时间压力因子
            'risk_escalation': 0.0,      # 风险升级因子
            'social_pressure': 0.0,      # 社会压力因子
            'experience_factor': 0.0     # 经验因子
        }
        
        # 决策阈值配置
        self.decision_thresholds = {
            DecisionType.STAY: 0.3,
            DecisionType.PREPARE: 0.5,
            DecisionType.EVACUATE: 0.7,
            DecisionType.EMERGENCY_EVACUATE: 0.9
        }
    
    async def assess_four_dimensional_criteria(
        self, 
        context: DecisionContext
    ) -> FourDimensionalCriteria:
        """
        评估四维决策标准
        
        Args:
            context: 决策上下文
            
        Returns:
            FourDimensionalCriteria: 四维标准评估结果
        """
        try:
            # 1. w₁: 安全性关注评估
            safety_concern = await self._assess_safety_concern(context)
            
            # 2. w₂: 信息信任评估
            trust_components = self.trust_assessor.calculate_information_trust(
                agent_profile=context.agent_profile,
                information_sources=context.information_sources,
                information_content=context.information_content,
                social_context=context.social_context
            )
            information_trust = trust_components.total_trust
            
            # 3. w₃: 疏散成本评估
            cost_components = self.cost_assessor.calculate_evacuation_cost(
                agent_profile=context.agent_profile,
                traffic_data=context.location_data.get('traffic_data', {}),
                distance=context.location_data.get('evacuation_distance', 50),
                weather_severity=self._extract_weather_severity(context.weather_data)
            )
            # 成本转换为阻力因子（成本越高，疏散意愿越低）
            evacuation_cost = 1.0 - min(cost_components.total_cost / 1000.0, 1.0)
            
            # 4. w₄: 财产依恋评估
            panic_level = self._calculate_panic_level(context)
            social_pressure = self._calculate_social_pressure(context)
            
            attachment_components = self.attachment_assessor.calculate_attachment_level(
                agent_profile=context.agent_profile,
                panic_level=panic_level,
                social_pressure=social_pressure
            )
            # 依恋转换为阻力因子（依恋越强，疏散意愿越低）
            property_attachment = 1.0 - attachment_components.total_attachment
            
            return FourDimensionalCriteria(
                safety_concern=safety_concern,
                information_trust=information_trust,
                evacuation_cost=evacuation_cost,
                property_attachment=property_attachment,
                trust_components=trust_components,
                cost_components=cost_components,
                attachment_components=attachment_components
            )
            
        except Exception as e:
            logger.error(f"四维标准评估失败: {e}")
            # 返回默认值
            return FourDimensionalCriteria(
                safety_concern=0.6,
                information_trust=0.6,
                evacuation_cost=0.6,
                property_attachment=0.6
            )
    
    async def make_four_dimensional_decision(
        self, 
        context: DecisionContext
    ) -> Dict[str, Any]:
        """
        基于四维标准制定疏散决策
        
        Args:
            context: 决策上下文
            
        Returns:
            Dict: 决策结果详情
        """
        try:
            # 1. 评估四维标准
            criteria = await self.assess_four_dimensional_criteria(context)
            
            # 2. 计算动态权重
            weights = self._calculate_dynamic_weights(context) if self.use_dynamic_weights else self.base_weights
            
            # 3. 计算综合决策分数
            decision_score = (
                criteria.safety_concern * weights['safety_concern'] +
                criteria.information_trust * weights['information_trust'] +
                criteria.evacuation_cost * weights['evacuation_cost'] +
                criteria.property_attachment * weights['property_attachment']
            )
            
            # 4. 应用EDFT动态调整
            adjusted_score = self._apply_edft_adjustment(decision_score, context, criteria)
            
            # 5. 决策分类
            decision_type = self._classify_four_dimensional_decision(adjusted_score, context)
            
            # 6. 计算决策信心
            confidence = self._calculate_decision_confidence(criteria, weights, context)
            
            # 7. 生成决策解释
            explanation = self._generate_four_dimensional_explanation(
                criteria, weights, decision_type, context
            )
            
            return {
                'decision': decision_type.value,
                'decision_score': adjusted_score,
                'confidence': confidence,
                'criteria': {
                    'safety_concern': criteria.safety_concern,
                    'information_trust': criteria.information_trust,
                    'evacuation_cost': criteria.evacuation_cost,
                    'property_attachment': criteria.property_attachment
                },
                'weights': weights,
                'explanation': explanation,
                'components': {
                    'trust_details': criteria.trust_components,
                    'cost_details': criteria.cost_components,
                    'attachment_details': criteria.attachment_components
                }
            }
            
        except Exception as e:
            logger.error(f"四维决策制定失败: {e}")
            return {
                'decision': DecisionType.PREPARE.value,
                'decision_score': 0.5,
                'confidence': 0.5,
                'explanation': "决策计算失败，采用默认准备策略"
            }
    
    async def _assess_safety_concern(self, context: DecisionContext) -> float:
        """
        评估安全性关注 (w₁)
        """
        # 客观风险评估
        objective_risk = self._calculate_objective_risk(
            context.weather_data, 
            context.location_data
        )
        
        # 主观风险感知
        perceived_risk = self._calculate_perceived_risk(
            objective_risk, 
            context.agent_profile
        )
        
        # 个人脆弱性
        personal_vulnerability = self._assess_personal_vulnerability(
            context.agent_profile
        )
        
        # 综合安全关注
        safety_concern = (
            objective_risk * 0.4 +
            perceived_risk * 0.4 +
            personal_vulnerability * 0.2
        )
        
        return min(safety_concern, 1.0)
    
    def _calculate_objective_risk(
        self, 
        weather_data: Dict[str, Any], 
        location_data: Dict[str, Any]
    ) -> float:
        """
        计算客观风险
        """
        # 飓风强度
        hurricane_category = weather_data.get('hurricane_category', 1)
        wind_speed = weather_data.get('wind_speed', 80)
        storm_surge = weather_data.get('storm_surge', 3)
        
        # 位置脆弱性
        elevation = location_data.get('elevation', 10)
        distance_to_coast = location_data.get('distance_to_coast', 10)
        flood_zone = location_data.get('flood_zone', 'X')
        
        # 风险计算
        weather_risk = min((hurricane_category / 5.0) * 0.4 + 
                          (wind_speed / 200.0) * 0.3 + 
                          (storm_surge / 20.0) * 0.3, 1.0)
        
        location_risk = min((1.0 - elevation / 50.0) * 0.4 + 
                           (1.0 - distance_to_coast / 50.0) * 0.3 + 
                           (1.0 if flood_zone in ['A', 'V'] else 0.5) * 0.3, 1.0)
        
        return (weather_risk + location_risk) / 2.0
    
    def _calculate_perceived_risk(
        self, 
        objective_risk: float, 
        agent_profile: Dict[str, Any]
    ) -> float:
        """
        计算主观风险感知
        """
        # 风险规避倾向
        risk_aversion = agent_profile.get('risk_aversion', 0.5)
        
        # 过往经验
        evacuation_experience = agent_profile.get('evacuation_experience', 0.0)
        disaster_experience = agent_profile.get('disaster_experience', 0.0)
        
        # 年龄影响
        age = agent_profile.get('age', 35)
        age_factor = 1.2 if age > 65 or age < 18 else 1.0
        
        # 感知调整
        experience_adjustment = 1.0 + (evacuation_experience + disaster_experience) * 0.2
        risk_adjustment = 1.0 + (risk_aversion - 0.5) * 0.4
        
        perceived_risk = objective_risk * experience_adjustment * risk_adjustment * age_factor
        
        return min(perceived_risk, 1.0)
    
    def _assess_personal_vulnerability(self, agent_profile: Dict[str, Any]) -> float:
        """
        评估个人脆弱性
        """
        # 身体条件
        age = agent_profile.get('age', 35)
        health_status = agent_profile.get('health_status', 'good')
        mobility = agent_profile.get('mobility', 1.0)
        
        # 家庭责任
        has_children = agent_profile.get('has_children', False)
        has_elderly = agent_profile.get('has_elderly_dependents', False)
        family_size = agent_profile.get('family_size', 2)
        
        # 资源能力
        income_level = agent_profile.get('income_level', 'medium')
        vehicle_access = agent_profile.get('vehicle_access', True)
        
        # 脆弱性计算
        age_vulnerability = 0.8 if age > 65 or age < 18 else 0.3
        health_vulnerability = {'poor': 0.9, 'fair': 0.6, 'good': 0.3}.get(health_status, 0.3)
        mobility_vulnerability = 1.0 - mobility
        
        family_vulnerability = (
            (0.3 if has_children else 0.0) +
            (0.4 if has_elderly else 0.0) +
            min(family_size / 10.0, 0.3)
        )
        
        resource_vulnerability = (
            {'low': 0.7, 'medium': 0.4, 'high': 0.1}.get(income_level, 0.4) +
            (0.0 if vehicle_access else 0.3)
        ) / 2.0
        
        total_vulnerability = (
            age_vulnerability * 0.2 +
            health_vulnerability * 0.2 +
            mobility_vulnerability * 0.2 +
            family_vulnerability * 0.2 +
            resource_vulnerability * 0.2
        )
        
        return min(total_vulnerability, 1.0)
    
    def _extract_weather_severity(self, weather_data: Dict[str, Any]) -> float:
        """
        提取天气严重程度
        """
        hurricane_category = weather_data.get('hurricane_category', 1)
        return min(hurricane_category / 5.0, 1.0)
    
    def _calculate_panic_level(self, context: DecisionContext) -> float:
        """
        计算恐慌程度
        """
        # 风险紧迫性
        time_to_impact = context.time_context.get('time_to_impact', 24)
        urgency = max(0, 1.0 - time_to_impact / 48.0)
        
        # 社会恐慌指标
        social_media_sentiment = context.social_context.get('social_media_sentiment', 0.0)
        neighbor_panic = context.social_context.get('neighbor_panic_level', 0.0)
        
        # 个人恐慌倾向
        anxiety_level = context.agent_profile.get('anxiety_level', 0.5)
        
        panic_level = (urgency * 0.4 + 
                      abs(social_media_sentiment) * 0.3 + 
                      neighbor_panic * 0.2 + 
                      anxiety_level * 0.1)
        
        return min(panic_level, 1.0)
    
    def _calculate_social_pressure(self, context: DecisionContext) -> float:
        """
        计算社会压力
        """
        # 官方建议压力
        official_pressure = 1.0 if context.social_context.get('mandatory_evacuation', False) else 0.5
        
        # 邻居行为压力
        neighbor_evacuation_rate = context.social_context.get('neighbor_evacuation_rate', 0.0)
        
        # 家庭压力
        family_pressure = context.social_context.get('family_evacuation_pressure', 0.0)
        
        social_pressure = (official_pressure * 0.4 + 
                          neighbor_evacuation_rate * 0.3 + 
                          family_pressure * 0.3)
        
        return min(social_pressure, 1.0)
    
    def _calculate_dynamic_weights(self, context: DecisionContext) -> Dict[str, float]:
        """
        计算动态权重
        基于EDFT框架
        """
        base_weights = self.base_weights.copy()
        
        # 时间压力调整
        time_to_impact = context.time_context.get('time_to_impact', 24)
        if time_to_impact < 6:  # 6小时内，安全性权重增加
            base_weights['safety_concern'] += 0.1
            base_weights['property_attachment'] -= 0.05
            base_weights['evacuation_cost'] -= 0.05
        
        # 风险升级调整
        risk_trend = context.weather_data.get('risk_trend', 'stable')
        if risk_trend == 'increasing':
            base_weights['safety_concern'] += 0.05
            base_weights['information_trust'] += 0.05
            base_weights['property_attachment'] -= 0.1
        
        # 社会压力调整
        if context.social_context.get('mandatory_evacuation', False):
            base_weights['information_trust'] += 0.1
            base_weights['property_attachment'] -= 0.1
        
        # 归一化权重
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        return normalized_weights
    
    def _apply_edft_adjustment(
        self, 
        base_score: float, 
        context: DecisionContext, 
        criteria: FourDimensionalCriteria
    ) -> float:
        """
        应用EDFT动态调整
        """
        # 学习因子
        experience_factor = context.agent_profile.get('evacuation_experience', 0.0)
        learning_adjustment = experience_factor * 0.1
        
        # 时间压力因子
        time_to_impact = context.time_context.get('time_to_impact', 24)
        time_pressure = max(0, 1.0 - time_to_impact / 24.0)
        time_adjustment = time_pressure * 0.15
        
        # 信息质量因子
        info_quality = criteria.trust_components.total_trust if criteria.trust_components else 0.6
        info_adjustment = (info_quality - 0.5) * 0.1
        
        adjusted_score = base_score + learning_adjustment + time_adjustment + info_adjustment
        
        return min(max(adjusted_score, 0.0), 1.0)
    
    def _classify_four_dimensional_decision(
        self, 
        decision_score: float, 
        context: DecisionContext
    ) -> DecisionType:
        """
        基于四维标准分类决策
        """
        # 强制疏散令优先
        if context.social_context.get('mandatory_evacuation', False):
            return DecisionType.EMERGENCY_EVACUATE
        
        # 基于分数分类
        if decision_score >= self.decision_thresholds[DecisionType.EMERGENCY_EVACUATE]:
            return DecisionType.EMERGENCY_EVACUATE
        elif decision_score >= self.decision_thresholds[DecisionType.EVACUATE]:
            return DecisionType.EVACUATE
        elif decision_score >= self.decision_thresholds[DecisionType.PREPARE]:
            return DecisionType.PREPARE
        else:
            return DecisionType.STAY
    
    def _calculate_decision_confidence(
        self, 
        criteria: FourDimensionalCriteria, 
        weights: Dict[str, float], 
        context: DecisionContext
    ) -> float:
        """
        计算决策信心
        """
        # 标准一致性
        criteria_values = [
            criteria.safety_concern,
            criteria.information_trust,
            criteria.evacuation_cost,
            criteria.property_attachment
        ]
        consistency = 1.0 - np.std(criteria_values)
        
        # 信息质量
        info_quality = criteria.trust_components.total_trust if criteria.trust_components else 0.6
        
        # 经验因子
        experience = context.agent_profile.get('evacuation_experience', 0.0)
        
        confidence = (consistency * 0.4 + info_quality * 0.4 + experience * 0.2)
        
        return min(confidence, 1.0)
    
    def _generate_four_dimensional_explanation(
        self, 
        criteria: FourDimensionalCriteria, 
        weights: Dict[str, float], 
        decision_type: DecisionType, 
        context: DecisionContext
    ) -> str:
        """
        生成四维决策解释
        """
        explanations = []
        
        # 主导因素分析
        dominant_factor = max(
            [
                ('安全性关注', criteria.safety_concern * weights['safety_concern']),
                ('信息信任', criteria.information_trust * weights['information_trust']),
                ('疏散成本', criteria.evacuation_cost * weights['evacuation_cost']),
                ('财产依恋', criteria.property_attachment * weights['property_attachment'])
            ],
            key=lambda x: x[1]
        )
        
        explanations.append(f"主导因素：{dominant_factor[0]}")
        
        # 具体因素解释
        if criteria.safety_concern > 0.7:
            explanations.append("安全风险较高")
        
        if criteria.trust_components:
            trust_explanation = self.trust_assessor.get_trust_explanation(criteria.trust_components)
            explanations.append(f"信息信任：{trust_explanation}")
        
        if criteria.cost_components:
            cost_explanation = self.cost_assessor.get_cost_explanation(criteria.cost_components)
            explanations.append(f"疏散成本：{cost_explanation}")
        
        if criteria.attachment_components:
            attachment_explanation = self.attachment_assessor.get_attachment_explanation(criteria.attachment_components)
            explanations.append(f"财产依恋：{attachment_explanation}")
        
        # 决策合理性
        explanations.append(f"综合评估建议：{decision_type.value}")
        
        return "；".join(explanations)