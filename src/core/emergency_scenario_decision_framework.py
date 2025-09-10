"""
特殊场景决策框架 (Emergency Scenario Decision Framework)
针对3.3特殊场景移动要求的完整决策链实现
整合风险评估、疏散决策、移动规划的综合决策系统
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
from datetime import datetime, timedelta

# 导入现有框架
from .enhanced_decision_framework import (
    EnhancedDecisionFramework, 
    RiskLevel, 
    DecisionType,
    SocialContext,
    PersonalCapacity,
    RiskPerception,
    CognitiveBias
)
from .four_dimensional_framework import (
    FourDimensionalDecisionFramework,
    FourDimensionalCriteria,
    DecisionContext as FourDimContext
)
from .evacuation_prediction_models import EvacuationPredictionFramework

logger = logging.getLogger(__name__)

class EmergencyScenarioType(Enum):
    """紧急场景类型"""
    HURRICANE_APPROACH = "hurricane_approach"      # 飓风接近
    FLOOD_RISING = "flood_rising"                 # 洪水上涨
    WILDFIRE_SPREAD = "wildfire_spread"           # 野火蔓延
    EARTHQUAKE_AFTERSHOCK = "earthquake_aftershock"  # 地震余震
    CHEMICAL_SPILL = "chemical_spill"             # 化学泄漏
    TERRORIST_THREAT = "terrorist_threat"         # 恐怖威胁

class MovementUrgency(Enum):
    """移动紧急程度"""
    ROUTINE = "routine"           # 常规移动
    CAUTIOUS = "cautious"         # 谨慎移动
    URGENT = "urgent"             # 紧急移动
    IMMEDIATE = "immediate"       # 立即移动

class MovementPattern(Enum):
    """移动模式"""
    NORMAL_ACTIVITY = "normal_activity"       # 正常活动
    PREPARATION = "preparation"               # 准备活动
    EVACUATION = "evacuation"                # 疏散移动
    SHELTER_SEEKING = "shelter_seeking"       # 寻找避难所
    RESOURCE_GATHERING = "resource_gathering" # 资源收集
    FAMILY_REUNION = "family_reunion"        # 家庭团聚

@dataclass
class EmergencyContext:
    """紧急场景上下文"""
    scenario_type: EmergencyScenarioType
    severity_level: float = 0.5              # 严重程度 (0-1)
    time_to_impact: float = 24.0             # 预计影响时间(小时)
    affected_areas: List[int] = field(default_factory=list)  # 受影响区域
    evacuation_zones: List[int] = field(default_factory=list)  # 疏散区域
    safe_zones: List[int] = field(default_factory=list)      # 安全区域
    
    # 官方信息
    official_warnings: List[str] = field(default_factory=list)
    evacuation_orders: Dict[int, str] = field(default_factory=dict)  # 区域->命令类型
    transportation_status: Dict[str, str] = field(default_factory=dict)
    
    # 实时状态
    current_conditions: Dict[str, Any] = field(default_factory=dict)
    forecast_conditions: Dict[str, Any] = field(default_factory=dict)
    infrastructure_status: Dict[str, str] = field(default_factory=dict)

@dataclass
class MovementDecision:
    """移动决策结果"""
    decision_type: DecisionType
    movement_pattern: MovementPattern
    urgency_level: MovementUrgency
    target_location: Optional[int] = None
    departure_time: Optional[datetime] = None
    travel_mode: str = "car"
    route_preference: str = "fastest"
    
    # 决策依据
    risk_score: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    
    # 四维标准评分
    safety_concern: float = 0.0
    information_trust: float = 0.0
    evacuation_cost: float = 0.0
    property_attachment: float = 0.0
    
    # 个人、社交、经济属性影响
    personal_influence: Dict[str, float] = field(default_factory=dict)
    social_influence: Dict[str, float] = field(default_factory=dict)
    economic_influence: Dict[str, float] = field(default_factory=dict)

class EmergencyScenarioDecisionFramework:
    """特殊场景决策框架"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化特殊场景决策框架"""
        self.config = config
        
        # 初始化子框架
        self.enhanced_framework = EnhancedDecisionFramework(config)
        self.four_dimensional_framework = FourDimensionalDecisionFramework(
            use_dynamic_weights=True
        )
        self.prediction_framework = EvacuationPredictionFramework()
        
        # 场景特定配置
        self.scenario_configs = {
            EmergencyScenarioType.HURRICANE_APPROACH: {
                'risk_escalation_rate': 0.1,
                'time_pressure_factor': 2.0,
                'social_influence_weight': 0.3,
                'preparation_time_hours': 48
            },
            EmergencyScenarioType.FLOOD_RISING: {
                'risk_escalation_rate': 0.15,
                'time_pressure_factor': 3.0,
                'social_influence_weight': 0.25,
                'preparation_time_hours': 24
            },
            EmergencyScenarioType.WILDFIRE_SPREAD: {
                'risk_escalation_rate': 0.2,
                'time_pressure_factor': 4.0,
                'social_influence_weight': 0.2,
                'preparation_time_hours': 12
            }
        }
        
        # 移动模式配置
        self.movement_patterns = {
            MovementPattern.NORMAL_ACTIVITY: {
                'frequency_multiplier': 1.0,
                'distance_preference': 'local',
                'time_flexibility': 'high'
            },
            MovementPattern.PREPARATION: {
                'frequency_multiplier': 1.5,
                'distance_preference': 'extended',
                'time_flexibility': 'medium'
            },
            MovementPattern.EVACUATION: {
                'frequency_multiplier': 0.3,
                'distance_preference': 'long_distance',
                'time_flexibility': 'low'
            }
        }
    
    async def make_emergency_decision(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        current_location: int,
        time_context: Dict[str, Any],
        social_context: Optional[SocialContext] = None
    ) -> MovementDecision:
        """
        制定紧急场景下的移动决策
        
        Args:
            agent_profile: 智能体档案
            emergency_context: 紧急场景上下文
            current_location: 当前位置
            time_context: 时间上下文
            social_context: 社会上下文
            
        Returns:
            MovementDecision: 移动决策结果
        """
        try:
            # 1. 综合风险评估
            risk_assessment = await self._assess_emergency_risk(
                agent_profile, emergency_context, current_location, time_context
            )
            
            # 2. 四维决策标准评估
            four_dim_context = self._build_four_dimensional_context(
                agent_profile, emergency_context, current_location, time_context
            )
            four_dim_criteria = await self.four_dimensional_framework.assess_four_dimensional_criteria(
                four_dim_context
            )
            
            # 3. 增强决策框架评估
            if social_context is None:
                social_context = self._build_social_context(emergency_context, agent_profile)
            
            enhanced_decision = await self.enhanced_framework.make_evacuation_decision(
                risk_assessment, agent_profile, social_context, time_context,
                weather_data=emergency_context.current_conditions,
                location_data={'current_location': current_location},
                use_four_dimensional=True
            )
            
            # 4. 预测模型验证
            prediction_result = await self._validate_with_prediction_models(
                agent_profile, emergency_context, risk_assessment
            )
            
            # 5. 综合决策融合
            final_decision = await self._integrate_decision_results(
                risk_assessment,
                four_dim_criteria,
                enhanced_decision,
                prediction_result,
                agent_profile,
                emergency_context,
                current_location,
                time_context
            )
            
            # 6. 移动规划生成
            movement_plan = await self._generate_movement_plan(
                final_decision, agent_profile, emergency_context, time_context
            )
            
            return movement_plan
            
        except Exception as e:
            logger.error(f"紧急决策制定失败: {e}")
            # 回退到保守决策
            return self._generate_conservative_decision(
                agent_profile, emergency_context, current_location
            )
    
    async def _assess_emergency_risk(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        current_location: int,
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估紧急场景风险"""
        
        # 基础风险评估
        base_risk = emergency_context.severity_level
        
        # 位置风险调整
        location_risk = self._calculate_location_risk(
            current_location, emergency_context
        )
        
        # 时间风险调整
        time_risk = self._calculate_time_risk(
            emergency_context.time_to_impact, emergency_context.scenario_type
        )
        
        # 个人脆弱性评估
        personal_vulnerability = self._assess_personal_vulnerability(
            agent_profile, emergency_context.scenario_type
        )
        
        # 综合风险计算
        composite_risk = (
            base_risk * 0.3 +
            location_risk * 0.3 +
            time_risk * 0.2 +
            personal_vulnerability * 0.2
        )
        
        return {
            'composite_risk': min(composite_risk, 1.0),
            'base_risk': base_risk,
            'location_risk': location_risk,
            'time_risk': time_risk,
            'personal_vulnerability': personal_vulnerability,
            'risk_level': self._classify_risk_level(composite_risk),
            'risk_factors': self._identify_risk_factors(
                agent_profile, emergency_context, current_location
            )
        }
    
    def _calculate_location_risk(
        self, 
        location: int, 
        emergency_context: EmergencyContext
    ) -> float:
        """计算位置相关风险"""
        
        # 检查是否在疏散区域
        if location in emergency_context.evacuation_zones:
            return 0.9
        
        # 检查是否在受影响区域
        if location in emergency_context.affected_areas:
            return 0.7
        
        # 检查是否在安全区域
        if location in emergency_context.safe_zones:
            return 0.1
        
        # 默认中等风险
        return 0.5
    
    def _calculate_time_risk(
        self, 
        time_to_impact: float, 
        scenario_type: EmergencyScenarioType
    ) -> float:
        """计算时间相关风险"""
        
        scenario_config = self.scenario_configs.get(scenario_type, {})
        escalation_rate = scenario_config.get('risk_escalation_rate', 0.1)
        
        # 时间越短，风险越高
        if time_to_impact <= 0:
            return 1.0
        elif time_to_impact <= 6:
            return 0.9
        elif time_to_impact <= 12:
            return 0.7
        elif time_to_impact <= 24:
            return 0.5
        else:
            return max(0.1, 0.5 - (time_to_impact - 24) * escalation_rate)
    
    def _assess_personal_vulnerability(
        self, 
        agent_profile: Dict[str, Any], 
        scenario_type: EmergencyScenarioType
    ) -> float:
        """评估个人脆弱性"""
        
        vulnerability = 0.0
        
        # 年龄因素
        age = agent_profile.get('age', 35)
        if age < 18 or age > 65:
            vulnerability += 0.2
        
        # 健康状况
        health = agent_profile.get('health_status', 'good')
        if health in ['poor', 'disabled']:
            vulnerability += 0.3
        elif health == 'fair':
            vulnerability += 0.1
        
        # 交通工具可达性
        has_car = agent_profile.get('has_car', True)
        if not has_car:
            vulnerability += 0.2
        
        # 经济资源
        income = agent_profile.get('income_level', 'medium')
        if income == 'low':
            vulnerability += 0.15
        
        # 社会支持网络
        social_support = agent_profile.get('social_support_level', 0.5)
        vulnerability += (1.0 - social_support) * 0.15
        
        return min(vulnerability, 1.0)
    
    def _build_four_dimensional_context(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        current_location: int,
        time_context: Dict[str, Any]
    ) -> FourDimContext:
        """构建四维决策上下文"""
        
        return FourDimContext(
            weather_data=emergency_context.current_conditions,
            agent_profile=agent_profile,
            social_context={
                'evacuation_orders': emergency_context.evacuation_orders,
                'official_warnings': emergency_context.official_warnings,
                'community_evacuation_rate': self._estimate_community_evacuation_rate(
                    current_location, emergency_context
                )
            },
            time_context=time_context,
            location_data={
                'current_location': current_location,
                'safe_zones': emergency_context.safe_zones,
                'evacuation_zones': emergency_context.evacuation_zones,
                'transportation_status': emergency_context.transportation_status
            },
            information_sources=self._build_information_sources(emergency_context),
            information_content={
                'warnings': emergency_context.official_warnings,
                'severity': emergency_context.severity_level,
                'time_to_impact': emergency_context.time_to_impact
            }
        )
    
    def _build_social_context(
        self, 
        emergency_context: EmergencyContext, 
        agent_profile: Dict[str, Any]
    ) -> SocialContext:
        """构建社会上下文"""
        
        # 检查强制疏散令
        mandatory_evacuation = any(
            order_type == 'mandatory' 
            for order_type in emergency_context.evacuation_orders.values()
        )
        
        return SocialContext(
            family_influence=agent_profile.get('family_influence', 0.0),
            neighbor_evacuation_rate=self._estimate_neighbor_evacuation_rate(emergency_context),
            community_cohesion=agent_profile.get('community_cohesion', 0.5),
            official_recommendation=self._get_official_recommendation(emergency_context),
            mandatory_evacuation=mandatory_evacuation,
            social_media_sentiment=self._estimate_social_media_sentiment(emergency_context)
        )
    
    async def _validate_with_prediction_models(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用预测模型验证决策"""
        
        try:
            # 构建预测输入
            prediction_input = {
                'risk_score': risk_assessment['composite_risk'],
                'time_to_impact': emergency_context.time_to_impact,
                'severity': emergency_context.severity_level,
                'has_car': agent_profile.get('has_car', True),
                'age': agent_profile.get('age', 35),
                'income': agent_profile.get('income_level', 'medium'),
                'education': agent_profile.get('education_level', 'high_school'),
                'evacuation_experience': agent_profile.get('evacuation_experience', False)
            }
            
            # 使用逻辑回归模型预测
            lr_prediction = self.prediction_framework.predict_evacuation_probability(
                prediction_input, model_type='logistic_regression'
            )
            
            # 使用动态离散选择模型预测
            ddc_prediction = self.prediction_framework.predict_evacuation_probability(
                prediction_input, model_type='dynamic_discrete_choice'
            )
            
            return {
                'lr_evacuation_probability': lr_prediction,
                'ddc_evacuation_probability': ddc_prediction,
                'model_consensus': abs(lr_prediction - ddc_prediction) < 0.2,
                'average_probability': (lr_prediction + ddc_prediction) / 2
            }
            
        except Exception as e:
            logger.error(f"预测模型验证失败: {e}")
            return {
                'lr_evacuation_probability': 0.5,
                'ddc_evacuation_probability': 0.5,
                'model_consensus': False,
                'average_probability': 0.5
            }
    
    async def _integrate_decision_results(
        self,
        risk_assessment: Dict[str, Any],
        four_dim_criteria: FourDimensionalCriteria,
        enhanced_decision: Dict[str, Any],
        prediction_result: Dict[str, Any],
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        current_location: int,
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """整合多个决策结果"""
        
        # 提取各框架的决策建议
        enhanced_decision_type = enhanced_decision.get('decision_type', 'stay')
        prediction_probability = prediction_result.get('average_probability', 0.5)
        
        # 计算综合决策分数
        decision_scores = {
            'stay': 0.0,
            'prepare': 0.0,
            'evacuate': 0.0,
            'emergency_evacuate': 0.0
        }
        
        # 基于风险评估的分数
        risk_score = risk_assessment['composite_risk']
        if risk_score < 0.3:
            decision_scores['stay'] += 0.4
        elif risk_score < 0.6:
            decision_scores['prepare'] += 0.4
        elif risk_score < 0.8:
            decision_scores['evacuate'] += 0.4
        else:
            decision_scores['emergency_evacuate'] += 0.4
        
        # 基于四维标准的分数
        four_dim_score = (
            four_dim_criteria.safety_concern * 0.4 +
            four_dim_criteria.information_trust * 0.25 +
            four_dim_criteria.evacuation_cost * 0.2 +
            four_dim_criteria.property_attachment * 0.15
        )
        
        if four_dim_score > 0.7:
            decision_scores['evacuate'] += 0.3
        elif four_dim_score > 0.5:
            decision_scores['prepare'] += 0.3
        else:
            decision_scores['stay'] += 0.3
        
        # 基于预测模型的分数
        if prediction_probability > 0.7:
            decision_scores['evacuate'] += 0.3
        elif prediction_probability > 0.4:
            decision_scores['prepare'] += 0.3
        else:
            decision_scores['stay'] += 0.3
        
        # 选择最高分数的决策
        final_decision_type = max(decision_scores, key=decision_scores.get)
        final_confidence = decision_scores[final_decision_type]
        
        # 分析个人、社交、经济属性影响
        personal_influence = self._analyze_personal_influence(
            agent_profile, emergency_context, risk_assessment
        )
        social_influence = self._analyze_social_influence(
            agent_profile, emergency_context, prediction_result
        )
        economic_influence = self._analyze_economic_influence(
            agent_profile, emergency_context, four_dim_criteria
        )
        
        return {
            'decision_type': final_decision_type,
            'confidence': final_confidence,
            'risk_score': risk_score,
            'four_dimensional_score': four_dim_score,
            'prediction_probability': prediction_probability,
            'decision_scores': decision_scores,
            'personal_influence': personal_influence,
            'social_influence': social_influence,
            'economic_influence': economic_influence,
            'reasoning': self._generate_integrated_reasoning(
                final_decision_type, risk_assessment, four_dim_criteria, 
                enhanced_decision, prediction_result
            )
        }
    
    async def _generate_movement_plan(
        self,
        decision_result: Dict[str, Any],
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        time_context: Dict[str, Any]
    ) -> MovementDecision:
        """生成移动规划"""
        
        decision_type_str = decision_result['decision_type']
        decision_type = DecisionType(decision_type_str.upper())
        
        # 确定移动模式
        movement_pattern = self._determine_movement_pattern(
            decision_type, emergency_context, agent_profile
        )
        
        # 确定紧急程度
        urgency_level = self._determine_urgency_level(
            decision_type, decision_result['risk_score'], emergency_context.time_to_impact
        )
        
        # 选择目标位置
        target_location = self._select_target_location(
            decision_type, agent_profile, emergency_context
        )
        
        # 确定出发时间
        departure_time = self._calculate_departure_time(
            decision_type, urgency_level, emergency_context.time_to_impact
        )
        
        # 选择交通方式
        travel_mode = self._select_travel_mode(
            agent_profile, urgency_level, emergency_context
        )
        
        # 路径偏好
        route_preference = self._determine_route_preference(
            urgency_level, emergency_context
        )
        
        return MovementDecision(
            decision_type=decision_type,
            movement_pattern=movement_pattern,
            urgency_level=urgency_level,
            target_location=target_location,
            departure_time=departure_time,
            travel_mode=travel_mode,
            route_preference=route_preference,
            risk_score=decision_result['risk_score'],
            confidence=decision_result['confidence'],
            reasoning=decision_result['reasoning'],
            safety_concern=decision_result.get('four_dimensional_score', 0.0),
            information_trust=0.0,  # 从四维标准中提取
            evacuation_cost=0.0,   # 从四维标准中提取
            property_attachment=0.0,  # 从四维标准中提取
            personal_influence=decision_result['personal_influence'],
            social_influence=decision_result['social_influence'],
            economic_influence=decision_result['economic_influence']
        )
    
    def _analyze_personal_influence(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, float]:
        """分析个人属性影响"""
        
        influence = {}
        
        # 年龄影响
        age = agent_profile.get('age', 35)
        if age < 25:
            influence['age_factor'] = 0.1  # 年轻人更愿意冒险
        elif age > 65:
            influence['age_factor'] = -0.2  # 老年人行动不便
        else:
            influence['age_factor'] = 0.0
        
        # 健康状况影响
        health = agent_profile.get('health_status', 'good')
        health_mapping = {'excellent': 0.1, 'good': 0.0, 'fair': -0.1, 'poor': -0.3}
        influence['health_factor'] = health_mapping.get(health, 0.0)
        
        # 教育水平影响
        education = agent_profile.get('education_level', 'high_school')
        education_mapping = {
            'elementary': -0.1, 'high_school': 0.0, 
            'college': 0.1, 'graduate': 0.15
        }
        influence['education_factor'] = education_mapping.get(education, 0.0)
        
        # 疏散经验影响
        experience = agent_profile.get('evacuation_experience', False)
        influence['experience_factor'] = 0.2 if experience else -0.1
        
        return influence
    
    def _analyze_social_influence(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        prediction_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """分析社交属性影响"""
        
        influence = {}
        
        # 家庭影响
        family_size = agent_profile.get('family_size', 1)
        influence['family_size_factor'] = min(family_size * 0.1, 0.3)
        
        # 社区凝聚力影响
        community_cohesion = agent_profile.get('community_cohesion', 0.5)
        influence['community_factor'] = community_cohesion * 0.2
        
        # 社会网络影响
        social_network_size = agent_profile.get('social_network_size', 10)
        influence['network_factor'] = min(social_network_size * 0.01, 0.2)
        
        # 邻居疏散率影响
        neighbor_evacuation = self._estimate_neighbor_evacuation_rate(emergency_context)
        influence['neighbor_factor'] = neighbor_evacuation * 0.3
        
        return influence
    
    def _analyze_economic_influence(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        four_dim_criteria: FourDimensionalCriteria
    ) -> Dict[str, float]:
        """分析经济属性影响"""
        
        influence = {}
        
        # 收入水平影响
        income = agent_profile.get('income_level', 'medium')
        income_mapping = {'low': -0.2, 'medium': 0.0, 'high': 0.1, 'very_high': 0.15}
        influence['income_factor'] = income_mapping.get(income, 0.0)
        
        # 房屋所有权影响
        home_ownership = agent_profile.get('home_ownership', False)
        influence['ownership_factor'] = -0.15 if home_ownership else 0.05
        
        # 交通工具影响
        has_car = agent_profile.get('has_car', True)
        influence['transportation_factor'] = 0.2 if has_car else -0.3
        
        # 疏散成本影响（从四维标准获取）
        if four_dim_criteria.cost_components:
            cost_ratio = four_dim_criteria.cost_components.total_cost / 1000.0
            influence['cost_factor'] = -min(cost_ratio, 0.3)
        else:
            influence['cost_factor'] = 0.0
        
        return influence
    
    # 辅助方法
    def _classify_risk_level(self, risk_score: float) -> str:
        """分类风险等级"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(
        self, 
        agent_profile: Dict[str, Any], 
        emergency_context: EmergencyContext, 
        current_location: int
    ) -> List[str]:
        """识别风险因素"""
        factors = []
        
        if current_location in emergency_context.evacuation_zones:
            factors.append("位于疏散区域")
        
        if emergency_context.time_to_impact < 12:
            factors.append("时间紧迫")
        
        if emergency_context.severity_level > 0.7:
            factors.append("灾害严重")
        
        age = agent_profile.get('age', 35)
        if age < 18 or age > 65:
            factors.append("年龄脆弱性")
        
        if not agent_profile.get('has_car', True):
            factors.append("交通不便")
        
        return factors
    
    def _estimate_community_evacuation_rate(
        self, location: int, emergency_context: EmergencyContext
    ) -> float:
        """估算社区疏散率"""
        if location in emergency_context.evacuation_zones:
            return 0.8
        elif location in emergency_context.affected_areas:
            return 0.4
        else:
            return 0.1
    
    def _estimate_neighbor_evacuation_rate(
        self, emergency_context: EmergencyContext
    ) -> float:
        """估算邻居疏散率"""
        return emergency_context.severity_level * 0.6
    
    def _get_official_recommendation(
        self, emergency_context: EmergencyContext
    ) -> str:
        """获取官方建议"""
        if emergency_context.evacuation_orders:
            return "evacuate"
        elif emergency_context.official_warnings:
            return "prepare"
        else:
            return "monitor"
    
    def _estimate_social_media_sentiment(
        self, emergency_context: EmergencyContext
    ) -> float:
        """估算社交媒体情绪"""
        return emergency_context.severity_level * 0.5 - 0.25
    
    def _build_information_sources(self, emergency_context: EmergencyContext) -> List:
        """构建信息源列表"""
        # 简化实现，返回空列表
        return []
    
    def _determine_movement_pattern(
        self, 
        decision_type: DecisionType, 
        emergency_context: EmergencyContext, 
        agent_profile: Dict[str, Any]
    ) -> MovementPattern:
        """确定移动模式"""
        if decision_type == DecisionType.EVACUATE:
            return MovementPattern.EVACUATION
        elif decision_type == DecisionType.PREPARE:
            return MovementPattern.PREPARATION
        else:
            return MovementPattern.NORMAL_ACTIVITY
    
    def _determine_urgency_level(
        self, 
        decision_type: DecisionType, 
        risk_score: float, 
        time_to_impact: float
    ) -> MovementUrgency:
        """确定紧急程度"""
        if decision_type == DecisionType.EMERGENCY_EVACUATE or time_to_impact < 6:
            return MovementUrgency.IMMEDIATE
        elif decision_type == DecisionType.EVACUATE or risk_score > 0.7:
            return MovementUrgency.URGENT
        elif decision_type == DecisionType.PREPARE:
            return MovementUrgency.CAUTIOUS
        else:
            return MovementUrgency.ROUTINE
    
    def _select_target_location(
        self, 
        decision_type: DecisionType, 
        agent_profile: Dict[str, Any], 
        emergency_context: EmergencyContext
    ) -> Optional[int]:
        """选择目标位置"""
        if decision_type in [DecisionType.EVACUATE, DecisionType.EMERGENCY_EVACUATE]:
            # 优先选择安全区域
            if emergency_context.safe_zones:
                return emergency_context.safe_zones[0]
            else:
                return None
        else:
            return None
    
    def _calculate_departure_time(
        self, 
        decision_type: DecisionType, 
        urgency_level: MovementUrgency, 
        time_to_impact: float
    ) -> Optional[datetime]:
        """计算出发时间"""
        if decision_type in [DecisionType.EVACUATE, DecisionType.EMERGENCY_EVACUATE]:
            if urgency_level == MovementUrgency.IMMEDIATE:
                return datetime.now()
            elif urgency_level == MovementUrgency.URGENT:
                return datetime.now() + timedelta(hours=1)
            else:
                return datetime.now() + timedelta(hours=min(6, time_to_impact * 0.5))
        else:
            return None
    
    def _select_travel_mode(
        self, 
        agent_profile: Dict[str, Any], 
        urgency_level: MovementUrgency, 
        emergency_context: EmergencyContext
    ) -> str:
        """选择交通方式"""
        has_car = agent_profile.get('has_car', True)
        
        if urgency_level == MovementUrgency.IMMEDIATE:
            return "car" if has_car else "public_transport"
        elif has_car:
            return "car"
        else:
            return "public_transport"
    
    def _determine_route_preference(
        self, urgency_level: MovementUrgency, emergency_context: EmergencyContext
    ) -> str:
        """确定路径偏好"""
        if urgency_level in [MovementUrgency.IMMEDIATE, MovementUrgency.URGENT]:
            return "fastest"
        else:
            return "safest"
    
    def _generate_integrated_reasoning(
        self,
        decision_type: str,
        risk_assessment: Dict[str, Any],
        four_dim_criteria: FourDimensionalCriteria,
        enhanced_decision: Dict[str, Any],
        prediction_result: Dict[str, Any]
    ) -> str:
        """生成综合推理"""
        
        reasoning_parts = []
        
        # 风险评估部分
        risk_level = risk_assessment['risk_level']
        reasoning_parts.append(f"综合风险评估为{risk_level}级别")
        
        # 四维标准部分
        reasoning_parts.append(
            f"四维决策标准显示安全关注度{four_dim_criteria.safety_concern:.2f}"
        )
        
        # 预测模型部分
        avg_prob = prediction_result['average_probability']
        reasoning_parts.append(f"预测模型显示疏散概率为{avg_prob:.2f}")
        
        # 最终决策
        reasoning_parts.append(f"综合分析建议{decision_type}")
        
        return "；".join(reasoning_parts)
    
    def _generate_conservative_decision(
        self,
        agent_profile: Dict[str, Any],
        emergency_context: EmergencyContext,
        current_location: int
    ) -> MovementDecision:
        """生成保守决策（回退方案）"""
        
        # 如果在疏散区域，建议疏散
        if current_location in emergency_context.evacuation_zones:
            decision_type = DecisionType.EVACUATE
            movement_pattern = MovementPattern.EVACUATION
            urgency_level = MovementUrgency.URGENT
        else:
            decision_type = DecisionType.PREPARE
            movement_pattern = MovementPattern.PREPARATION
            urgency_level = MovementUrgency.CAUTIOUS
        
        return MovementDecision(
            decision_type=decision_type,
            movement_pattern=movement_pattern,
            urgency_level=urgency_level,
            target_location=emergency_context.safe_zones[0] if emergency_context.safe_zones else None,
            departure_time=datetime.now() + timedelta(hours=2),
            travel_mode="car" if agent_profile.get('has_car', True) else "public_transport",
            route_preference="safest",
            risk_score=0.7,
            confidence=0.6,
            reasoning="基于保守策略的回退决策",
            personal_influence={'conservative_bias': 0.3},
            social_influence={'safety_priority': 0.4},
            economic_influence={'risk_aversion': 0.2}
        )