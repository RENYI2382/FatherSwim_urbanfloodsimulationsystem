"""
增强决策框架模块
基于飓风人类行为逻辑的决策框架重构
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from .deepseek_integration import get_deepseek_client
from .four_dimensional_framework import FourDimensionalDecisionFramework, DecisionContext

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
class CognitiveBias:
    """认知偏差参数"""
    availability_heuristic: float = 0.3  # 可得性启发式
    anchoring_effect: float = 0.2        # 锚定效应
    loss_aversion: float = 0.4           # 损失厌恶
    optimism_bias: float = 0.25          # 乐观偏差
    social_proof: float = 0.35           # 社会认同

@dataclass
class SocialContext:
    """社会环境上下文"""
    family_influence: float = 0.0        # 家庭影响 (-1到1)
    neighbor_evacuation_rate: float = 0.0  # 邻居疏散率
    community_cohesion: float = 0.5      # 社区凝聚力
    official_recommendation: str = "none"  # 官方建议
    mandatory_evacuation: bool = False    # 强制疏散令
    social_media_sentiment: float = 0.0   # 社交媒体情绪

@dataclass
class PersonalCapacity:
    """个人能力评估"""
    transportation_access: float = 1.0    # 交通工具可得性
    financial_resources: float = 0.5      # 经济资源
    physical_mobility: float = 1.0        # 身体机动性
    information_access: float = 0.8       # 信息获取能力
    social_support: float = 0.6           # 社会支持网络
    evacuation_experience: float = 0.3    # 疏散经验

@dataclass
class RiskPerception:
    """风险感知评估"""
    objective_risk: float = 0.5           # 客观风险
    perceived_risk: float = 0.5           # 感知风险
    risk_confidence: float = 0.5          # 风险判断信心
    temporal_distance: float = 0.5        # 时间距离感知
    spatial_distance: float = 0.5         # 空间距离感知

class EnhancedDecisionFramework:
    """增强决策框架"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化增强决策框架"""
        self.config = config
        self.use_deepseek = config.get('use_deepseek', True)
        
        # 初始化四维决策框架
        self.four_dimensional_framework = FourDimensionalDecisionFramework(
            use_dynamic_weights=config.get('use_dynamic_weights', True)
        )
        
        # 认知偏差参数
        self.cognitive_bias = CognitiveBias()
        
        # 传统五维权重（向后兼容）
        self.decision_weights = config.get('decision_weights', {
            'risk_perception': 0.35,
            'personal_capacity': 0.25,
            'social_influence': 0.20,
            'cognitive_bias': 0.15,
            'situational_factors': 0.05
        })
        
        # 风险阈值配置
        self.risk_thresholds = {
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.95
        }
    
    async def assess_comprehensive_risk(
        self,
        weather_data: Dict[str, Any],
        agent_profile: Dict[str, Any],
        current_location: int,
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """综合风险评估"""
        
        # 1. 客观风险评估
        objective_risk = self._calculate_objective_risk(weather_data, current_location)
        
        # 2. 个人风险感知
        risk_perception = self._assess_risk_perception(
            objective_risk, agent_profile, time_context
        )
        
        # 3. DeepSeek-R1增强评估（如果启用）
        deepseek_assessment = None
        if self.use_deepseek:
            try:
                client = await get_deepseek_client()
                async with client:
                    deepseek_assessment = await client.assess_hurricane_risk(
                        weather_data, agent_profile, current_location
                    )
            except Exception as e:
                logger.warning(f"DeepSeek评估失败，使用传统方法: {e}")
        
        # 4. 综合风险评分
        final_risk_score = self._integrate_risk_assessments(
            objective_risk, risk_perception, deepseek_assessment
        )
        
        # 5. 风险等级分类
        risk_level = self._classify_risk_level(final_risk_score)
        
        return {
            'risk_score': final_risk_score,
            'risk_level': risk_level.value,
            'objective_risk': objective_risk,
            'perceived_risk': risk_perception.perceived_risk,
            'risk_confidence': risk_perception.risk_confidence,
            'deepseek_assessment': deepseek_assessment,
            'assessment_components': {
                'weather_severity': self._get_weather_severity(weather_data),
                'location_vulnerability': self._get_location_vulnerability(current_location),
                'personal_vulnerability': self._get_personal_vulnerability(agent_profile)
            }
        }
    
    async def make_evacuation_decision(
        self,
        risk_assessment: Dict[str, Any],
        agent_profile: Dict[str, Any],
        social_context: SocialContext,
        time_context: Dict[str, Any],
        weather_data: Dict[str, Any] = None,
        location_data: Dict[str, Any] = None,
        use_four_dimensional: bool = True
    ) -> Dict[str, Any]:
        """疏散决策制定"""
        
        # 优先使用四维决策标准
        if use_four_dimensional and self.config.get('use_four_dimensional', True):
            try:
                # 构建决策上下文
                context = DecisionContext(
                    agent_profile=agent_profile,
                    weather_data=weather_data or {},
                    location_data=location_data or {},
                    social_context=social_context.__dict__ if hasattr(social_context, '__dict__') else social_context,
                    time_context=time_context
                )
                
                # 使用四维决策框架
                four_dim_result = await self.four_dimensional_framework.make_four_dimensional_decision(context)
                
                # 添加传统决策信息作为补充
                traditional_result = await self._make_traditional_decision(
                    risk_assessment, agent_profile, social_context, time_context
                )
                
                # 合并结果
                return {
                    **four_dim_result,
                    'decision_method': 'four_dimensional',
                    'traditional_backup': traditional_result,
                    'four_dimensional_details': four_dim_result.get('detailed_analysis', {})
                }
                
            except Exception as e:
                logger.warning(f"四维决策失败，回退到传统方法: {e}")
                use_four_dimensional = False
        
        # 传统五维决策方法（向后兼容）
        return await self._make_traditional_decision(
            risk_assessment, agent_profile, social_context, time_context
        )
    
    async def _make_traditional_decision(
        self,
        risk_assessment: Dict[str, Any],
        agent_profile: Dict[str, Any],
        social_context: SocialContext,
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """传统五维决策方法"""
        
        # 1. 个人能力评估
        personal_capacity = self._assess_personal_capacity(agent_profile)
        
        # 2. 社会影响评估
        social_influence_score = self._calculate_social_influence(social_context)
        
        # 3. 认知偏差调整
        bias_adjusted_risk = self._apply_cognitive_bias(
            risk_assessment['risk_score'], agent_profile
        )
        
        # 4. DeepSeek-R1决策增强
        deepseek_decision = None
        if self.use_deepseek:
            try:
                client = await get_deepseek_client()
                async with client:
                    deepseek_decision = await client.make_evacuation_decision(
                        risk_assessment, agent_profile, social_context.__dict__
                    )
            except Exception as e:
                logger.warning(f"DeepSeek决策失败，使用传统方法: {e}")
        
        # 5. 综合决策计算
        decision_score = self._calculate_decision_score(
            bias_adjusted_risk,
            personal_capacity,
            social_influence_score,
            social_context,
            deepseek_decision
        )
        
        # 6. 决策分类
        decision_type = self._classify_decision(decision_score, social_context)
        
        # 7. 决策时机评估
        timing = self._assess_decision_timing(
            decision_type, risk_assessment, time_context
        )
        
        return {
            'decision': decision_type.value,
            'decision_score': decision_score,
            'confidence': self._calculate_decision_confidence(
                decision_score, risk_assessment, personal_capacity
            ),
            'timing': timing,
            'reasoning': self._generate_decision_reasoning(
                decision_type, risk_assessment, personal_capacity, social_context
            ),
            'deepseek_decision': deepseek_decision,
            'decision_method': 'traditional_five_dimensional',
            'decision_factors': {
                'risk_factor': bias_adjusted_risk,
                'capacity_factor': self._capacity_to_score(personal_capacity),
                'social_factor': social_influence_score,
                'bias_adjustment': bias_adjusted_risk - risk_assessment['risk_score']
            }
        }
    
    async def generate_mobility_pattern(
        self,
        decision: Dict[str, Any],
        agent_profile: Dict[str, Any],
        time_context: Dict[str, Any],
        location_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成移动模式"""
        
        # 1. 基础移动参数
        base_mobility = self._calculate_base_mobility(
            decision, agent_profile, time_context
        )
        
        # 2. DeepSeek-R1移动模式增强
        deepseek_mobility = None
        if self.use_deepseek:
            try:
                client = await get_deepseek_client()
                async with client:
                    deepseek_mobility = await client.generate_mobility_pattern(
                        decision, agent_profile, time_context
                    )
            except Exception as e:
                logger.warning(f"DeepSeek移动模式生成失败: {e}")
        
        # 3. 目的地选择
        destination = self._select_destination(
            decision, agent_profile, location_context, deepseek_mobility
        )
        
        # 4. 交通方式选择
        travel_mode = self._select_travel_mode(
            agent_profile, time_context, destination, deepseek_mobility
        )
        
        # 5. 路径偏好
        route_preference = self._determine_route_preference(
            decision, agent_profile, time_context
        )
        
        return {
            'movement_type': decision['decision'],
            'destination_aoi': destination,
            'travel_mode': travel_mode,
            'route_preference': route_preference,
            'urgency_level': self._calculate_urgency(decision, time_context),
            'estimated_duration': self._estimate_travel_duration(
                destination, travel_mode, time_context
            ),
            'deepseek_mobility': deepseek_mobility,
            'mobility_factors': {
                'decision_urgency': decision.get('timing', 'monitor'),
                'personal_mobility': agent_profile.get('mobility_score', 0.5),
                'time_pressure': self._calculate_time_pressure(time_context),
                'resource_availability': agent_profile.get('resource_score', 0.5)
            }
        }
    
    def _calculate_objective_risk(
        self, weather_data: Dict[str, Any], location: int
    ) -> float:
        """计算客观风险"""
        # 天气严重程度
        weather_severity = 0.0
        if 'hurricane_category' in weather_data:
            category = weather_data['hurricane_category']
            weather_severity = min(category / 5.0, 1.0) if category else 0.3
        
        # 风速影响
        wind_speed = weather_data.get('wind_speed', 0)
        wind_factor = min(wind_speed / 150.0, 1.0) if wind_speed else 0.3
        
        # 风暴潮影响
        storm_surge = weather_data.get('storm_surge', 0)
        surge_factor = min(storm_surge / 20.0, 1.0) if storm_surge else 0.2
        
        # 位置脆弱性（简化）
        location_vulnerability = 0.5  # 可以基于历史数据或地理信息改进
        
        # 综合客观风险
        objective_risk = (
            weather_severity * 0.4 +
            wind_factor * 0.3 +
            surge_factor * 0.2 +
            location_vulnerability * 0.1
        )
        
        return min(objective_risk, 1.0)
    
    def _assess_risk_perception(
        self,
        objective_risk: float,
        agent_profile: Dict[str, Any],
        time_context: Dict[str, Any]
    ) -> RiskPerception:
        """评估个人风险感知"""
        
        # 基础感知风险
        base_perceived_risk = objective_risk
        
        # 个人特征调整
        risk_aversion = agent_profile.get('risk_aversion', 0.5)
        experience = agent_profile.get('hurricane_experience', 0.0)
        age_factor = self._get_age_risk_factor(agent_profile.get('age', 35))
        
        # 感知调整
        perception_adjustment = (
            (risk_aversion - 0.5) * 0.3 +
            (experience - 0.5) * 0.2 +
            age_factor * 0.1
        )
        
        perceived_risk = np.clip(
            base_perceived_risk + perception_adjustment, 0.0, 1.0
        )
        
        # 信心评估
        education_score = self._education_to_score(agent_profile.get('education_level', 'medium'))
        risk_confidence = min(
            0.5 + experience * 0.3 + education_score * 0.2,
            1.0
        )
        
        return RiskPerception(
            objective_risk=objective_risk,
            perceived_risk=perceived_risk,
            risk_confidence=risk_confidence,
            temporal_distance=self._calculate_temporal_distance(time_context),
            spatial_distance=0.5  # 简化处理
        )
    
    def _integrate_risk_assessments(
        self,
        objective_risk: float,
        risk_perception: RiskPerception,
        deepseek_assessment: Optional[Dict[str, Any]]
    ) -> float:
        """整合多种风险评估结果"""
        
        # 基础权重
        weights = {
            'objective': 0.4,
            'perceived': 0.4,
            'deepseek': 0.2
        }
        
        # 基础分数
        integrated_score = (
            objective_risk * weights['objective'] +
            risk_perception.perceived_risk * weights['perceived']
        )
        
        # DeepSeek增强
        if deepseek_assessment and 'risk_score' in deepseek_assessment:
            deepseek_score = deepseek_assessment['risk_score']
            confidence = deepseek_assessment.get('confidence', 0.5)
            
            # 根据置信度调整权重
            adjusted_deepseek_weight = weights['deepseek'] * confidence
            remaining_weight = 1.0 - adjusted_deepseek_weight
            
            integrated_score = (
                integrated_score * remaining_weight +
                deepseek_score * adjusted_deepseek_weight
            )
        
        return np.clip(integrated_score, 0.0, 1.0)
    
    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """分类风险等级"""
        if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _assess_personal_capacity(self, agent_profile: Dict[str, Any]) -> PersonalCapacity:
        """评估个人能力"""
        return PersonalCapacity(
            transportation_access=1.0 if agent_profile.get('has_vehicle', True) else 0.3,
            financial_resources=self._income_to_score(agent_profile.get('income_level', 'medium')),
            physical_mobility=self._age_to_mobility(agent_profile.get('age', 35)),
            information_access=agent_profile.get('tech_savvy', 0.7),
            social_support=agent_profile.get('social_network', 0.6),
            evacuation_experience=agent_profile.get('hurricane_experience', 0.3)
        )
    
    def _calculate_social_influence(self, social_context: SocialContext) -> float:
        """计算社会影响分数"""
        influence_score = (
            social_context.family_influence * 0.3 +
            social_context.neighbor_evacuation_rate * 0.25 +
            social_context.community_cohesion * 0.2 +
            (1.0 if social_context.mandatory_evacuation else 0.0) * 0.15 +
            social_context.social_media_sentiment * 0.1
        )
        return np.clip(influence_score, -1.0, 1.0)
    
    def _apply_cognitive_bias(
        self, risk_score: float, agent_profile: Dict[str, Any]
    ) -> float:
        """应用认知偏差调整"""
        
        # 可得性启发式：最近经历影响判断
        recent_experience = agent_profile.get('recent_hurricane_experience', 0.0)
        availability_adjustment = (recent_experience - 0.5) * self.cognitive_bias.availability_heuristic
        
        # 乐观偏差：倾向于低估风险
        optimism_adjustment = -self.cognitive_bias.optimism_bias * (1 - agent_profile.get('pessimism', 0.3))
        
        # 损失厌恶：对损失的敏感性
        loss_aversion_factor = agent_profile.get('loss_aversion', 0.5)
        loss_adjustment = risk_score * self.cognitive_bias.loss_aversion * loss_aversion_factor
        
        # 综合调整
        adjusted_risk = risk_score + availability_adjustment + optimism_adjustment + loss_adjustment
        
        return np.clip(adjusted_risk, 0.0, 1.0)
    
    def _calculate_decision_score(
        self,
        risk_score: float,
        capacity: PersonalCapacity,
        social_influence: float,
        social_context: SocialContext,
        deepseek_decision: Optional[Dict[str, Any]]
    ) -> float:
        """计算决策分数"""
        
        # 基础决策分数
        capacity_score = self._capacity_to_score(capacity)
        
        base_score = (
            risk_score * self.decision_weights['risk_perception'] +
            capacity_score * self.decision_weights['personal_capacity'] +
            max(social_influence, 0) * self.decision_weights['social_influence']
        )
        
        # 强制疏散令的影响
        if social_context.mandatory_evacuation:
            base_score = max(base_score, 0.8)
        
        # DeepSeek决策增强
        if deepseek_decision and 'decision' in deepseek_decision:
            deepseek_score = self._decision_to_score(deepseek_decision['decision'])
            deepseek_confidence = deepseek_decision.get('confidence', 0.5)
            
            # 加权平均
            base_score = (
                base_score * (1 - deepseek_confidence * 0.3) +
                deepseek_score * deepseek_confidence * 0.3
            )
        
        return np.clip(base_score, 0.0, 1.0)
    
    def _classify_decision(
        self, decision_score: float, social_context: SocialContext
    ) -> DecisionType:
        """分类决策类型"""
        
        # 强制疏散令
        if social_context.mandatory_evacuation:
            return DecisionType.EVACUATE
        
        # 基于分数分类
        if decision_score >= 0.8:
            return DecisionType.EVACUATE
        elif decision_score >= 0.6:
            return DecisionType.PREPARE
        elif decision_score >= 0.3:
            return DecisionType.PREPARE
        else:
            return DecisionType.STAY
    
    # 辅助方法
    def _get_weather_severity(self, weather_data: Dict[str, Any]) -> float:
        """获取天气严重程度"""
        category = weather_data.get('hurricane_category', 0)
        return min(category / 5.0, 1.0) if category else 0.3
    
    def _get_location_vulnerability(self, location: int) -> float:
        """获取位置脆弱性（简化）"""
        # 这里可以基于历史数据或地理信息系统改进
        return 0.5
    
    def _get_personal_vulnerability(self, agent_profile: Dict[str, Any]) -> float:
        """获取个人脆弱性"""
        age = agent_profile.get('age', 35)
        income = agent_profile.get('income_level', 'medium')
        family_size = agent_profile.get('family_size', 2)
        
        age_vulnerability = 0.3 if age > 65 or age < 18 else 0.1
        income_vulnerability = 0.4 if income == 'low' else 0.2 if income == 'medium' else 0.1
        family_vulnerability = min(family_size / 10.0, 0.3)
        
        return (age_vulnerability + income_vulnerability + family_vulnerability) / 3
    
    def _get_age_risk_factor(self, age: int) -> float:
        """年龄风险因子"""
        if age < 25:
            return -0.1  # 年轻人可能低估风险
        elif age > 65:
            return 0.1   # 老年人可能高估风险
        else:
            return 0.0
    
    def _calculate_temporal_distance(self, time_context: Dict[str, Any]) -> float:
        """计算时间距离感知"""
        hours_to_landfall = time_context.get('hours_to_landfall', 24)
        return max(0.0, min(1.0, hours_to_landfall / 48.0))
    
    def _education_to_score(self, education_level: str) -> float:
        """教育水平转分数"""
        mapping = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        return mapping.get(education_level, 0.6)
    
    def _get_age_risk_factor(self, age: int) -> float:
        """获取年龄风险因子"""
        if age < 18 or age > 75:
            return 0.3  # 高风险
        elif age > 65:
            return 0.1  # 中等风险
        else:
            return 0.0  # 低风险
    
    def _calculate_temporal_distance(self, time_context: Dict[str, Any]) -> float:
        """计算时间距离感知"""
        hours_to_landfall = time_context.get('time_to_landfall', 24)
        return max(0.0, min(1.0, hours_to_landfall / 48.0))
    
    def _income_to_score(self, income_level: str) -> float:
        """收入水平转分数"""
        mapping = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        return mapping.get(income_level, 0.6)
    
    def _age_to_mobility(self, age: int) -> float:
        """年龄转移动能力"""
        if age < 18 or age > 75:
            return 0.6
        elif age > 65:
            return 0.8
        else:
            return 1.0
    
    def _capacity_to_score(self, capacity: PersonalCapacity) -> float:
        """能力转分数"""
        return (
            capacity.transportation_access * 0.25 +
            capacity.financial_resources * 0.2 +
            capacity.physical_mobility * 0.2 +
            capacity.information_access * 0.15 +
            capacity.social_support * 0.15 +
            capacity.evacuation_experience * 0.05
        )
    
    def _decision_to_score(self, decision: str) -> float:
        """决策转分数"""
        mapping = {
            'stay': 0.2,
            'prepare': 0.5,
            'evacuate': 0.8,
            'emergency_evacuate': 1.0
        }
        return mapping.get(decision, 0.5)
    
    def _assess_decision_timing(
        self,
        decision_type: DecisionType,
        risk_assessment: Dict[str, Any],
        time_context: Dict[str, Any]
    ) -> str:
        """评估决策时机"""
        if decision_type == DecisionType.EVACUATE:
            if risk_assessment['risk_level'] == 'critical':
                return 'immediate'
            else:
                return 'within_hours'
        elif decision_type == DecisionType.PREPARE:
            return 'monitor'
        else:
            return 'normal'
    
    def _calculate_decision_confidence(
        self,
        decision_score: float,
        risk_assessment: Dict[str, Any],
        capacity: PersonalCapacity
    ) -> float:
        """计算决策信心"""
        # 基于风险评估信心和个人能力
        risk_confidence = risk_assessment.get('risk_confidence', 0.5)
        capacity_confidence = min(capacity.information_access, capacity.evacuation_experience)
        
        return (risk_confidence + capacity_confidence) / 2
    
    def _generate_decision_reasoning(
        self,
        decision_type: DecisionType,
        risk_assessment: Dict[str, Any],
        capacity: PersonalCapacity,
        social_context: SocialContext
    ) -> str:
        """生成决策推理"""
        risk_level = risk_assessment['risk_level']
        
        if decision_type == DecisionType.EVACUATE:
            return f"基于{risk_level}风险等级和个人疏散能力，建议立即疏散"
        elif decision_type == DecisionType.PREPARE:
            return f"风险等级为{risk_level}，建议做好疏散准备并密切监控"
        else:
            return f"当前风险等级为{risk_level}，可以选择就地避险"
    
    def _calculate_base_mobility(
        self,
        decision: Dict[str, Any],
        agent_profile: Dict[str, Any],
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算基础移动参数"""
        decision_type = decision['decision']
        
        if decision_type == 'evacuate':
            return {
                'urgency': 0.8,
                'distance_preference': 'far',
                'speed_preference': 'fast'
            }
        elif decision_type == 'prepare':
            return {
                'urgency': 0.4,
                'distance_preference': 'near',
                'speed_preference': 'normal'
            }
        else:
            return {
                'urgency': 0.1,
                'distance_preference': 'minimal',
                'speed_preference': 'slow'
            }
    
    def _select_destination(
        self,
        decision: Dict[str, Any],
        agent_profile: Dict[str, Any],
        location_context: Dict[str, Any],
        deepseek_mobility: Optional[Dict[str, Any]]
    ) -> Optional[int]:
        """选择目的地"""
        if decision['decision'] == 'evacuate':
            # 优先使用DeepSeek建议
            if deepseek_mobility and deepseek_mobility.get('destination_aoi'):
                return deepseek_mobility['destination_aoi']
            
            # 否则使用默认逻辑
            return location_context.get('safe_zones', [1])[0]
        
        return None
    
    def _select_travel_mode(
        self,
        agent_profile: Dict[str, Any],
        time_context: Dict[str, Any],
        destination: Optional[int],
        deepseek_mobility: Optional[Dict[str, Any]]
    ) -> str:
        """选择交通方式"""
        # 优先使用DeepSeek建议
        if deepseek_mobility and deepseek_mobility.get('travel_mode'):
            return deepseek_mobility['travel_mode']
        
        # 默认逻辑
        if agent_profile.get('has_vehicle', True):
            return 'driving'
        else:
            return 'walking'
    
    def _determine_route_preference(
        self,
        decision: Dict[str, Any],
        agent_profile: Dict[str, Any],
        time_context: Dict[str, Any]
    ) -> str:
        """确定路径偏好"""
        if decision['decision'] == 'evacuate':
            return 'safest'
        else:
            return 'familiar'
    
    def _calculate_urgency(
        self, decision: Dict[str, Any], time_context: Dict[str, Any]
    ) -> float:
        """计算紧急程度"""
        base_urgency = {
            'stay': 0.1,
            'prepare': 0.4,
            'evacuate': 0.8,
            'emergency_evacuate': 1.0
        }.get(decision['decision'], 0.3)
        
        # 时间压力调整
        hours_to_landfall = time_context.get('hours_to_landfall', 24)
        time_pressure = max(0.0, 1.0 - hours_to_landfall / 24.0)
        
        return min(base_urgency + time_pressure * 0.2, 1.0)
    
    def _estimate_travel_duration(
        self, destination: Optional[int], travel_mode: str, time_context: Dict[str, Any]
    ) -> int:
        """估算出行时长（分钟）"""
        if not destination:
            return 0
        
        base_duration = {
            'walking': 60,
            'driving': 30,
            'public_transport': 45
        }.get(travel_mode, 30)
        
        # 交通拥堵调整
        congestion = time_context.get('traffic_congestion', 0.5)
        adjusted_duration = base_duration * (1 + congestion)
        
        return int(adjusted_duration)
    
    def _calculate_time_pressure(self, time_context: Dict[str, Any]) -> float:
        """计算时间压力"""
        hours_to_landfall = time_context.get('hours_to_landfall', 24)
        return max(0.0, 1.0 - hours_to_landfall / 48.0)