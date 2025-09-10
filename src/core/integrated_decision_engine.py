"""
集成决策引擎 (Integrated Decision Engine)
整合所有决策框架和适配器，提供统一的决策接口
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import json

from .emergency_scenario_decision_framework import (
    EmergencyScenarioDecisionFramework,
    EmergencyScenarioType,
    EmergencyContext,
    MovementDecision,
    MovementPattern,
    MovementUrgency,
    DecisionType
)
from .scenario_specific_adapters import (
    ScenarioAdapterFactory,
    BaseScenarioAdapter,
    ScenarioParameters
)
from .enhanced_decision_framework import EnhancedDecisionFramework
from .four_dimensional_framework import FourDimensionalDecisionFramework

logger = logging.getLogger(__name__)

@dataclass
class DecisionRequest:
    """决策请求"""
    agent_id: str
    agent_profile: Dict[str, Any]
    current_location: Tuple[float, float]
    scenario_type: EmergencyScenarioType
    scenario_context: Dict[str, Any]
    weather_data: Dict[str, Any]
    social_context: Dict[str, Any]
    time_context: Dict[str, Any]
    information_sources: List[Dict[str, Any]]
    request_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DecisionResponse:
    """决策响应"""
    agent_id: str
    decision: MovementDecision
    confidence_score: float
    reasoning: str
    scenario_advice: Dict[str, Any]
    alternative_options: List[MovementDecision]
    risk_assessment: Dict[str, float]
    decision_timestamp: datetime = field(default_factory=datetime.now)
    processing_time_ms: float = 0.0

class IntegratedDecisionEngine:
    """集成决策引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化集成决策引擎"""
        
        self.config = config or {}
        
        # 初始化核心框架
        emergency_config = self.config.get('emergency_framework', {})
        self.emergency_framework = EmergencyScenarioDecisionFramework(emergency_config)
        enhanced_config = self.config.get('enhanced_framework', {})
        self.enhanced_framework = EnhancedDecisionFramework(enhanced_config)
        four_dim_config = self.config.get('four_dimensional_framework', {})
        self.four_dimensional_framework = FourDimensionalDecisionFramework(
            use_dynamic_weights=four_dim_config.get('use_dynamic_weights', True)
        )
        
        # 初始化适配器工厂
        self.adapter_factory = ScenarioAdapterFactory()
        
        # 决策历史记录
        self.decision_history: List[DecisionResponse] = []
        
        # 性能统计
        self.performance_stats = {
            'total_decisions': 0,
            'average_processing_time': 0.0,
            'scenario_distribution': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        logger.info("集成决策引擎初始化完成")
    
    async def make_decision(self, request: DecisionRequest) -> DecisionResponse:
        """执行决策"""
        
        start_time = datetime.now()
        
        try:
            # 1. 获取场景适配器
            adapter = self.adapter_factory.create_adapter(request.scenario_type)
            
            # 2. 构建紧急场景上下文
            emergency_context = self._build_emergency_context(request, adapter)
            
            # 3. 执行多框架决策
            decisions = await self._execute_multi_framework_decision(
                request, emergency_context, adapter
            )
            
            # 4. 决策融合
            final_decision = self._fuse_decisions(decisions, request, adapter)
            
            # 5. 生成场景特定建议
            scenario_advice = adapter.generate_scenario_specific_advice(
                final_decision, request.scenario_context
            )
            
            # 6. 计算置信度
            confidence_score = self._calculate_confidence_score(
                decisions, final_decision, request
            )
            
            # 7. 生成推理解释
            reasoning = self._generate_reasoning(
                final_decision, decisions, request, adapter
            )
            
            # 8. 生成备选方案
            alternatives = self._generate_alternatives(
                decisions, final_decision, request
            )
            
            # 9. 风险评估
            risk_assessment = self._assess_comprehensive_risk(
                request, emergency_context, adapter
            )
            
            # 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # 构建响应
            response = DecisionResponse(
                agent_id=request.agent_id,
                decision=final_decision,
                confidence_score=confidence_score,
                reasoning=reasoning,
                scenario_advice=scenario_advice,
                alternative_options=alternatives,
                risk_assessment=risk_assessment,
                processing_time_ms=processing_time
            )
            
            # 更新统计信息
            self._update_statistics(request, response)
            
            # 记录决策历史
            self.decision_history.append(response)
            
            logger.info(f"决策完成 - Agent: {request.agent_id}, "
                       f"场景: {request.scenario_type.value}, "
                       f"决策: {final_decision.movement_pattern.value}, "
                       f"置信度: {confidence_score:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"决策执行失败: {str(e)}")
            # 返回保守决策
            return self._create_fallback_response(request, str(e))
    
    def _build_emergency_context(
        self, 
        request: DecisionRequest, 
        adapter: BaseScenarioAdapter
    ) -> EmergencyContext:
        """构建紧急场景上下文"""
        
        return EmergencyContext(
            scenario_type=request.scenario_type,
            severity_level=request.scenario_context.get('severity', 0.5),
            time_to_impact=request.scenario_context.get('time_to_impact', 24.0),
            affected_areas=request.scenario_context.get('affected_areas', []),
            current_conditions=request.weather_data,
            official_warnings=request.scenario_context.get('official_warnings', []),
            evacuation_orders=request.scenario_context.get('evacuation_orders', {}),
            infrastructure_status=request.scenario_context.get('infrastructure_status', {})
        )
    
    async def _execute_multi_framework_decision(
        self, 
        request: DecisionRequest, 
        emergency_context: EmergencyContext, 
        adapter: BaseScenarioAdapter
    ) -> Dict[str, MovementDecision]:
        """执行多框架决策"""
        
        decisions = {}
        
        # 1. 紧急场景决策框架
        try:
            emergency_decision = await self.emergency_framework.make_emergency_decision(
                agent_profile=request.agent_profile,
                emergency_context=emergency_context,
                current_location=request.current_location,
                time_context=request.time_context,
                social_context=None  # 可选参数
            )
            decisions['emergency_framework'] = emergency_decision
        except Exception as e:
            logger.warning(f"紧急场景框架决策失败: {str(e)}")
        
        # 2. 增强决策框架
        try:
            enhanced_decision = await self._make_enhanced_decision(request, adapter)
            decisions['enhanced_framework'] = enhanced_decision
        except Exception as e:
            logger.warning(f"增强决策框架决策失败: {str(e)}")
        
        # 3. 四维决策框架
        try:
            four_dim_decision = self._make_four_dimensional_decision(request, adapter)
            decisions['four_dimensional_framework'] = four_dim_decision
        except Exception as e:
            logger.warning(f"四维决策框架决策失败: {str(e)}")
        
        return decisions
    
    async def _make_enhanced_decision(
        self, 
        request: DecisionRequest, 
        adapter: BaseScenarioAdapter
    ) -> MovementDecision:
        """使用增强决策框架做决策"""
        
        # 调整风险评估
        base_risk = request.scenario_context.get('base_risk', 0.5)
        time_to_impact = request.scenario_context.get('time_to_impact', 24.0)
        location_distance = request.scenario_context.get('distance_to_hazard', 10.0)
        
        adjusted_risk = adapter.adjust_risk_assessment(
            base_risk, time_to_impact, location_distance, request.agent_profile
        )
        
        # 构建风险评估结果
        risk_assessment = {
            'risk_score': adjusted_risk,
            'risk_level': 'high' if adjusted_risk > 0.7 else 'medium' if adjusted_risk > 0.4 else 'low',
            'evacuation_recommendation': 'evacuate' if adjusted_risk > 0.6 else 'prepare' if adjusted_risk > 0.3 else 'stay'
        }
        
        # 使用增强框架生成决策
        decision_result = await self.enhanced_framework.make_evacuation_decision(
            risk_assessment,
            request.agent_profile,
            request.social_context,
            request.time_context,
            request.weather_data
        )
        
        # 转换为MovementDecision格式
        return self._convert_to_movement_decision(decision_result, request)
    
    def _make_four_dimensional_decision(
        self, 
        request: DecisionRequest, 
        adapter: BaseScenarioAdapter
    ) -> MovementDecision:
        """使用四维决策框架做决策"""
        
        # 构建决策上下文
        from .four_dimensional_framework import DecisionContext
        
        decision_context = DecisionContext(
            weather_data=request.weather_data,
            agent_profile=request.agent_profile,
            social_context=request.social_context,
            time_context=request.time_context,
            location_data={'current_location': request.current_location},
            information_sources=request.information_sources
        )
        
        # 调整决策权重
        base_weights = {
            'safety_concern': 0.4,
            'information_trust': 0.25,
            'evacuation_cost': 0.2,
            'property_attachment': 0.15
        }
        
        adjusted_weights = adapter.adjust_decision_weights(
            base_weights, request.scenario_context
        )
        
        # 执行四维决策
        decision_result = self.four_dimensional_framework.make_four_dimensional_decision(
            decision_context, custom_weights=adjusted_weights
        )
        
        # 转换为MovementDecision格式
        return self._convert_four_dim_to_movement_decision(decision_result, request)
    
    def _fuse_decisions(
        self, 
        decisions: Dict[str, MovementDecision], 
        request: DecisionRequest, 
        adapter: BaseScenarioAdapter
    ) -> MovementDecision:
        """融合多个决策结果"""
        
        if not decisions:
            return self._create_default_decision(request)
        
        # 权重配置
        framework_weights = {
            'emergency_framework': 0.4,
            'enhanced_framework': 0.35,
            'four_dimensional_framework': 0.25
        }
        
        # 场景特定权重调整
        if request.scenario_type == EmergencyScenarioType.WILDFIRE_SPREAD:
            framework_weights['emergency_framework'] = 0.5
            framework_weights['enhanced_framework'] = 0.3
            framework_weights['four_dimensional_framework'] = 0.2
        
        # 计算加权决策
        pattern_scores = {}
        urgency_scores = {}
        
        for framework, decision in decisions.items():
            weight = framework_weights.get(framework, 0.0)
            
            # 移动模式评分
            pattern = decision.movement_pattern
            if pattern not in pattern_scores:
                pattern_scores[pattern] = 0.0
            pattern_scores[pattern] += weight
            
            # 紧急程度评分
            urgency = decision.urgency_level
            if urgency not in urgency_scores:
                urgency_scores[urgency] = 0.0
            urgency_scores[urgency] += weight
        
        # 选择最高分的模式和紧急程度
        final_pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
        final_urgency = max(urgency_scores.items(), key=lambda x: x[1])[0]
        
        # 融合其他属性
        final_decision = self._create_fused_decision(
            decisions, final_pattern, final_urgency, request
        )
        
        return final_decision
    
    def _create_fused_decision(
        self, 
        decisions: Dict[str, MovementDecision], 
        pattern: MovementPattern, 
        urgency: MovementUrgency, 
        request: DecisionRequest
    ) -> MovementDecision:
        """创建融合决策"""
        
        # 收集所有决策的属性
        all_destinations = []
        all_departure_times = []
        all_travel_modes = []
        all_route_preferences = []
        
        for decision in decisions.values():
            if decision.target_location:
                all_destinations.append(decision.target_location)
            if decision.departure_time:
                all_departure_times.append(decision.departure_time)
            if decision.travel_mode:
                all_travel_modes.append(decision.travel_mode)
            if decision.route_preference:
                all_route_preferences.append(decision.route_preference)
        
        # 选择最常见的属性
        target_location = self._select_most_common(all_destinations) if all_destinations else None
        departure_time = self._select_earliest_time(all_departure_times) if all_departure_times else None
        travel_mode = self._select_most_common(all_travel_modes) if all_travel_modes else "car"
        route_preference = self._select_most_common(all_route_preferences) if all_route_preferences else "fastest"
        
        return MovementDecision(
            decision_type=DecisionType.STAY,
            movement_pattern=pattern,
            urgency_level=urgency,
            target_location=target_location,
            departure_time=departure_time,
            travel_mode=travel_mode,
            route_preference=route_preference,
            confidence=0.8,  # 将在后续计算中更新
            reasoning="多框架融合决策结果"
        )
    
    def _calculate_confidence_score(
        self, 
        decisions: Dict[str, MovementDecision], 
        final_decision: MovementDecision, 
        request: DecisionRequest
    ) -> float:
        """计算置信度分数"""
        
        if not decisions:
            return 0.5
        
        # 一致性评分
        consistency_score = self._calculate_consistency_score(decisions, final_decision)
        
        # 信息质量评分
        info_quality_score = self._calculate_info_quality_score(request)
        
        # 时间压力调整
        time_pressure = request.scenario_context.get('time_pressure', 0.5)
        time_adjustment = 1.0 - (time_pressure * 0.2)
        
        # 综合置信度
        confidence = (consistency_score * 0.5 + info_quality_score * 0.3) * time_adjustment
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_consistency_score(
        self, 
        decisions: Dict[str, MovementDecision], 
        final_decision: MovementDecision
    ) -> float:
        """计算决策一致性分数"""
        
        if len(decisions) <= 1:
            return 1.0
        
        pattern_matches = 0
        urgency_matches = 0
        
        for decision in decisions.values():
            if decision.movement_pattern == final_decision.movement_pattern:
                pattern_matches += 1
            if decision.urgency_level == final_decision.urgency_level:
                urgency_matches += 1
        
        pattern_consistency = pattern_matches / len(decisions)
        urgency_consistency = urgency_matches / len(decisions)
        
        return (pattern_consistency + urgency_consistency) / 2
    
    def _calculate_info_quality_score(self, request: DecisionRequest) -> float:
        """计算信息质量分数"""
        
        quality_factors = []
        
        # 信息来源可靠性
        info_reliability = request.scenario_context.get('information_reliability', 0.8)
        quality_factors.append(info_reliability)
        
        # 数据完整性
        required_fields = ['severity', 'time_to_impact', 'base_risk']
        available_fields = sum(1 for field in required_fields 
                             if field in request.scenario_context)
        completeness = available_fields / len(required_fields)
        quality_factors.append(completeness)
        
        # 时效性
        time_freshness = min(1.0, 24.0 / max(1.0, 
            request.scenario_context.get('data_age_hours', 1.0)))
        quality_factors.append(time_freshness)
        
        return sum(quality_factors) / len(quality_factors)
    
    def _generate_reasoning(
        self, 
        final_decision: MovementDecision, 
        decisions: Dict[str, MovementDecision], 
        request: DecisionRequest, 
        adapter: BaseScenarioAdapter
    ) -> str:
        """生成决策推理"""
        
        reasoning_parts = []
        
        # 场景分析
        scenario_name = request.scenario_type.value.replace('_', ' ').title()
        severity = request.scenario_context.get('severity', 0.5)
        time_to_impact = request.scenario_context.get('time_to_impact', 24.0)
        
        reasoning_parts.append(
            f"面对{scenario_name}场景（严重程度: {severity:.1f}, "
            f"预计影响时间: {time_to_impact:.1f}小时）"
        )
        
        # 决策框架分析
        if len(decisions) > 1:
            reasoning_parts.append(
                f"综合{len(decisions)}个决策框架的分析结果"
            )
        
        # 最终决策说明
        pattern_name = final_decision.movement_pattern.value.replace('_', ' ')
        urgency_name = final_decision.urgency_level.value.replace('_', ' ')
        
        reasoning_parts.append(
            f"建议采取{pattern_name}行动，紧急程度为{urgency_name}"
        )
        
        # 关键因素
        key_factors = []
        
        if severity > 0.7:
            key_factors.append("高风险等级")
        
        if time_to_impact < 12:
            key_factors.append("时间紧迫")
        
        agent_age = request.agent_profile.get('age', 35)
        if agent_age < 18 or agent_age > 65:
            key_factors.append("人员脆弱性")
        
        if key_factors:
            reasoning_parts.append(f"主要考虑因素: {', '.join(key_factors)}")
        
        return "。".join(reasoning_parts) + "。"
    
    def _generate_alternatives(
        self, 
        decisions: Dict[str, MovementDecision], 
        final_decision: MovementDecision, 
        request: DecisionRequest
    ) -> List[MovementDecision]:
        """生成备选方案"""
        
        alternatives = []
        
        # 从其他框架的决策中选择备选方案
        for framework, decision in decisions.items():
            if (decision.movement_pattern != final_decision.movement_pattern or 
                decision.urgency_level != final_decision.urgency_level):
                alternatives.append(decision)
        
        # 生成保守和激进备选方案
        if final_decision.movement_pattern != MovementPattern.SHELTER_SEEKING:
            conservative_alternative = MovementDecision(
                decision_type=DecisionType.STAY,
                movement_pattern=MovementPattern.SHELTER_SEEKING,
                urgency_level=MovementUrgency.ROUTINE,
                reasoning="保守选择：就地避难"
            )
            alternatives.append(conservative_alternative)
        
        if final_decision.movement_pattern != MovementPattern.EVACUATION:
            aggressive_alternative = MovementDecision(
                decision_type=DecisionType.EMERGENCY_EVACUATE,
                movement_pattern=MovementPattern.EVACUATION,
                urgency_level=MovementUrgency.IMMEDIATE,
                reasoning="激进选择：立即疏散"
            )
            alternatives.append(aggressive_alternative)
        
        return alternatives[:3]  # 最多返回3个备选方案
    
    def _assess_comprehensive_risk(
        self, 
        request: DecisionRequest, 
        emergency_context: EmergencyContext, 
        adapter: BaseScenarioAdapter
    ) -> Dict[str, float]:
        """综合风险评估"""
        
        risk_assessment = {}
        
        # 基础风险
        base_risk = request.scenario_context.get('base_risk', 0.5)
        time_to_impact = request.scenario_context.get('time_to_impact', 24.0)
        location_distance = request.scenario_context.get('distance_to_hazard', 10.0)
        
        # 场景调整风险
        adjusted_risk = adapter.adjust_risk_assessment(
            base_risk, time_to_impact, location_distance, request.agent_profile
        )
        
        risk_assessment['overall_risk'] = adjusted_risk
        risk_assessment['base_risk'] = base_risk
        risk_assessment['time_risk'] = max(0.0, 1.0 - time_to_impact / 48.0)
        risk_assessment['location_risk'] = max(0.0, 1.0 - location_distance / 50.0)
        
        # 个人脆弱性风险
        vulnerability_risk = self._calculate_vulnerability_risk(request.agent_profile)
        risk_assessment['vulnerability_risk'] = vulnerability_risk
        
        # 社会风险
        social_risk = self._calculate_social_risk(request.social_context)
        risk_assessment['social_risk'] = social_risk
        
        return risk_assessment
    
    def _calculate_vulnerability_risk(self, agent_profile: Dict[str, Any]) -> float:
        """计算个人脆弱性风险"""
        
        vulnerability = 0.0
        
        # 年龄脆弱性
        age = agent_profile.get('age', 35)
        if age < 18:
            vulnerability += 0.3
        elif age > 65:
            vulnerability += 0.4
        
        # 健康脆弱性
        health = agent_profile.get('health_status', 'good')
        if health == 'poor':
            vulnerability += 0.4
        elif health == 'disabled':
            vulnerability += 0.5
        
        # 经济脆弱性
        income = agent_profile.get('income_level', 'medium')
        if income == 'low':
            vulnerability += 0.2
        
        # 交通脆弱性
        if not agent_profile.get('has_car', True):
            vulnerability += 0.2
        
        return min(vulnerability, 1.0)
    
    def _calculate_social_risk(self, social_context: Dict[str, Any]) -> float:
        """计算社会风险"""
        
        social_risk = 0.0
        
        # 社会网络风险
        network_size = social_context.get('network_size', 10)
        if network_size < 5:
            social_risk += 0.2
        
        # 信息获取风险
        info_access = social_context.get('information_access', 0.8)
        social_risk += (1.0 - info_access) * 0.3
        
        # 社会支持风险
        social_support = social_context.get('social_support', 0.7)
        social_risk += (1.0 - social_support) * 0.2
        
        return min(social_risk, 1.0)
    
    def _convert_to_movement_decision(
        self, 
        decision_result: Dict[str, Any], 
        request: DecisionRequest
    ) -> MovementDecision:
        """转换增强框架决策结果为MovementDecision"""
        
        # 根据决策类型映射移动模式
        decision_type = decision_result.get('decision_type', 'stay')
        
        if decision_type == 'evacuate':
            pattern = MovementPattern.EVACUATION
            urgency = MovementUrgency.URGENT
            decision_type_enum = DecisionType.EVACUATE
        elif decision_type == 'prepare':
            pattern = MovementPattern.PREPARATION
            urgency = MovementUrgency.CAUTIOUS
            decision_type_enum = DecisionType.PREPARE
        else:
            pattern = MovementPattern.SHELTER_SEEKING
            urgency = MovementUrgency.ROUTINE
            decision_type_enum = DecisionType.STAY
        
        return MovementDecision(
            decision_type=decision_type_enum,
            movement_pattern=pattern,
            urgency_level=urgency,
            confidence=decision_result.get('confidence', 0.7),
            reasoning=decision_result.get('reasoning', '增强框架决策')
        )
    
    def _convert_four_dim_to_movement_decision(
        self, 
        decision_result: Dict[str, Any], 
        request: DecisionRequest
    ) -> MovementDecision:
        """转换四维框架决策结果为MovementDecision"""
        
        decision_type = decision_result.get('decision_type', 'stay')
        
        if decision_type == 'emergency_evacuate':
            pattern = MovementPattern.EVACUATION
            urgency = MovementUrgency.IMMEDIATE
            decision_type_enum = DecisionType.EMERGENCY_EVACUATE
        elif decision_type == 'evacuate':
            pattern = MovementPattern.EVACUATION
            urgency = MovementUrgency.URGENT
            decision_type_enum = DecisionType.EVACUATE
        elif decision_type == 'prepare':
            pattern = MovementPattern.PREPARATION
            urgency = MovementUrgency.CAUTIOUS
            decision_type_enum = DecisionType.PREPARE
        else:
            pattern = MovementPattern.SHELTER_SEEKING
            urgency = MovementUrgency.ROUTINE
            decision_type_enum = DecisionType.STAY
        
        return MovementDecision(
            decision_type=decision_type_enum,
            movement_pattern=pattern,
            urgency_level=urgency,
            confidence=decision_result.get('confidence', 0.7),
            reasoning=decision_result.get('reasoning', '四维框架决策')
        )
    
    def _create_default_decision(self, request: DecisionRequest) -> MovementDecision:
        """创建默认决策"""
        
        severity = request.scenario_context.get('severity', 0.5)
        
        if severity > 0.8:
            pattern = MovementPattern.EVACUATION
            urgency = MovementUrgency.IMMEDIATE
            decision_type = DecisionType.EMERGENCY_EVACUATE
        elif severity > 0.6:
            pattern = MovementPattern.EVACUATION
            urgency = MovementUrgency.URGENT
            decision_type = DecisionType.EVACUATE
        elif severity > 0.4:
            pattern = MovementPattern.PREPARATION
            urgency = MovementUrgency.CAUTIOUS
            decision_type = DecisionType.PREPARE
        else:
            pattern = MovementPattern.SHELTER_SEEKING
            urgency = MovementUrgency.ROUTINE
            decision_type = DecisionType.STAY
        
        return MovementDecision(
            decision_type=decision_type,
            movement_pattern=pattern,
            urgency_level=urgency,
            confidence=0.6,
            reasoning="基于严重程度的默认决策"
        )
    
    def _create_fallback_response(
        self, 
        request: DecisionRequest, 
        error_message: str
    ) -> DecisionResponse:
        """创建回退响应"""
        
        fallback_decision = MovementDecision(
            decision_type=DecisionType.STAY,
            movement_pattern=MovementPattern.SHELTER_SEEKING,
            urgency_level=MovementUrgency.CAUTIOUS,
            confidence=0.5,
            reasoning=f"决策失败，采用保守策略: {error_message}"
        )
        
        return DecisionResponse(
            agent_id=request.agent_id,
            decision=fallback_decision,
            confidence_score=0.5,
            reasoning="系统错误，采用保守决策",
            scenario_advice={},
            alternative_options=[],
            risk_assessment={'overall_risk': 0.5}
        )
    
    def _select_most_common(self, items: List[Any]) -> Any:
        """选择最常见的项目"""
        if not items:
            return None
        
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(1)[0][0]
    
    def _select_earliest_time(self, times: List[datetime]) -> datetime:
        """选择最早的时间"""
        if not times:
            return None
        
        return min(times)
    
    def _update_statistics(self, request: DecisionRequest, response: DecisionResponse):
        """更新性能统计"""
        
        self.performance_stats['total_decisions'] += 1
        
        # 更新平均处理时间
        total_time = (self.performance_stats['average_processing_time'] * 
                     (self.performance_stats['total_decisions'] - 1) + 
                     response.processing_time_ms)
        self.performance_stats['average_processing_time'] = (
            total_time / self.performance_stats['total_decisions']
        )
        
        # 更新场景分布
        scenario = request.scenario_type.value
        if scenario not in self.performance_stats['scenario_distribution']:
            self.performance_stats['scenario_distribution'][scenario] = 0
        self.performance_stats['scenario_distribution'][scenario] += 1
        
        # 更新置信度分布
        if response.confidence_score >= 0.8:
            self.performance_stats['confidence_distribution']['high'] += 1
        elif response.confidence_score >= 0.6:
            self.performance_stats['confidence_distribution']['medium'] += 1
        else:
            self.performance_stats['confidence_distribution']['low'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        
        return {
            'statistics': self.performance_stats.copy(),
            'recent_decisions': len(self.decision_history),
            'available_scenarios': [s.value for s in 
                                  self.adapter_factory.get_available_scenarios()],
            'framework_status': {
                'emergency_framework': 'active',
                'enhanced_framework': 'active',
                'four_dimensional_framework': 'active'
            }
        }
    
    def clear_history(self):
        """清除决策历史"""
        self.decision_history.clear()
        logger.info("决策历史已清除")