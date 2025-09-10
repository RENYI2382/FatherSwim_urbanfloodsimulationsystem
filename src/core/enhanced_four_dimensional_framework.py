"""
增强四维决策框架
集成实证研究的预测模型，提升决策准确性和可靠性
基于论文: "Prediction of population behavior in hurricane evacuations" (2022)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from .four_dimensional_framework import (
    FourDimensionalDecisionFramework, 
    DecisionContext, 
    FourDimensionalCriteria,
    DecisionType
)
from .evacuation_prediction_models import (
    EvacuationPredictionFramework,
    PredictionContext,
    PredictionResult,
    EvacuationDecision,
    PredictionModelType
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedDecisionResult:
    """增强决策结果"""
    # 原四维决策结果
    four_dimensional_decision: str
    four_dimensional_score: float
    four_dimensional_confidence: float
    four_dimensional_criteria: Dict[str, float]
    
    # 实证模型预测结果
    lr_prediction: PredictionResult
    ddc_prediction: PredictionResult
    
    # 集成决策结果
    final_decision: str
    final_confidence: float
    consensus_score: float  # 模型一致性评分
    
    # 详细分析
    prediction_analysis: Dict[str, Any]
    model_weights: Dict[str, float]
    explanation: str

class EnhancedFourDimensionalFramework:
    """
    增强四维决策框架
    集成实证研究预测模型，提升决策准确性
    """
    
    def __init__(self, use_dynamic_weights: bool = True):
        # 初始化原四维框架
        self.four_dimensional_framework = FourDimensionalDecisionFramework(use_dynamic_weights)
        
        # 初始化实证预测框架
        self.prediction_framework = EvacuationPredictionFramework()
        
        # 模型权重配置 (基于论文性能评估)
        self.model_weights = {
            'four_dimensional': 0.4,  # 原四维框架权重
            'logistic_regression': 0.3,  # LR模型权重 (适用于总体预测)
            'dynamic_discrete_choice': 0.3  # DDC模型权重 (适用于时空预测)
        }
        
        # 决策映射
        self.decision_mapping = {
            'stay': DecisionType.STAY,
            'prepare': DecisionType.PREPARE,
            'evacuate': DecisionType.EVACUATE,
            'wait': DecisionType.PREPARE  # 等待映射为准备
        }
        
        # 一致性阈值
        self.consensus_threshold = 0.7
    
    async def make_enhanced_decision(
        self, 
        context: DecisionContext,
        prediction_type: str = "individual_decision"
    ) -> EnhancedDecisionResult:
        """
        制定增强决策
        
        Args:
            context: 决策上下文
            prediction_type: 预测类型
            
        Returns:
            EnhancedDecisionResult: 增强决策结果
        """
        try:
            # 1. 执行原四维决策
            four_dim_result = await self.four_dimensional_framework.make_four_dimensional_decision(context)
            
            # 2. 转换上下文为预测模型格式
            prediction_context = self._convert_to_prediction_context(context)
            
            # 3. 执行实证模型预测
            lr_result = self.prediction_framework.lr_model.predict(prediction_context)
            ddc_result = await self.prediction_framework.ddc_model.predict_dynamic(prediction_context)
            
            # 4. 集成决策分析
            final_decision, final_confidence, consensus_score = self._integrate_decisions(
                four_dim_result, lr_result, ddc_result
            )
            
            # 5. 生成预测分析
            prediction_analysis = self._analyze_predictions(
                four_dim_result, lr_result, ddc_result, context
            )
            
            # 6. 生成综合解释
            explanation = self._generate_enhanced_explanation(
                four_dim_result, lr_result, ddc_result, final_decision, consensus_score
            )
            
            return EnhancedDecisionResult(
                four_dimensional_decision=four_dim_result['decision'],
                four_dimensional_score=four_dim_result['decision_score'],
                four_dimensional_confidence=four_dim_result['confidence'],
                four_dimensional_criteria=four_dim_result['criteria'],
                lr_prediction=lr_result,
                ddc_prediction=ddc_result,
                final_decision=final_decision,
                final_confidence=final_confidence,
                consensus_score=consensus_score,
                prediction_analysis=prediction_analysis,
                model_weights=self.model_weights.copy(),
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"增强决策制定失败: {e}")
            # 返回默认结果
            return self._create_default_result()
    
    def _convert_to_prediction_context(self, context: DecisionContext) -> PredictionContext:
        """
        将决策上下文转换为预测模型上下文
        
        Args:
            context: 原决策上下文
            
        Returns:
            PredictionContext: 预测模型上下文
        """
        agent_profile = context.agent_profile
        weather_data = context.weather_data
        social_context = context.social_context
        time_context = context.time_context
        location_data = context.location_data
        
        return PredictionContext(
            # 个人属性
            homeownership=agent_profile.get('homeownership', True),
            vehicle_access=agent_profile.get('vehicle_access', True),
            has_children=agent_profile.get('has_children', False),
            employed=agent_profile.get('employed', True),
            mobile_home=agent_profile.get('mobile_home', False),
            
            # 官方警告
            official_order=social_context.get('evacuation_order', 'none'),
            
            # 飓风属性
            hurricane_category=weather_data.get('hurricane_category', 2.0),
            distance_to_hurricane=location_data.get('distance_to_hurricane', 10.0),
            distance_to_forecast_landfall=location_data.get('distance_to_landfall', 5.0),
            wind_probability=weather_data.get('wind_probability', 0.3),
            storm_surge_forecast=weather_data.get('storm_surge', 2.0),
            
            # 时间因素
            time_of_day=time_context.get('time_of_day', 'day'),
            time_to_landfall=time_context.get('time_to_impact', 24.0),
            
            # 动态因素
            current_time_step=time_context.get('current_step', 1),
            max_time_steps=time_context.get('max_steps', 24)
        )
    
    def _integrate_decisions(
        self, 
        four_dim_result: Dict[str, Any],
        lr_result: PredictionResult,
        ddc_result: PredictionResult
    ) -> Tuple[str, float, float]:
        """
        集成多个模型的决策结果
        
        Args:
            four_dim_result: 四维决策结果
            lr_result: 逻辑回归结果
            ddc_result: 动态离散选择结果
            
        Returns:
            Tuple: (最终决策, 最终置信度, 一致性评分)
        """
        # 收集所有决策
        decisions = {
            'four_dimensional': four_dim_result['decision'],
            'logistic_regression': lr_result.decision.value,
            'dynamic_discrete_choice': ddc_result.decision.value
        }
        
        # 收集置信度
        confidences = {
            'four_dimensional': four_dim_result['confidence'],
            'logistic_regression': lr_result.confidence,
            'dynamic_discrete_choice': ddc_result.confidence
        }
        
        # 计算加权决策分数
        decision_scores = {}
        
        # 为每个可能的决策计算加权分数
        possible_decisions = ['stay', 'prepare', 'evacuate']
        
        for decision in possible_decisions:
            score = 0.0
            
            # 四维框架贡献
            if decisions['four_dimensional'] == decision:
                score += self.model_weights['four_dimensional'] * confidences['four_dimensional']
            
            # LR模型贡献
            if decisions['logistic_regression'] == decision:
                score += self.model_weights['logistic_regression'] * confidences['logistic_regression']
            elif decisions['logistic_regression'] == 'evacuate' and decision == 'prepare':
                # 疏散决策的部分支持准备决策
                score += self.model_weights['logistic_regression'] * confidences['logistic_regression'] * 0.5
            
            # DDC模型贡献
            if decisions['dynamic_discrete_choice'] == decision:
                score += self.model_weights['dynamic_discrete_choice'] * confidences['dynamic_discrete_choice']
            elif decisions['dynamic_discrete_choice'] == 'wait' and decision == 'prepare':
                # 等待决策映射为准备决策
                score += self.model_weights['dynamic_discrete_choice'] * confidences['dynamic_discrete_choice']
            
            decision_scores[decision] = score
        
        # 选择最高分数的决策
        final_decision = max(decision_scores.keys(), key=lambda k: decision_scores[k])
        final_confidence = decision_scores[final_decision]
        
        # 计算一致性评分
        consensus_score = self._calculate_consensus_score(decisions, confidences)
        
        return final_decision, final_confidence, consensus_score
    
    def _calculate_consensus_score(
        self, 
        decisions: Dict[str, str], 
        confidences: Dict[str, float]
    ) -> float:
        """
        计算模型一致性评分
        
        Args:
            decisions: 各模型决策
            confidences: 各模型置信度
            
        Returns:
            float: 一致性评分 (0-1)
        """
        # 统计决策一致性
        decision_list = list(decisions.values())
        unique_decisions = set(decision_list)
        
        if len(unique_decisions) == 1:
            # 完全一致
            avg_confidence = np.mean(list(confidences.values()))
            return min(avg_confidence * 1.2, 1.0)  # 一致性奖励
        elif len(unique_decisions) == 2:
            # 部分一致
            avg_confidence = np.mean(list(confidences.values()))
            return avg_confidence * 0.8
        else:
            # 完全不一致
            avg_confidence = np.mean(list(confidences.values()))
            return avg_confidence * 0.5
    
    def _analyze_predictions(
        self,
        four_dim_result: Dict[str, Any],
        lr_result: PredictionResult,
        ddc_result: PredictionResult,
        context: DecisionContext
    ) -> Dict[str, Any]:
        """
        分析预测结果
        
        Args:
            four_dim_result: 四维决策结果
            lr_result: 逻辑回归结果
            ddc_result: 动态离散选择结果
            context: 决策上下文
            
        Returns:
            Dict: 预测分析结果
        """
        analysis = {
            'model_agreement': {
                'four_dim_vs_lr': four_dim_result['decision'] == lr_result.decision.value,
                'four_dim_vs_ddc': four_dim_result['decision'] == ddc_result.decision.value,
                'lr_vs_ddc': lr_result.decision.value == ddc_result.decision.value
            },
            'confidence_comparison': {
                'four_dimensional': four_dim_result['confidence'],
                'logistic_regression': lr_result.confidence,
                'dynamic_discrete_choice': ddc_result.confidence
            },
            'risk_assessment': {
                'objective_risk': self._assess_objective_risk(context),
                'temporal_urgency': self._assess_temporal_urgency(context),
                'social_pressure': self._assess_social_pressure(context)
            },
            'prediction_reliability': {
                'lr_applicability': self._assess_lr_applicability(context),
                'ddc_applicability': self._assess_ddc_applicability(context),
                'four_dim_completeness': self._assess_four_dim_completeness(four_dim_result)
            }
        }
        
        # 添加DDC特有的时空预测
        if ddc_result.timing_prediction:
            analysis['temporal_prediction'] = {
                'evacuation_timing': ddc_result.timing_prediction,
                'time_to_landfall': context.time_context.get('time_to_impact', 24)
            }
        
        if ddc_result.spatial_factors:
            analysis['spatial_analysis'] = ddc_result.spatial_factors
        
        return analysis
    
    def _assess_objective_risk(self, context: DecisionContext) -> float:
        """评估客观风险水平"""
        weather_data = context.weather_data
        hurricane_category = weather_data.get('hurricane_category', 1)
        wind_speed = weather_data.get('wind_speed', 80)
        storm_surge = weather_data.get('storm_surge', 2)
        
        risk_score = (
            (hurricane_category / 5.0) * 0.4 +
            (min(wind_speed / 200.0, 1.0)) * 0.3 +
            (min(storm_surge / 20.0, 1.0)) * 0.3
        )
        
        return min(risk_score, 1.0)
    
    def _assess_temporal_urgency(self, context: DecisionContext) -> float:
        """评估时间紧迫性"""
        time_to_impact = context.time_context.get('time_to_impact', 48)
        return max(0, 1.0 - time_to_impact / 48.0)
    
    def _assess_social_pressure(self, context: DecisionContext) -> float:
        """评估社会压力"""
        evacuation_order = context.social_context.get('evacuation_order', 'none')
        pressure_map = {'none': 0.0, 'voluntary': 0.5, 'mandatory': 1.0}
        return pressure_map.get(evacuation_order, 0.0)
    
    def _assess_lr_applicability(self, context: DecisionContext) -> float:
        """评估LR模型适用性"""
        # LR模型适用于总体预测，数据完整性要求较低
        required_fields = ['hurricane_category', 'wind_probability']
        available_fields = sum(1 for field in required_fields 
                             if context.weather_data.get(field) is not None)
        return available_fields / len(required_fields)
    
    def _assess_ddc_applicability(self, context: DecisionContext) -> float:
        """评估DDC模型适用性"""
        # DDC模型适用于时空预测，需要更完整的时间和空间数据
        required_fields = ['time_to_impact', 'current_step', 'distance_to_hurricane']
        available_data = 0
        
        if context.time_context.get('time_to_impact') is not None:
            available_data += 1
        if context.time_context.get('current_step') is not None:
            available_data += 1
        if context.location_data.get('distance_to_hurricane') is not None:
            available_data += 1
            
        return available_data / len(required_fields)
    
    def _assess_four_dim_completeness(self, four_dim_result: Dict[str, Any]) -> float:
        """评估四维框架完整性"""
        criteria = four_dim_result.get('criteria', {})
        required_criteria = ['safety_concern', 'information_trust', 'evacuation_cost', 'property_attachment']
        
        available_criteria = sum(1 for criterion in required_criteria 
                               if criteria.get(criterion) is not None)
        return available_criteria / len(required_criteria)
    
    def _generate_enhanced_explanation(
        self,
        four_dim_result: Dict[str, Any],
        lr_result: PredictionResult,
        ddc_result: PredictionResult,
        final_decision: str,
        consensus_score: float
    ) -> str:
        """
        生成增强决策解释
        
        Args:
            four_dim_result: 四维决策结果
            lr_result: 逻辑回归结果
            ddc_result: 动态离散选择结果
            final_decision: 最终决策
            consensus_score: 一致性评分
            
        Returns:
            str: 决策解释
        """
        explanation_parts = []
        
        # 最终决策
        explanation_parts.append(f"增强决策结果: {final_decision}")
        
        # 一致性分析
        if consensus_score > self.consensus_threshold:
            explanation_parts.append(f"模型高度一致 (一致性: {consensus_score:.2f})")
        else:
            explanation_parts.append(f"模型存在分歧 (一致性: {consensus_score:.2f})")
        
        # 各模型贡献
        model_decisions = [
            f"四维框架: {four_dim_result['decision']} (置信度: {four_dim_result['confidence']:.2f})",
            f"逻辑回归: {lr_result.decision.value} (置信度: {lr_result.confidence:.2f})",
            f"动态选择: {ddc_result.decision.value} (置信度: {ddc_result.confidence:.2f})"
        ]
        explanation_parts.append("模型预测: " + "; ".join(model_decisions))
        
        # 关键影响因素
        key_factors = []
        if four_dim_result['criteria']['safety_concern'] > 0.7:
            key_factors.append("高安全关注")
        if lr_result.probability > 0.7:
            key_factors.append("高疏散概率(LR)")
        if ddc_result.timing_prediction:
            key_factors.append(f"预测疏散时间步: {ddc_result.timing_prediction}")
        
        if key_factors:
            explanation_parts.append("关键因素: " + ", ".join(key_factors))
        
        return "; ".join(explanation_parts)
    
    def _create_default_result(self) -> EnhancedDecisionResult:
        """创建默认结果"""
        from .evacuation_prediction_models import PredictionResult, EvacuationDecision, PredictionModelType
        
        default_lr = PredictionResult(
            decision=EvacuationDecision.STAY,
            probability=0.5,
            confidence=0.0,
            model_type=PredictionModelType.LOGISTIC_REGRESSION,
            explanation="默认LR结果"
        )
        
        default_ddc = PredictionResult(
            decision=EvacuationDecision.WAIT,
            probability=0.33,
            confidence=0.0,
            model_type=PredictionModelType.DYNAMIC_DISCRETE_CHOICE,
            explanation="默认DDC结果"
        )
        
        return EnhancedDecisionResult(
            four_dimensional_decision="prepare",
            four_dimensional_score=0.5,
            four_dimensional_confidence=0.5,
            four_dimensional_criteria={
                'safety_concern': 0.5,
                'information_trust': 0.5,
                'evacuation_cost': 0.5,
                'property_attachment': 0.5
            },
            lr_prediction=default_lr,
            ddc_prediction=default_ddc,
            final_decision="prepare",
            final_confidence=0.5,
            consensus_score=0.5,
            prediction_analysis={},
            model_weights=self.model_weights.copy(),
            explanation="系统错误，返回默认决策"
        )
    
    async def batch_predict(
        self, 
        contexts: List[DecisionContext],
        prediction_type: str = "individual_decision"
    ) -> List[EnhancedDecisionResult]:
        """
        批量预测
        
        Args:
            contexts: 决策上下文列表
            prediction_type: 预测类型
            
        Returns:
            List[EnhancedDecisionResult]: 预测结果列表
        """
        results = []
        
        for context in contexts:
            try:
                result = await self.make_enhanced_decision(context, prediction_type)
                results.append(result)
            except Exception as e:
                logger.error(f"批量预测中单个预测失败: {e}")
                results.append(self._create_default_result())
        
        return results
    
    def get_performance_metrics(self, results: List[EnhancedDecisionResult]) -> Dict[str, float]:
        """
        计算性能指标
        
        Args:
            results: 预测结果列表
            
        Returns:
            Dict: 性能指标
        """
        if not results:
            return {}
        
        # 一致性指标
        consensus_scores = [r.consensus_score for r in results]
        avg_consensus = np.mean(consensus_scores)
        
        # 置信度指标
        confidences = [r.final_confidence for r in results]
        avg_confidence = np.mean(confidences)
        
        # 模型一致性统计
        four_dim_lr_agreement = sum(1 for r in results 
                                   if r.four_dimensional_decision == r.lr_prediction.decision.value)
        four_dim_ddc_agreement = sum(1 for r in results 
                                    if r.four_dimensional_decision == r.ddc_prediction.decision.value)
        lr_ddc_agreement = sum(1 for r in results 
                              if r.lr_prediction.decision.value == r.ddc_prediction.decision.value)
        
        total_results = len(results)
        
        return {
            'average_consensus_score': avg_consensus,
            'average_confidence': avg_confidence,
            'four_dim_lr_agreement_rate': four_dim_lr_agreement / total_results,
            'four_dim_ddc_agreement_rate': four_dim_ddc_agreement / total_results,
            'lr_ddc_agreement_rate': lr_ddc_agreement / total_results,
            'high_consensus_rate': sum(1 for score in consensus_scores if score > self.consensus_threshold) / total_results
        }