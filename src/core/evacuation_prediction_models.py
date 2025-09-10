"""
基于实证研究的飓风疏散预测模型
基于论文: "Prediction of population behavior in hurricane evacuations" (2022)
集成逻辑回归(LR-S)和动态离散选择(DDC)模型
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
from scipy.special import expit  # sigmoid function

logger = logging.getLogger(__name__)

class PredictionModelType(Enum):
    """预测模型类型"""
    LOGISTIC_REGRESSION = "logistic_regression"  # LR-S: 适用于总疏散率预测
    DYNAMIC_DISCRETE_CHOICE = "dynamic_discrete_choice"  # DDC: 适用于时空分布预测
    PARTICIPATION_RATE = "participation_rate"  # PR-S: 基础参与率模型

class EvacuationDecision(Enum):
    """疏散决策类型"""
    STAY = "stay"
    EVACUATE = "evacuate"
    WAIT = "wait"  # DDC模型中的等待决策

@dataclass
class PredictionContext:
    """预测上下文数据"""
    # 个人属性 (基于论文Table 3)
    homeownership: bool = True  # 房屋所有权 (1=租房, 0=自有)
    vehicle_access: bool = True  # 车辆使用权
    has_children: bool = False  # 有儿童
    employed: bool = True  # 就业状态
    mobile_home: bool = False  # 移动房屋
    
    # 官方警告 (基于论文)
    official_order: str = "none"  # none/voluntary/mandatory
    
    # 飓风属性 (基于论文Table 4)
    hurricane_category: float = 2.0  # 飓风等级
    distance_to_hurricane: float = 10.0  # 到飓风距离(km)
    distance_to_forecast_landfall: float = 5.0  # 到预测登陆点距离(km)
    wind_probability: float = 0.3  # >74mph风速概率
    storm_surge_forecast: float = 2.0  # 风暴潮预测(ft)
    
    # 时间因素
    time_of_day: str = "day"  # night/evening/day
    time_to_landfall: float = 24.0  # 到登陆时间(小时)
    
    # 动态因素 (DDC模型)
    current_time_step: int = 1
    max_time_steps: int = 24  # 6小时为一个时间步，共6天

@dataclass
class PredictionResult:
    """预测结果"""
    decision: EvacuationDecision
    probability: float
    confidence: float
    model_type: PredictionModelType
    explanation: str
    timing_prediction: Optional[int] = None  # 疏散时间步预测
    spatial_factors: Optional[Dict[str, float]] = None

class LogisticRegressionModel:
    """
    逻辑回归模型 (LR-S)
    基于论文实证系数，适用于总疏散率预测
    """
    
    def __init__(self):
        # 基于论文实证研究的系数 (简化版本)
        self.coefficients = {
            'intercept': -1.2,
            'official_voluntary': 1.8,
            'official_mandatory': 2.5,
            'mobile_home': 0.8,
            'homeownership_rental': 0.4,
            'has_children': 0.3,
            'hurricane_category': 0.6,
            'distance_to_hurricane': -0.05,
            'wind_probability': 1.5,
            'storm_surge': 0.2
        }
    
    def predict(self, context: PredictionContext) -> PredictionResult:
        """
        使用逻辑回归预测疏散概率
        
        Args:
            context: 预测上下文
            
        Returns:
            PredictionResult: 预测结果
        """
        try:
            # 计算线性组合
            linear_combination = self.coefficients['intercept']
            
            # 官方命令
            if context.official_order == "voluntary":
                linear_combination += self.coefficients['official_voluntary']
            elif context.official_order == "mandatory":
                linear_combination += self.coefficients['official_mandatory']
            
            # 个人属性
            if context.mobile_home:
                linear_combination += self.coefficients['mobile_home']
            if not context.homeownership:  # 租房
                linear_combination += self.coefficients['homeownership_rental']
            if context.has_children:
                linear_combination += self.coefficients['has_children']
            
            # 飓风属性
            linear_combination += (
                self.coefficients['hurricane_category'] * context.hurricane_category +
                self.coefficients['distance_to_hurricane'] * context.distance_to_hurricane +
                self.coefficients['wind_probability'] * context.wind_probability +
                self.coefficients['storm_surge'] * context.storm_surge_forecast
            )
            
            # 应用sigmoid函数
            evacuation_probability = expit(linear_combination)
            
            # 决策阈值 (基于论文建议)
            decision = EvacuationDecision.EVACUATE if evacuation_probability > 0.5 else EvacuationDecision.STAY
            
            # 计算置信度
            confidence = abs(evacuation_probability - 0.5) * 2
            
            explanation = self._generate_lr_explanation(context, evacuation_probability)
            
            return PredictionResult(
                decision=decision,
                probability=evacuation_probability,
                confidence=confidence,
                model_type=PredictionModelType.LOGISTIC_REGRESSION,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"逻辑回归预测失败: {e}")
            return PredictionResult(
                decision=EvacuationDecision.STAY,
                probability=0.5,
                confidence=0.0,
                model_type=PredictionModelType.LOGISTIC_REGRESSION,
                explanation="预测计算失败，返回默认结果"
            )
    
    def _generate_lr_explanation(self, context: PredictionContext, probability: float) -> str:
        """生成逻辑回归预测解释"""
        factors = []
        
        if context.official_order == "mandatory":
            factors.append("强制疏散令(+)")
        elif context.official_order == "voluntary":
            factors.append("自愿疏散令(+)")
        
        if context.hurricane_category >= 3:
            factors.append("高强度飓风(+)")
        
        if context.wind_probability > 0.5:
            factors.append("高风速概率(+)")
        
        if context.mobile_home:
            factors.append("移动房屋(+)")
        
        if context.distance_to_hurricane < 20:
            factors.append("距离飓风较近(+)")
        
        return f"疏散概率: {probability:.2f}, 主要影响因素: {', '.join(factors) if factors else '无显著因素'}"

class DynamicDiscreteChoiceModel:
    """
    动态离散选择模型 (DDC)
    基于论文方法，适用于时空分布和时序预测
    """
    
    def __init__(self):
        # 基于论文的动态系数
        self.utility_coefficients = {
            'evacuate': {
                'intercept': -2.0,
                'official_order': 2.0,
                'hurricane_category': 0.8,
                'time_pressure': 1.5,
                'distance_factor': -0.03
            },
            'wait': {
                'intercept': -0.5,
                'uncertainty': 0.5,
                'information_value': 0.3
            },
            'stay': {
                'intercept': 0.0,  # 基准选择
                'property_attachment': 0.8,
                'evacuation_cost': -0.6
            }
        }
        
        # 动态调整参数
        self.discount_factor = 0.95  # 未来效用折扣因子
        self.learning_rate = 0.1
    
    async def predict_dynamic(self, context: PredictionContext) -> PredictionResult:
        """
        使用DDC模型进行动态预测
        
        Args:
            context: 预测上下文
            
        Returns:
            PredictionResult: 预测结果
        """
        try:
            # 计算当前时间步的选择概率
            utilities = self._calculate_utilities(context)
            probabilities = self._softmax(utilities)
            
            # 选择最优决策
            best_choice = max(probabilities.keys(), key=lambda k: probabilities[k])
            best_probability = probabilities[best_choice]
            
            # 预测疏散时间
            timing_prediction = self._predict_evacuation_timing(context, probabilities)
            
            # 空间因素分析
            spatial_factors = self._analyze_spatial_factors(context)
            
            # 计算置信度
            confidence = self._calculate_ddc_confidence(probabilities)
            
            explanation = self._generate_ddc_explanation(context, probabilities, timing_prediction)
            
            return PredictionResult(
                decision=EvacuationDecision(best_choice),
                probability=best_probability,
                confidence=confidence,
                model_type=PredictionModelType.DYNAMIC_DISCRETE_CHOICE,
                explanation=explanation,
                timing_prediction=timing_prediction,
                spatial_factors=spatial_factors
            )
            
        except Exception as e:
            logger.error(f"DDC模型预测失败: {e}")
            return PredictionResult(
                decision=EvacuationDecision.WAIT,
                probability=0.33,
                confidence=0.0,
                model_type=PredictionModelType.DYNAMIC_DISCRETE_CHOICE,
                explanation="DDC预测计算失败"
            )
    
    def _calculate_utilities(self, context: PredictionContext) -> Dict[str, float]:
        """计算各选择的效用值"""
        utilities = {}
        
        # 时间压力因子
        time_pressure = max(0, 1.0 - context.time_to_landfall / 48.0)
        
        # 官方命令强度
        order_strength = {
            'none': 0.0,
            'voluntary': 0.5,
            'mandatory': 1.0
        }.get(context.official_order, 0.0)
        
        # 疏散效用
        utilities['evacuate'] = (
            self.utility_coefficients['evacuate']['intercept'] +
            self.utility_coefficients['evacuate']['official_order'] * order_strength +
            self.utility_coefficients['evacuate']['hurricane_category'] * context.hurricane_category +
            self.utility_coefficients['evacuate']['time_pressure'] * time_pressure +
            self.utility_coefficients['evacuate']['distance_factor'] * context.distance_to_hurricane
        )
        
        # 等待效用
        uncertainty = 1.0 - min(context.wind_probability, 1.0)
        information_value = 1.0 - (context.current_time_step / context.max_time_steps)
        
        utilities['wait'] = (
            self.utility_coefficients['wait']['intercept'] +
            self.utility_coefficients['wait']['uncertainty'] * uncertainty +
            self.utility_coefficients['wait']['information_value'] * information_value
        )
        
        # 留守效用
        property_attachment = 0.8 if context.homeownership else 0.4
        evacuation_cost = 0.6 if context.vehicle_access else 1.0
        
        utilities['stay'] = (
            self.utility_coefficients['stay']['intercept'] +
            self.utility_coefficients['stay']['property_attachment'] * property_attachment +
            self.utility_coefficients['stay']['evacuation_cost'] * evacuation_cost
        )
        
        return utilities
    
    def _softmax(self, utilities: Dict[str, float]) -> Dict[str, float]:
        """计算softmax概率"""
        exp_utilities = {k: math.exp(v) for k, v in utilities.items()}
        total = sum(exp_utilities.values())
        return {k: v / total for k, v in exp_utilities.items()}
    
    def _predict_evacuation_timing(self, context: PredictionContext, probabilities: Dict[str, float]) -> Optional[int]:
        """预测疏散时间步"""
        if probabilities.get('evacuate', 0) > 0.5:
            # 基于时间压力和风险等级预测疏散时间
            urgency_factor = 1.0 - (context.time_to_landfall / 48.0)
            risk_factor = context.hurricane_category / 5.0
            
            # 时间步预测 (越紧急越早疏散)
            predicted_step = max(1, int(context.max_time_steps * (1.0 - urgency_factor - risk_factor)))
            return min(predicted_step, context.max_time_steps)
        
        return None
    
    def _analyze_spatial_factors(self, context: PredictionContext) -> Dict[str, float]:
        """分析空间分布因素"""
        return {
            'coastal_proximity': 1.0 - min(context.distance_to_hurricane / 50.0, 1.0),
            'landfall_proximity': 1.0 - min(context.distance_to_forecast_landfall / 30.0, 1.0),
            'surge_risk': min(context.storm_surge_forecast / 10.0, 1.0),
            'wind_risk': context.wind_probability
        }
    
    def _calculate_ddc_confidence(self, probabilities: Dict[str, float]) -> float:
        """计算DDC模型置信度"""
        # 基于概率分布的熵计算置信度
        entropy = -sum(p * math.log(p + 1e-10) for p in probabilities.values())
        max_entropy = math.log(len(probabilities))
        return 1.0 - (entropy / max_entropy)
    
    def _generate_ddc_explanation(
        self, 
        context: PredictionContext, 
        probabilities: Dict[str, float],
        timing: Optional[int]
    ) -> str:
        """生成DDC预测解释"""
        best_choice = max(probabilities.keys(), key=lambda k: probabilities[k])
        
        explanation = f"DDC预测: {best_choice} (概率: {probabilities[best_choice]:.2f})"
        
        if timing:
            explanation += f", 预测疏散时间步: {timing}"
        
        # 添加主要影响因素
        factors = []
        if context.official_order != "none":
            factors.append(f"官方命令({context.official_order})")
        if context.hurricane_category >= 3:
            factors.append("高强度飓风")
        if context.time_to_landfall < 24:
            factors.append("时间紧迫")
        
        if factors:
            explanation += f", 主要因素: {', '.join(factors)}"
        
        return explanation

class EvacuationPredictionFramework:
    """
    疏散预测框架
    集成多种预测模型，根据需求选择最适合的模型
    """
    
    def __init__(self):
        self.lr_model = LogisticRegressionModel()
        self.ddc_model = DynamicDiscreteChoiceModel()
        
        # 模型选择策略
        self.model_selection_strategy = {
            'total_evacuation_rate': PredictionModelType.LOGISTIC_REGRESSION,
            'spatial_temporal_prediction': PredictionModelType.DYNAMIC_DISCRETE_CHOICE,
            'individual_decision': PredictionModelType.DYNAMIC_DISCRETE_CHOICE
        }
    
    async def predict_evacuation_behavior(
        self, 
        context: PredictionContext,
        prediction_type: str = "individual_decision"
    ) -> PredictionResult:
        """
        预测疏散行为
        
        Args:
            context: 预测上下文
            prediction_type: 预测类型 (total_evacuation_rate/spatial_temporal_prediction/individual_decision)
            
        Returns:
            PredictionResult: 预测结果
        """
        try:
            # 根据预测类型选择模型
            model_type = self.model_selection_strategy.get(
                prediction_type, 
                PredictionModelType.DYNAMIC_DISCRETE_CHOICE
            )
            
            if model_type == PredictionModelType.LOGISTIC_REGRESSION:
                return self.lr_model.predict(context)
            elif model_type == PredictionModelType.DYNAMIC_DISCRETE_CHOICE:
                return await self.ddc_model.predict_dynamic(context)
            else:
                # 默认使用LR模型
                return self.lr_model.predict(context)
                
        except Exception as e:
            logger.error(f"疏散预测失败: {e}")
            return PredictionResult(
                decision=EvacuationDecision.WAIT,
                probability=0.5,
                confidence=0.0,
                model_type=PredictionModelType.LOGISTIC_REGRESSION,
                explanation="预测框架执行失败"
            )
    
    def compare_model_predictions(self, context: PredictionContext) -> Dict[str, PredictionResult]:
        """
        比较不同模型的预测结果
        
        Args:
            context: 预测上下文
            
        Returns:
            Dict: 各模型预测结果
        """
        results = {}
        
        try:
            # LR模型预测
            results['logistic_regression'] = self.lr_model.predict(context)
            
            # DDC模型预测 (需要异步)
            loop = asyncio.get_event_loop()
            results['dynamic_discrete_choice'] = loop.run_until_complete(
                self.ddc_model.predict_dynamic(context)
            )
            
        except Exception as e:
            logger.error(f"模型比较失败: {e}")
        
        return results
    
    def get_model_recommendation(self, use_case: str) -> str:
        """
        根据使用场景推荐模型
        
        Args:
            use_case: 使用场景
            
        Returns:
            str: 模型推荐说明
        """
        recommendations = {
            'emergency_management': "推荐使用DDC模型进行详细的时空分布预测，支持动态决策调整",
            'policy_planning': "推荐使用LR模型进行总体疏散率估算，简单高效",
            'real_time_response': "推荐使用DDC模型，能够处理动态变化的情况信息",
            'research_analysis': "建议同时使用LR和DDC模型进行对比分析"
        }
        
        return recommendations.get(use_case, "建议根据具体需求选择合适的模型")