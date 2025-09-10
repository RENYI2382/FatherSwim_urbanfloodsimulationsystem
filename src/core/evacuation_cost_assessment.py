"""
疏散成本评估模块 (w₃: 疏散成本或努力)
基于DDABM论文和博弈论模型实现
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CostComponents:
    """成本组件数据类"""
    economic_cost: float = 0.0      # 经济成本
    time_cost: float = 0.0          # 时间成本
    reliability_cost: float = 0.0   # 可靠性成本
    opportunity_cost: float = 0.0   # 机会成本
    total_cost: float = 0.0         # 总成本

class EvacuationCostAssessment:
    """
    疏散成本评估类
    基于DDABM论文的w₃标准：疏散成本高、不想堵车、不想离开
    """
    
    def __init__(self):
        # 基于文献的成本权重配置
        self.cost_weights = {
            'economic_cost': 0.35,      # 经济成本权重
            'time_cost': 0.30,          # 时间成本权重  
            'reliability_cost': 0.25,   # 可靠性成本权重
            'opportunity_cost': 0.10    # 机会成本权重
        }
        
        # BPR函数参数（基于博弈论文献）
        self.bpr_params = {
            'alpha': 0.15,  # BPR参数α
            'beta': 4.0     # BPR参数β
        }
        
        # 收入水平对成本敏感性映射
        self.income_sensitivity = {
            'low': 0.85,     # 低收入对成本高度敏感
            'medium': 0.55,  # 中等收入中度敏感
            'high': 0.25     # 高收入对成本不敏感
        }
        
        # 年龄对时间成本的影响
        self.age_time_factors = {
            'young': 0.6,    # 年轻人时间成本相对较低
            'middle': 0.8,   # 中年人时间成本中等
            'elderly': 1.2   # 老年人时间成本较高
        }
    
    def calculate_evacuation_cost(
        self, 
        agent_profile: Dict[str, Any], 
        traffic_data: Dict[str, Any], 
        distance: float,
        weather_severity: float = 0.5
    ) -> CostComponents:
        """
        计算综合疏散成本
        
        Args:
            agent_profile: 智能体档案
            traffic_data: 交通数据
            distance: 疏散距离（英里）
            weather_severity: 天气严重程度 [0,1]
            
        Returns:
            CostComponents: 成本组件详情
        """
        try:
            # 1. 计算经济成本
            economic_cost = self._calculate_economic_cost(
                agent_profile, distance, weather_severity
            )
            
            # 2. 计算时间成本（基于BPR函数）
            time_cost = self._calculate_time_cost(
                agent_profile, traffic_data, distance
            )
            
            # 3. 计算可靠性成本（基于对数函数）
            reliability_cost = self._calculate_reliability_cost(
                traffic_data, weather_severity
            )
            
            # 4. 计算机会成本
            opportunity_cost = self._calculate_opportunity_cost(
                agent_profile, weather_severity
            )
            
            # 5. 综合成本计算
            total_cost = (
                economic_cost * self.cost_weights['economic_cost'] +
                time_cost * self.cost_weights['time_cost'] +
                reliability_cost * self.cost_weights['reliability_cost'] +
                opportunity_cost * self.cost_weights['opportunity_cost']
            )
            
            return CostComponents(
                economic_cost=economic_cost,
                time_cost=time_cost,
                reliability_cost=reliability_cost,
                opportunity_cost=opportunity_cost,
                total_cost=min(total_cost, 1.0)  # 归一化到[0,1]
            )
            
        except Exception as e:
            logger.error(f"疏散成本计算失败: {e}")
            return CostComponents(total_cost=0.5)  # 返回默认中等成本
    
    def _calculate_economic_cost(
        self, 
        agent_profile: Dict[str, Any], 
        distance: float, 
        weather_severity: float
    ) -> float:
        """
        计算经济成本
        基于收入水平、距离和天气严重程度
        """
        # 基础距离成本
        base_distance_cost = min(distance / 500.0, 1.0)
        
        # 收入水平调整
        income_level = agent_profile.get('income_level', 'medium')
        income_multiplier = self.income_sensitivity.get(income_level, 0.55)
        
        # 车辆拥有情况影响
        has_vehicle = agent_profile.get('has_vehicle', True)
        vehicle_factor = 0.3 if has_vehicle else 0.8  # 无车成本更高
        
        # 家庭规模影响
        family_size = agent_profile.get('family_size', 2)
        family_factor = min(1.0 + (family_size - 1) * 0.15, 2.0)
        
        # 天气严重程度影响（恶劣天气增加成本）
        weather_factor = 1.0 + weather_severity * 0.3
        
        economic_cost = (
            base_distance_cost * 
            income_multiplier * 
            vehicle_factor * 
            family_factor * 
            weather_factor
        )
        
        return min(economic_cost, 1.0)
    
    def _calculate_time_cost(
        self, 
        agent_profile: Dict[str, Any], 
        traffic_data: Dict[str, Any], 
        distance: float
    ) -> float:
        """
        基于BPR函数计算时间成本
        t = t₀(1 + α(x/c)^β)
        """
        # 基础自由流时间（小时）
        base_time = distance / 60.0  # 假设自由流速度60mph
        
        # 交通拥堵水平
        congestion_level = traffic_data.get('congestion_level', 0.5)
        capacity_ratio = traffic_data.get('capacity_ratio', 0.7)
        
        # BPR函数计算实际时间
        alpha = self.bpr_params['alpha']
        beta = self.bpr_params['beta']
        
        actual_time = base_time * (
            1 + alpha * (capacity_ratio ** beta) * (1 + congestion_level)
        )
        
        # 年龄对时间成本的影响
        age = agent_profile.get('age', 35)
        if age < 30:
            age_category = 'young'
        elif age < 65:
            age_category = 'middle'
        else:
            age_category = 'elderly'
        
        age_factor = self.age_time_factors[age_category]
        
        # 工作状态影响
        has_work_responsibility = agent_profile.get('work_responsibility', False)
        work_factor = 1.3 if has_work_responsibility else 1.0
        
        time_cost = (actual_time / 12.0) * age_factor * work_factor  # 归一化
        
        return min(time_cost, 1.0)
    
    def _calculate_reliability_cost(
        self, 
        traffic_data: Dict[str, Any], 
        weather_severity: float
    ) -> float:
        """
        基于对数函数计算可靠性成本
        r = -ln(R), R = P(t ≤ θt₀)
        """
        # 基础可靠性
        base_reliability = traffic_data.get('reliability', 0.8)
        
        # 天气对可靠性的影响
        weather_impact = weather_severity * 0.4
        adjusted_reliability = max(base_reliability - weather_impact, 0.1)
        
        # 交通拥堵对可靠性的影响
        congestion_impact = traffic_data.get('congestion_level', 0.5) * 0.2
        final_reliability = max(adjusted_reliability - congestion_impact, 0.1)
        
        # 对数可靠性成本
        reliability_cost = -np.log(final_reliability) / 3.0  # 归一化
        
        return min(reliability_cost, 1.0)
    
    def _calculate_opportunity_cost(
        self, 
        agent_profile: Dict[str, Any], 
        weather_severity: float
    ) -> float:
        """
        计算机会成本
        包括工作损失、社会活动中断等
        """
        # 工作职责影响
        work_responsibility = agent_profile.get('work_responsibility', False)
        work_cost = 0.7 if work_responsibility else 0.2
        
        # 教育水平影响（高教育水平通常机会成本更高）
        education_level = agent_profile.get('education_level', 'medium')
        education_factors = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        education_cost = education_factors.get(education_level, 0.6)
        
        # 社会网络影响
        social_network = agent_profile.get('social_network', 0.5)
        social_cost = social_network * 0.4
        
        # 天气严重程度降低机会成本（紧急情况下其他活动不重要）
        weather_reduction = weather_severity * 0.3
        
        opportunity_cost = max(
            (work_cost * 0.5 + education_cost * 0.3 + social_cost * 0.2) - weather_reduction,
            0.0
        )
        
        return min(opportunity_cost, 1.0)
    
    def get_cost_explanation(self, cost_components: CostComponents) -> str:
        """
        生成成本评估的解释说明
        """
        explanations = []
        
        if cost_components.economic_cost > 0.7:
            explanations.append("经济负担较重")
        elif cost_components.economic_cost > 0.4:
            explanations.append("经济成本适中")
        else:
            explanations.append("经济成本较低")
        
        if cost_components.time_cost > 0.7:
            explanations.append("时间成本很高")
        elif cost_components.time_cost > 0.4:
            explanations.append("时间成本中等")
        else:
            explanations.append("时间成本较低")
        
        if cost_components.reliability_cost > 0.6:
            explanations.append("路况不确定性大")
        
        if cost_components.opportunity_cost > 0.6:
            explanations.append("机会成本较高")
        
        return "、".join(explanations)
    
    def apply_dynamic_adjustment(
        self, 
        base_cost: float, 
        time_factor: float, 
        social_influence: float
    ) -> float:
        """
        应用动态调整因子
        基于EDFT框架的动态更新
        """
        # 时间因子影响（随时间推移，成本感知可能变化）
        time_adjustment = 1.0 + (time_factor - 0.5) * 0.2
        
        # 社会影响调整（他人疏散行为影响成本感知）
        social_adjustment = 1.0 + social_influence * 0.15
        
        adjusted_cost = base_cost * time_adjustment * social_adjustment
        
        return min(adjusted_cost, 1.0)