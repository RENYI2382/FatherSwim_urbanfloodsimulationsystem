"""
Hurricane Mobility Agent 核心模块
包含智能体的核心行为逻辑和决策算法
"""

import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HurricanePhase(Enum):
    """飓风阶段枚举"""
    PRE_HURRICANE = "pre"      # 飓风前
    DURING_HURRICANE = "during"  # 飓风中
    POST_HURRICANE = "post"    # 飓风后


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"        # 低风险
    MEDIUM = "medium"  # 中等风险
    HIGH = "high"      # 高风险
    CRITICAL = "critical"  # 极高风险


@dataclass
class WeatherCondition:
    """天气条件数据类"""
    wind_speed: float = 0.0
    precipitation: float = 0.0
    temperature: float = 20.0
    humidity: float = 0.5
    pressure: float = 1013.25
    phase: HurricanePhase = HurricanePhase.PRE_HURRICANE


@dataclass
class AgentProfile:
    """智能体个人档案"""
    age: int = 35
    income_level: str = "medium"
    family_size: int = 2
    has_vehicle: bool = True
    mobility_impairment: bool = False
    risk_aversion: float = 0.5  # 0-1, 越高越规避风险
    social_connectivity: float = 0.5  # 0-1, 社交连接度


class RiskAssessment:
    """风险评估模块"""
    
    @staticmethod
    def calculate_weather_risk(weather: WeatherCondition) -> float:
        """
        计算天气风险
        
        Args:
            weather: 天气条件
            
        Returns:
            float: 风险评分 (0-1)
        """
        # 风速风险 (0-150 mph)
        wind_risk = min(weather.wind_speed / 150.0, 1.0)
        
        # 降雨风险 (0-100 mm/h)
        rain_risk = min(weather.precipitation / 100.0, 1.0)
        
        # 飓风阶段权重
        phase_weights = {
            HurricanePhase.PRE_HURRICANE: 0.3,
            HurricanePhase.DURING_HURRICANE: 1.0,
            HurricanePhase.POST_HURRICANE: 0.6
        }
        
        base_risk = wind_risk * 0.7 + rain_risk * 0.3
        phase_weight = phase_weights.get(weather.phase, 0.5)
        
        return min(base_risk * phase_weight, 1.0)
    
    @staticmethod
    def calculate_personal_risk(profile: AgentProfile, weather_risk: float) -> float:
        """
        计算个人风险
        
        Args:
            profile: 个人档案
            weather_risk: 天气风险
            
        Returns:
            float: 个人风险评分 (0-1)
        """
        # 年龄风险调整
        age_factor = 1.0
        if profile.age < 18 or profile.age > 65:
            age_factor = 1.2
        
        # 移动能力调整
        mobility_factor = 1.3 if profile.mobility_impairment else 1.0
        
        # 车辆可用性调整
        vehicle_factor = 0.8 if profile.has_vehicle else 1.2
        
        # 家庭规模调整
        family_factor = 1.0 + (profile.family_size - 1) * 0.1
        
        personal_risk = weather_risk * age_factor * mobility_factor * vehicle_factor * family_factor
        
        return min(personal_risk, 1.0)
    
    @staticmethod
    def get_risk_level(risk_score: float) -> RiskLevel:
        """
        获取风险等级
        
        Args:
            risk_score: 风险评分
            
        Returns:
            RiskLevel: 风险等级
        """
        if risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


class EvacuationDecision:
    """疏散决策模块"""
    
    @staticmethod
    def should_evacuate(
        risk_score: float,
        profile: AgentProfile,
        social_influence: float = 0.0
    ) -> Tuple[bool, str]:
        """
        决定是否疏散
        
        Args:
            risk_score: 风险评分
            profile: 个人档案
            social_influence: 社会影响因子
            
        Returns:
            Tuple[bool, str]: (是否疏散, 决策原因)
        """
        # 基础疏散阈值
        base_threshold = 0.6
        
        # 个人风险规避调整
        personal_threshold = base_threshold * (1 - profile.risk_aversion * 0.3)
        
        # 社会影响调整
        social_adjustment = social_influence * 0.2
        adjusted_threshold = personal_threshold - social_adjustment
        
        # 特殊情况调整
        if profile.mobility_impairment:
            adjusted_threshold -= 0.1  # 行动不便者更早疏散
        
        if profile.family_size > 3:
            adjusted_threshold -= 0.05  # 大家庭更早疏散
        
        should_evacuate = risk_score > adjusted_threshold
        
        # 决策原因
        if should_evacuate:
            if risk_score > 0.8:
                reason = "极高风险，立即疏散"
            elif risk_score > 0.6:
                reason = "高风险，建议疏散"
            else:
                reason = "基于个人情况决定疏散"
        else:
            reason = "风险可控，暂不疏散"
        
        return should_evacuate, reason


class MobilityPattern:
    """移动模式模块"""
    
    @staticmethod
    def generate_pattern(
        weather: WeatherCondition,
        profile: AgentProfile,
        phase: HurricanePhase = None
    ) -> Dict[str, Any]:
        """
        生成完整的移动模式
        
        Args:
            weather: 天气条件
            profile: 个人档案
            phase: 飓风阶段（可选）
            
        Returns:
            Dict[str, Any]: 包含移动模式数据的字典
        """
        try:
            # 使用传入的phase或从weather中获取
            current_phase = phase or getattr(weather, 'phase', HurricanePhase.PRE_HURRICANE)
            
            # 生成基础旅行时间
            base_time = 1.0  # 基础1小时
            travel_time = MobilityPattern.generate_travel_time(base_time, weather, profile)
            
            # 生成24小时分布
            hourly_distribution = MobilityPattern.generate_hourly_distribution(weather, profile)
            
            # 计算移动倾向
            mobility_tendency = MobilityPattern._calculate_mobility_tendency(weather, profile, current_phase)
            
            # 生成目的地偏好
            destination_preference = MobilityPattern._calculate_destination_preference(profile, current_phase)
            
            return {
                'travel_time': round(travel_time, 2),
                'hourly_distribution': [round(x, 3) for x in hourly_distribution],
                'mobility_tendency': round(mobility_tendency, 3),
                'destination_preference': destination_preference,
                'phase': current_phase.name if hasattr(current_phase, 'name') else str(current_phase),
                'weather_impact': MobilityPattern._assess_weather_impact(weather),
                'personal_factors': MobilityPattern._assess_personal_factors(profile)
            }
        except Exception as e:
            # 返回默认模式
            return {
                'travel_time': 1.0,
                'hourly_distribution': [1/24] * 24,
                'mobility_tendency': 0.5,
                'destination_preference': 'home',
                'phase': 'unknown',
                'weather_impact': 'moderate',
                'personal_factors': 'average',
                'error': str(e)
            }
    
    @staticmethod
    def _calculate_mobility_tendency(weather: WeatherCondition, profile: AgentProfile, phase: HurricanePhase) -> float:
        """计算移动倾向"""
        base_tendency = 0.5
        
        # 天气影响
        if weather.wind_speed > 50:
            base_tendency -= 0.3
        if weather.precipitation > 10:
            base_tendency -= 0.2
            
        # 个人因素影响
        if profile.mobility_impairment:
            base_tendency -= 0.2
        if not profile.has_vehicle:
            base_tendency -= 0.1
            
        # 飓风阶段影响
        phase_adjustments = {
            HurricanePhase.PRE_HURRICANE: 0.1,   # 准备期间稍微增加移动
            HurricanePhase.DURING_HURRICANE: -0.4,  # 飓风期间大幅减少移动
            HurricanePhase.POST_HURRICANE: -0.1   # 灾后谨慎移动
        }
        
        base_tendency += phase_adjustments.get(phase, 0)
        
        return max(0.0, min(1.0, base_tendency))
    
    @staticmethod
    def _calculate_destination_preference(profile: AgentProfile, phase: HurricanePhase) -> str:
        """计算目的地偏好"""
        if phase == HurricanePhase.DURING_HURRICANE:
            return 'shelter'
        elif phase == HurricanePhase.PRE_HURRICANE:
            if profile.family_size > 2:
                return 'family_gathering'
            else:
                return 'preparation_sites'
        elif phase == HurricanePhase.POST_HURRICANE:
            return 'recovery_sites'
        else:
            return 'home'
    
    @staticmethod
    def _assess_weather_impact(weather: WeatherCondition) -> str:
        """评估天气影响"""
        if weather.wind_speed > 70 or weather.precipitation > 20:
            return 'severe'
        elif weather.wind_speed > 40 or weather.precipitation > 10:
            return 'moderate'
        else:
            return 'mild'
    
    @staticmethod
    def _assess_personal_factors(profile: AgentProfile) -> str:
        """评估个人因素"""
        factors = []
        if profile.mobility_impairment:
            factors.append('mobility_limited')
        if not profile.has_vehicle:
            factors.append('no_vehicle')
        if profile.age > 65:
            factors.append('elderly')
        if profile.family_size > 3:
            factors.append('large_family')
            
        return ','.join(factors) if factors else 'average'
    
    @staticmethod
    def generate_travel_time(
        base_time: float,
        weather: WeatherCondition,
        profile: AgentProfile
    ) -> float:
        """
        生成旅行时间
        
        Args:
            base_time: 基础旅行时间
            weather: 天气条件
            profile: 个人档案
            
        Returns:
            float: 调整后的旅行时间
        """
        # 天气影响因子
        weather_factor = 1.0
        if weather.wind_speed > 50:
            weather_factor += weather.wind_speed / 100.0
        if weather.precipitation > 10:
            weather_factor += weather.precipitation / 50.0
        
        # 个人因子
        personal_factor = 1.0
        if profile.mobility_impairment:
            personal_factor *= 1.5
        if not profile.has_vehicle:
            personal_factor *= 2.0
        
        # 飓风阶段影响
        phase_factors = {
            HurricanePhase.PRE_HURRICANE: 1.2,  # 准备阶段，交通拥堵
            HurricanePhase.DURING_HURRICANE: 3.0,  # 飓风期间，严重延误
            HurricanePhase.POST_HURRICANE: 2.0   # 灾后，道路受损
        }
        
        phase_factor = phase_factors.get(weather.phase, 1.0)
        
        adjusted_time = base_time * weather_factor * personal_factor * phase_factor
        
        # 添加随机变异
        variation = random.uniform(0.8, 1.2)
        
        return adjusted_time * variation
    
    @staticmethod
    def generate_hourly_distribution(
        weather: WeatherCondition,
        profile: AgentProfile
    ) -> List[float]:
        """
        生成24小时出行分布
        
        Args:
            weather: 天气条件
            profile: 个人档案
            
        Returns:
            List[float]: 24小时的出行概率分布
        """
        # 基础分布模式（正常情况下的出行模式）
        base_pattern = [
            0.02, 0.01, 0.01, 0.01, 0.02, 0.05,  # 0-5时
            0.08, 0.12, 0.15, 0.10, 0.08, 0.09,  # 6-11时
            0.10, 0.08, 0.06, 0.05, 0.08, 0.12,  # 12-17时
            0.10, 0.08, 0.06, 0.04, 0.03, 0.02   # 18-23时
        ]
        
        # 根据飓风阶段调整
        if weather.phase == HurricanePhase.PRE_HURRICANE:
            # 飓风前：白天活动增加（准备工作）
            for i in range(8, 18):
                base_pattern[i] *= 1.3
        elif weather.phase == HurricanePhase.DURING_HURRICANE:
            # 飓风期间：大幅减少出行
            base_pattern = [x * 0.2 for x in base_pattern]
        elif weather.phase == HurricanePhase.POST_HURRICANE:
            # 飓风后：逐步恢复，但仍然谨慎
            base_pattern = [x * 0.6 for x in base_pattern]
        
        # 个人特征调整
        if profile.age > 65:
            # 老年人避免夜间出行
            for i in range(20, 24):
                base_pattern[i] *= 0.5
            for i in range(0, 6):
                base_pattern[i] *= 0.5
        
        if profile.family_size > 2:
            # 有家庭的人更规律的出行模式
            for i in range(7, 9):  # 早高峰
                base_pattern[i] *= 1.2
            for i in range(17, 19):  # 晚高峰
                base_pattern[i] *= 1.2
        
        # 归一化
        total = sum(base_pattern)
        if total > 0:
            base_pattern = [x / total for x in base_pattern]
        
        return base_pattern


class SocialInfluence:
    """社会影响模块"""
    
    @staticmethod
    def calculate_social_pressure(
        neighbor_evacuation_rate: float,
        social_connectivity: float
    ) -> float:
        """
        计算社会压力
        
        Args:
            neighbor_evacuation_rate: 邻居疏散率
            social_connectivity: 社交连接度
            
        Returns:
            float: 社会压力值 (0-1)
        """
        # 社会压力与邻居疏散率和个人社交连接度相关
        social_pressure = neighbor_evacuation_rate * social_connectivity
        
        # 非线性调整：当大多数人疏散时，压力急剧增加
        if neighbor_evacuation_rate > 0.7:
            social_pressure *= 1.5
        
        return min(social_pressure, 1.0)
    
    @staticmethod
    def get_information_credibility(source: str) -> float:
        """
        获取信息源可信度
        
        Args:
            source: 信息源类型
            
        Returns:
            float: 可信度 (0-1)
        """
        credibility_map = {
            "official": 0.9,      # 官方信息
            "news": 0.8,          # 新闻媒体
            "social_media": 0.6,  # 社交媒体
            "neighbor": 0.7,      # 邻居
            "family": 0.8,        # 家人
            "unknown": 0.3        # 未知来源
        }
        
        return credibility_map.get(source, 0.5)