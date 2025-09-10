"""
场景特定适配器 (Scenario-Specific Adapters)
为不同类型的紧急场景提供专门的决策逻辑和参数配置
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from .emergency_scenario_decision_framework import (
    EmergencyScenarioType, 
    EmergencyContext, 
    MovementDecision,
    MovementPattern,
    MovementUrgency
)

logger = logging.getLogger(__name__)

class ScenarioCharacteristics(Enum):
    """场景特征"""
    PREDICTABLE = "predictable"           # 可预测的
    SUDDEN_ONSET = "sudden_onset"         # 突发性的
    SLOW_ONSET = "slow_onset"            # 缓慢发展的
    LOCALIZED = "localized"              # 局部性的
    WIDESPREAD = "widespread"            # 广泛性的
    SHORT_DURATION = "short_duration"    # 短期的
    LONG_DURATION = "long_duration"      # 长期的

@dataclass
class ScenarioParameters:
    """场景参数配置"""
    # 时间特征
    typical_warning_time: float = 24.0    # 典型预警时间(小时)
    escalation_rate: float = 0.1          # 风险升级速率
    peak_danger_duration: float = 6.0     # 峰值危险持续时间
    
    # 空间特征
    affected_radius: float = 50.0         # 影响半径(公里)
    evacuation_radius: float = 30.0       # 疏散半径(公里)
    safe_distance: float = 100.0          # 安全距离(公里)
    
    # 决策权重调整
    safety_weight_multiplier: float = 1.0
    time_pressure_multiplier: float = 1.0
    social_influence_multiplier: float = 1.0
    cost_sensitivity_multiplier: float = 1.0
    
    # 行为特征
    panic_tendency: float = 0.3           # 恐慌倾向
    preparation_importance: float = 0.5   # 准备重要性
    evacuation_threshold_adjustment: float = 0.0  # 疏散阈值调整
    
    # 特殊考虑因素
    infrastructure_vulnerability: float = 0.5  # 基础设施脆弱性
    communication_reliability: float = 0.8     # 通信可靠性
    resource_scarcity_factor: float = 0.3      # 资源稀缺因子

class BaseScenarioAdapter(ABC):
    """场景适配器基类"""
    
    def __init__(self, scenario_type: EmergencyScenarioType):
        self.scenario_type = scenario_type
        self.parameters = self._initialize_parameters()
        self.characteristics = self._define_characteristics()
    
    @abstractmethod
    def _initialize_parameters(self) -> ScenarioParameters:
        """初始化场景参数"""
        pass
    
    @abstractmethod
    def _define_characteristics(self) -> List[ScenarioCharacteristics]:
        """定义场景特征"""
        pass
    
    def adjust_risk_assessment(
        self, 
        base_risk: float, 
        time_to_impact: float, 
        location_distance: float,
        agent_profile: Dict[str, Any]
    ) -> float:
        """调整风险评估"""
        
        adjusted_risk = base_risk
        
        # 时间因素调整
        time_factor = self._calculate_time_factor(time_to_impact)
        adjusted_risk *= time_factor
        
        # 距离因素调整
        distance_factor = self._calculate_distance_factor(location_distance)
        adjusted_risk *= distance_factor
        
        # 个人脆弱性调整
        vulnerability_factor = self._calculate_vulnerability_factor(agent_profile)
        adjusted_risk *= vulnerability_factor
        
        return min(adjusted_risk, 1.0)
    
    def adjust_decision_weights(
        self, 
        base_weights: Dict[str, float], 
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """调整决策权重"""
        
        adjusted_weights = base_weights.copy()
        
        # 安全权重调整
        adjusted_weights['safety_concern'] *= self.parameters.safety_weight_multiplier
        
        # 时间压力权重调整
        time_pressure = context.get('time_pressure', 0.5)
        if time_pressure > 0.7:
            adjusted_weights['safety_concern'] *= self.parameters.time_pressure_multiplier
        
        # 社会影响权重调整
        social_pressure = context.get('social_pressure', 0.5)
        if social_pressure > 0.6:
            adjusted_weights['social_influence'] *= self.parameters.social_influence_multiplier
        
        # 成本敏感性调整
        economic_stress = context.get('economic_stress', 0.5)
        if economic_stress > 0.6:
            adjusted_weights['evacuation_cost'] *= self.parameters.cost_sensitivity_multiplier
        
        # 归一化权重
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def generate_scenario_specific_advice(
        self, 
        decision: MovementDecision, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成场景特定建议"""
        
        advice = {
            'preparation_actions': self._get_preparation_actions(),
            'evacuation_timing': self._get_evacuation_timing_advice(context),
            'route_considerations': self._get_route_considerations(),
            'resource_priorities': self._get_resource_priorities(),
            'communication_plan': self._get_communication_plan(),
            'special_warnings': self._get_special_warnings(context)
        }
        
        return advice
    
    def _calculate_time_factor(self, time_to_impact: float) -> float:
        """计算时间因子"""
        if time_to_impact <= 0:
            return 2.0
        
        # 基于场景特征调整时间敏感性
        if ScenarioCharacteristics.SUDDEN_ONSET in self.characteristics:
            return max(1.0, 2.0 - time_to_impact / 12.0)
        elif ScenarioCharacteristics.SLOW_ONSET in self.characteristics:
            return max(0.8, 1.2 - time_to_impact / 48.0)
        else:
            return max(0.9, 1.5 - time_to_impact / 24.0)
    
    def _calculate_distance_factor(self, distance: float) -> float:
        """计算距离因子"""
        if distance <= self.parameters.evacuation_radius:
            return 1.5
        elif distance <= self.parameters.affected_radius:
            return 1.2
        elif distance <= self.parameters.safe_distance:
            return 1.0
        else:
            return 0.8
    
    def _calculate_vulnerability_factor(self, agent_profile: Dict[str, Any]) -> float:
        """计算脆弱性因子"""
        vulnerability = 1.0
        
        # 年龄脆弱性
        age = agent_profile.get('age', 35)
        if age < 18 or age > 65:
            vulnerability *= 1.2
        
        # 健康脆弱性
        health = agent_profile.get('health_status', 'good')
        if health in ['poor', 'disabled']:
            vulnerability *= 1.3
        
        # 交通脆弱性
        if not agent_profile.get('has_car', True):
            vulnerability *= 1.1
        
        return vulnerability
    
    @abstractmethod
    def _get_preparation_actions(self) -> List[str]:
        """获取准备行动建议"""
        pass
    
    @abstractmethod
    def _get_evacuation_timing_advice(self, context: Dict[str, Any]) -> str:
        """获取疏散时机建议"""
        pass
    
    @abstractmethod
    def _get_route_considerations(self) -> List[str]:
        """获取路径考虑因素"""
        pass
    
    @abstractmethod
    def _get_resource_priorities(self) -> List[str]:
        """获取资源优先级"""
        pass
    
    @abstractmethod
    def _get_communication_plan(self) -> Dict[str, str]:
        """获取通信计划"""
        pass
    
    @abstractmethod
    def _get_special_warnings(self, context: Dict[str, Any]) -> List[str]:
        """获取特殊警告"""
        pass

class HurricaneAdapter(BaseScenarioAdapter):
    """飓风场景适配器"""
    
    def __init__(self):
        super().__init__(EmergencyScenarioType.HURRICANE_APPROACH)
    
    def _initialize_parameters(self) -> ScenarioParameters:
        return ScenarioParameters(
            typical_warning_time=72.0,        # 飓风通常有72小时预警
            escalation_rate=0.08,             # 相对缓慢的升级
            peak_danger_duration=12.0,        # 峰值危险持续12小时
            affected_radius=200.0,            # 大范围影响
            evacuation_radius=100.0,          # 大范围疏散
            safe_distance=300.0,              # 需要远距离撤离
            safety_weight_multiplier=1.2,     # 强调安全
            time_pressure_multiplier=1.5,     # 时间压力重要
            social_influence_multiplier=1.3,  # 社会影响显著
            cost_sensitivity_multiplier=0.8,  # 成本敏感性降低
            panic_tendency=0.4,               # 中等恐慌倾向
            preparation_importance=0.8,       # 准备非常重要
            evacuation_threshold_adjustment=-0.1,  # 降低疏散阈值
            infrastructure_vulnerability=0.7,  # 基础设施较脆弱
            communication_reliability=0.6,     # 通信可能中断
            resource_scarcity_factor=0.4       # 资源相对充足
        )
    
    def _define_characteristics(self) -> List[ScenarioCharacteristics]:
        return [
            ScenarioCharacteristics.PREDICTABLE,
            ScenarioCharacteristics.SLOW_ONSET,
            ScenarioCharacteristics.WIDESPREAD,
            ScenarioCharacteristics.LONG_DURATION
        ]
    
    def _get_preparation_actions(self) -> List[str]:
        return [
            "储备至少7天的食物和水",
            "准备应急医疗用品",
            "加固门窗，清理排水系统",
            "准备备用电源和通信设备",
            "制定家庭疏散计划",
            "准备重要文件副本",
            "加满汽车油箱"
        ]
    
    def _get_evacuation_timing_advice(self, context: Dict[str, Any]) -> str:
        time_to_impact = context.get('time_to_impact', 24)
        
        if time_to_impact > 48:
            return "现在是制定疏散计划的最佳时机"
        elif time_to_impact > 24:
            return "建议在未来12-24小时内开始疏散"
        elif time_to_impact > 12:
            return "应立即开始疏散，避免交通拥堵"
        else:
            return "时间紧迫，立即疏散到最近的安全区域"
    
    def _get_route_considerations(self) -> List[str]:
        return [
            "避开沿海和低洼地区",
            "选择内陆高地路线",
            "避免跨越大桥和易受洪水影响的道路",
            "准备备用路线以应对道路封闭",
            "关注交通管制和疏散路线指引"
        ]
    
    def _get_resource_priorities(self) -> List[str]:
        return [
            "饮用水（每人每天4升）",
            "非易腐食品",
            "处方药物",
            "手电筒和电池",
            "急救包",
            "现金",
            "重要文件"
        ]
    
    def _get_communication_plan(self) -> Dict[str, str]:
        return {
            "primary": "手机通信",
            "backup": "收音机广播",
            "emergency": "卫星电话或对讲机",
            "family_contact": "指定外地联系人",
            "meeting_point": "预定家庭集合点"
        }
    
    def _get_special_warnings(self, context: Dict[str, Any]) -> List[str]:
        warnings = []
        
        severity = context.get('severity', 0.5)
        if severity > 0.8:
            warnings.append("预计将有极端风速和暴雨")
        
        if context.get('storm_surge_risk', False):
            warnings.append("警惕风暴潮威胁，远离海岸线")
        
        if context.get('power_outage_expected', True):
            warnings.append("预计长时间停电，准备备用电源")
        
        return warnings

class FloodAdapter(BaseScenarioAdapter):
    """洪水场景适配器"""
    
    def __init__(self):
        super().__init__(EmergencyScenarioType.FLOOD_RISING)
    
    def _initialize_parameters(self) -> ScenarioParameters:
        return ScenarioParameters(
            typical_warning_time=24.0,        # 洪水预警时间较短
            escalation_rate=0.15,             # 快速升级
            peak_danger_duration=8.0,         # 峰值危险持续8小时
            affected_radius=50.0,             # 沿河流域影响
            evacuation_radius=20.0,           # 局部疏散
            safe_distance=50.0,               # 中等距离撤离
            safety_weight_multiplier=1.5,     # 极度强调安全
            time_pressure_multiplier=2.0,     # 时间极其紧迫
            social_influence_multiplier=1.1,  # 社会影响中等
            cost_sensitivity_multiplier=0.6,  # 成本考虑较少
            panic_tendency=0.6,               # 较高恐慌倾向
            preparation_importance=0.6,       # 准备时间有限
            evacuation_threshold_adjustment=-0.2,  # 大幅降低疏散阈值
            infrastructure_vulnerability=0.8,  # 基础设施极脆弱
            communication_reliability=0.5,     # 通信容易中断
            resource_scarcity_factor=0.6       # 资源获取困难
        )
    
    def _define_characteristics(self) -> List[ScenarioCharacteristics]:
        return [
            ScenarioCharacteristics.SUDDEN_ONSET,
            ScenarioCharacteristics.LOCALIZED,
            ScenarioCharacteristics.SHORT_DURATION
        ]
    
    def _get_preparation_actions(self) -> List[str]:
        return [
            "立即移动贵重物品到高处",
            "准备救生衣和漂浮设备",
            "储备3天紧急用品",
            "关闭电源和燃气",
            "准备防水袋保护重要文件",
            "了解最近的高地位置"
        ]
    
    def _get_evacuation_timing_advice(self, context: Dict[str, Any]) -> str:
        water_level = context.get('water_level', 0.5)
        
        if water_level > 0.8:
            return "水位危险，立即撤离到高地"
        elif water_level > 0.6:
            return "水位上涨，建议立即疏散"
        elif water_level > 0.4:
            return "密切监控水位，准备随时疏散"
        else:
            return "保持警惕，制定疏散路线"
    
    def _get_route_considerations(self) -> List[str]:
        return [
            "避开所有低洼地区和河流附近",
            "选择高地路线",
            "避免穿越积水区域",
            "注意道路冲刷和桥梁安全",
            "准备步行路线以防车辆无法通行"
        ]
    
    def _get_resource_priorities(self) -> List[str]:
        return [
            "救生衣和漂浮设备",
            "防水手电筒",
            "饮用水",
            "高能量食品",
            "急救包",
            "防水通信设备",
            "绳索和工具"
        ]
    
    def _get_communication_plan(self) -> Dict[str, str]:
        return {
            "primary": "防水手机",
            "backup": "防水对讲机",
            "emergency": "信号弹或哨子",
            "rescue_signal": "在屋顶或高处挥舞明亮物品",
            "location_sharing": "GPS定位分享"
        }
    
    def _get_special_warnings(self, context: Dict[str, Any]) -> List[str]:
        warnings = []
        
        if context.get('flash_flood_risk', False):
            warnings.append("警惕山洪暴发，避开河谷地区")
        
        if context.get('dam_failure_risk', False):
            warnings.append("上游水库存在溃坝风险")
        
        warnings.append("避免在积水中行走，水深超过膝盖极其危险")
        warnings.append("远离电线和电气设备")
        
        return warnings

class WildfireAdapter(BaseScenarioAdapter):
    """野火场景适配器"""
    
    def __init__(self):
        super().__init__(EmergencyScenarioType.WILDFIRE_SPREAD)
    
    def _initialize_parameters(self) -> ScenarioParameters:
        return ScenarioParameters(
            typical_warning_time=12.0,        # 野火预警时间很短
            escalation_rate=0.25,             # 极快升级
            peak_danger_duration=4.0,         # 峰值危险持续4小时
            affected_radius=30.0,             # 中等范围影响
            evacuation_radius=15.0,           # 紧急疏散范围
            safe_distance=30.0,               # 需要足够距离
            safety_weight_multiplier=1.8,     # 极度强调安全
            time_pressure_multiplier=3.0,     # 时间极度紧迫
            social_influence_multiplier=0.9,  # 社会影响较小
            cost_sensitivity_multiplier=0.4,  # 几乎不考虑成本
            panic_tendency=0.8,               # 高恐慌倾向
            preparation_importance=0.3,       # 准备时间极少
            evacuation_threshold_adjustment=-0.3,  # 极大降低疏散阈值
            infrastructure_vulnerability=0.9,  # 基础设施极易损坏
            communication_reliability=0.4,     # 通信很容易中断
            resource_scarcity_factor=0.8       # 资源极度稀缺
        )
    
    def _define_characteristics(self) -> List[ScenarioCharacteristics]:
        return [
            ScenarioCharacteristics.SUDDEN_ONSET,
            ScenarioCharacteristics.LOCALIZED,
            ScenarioCharacteristics.SHORT_DURATION
        ]
    
    def _get_preparation_actions(self) -> List[str]:
        return [
            "立即准备逃生包",
            "湿润毛巾准备防烟",
            "关闭燃气阀门",
            "准备N95口罩",
            "清理车辆周围可燃物",
            "准备大量饮用水"
        ]
    
    def _get_evacuation_timing_advice(self, context: Dict[str, Any]) -> str:
        fire_distance = context.get('fire_distance', 10)
        wind_speed = context.get('wind_speed', 10)
        
        if fire_distance < 2 or wind_speed > 30:
            return "火势逼近，立即撤离"
        elif fire_distance < 5:
            return "准备立即疏散，不要等待"
        elif fire_distance < 10:
            return "密切关注火势，随时准备撤离"
        else:
            return "制定疏散计划，监控火势发展"
    
    def _get_route_considerations(self) -> List[str]:
        return [
            "选择远离植被茂密区域的路线",
            "避开峡谷和狭窄山谷",
            "选择宽阔的道路",
            "避免逆风行驶",
            "准备多条备用路线"
        ]
    
    def _get_resource_priorities(self) -> List[str]:
        return [
            "N95口罩或湿毛巾",
            "大量饮用水",
            "手电筒",
            "急救包",
            "重要文件",
            "手机充电器",
            "现金"
        ]
    
    def _get_communication_plan(self) -> Dict[str, str]:
        return {
            "primary": "手机（电池充足）",
            "backup": "车载收音机",
            "emergency": "汽车喇叭",
            "location_update": "定期向家人报告位置",
            "evacuation_center": "前往指定疏散中心"
        }
    
    def _get_special_warnings(self, context: Dict[str, Any]) -> List[str]:
        warnings = []
        
        wind_speed = context.get('wind_speed', 10)
        if wind_speed > 25:
            warnings.append("强风将加速火势蔓延")
        
        if context.get('dry_conditions', True):
            warnings.append("干燥条件下火势蔓延极快")
        
        warnings.append("避免吸入烟雾，使用湿毛巾遮住口鼻")
        warnings.append("如被困，寻找开阔地带或水源附近")
        
        return warnings

class ScenarioAdapterFactory:
    """场景适配器工厂"""
    
    _adapters = {
        EmergencyScenarioType.HURRICANE_APPROACH: HurricaneAdapter,
        EmergencyScenarioType.FLOOD_RISING: FloodAdapter,
        EmergencyScenarioType.WILDFIRE_SPREAD: WildfireAdapter,
    }
    
    @classmethod
    def create_adapter(cls, scenario_type: EmergencyScenarioType) -> BaseScenarioAdapter:
        """创建场景适配器"""
        
        adapter_class = cls._adapters.get(scenario_type)
        if adapter_class:
            return adapter_class()
        else:
            # 返回默认适配器
            logger.warning(f"未找到场景类型 {scenario_type} 的适配器，使用默认配置")
            return cls._create_default_adapter(scenario_type)
    
    @classmethod
    def _create_default_adapter(cls, scenario_type: EmergencyScenarioType) -> BaseScenarioAdapter:
        """创建默认适配器"""
        
        class DefaultAdapter(BaseScenarioAdapter):
            def _initialize_parameters(self) -> ScenarioParameters:
                return ScenarioParameters()
            
            def _define_characteristics(self) -> List[ScenarioCharacteristics]:
                return [ScenarioCharacteristics.PREDICTABLE]
            
            def _get_preparation_actions(self) -> List[str]:
                return ["准备应急用品", "制定疏散计划"]
            
            def _get_evacuation_timing_advice(self, context: Dict[str, Any]) -> str:
                return "根据官方指引决定疏散时机"
            
            def _get_route_considerations(self) -> List[str]:
                return ["选择安全路线", "避开危险区域"]
            
            def _get_resource_priorities(self) -> List[str]:
                return ["水", "食物", "急救包"]
            
            def _get_communication_plan(self) -> Dict[str, str]:
                return {"primary": "手机", "backup": "收音机"}
            
            def _get_special_warnings(self, context: Dict[str, Any]) -> List[str]:
                return ["保持警惕", "听从官方指引"]
        
        return DefaultAdapter(scenario_type)
    
    @classmethod
    def get_available_scenarios(cls) -> List[EmergencyScenarioType]:
        """获取可用的场景类型"""
        return list(cls._adapters.keys())