#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市空间移动行为建模模块

本模块实现智能体在城市空间中的移动行为建模，包括：
1. 空间路径选择算法
2. 拥堵响应策略
3. 社会影响下的移动模式
4. 特殊场景下的空间导航

作者: 城市智能体系统
日期: 2025-01-20
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class MovementMode(Enum):
    """移动模式枚举"""
    NORMAL = "normal"           # 正常移动
    EVACUATION = "evacuation"   # 疏散移动
    SHELTER_SEEKING = "shelter_seeking"  # 寻找避难所
    EMERGENCY = "emergency"     # 紧急移动

class RouteType(Enum):
    """路径类型枚举"""
    FASTEST = "fastest"         # 最快路径
    SAFEST = "safest"           # 最安全路径
    FAMILIAR = "familiar"       # 熟悉路径
    LEAST_CONGESTED = "least_congested"  # 最少拥堵路径

class TransportMode(Enum):
    """交通方式枚举"""
    WALKING = "walking"
    DRIVING = "driving"
    PUBLIC_TRANSPORT = "public_transport"
    CYCLING = "cycling"

@dataclass
class SpatialLocation:
    """空间位置数据结构"""
    aoi_id: int
    x: float
    y: float
    land_use_type: str
    safety_level: float = 0.5
    congestion_level: float = 0.5
    accessibility: float = 1.0

@dataclass
class MovementPath:
    """移动路径数据结构"""
    origin: SpatialLocation
    destination: SpatialLocation
    waypoints: List[SpatialLocation]
    route_type: RouteType
    transport_mode: TransportMode
    estimated_duration: int  # 分钟
    estimated_distance: float  # 公里
    safety_score: float
    congestion_factor: float
    social_influence_factor: float

@dataclass
class MovementConstraints:
    """移动约束条件"""
    max_travel_time: int = 120  # 最大出行时间（分钟）
    max_travel_distance: float = 50.0  # 最大出行距离（公里）
    avoid_high_risk_areas: bool = True
    consider_traffic: bool = True
    follow_social_influence: bool = True
    mobility_impairment: float = 0.0  # 行动不便程度 (0-1)

class SpatialMovementModeling:
    """空间移动行为建模核心类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grid_size = config.get('grid_size', 100)
        self.cell_size = config.get('cell_size', 100)  # 每个网格单元大小（米）
        
        # 地理信息缓存
        self.location_cache = {}
        self.route_cache = {}
        
        # 拥堵模型参数
        self.congestion_params = {
            'base_speed': {'walking': 5, 'driving': 30, 'public_transport': 20, 'cycling': 15},  # km/h
            'congestion_impact': {'walking': 0.2, 'driving': 0.8, 'public_transport': 0.5, 'cycling': 0.3},
            'peak_hours': [7, 8, 9, 17, 18, 19],  # 高峰时段
            'congestion_multiplier': 2.5
        }
        
        # 安全评估参数
        self.safety_params = {
            'land_use_safety': {
                'residential': 0.8,
                'commercial': 0.6,
                'industrial': 0.4,
                'park': 0.9,
                'water': 0.2,
                'road': 0.5
            },
            'weather_impact': {
                'normal': 1.0,
                'rain': 0.8,
                'storm': 0.3,
                'hurricane': 0.1
            }
        }
        
        logger.info("空间移动行为建模模块初始化完成")
    
    def generate_movement_plan(
        self,
        agent_profile: Dict[str, Any],
        current_location: SpatialLocation,
        target_location: SpatialLocation,
        movement_mode: MovementMode,
        time_context: Dict[str, Any],
        constraints: Optional[MovementConstraints] = None
    ) -> Dict[str, Any]:
        """生成移动计划"""
        
        if constraints is None:
            constraints = MovementConstraints()
        
        # 1. 选择交通方式
        transport_mode = self._select_transport_mode(
            agent_profile, current_location, target_location, movement_mode, time_context
        )
        
        # 2. 选择路径类型
        route_type = self._select_route_type(
            agent_profile, movement_mode, time_context
        )
        
        # 3. 计算最优路径
        optimal_path = self._calculate_optimal_path(
            current_location, target_location, transport_mode, route_type, 
            time_context, constraints
        )
        
        # 4. 评估社会影响
        social_influence = self._assess_social_influence(
            agent_profile, current_location, target_location, movement_mode, time_context
        )
        
        # 5. 生成移动计划
        movement_plan = {
            'movement_mode': movement_mode.value,
            'transport_mode': transport_mode.value,
            'route_type': route_type.value,
            'path': optimal_path,
            'social_influence': social_influence,
            'estimated_duration': optimal_path.estimated_duration,
            'estimated_distance': optimal_path.estimated_distance,
            'safety_assessment': {
                'overall_safety': optimal_path.safety_score,
                'risk_factors': self._identify_risk_factors(optimal_path, time_context),
                'safety_recommendations': self._generate_safety_recommendations(optimal_path)
            },
            'congestion_analysis': {
                'congestion_level': optimal_path.congestion_factor,
                'peak_hour_impact': self._assess_peak_hour_impact(time_context),
                'alternative_routes': self._suggest_alternative_routes(optimal_path, time_context)
            },
            'adaptive_strategies': self._generate_adaptive_strategies(
                agent_profile, optimal_path, movement_mode, time_context
            )
        }
        
        logger.info(f"生成移动计划: {movement_mode.value} 从AOI {current_location.aoi_id} 到AOI {target_location.aoi_id}")
        return movement_plan
    
    def _select_transport_mode(
        self,
        agent_profile: Dict[str, Any],
        current_location: SpatialLocation,
        target_location: SpatialLocation,
        movement_mode: MovementMode,
        time_context: Dict[str, Any]
    ) -> TransportMode:
        """选择交通方式"""
        
        # 计算距离
        distance = self._calculate_distance(current_location, target_location)
        
        # 个人能力评估
        has_vehicle = agent_profile.get('has_vehicle', True)
        mobility_score = agent_profile.get('mobility_score', 0.8)
        age = agent_profile.get('age', 35)
        income_level = agent_profile.get('income_level', 'medium')
        
        # 紧急情况优先级
        if movement_mode in [MovementMode.EVACUATION, MovementMode.EMERGENCY]:
            if has_vehicle and distance > 2.0:
                return TransportMode.DRIVING
            elif distance > 5.0:
                return TransportMode.PUBLIC_TRANSPORT
            else:
                return TransportMode.WALKING
        
        # 正常情况下的选择逻辑
        transport_scores = {
            TransportMode.WALKING: self._score_walking(distance, mobility_score, age),
            TransportMode.DRIVING: self._score_driving(distance, has_vehicle, income_level, time_context),
            TransportMode.PUBLIC_TRANSPORT: self._score_public_transport(distance, income_level),
            TransportMode.CYCLING: self._score_cycling(distance, mobility_score, age)
        }
        
        # 选择得分最高的交通方式
        best_mode = max(transport_scores.items(), key=lambda x: x[1])[0]
        return best_mode
    
    def _select_route_type(
        self,
        agent_profile: Dict[str, Any],
        movement_mode: MovementMode,
        time_context: Dict[str, Any]
    ) -> RouteType:
        """选择路径类型"""
        
        # 紧急情况优先安全
        if movement_mode in [MovementMode.EVACUATION, MovementMode.EMERGENCY]:
            return RouteType.SAFEST
        
        # 个人偏好评估
        risk_aversion = agent_profile.get('risk_aversion', 0.5)
        time_pressure = time_context.get('time_pressure', 0.3)
        familiarity = agent_profile.get('area_familiarity', 0.5)
        
        # 路径类型评分
        route_scores = {
            RouteType.FASTEST: (1 - risk_aversion) * 0.6 + time_pressure * 0.4,
            RouteType.SAFEST: risk_aversion * 0.8 + (1 - time_pressure) * 0.2,
            RouteType.FAMILIAR: familiarity * 0.7 + risk_aversion * 0.3,
            RouteType.LEAST_CONGESTED: (1 - time_pressure) * 0.5 + (1 - risk_aversion) * 0.5
        }
        
        best_route = max(route_scores.items(), key=lambda x: x[1])[0]
        return best_route
    
    def _calculate_optimal_path(
        self,
        origin: SpatialLocation,
        destination: SpatialLocation,
        transport_mode: TransportMode,
        route_type: RouteType,
        time_context: Dict[str, Any],
        constraints: MovementConstraints
    ) -> MovementPath:
        """计算最优路径"""
        
        # 简化的路径计算（实际应用中可以集成更复杂的路径规划算法）
        distance = self._calculate_distance(origin, destination)
        
        # 生成中间路径点（简化处理）
        waypoints = self._generate_waypoints(origin, destination, route_type)
        
        # 计算出行时间
        base_speed = self.congestion_params['base_speed'][transport_mode.value]
        congestion_factor = self._calculate_congestion_factor(time_context, transport_mode)
        actual_speed = base_speed / congestion_factor
        estimated_duration = int((distance / actual_speed) * 60)  # 转换为分钟
        
        # 安全评估
        safety_score = self._calculate_path_safety(waypoints, time_context)
        
        # 社会影响因子
        social_influence_factor = self._calculate_social_influence_factor(
            origin, destination, time_context
        )
        
        return MovementPath(
            origin=origin,
            destination=destination,
            waypoints=waypoints,
            route_type=route_type,
            transport_mode=transport_mode,
            estimated_duration=estimated_duration,
            estimated_distance=distance,
            safety_score=safety_score,
            congestion_factor=congestion_factor,
            social_influence_factor=social_influence_factor
        )
    
    def _assess_social_influence(
        self,
        agent_profile: Dict[str, Any],
        current_location: SpatialLocation,
        target_location: SpatialLocation,
        movement_mode: MovementMode,
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估社会影响"""
        
        social_trust = agent_profile.get('social_trust', 0.5)
        community_attachment = agent_profile.get('community_attachment', 0.5)
        neighbor_influence = time_context.get('neighbor_evacuation_rate', 0.0)
        
        # 群体行为影响
        herd_behavior_strength = 0.0
        if movement_mode == MovementMode.EVACUATION:
            herd_behavior_strength = neighbor_influence * social_trust * 0.8
        
        # 权威影响
        authority_influence = 0.0
        if time_context.get('official_evacuation_order', False):
            authority_trust = agent_profile.get('authority_trust', 0.6)
            authority_influence = authority_trust * 0.9
        
        # 社区依恋影响
        community_resistance = 0.0
        if movement_mode == MovementMode.EVACUATION:
            community_resistance = community_attachment * 0.6
        
        return {
            'herd_behavior_strength': herd_behavior_strength,
            'authority_influence': authority_influence,
            'community_resistance': community_resistance,
            'overall_social_pressure': max(herd_behavior_strength, authority_influence) - community_resistance
        }
    
    def _calculate_distance(self, loc1: SpatialLocation, loc2: SpatialLocation) -> float:
        """计算两点间距离（公里）"""
        dx = (loc1.x - loc2.x) * self.cell_size / 1000  # 转换为公里
        dy = (loc1.y - loc2.y) * self.cell_size / 1000
        return math.sqrt(dx * dx + dy * dy)
    
    def _generate_waypoints(
        self, origin: SpatialLocation, destination: SpatialLocation, route_type: RouteType
    ) -> List[SpatialLocation]:
        """生成路径中间点"""
        waypoints = [origin]
        
        # 简化的中间点生成（实际应用中应基于真实路网）
        if route_type == RouteType.FASTEST:
            # 直线路径
            mid_x = (origin.x + destination.x) / 2
            mid_y = (origin.y + destination.y) / 2
            waypoints.append(SpatialLocation(
                aoi_id=-1, x=mid_x, y=mid_y, land_use_type='road'
            ))
        elif route_type == RouteType.SAFEST:
            # 避开高风险区域的路径
            safe_x = origin.x + (destination.x - origin.x) * 0.3
            safe_y = origin.y + (destination.y - origin.y) * 0.7
            waypoints.append(SpatialLocation(
                aoi_id=-1, x=safe_x, y=safe_y, land_use_type='residential', safety_level=0.8
            ))
        
        waypoints.append(destination)
        return waypoints
    
    def _calculate_congestion_factor(
        self, time_context: Dict[str, Any], transport_mode: TransportMode
    ) -> float:
        """计算拥堵因子"""
        current_hour = time_context.get('current_hour', 12)
        base_congestion = time_context.get('traffic_congestion', 0.5)
        
        # 高峰时段影响
        peak_multiplier = 1.0
        if current_hour in self.congestion_params['peak_hours']:
            peak_multiplier = self.congestion_params['congestion_multiplier']
        
        # 交通方式影响
        mode_impact = self.congestion_params['congestion_impact'][transport_mode.value]
        
        congestion_factor = 1.0 + (base_congestion * peak_multiplier * mode_impact)
        return min(congestion_factor, 5.0)  # 限制最大拥堵因子
    
    def _calculate_path_safety(
        self, waypoints: List[SpatialLocation], time_context: Dict[str, Any]
    ) -> float:
        """计算路径安全性"""
        if not waypoints:
            return 0.5
        
        weather_condition = time_context.get('weather_condition', 'normal')
        weather_factor = self.safety_params['weather_impact'][weather_condition]
        
        # 计算路径各段的安全性
        safety_scores = []
        for waypoint in waypoints:
            land_use_safety = self.safety_params['land_use_safety'].get(
                waypoint.land_use_type, 0.5
            )
            location_safety = waypoint.safety_level
            segment_safety = (land_use_safety + location_safety) / 2 * weather_factor
            safety_scores.append(segment_safety)
        
        return np.mean(safety_scores)
    
    def _calculate_social_influence_factor(
        self, origin: SpatialLocation, destination: SpatialLocation, time_context: Dict[str, Any]
    ) -> float:
        """计算社会影响因子"""
        neighbor_evacuation_rate = time_context.get('neighbor_evacuation_rate', 0.0)
        social_media_sentiment = time_context.get('social_media_sentiment', 0.0)
        
        # 社会影响强度
        social_influence = (neighbor_evacuation_rate * 0.6 + 
                          abs(social_media_sentiment) * 0.4)
        
        return min(social_influence, 1.0)
    
    # 交通方式评分函数
    def _score_walking(self, distance: float, mobility_score: float, age: int) -> float:
        if distance > 5.0:
            return 0.1  # 距离太远不适合步行
        
        age_factor = 1.0 if age < 65 else 0.7
        distance_factor = max(0.1, 1.0 - distance / 5.0)
        
        return mobility_score * age_factor * distance_factor * 0.8
    
    def _score_driving(self, distance: float, has_vehicle: bool, income_level: str, time_context: Dict[str, Any]) -> float:
        if not has_vehicle:
            return 0.0
        
        income_factor = {'low': 0.6, 'medium': 0.8, 'high': 1.0}.get(income_level, 0.8)
        congestion = time_context.get('traffic_congestion', 0.5)
        congestion_penalty = congestion * 0.3
        
        base_score = 0.9 if distance > 2.0 else 0.6
        return base_score * income_factor * (1 - congestion_penalty)
    
    def _score_public_transport(self, distance: float, income_level: str) -> float:
        if distance < 1.0:
            return 0.3  # 距离太近不需要公共交通
        
        income_factor = {'low': 1.0, 'medium': 0.8, 'high': 0.6}.get(income_level, 0.8)
        distance_factor = min(1.0, distance / 10.0)
        
        return 0.7 * income_factor * distance_factor
    
    def _score_cycling(self, distance: float, mobility_score: float, age: int) -> float:
        if distance > 10.0 or age > 60:
            return 0.2
        
        distance_factor = max(0.3, 1.0 - distance / 10.0)
        return mobility_score * distance_factor * 0.6
    
    # 辅助分析函数
    def _identify_risk_factors(self, path: MovementPath, time_context: Dict[str, Any]) -> List[str]:
        """识别路径风险因素"""
        risk_factors = []
        
        if path.safety_score < 0.5:
            risk_factors.append("路径安全性较低")
        
        if path.congestion_factor > 2.0:
            risk_factors.append("交通拥堵严重")
        
        weather_condition = time_context.get('weather_condition', 'normal')
        if weather_condition in ['storm', 'hurricane']:
            risk_factors.append(f"恶劣天气条件: {weather_condition}")
        
        if path.estimated_duration > 120:
            risk_factors.append("出行时间过长")
        
        return risk_factors
    
    def _generate_safety_recommendations(self, path: MovementPath) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        if path.safety_score < 0.6:
            recommendations.append("建议选择更安全的路径")
        
        if path.transport_mode == TransportMode.WALKING and path.estimated_distance > 3.0:
            recommendations.append("距离较远，建议考虑其他交通方式")
        
        if path.congestion_factor > 2.5:
            recommendations.append("建议避开高峰时段出行")
        
        return recommendations
    
    def _assess_peak_hour_impact(self, time_context: Dict[str, Any]) -> Dict[str, Any]:
        """评估高峰时段影响"""
        current_hour = time_context.get('current_hour', 12)
        is_peak_hour = current_hour in self.congestion_params['peak_hours']
        
        return {
            'is_peak_hour': is_peak_hour,
            'congestion_multiplier': self.congestion_params['congestion_multiplier'] if is_peak_hour else 1.0,
            'recommended_departure_time': self._suggest_optimal_departure_time(current_hour)
        }
    
    def _suggest_alternative_routes(self, primary_path: MovementPath, time_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """建议替代路径"""
        alternatives = []
        
        # 如果主路径拥堵严重，建议其他路径类型
        if primary_path.congestion_factor > 2.0:
            alternatives.append({
                'route_type': 'least_congested',
                'description': '选择较少拥堵的路径',
                'estimated_time_saving': '15-30分钟'
            })
        
        # 如果安全性较低，建议安全路径
        if primary_path.safety_score < 0.6:
            alternatives.append({
                'route_type': 'safest',
                'description': '选择更安全的路径',
                'safety_improvement': '提升30-50%安全性'
            })
        
        return alternatives
    
    def _generate_adaptive_strategies(
        self,
        agent_profile: Dict[str, Any],
        path: MovementPath,
        movement_mode: MovementMode,
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成自适应策略"""
        
        strategies = {
            'congestion_response': self._generate_congestion_response_strategy(path, time_context),
            'weather_adaptation': self._generate_weather_adaptation_strategy(path, time_context),
            'social_coordination': self._generate_social_coordination_strategy(
                agent_profile, movement_mode, time_context
            ),
            'emergency_fallback': self._generate_emergency_fallback_strategy(path, agent_profile)
        }
        
        return strategies
    
    def _generate_congestion_response_strategy(self, path: MovementPath, time_context: Dict[str, Any]) -> Dict[str, Any]:
        """生成拥堵响应策略"""
        return {
            'threshold': 2.0,  # 拥堵因子阈值
            'actions': [
                '实时监控交通状况',
                '考虑改变出发时间',
                '选择替代路径',
                '切换交通方式'
            ],
            'fallback_routes': ['least_congested', 'familiar']
        }
    
    def _generate_weather_adaptation_strategy(self, path: MovementPath, time_context: Dict[str, Any]) -> Dict[str, Any]:
        """生成天气适应策略"""
        weather_condition = time_context.get('weather_condition', 'normal')
        
        strategies = {
            'normal': {'actions': ['正常出行'], 'precautions': []},
            'rain': {
                'actions': ['携带雨具', '降低行驶速度'],
                'precautions': ['避免低洼路段', '注意路面湿滑']
            },
            'storm': {
                'actions': ['推迟非必要出行', '选择室内路径'],
                'precautions': ['避开树木密集区域', '远离广告牌等高空设施']
            },
            'hurricane': {
                'actions': ['仅紧急疏散时出行', '寻找最近避难所'],
                'precautions': ['避免所有户外活动', '跟随官方疏散指引']
            }
        }
        
        return strategies.get(weather_condition, strategies['normal'])
    
    def _generate_social_coordination_strategy(
        self, agent_profile: Dict[str, Any], movement_mode: MovementMode, time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成社会协调策略"""
        
        social_trust = agent_profile.get('social_trust', 0.5)
        family_size = agent_profile.get('family_size', 1)
        
        coordination_level = 'low'
        if social_trust > 0.7:
            coordination_level = 'high'
        elif social_trust > 0.4:
            coordination_level = 'medium'
        
        strategies = {
            'low': {
                'approach': 'individual_decision',
                'actions': ['独立决策', '最小化社会互动']
            },
            'medium': {
                'approach': 'selective_coordination',
                'actions': ['与邻居交流信息', '参考他人经验']
            },
            'high': {
                'approach': 'collective_coordination',
                'actions': ['组织集体行动', '共享资源和信息', '协调出行时间']
            }
        }
        
        strategy = strategies[coordination_level].copy()
        
        # 家庭协调
        if family_size > 1:
            strategy['family_coordination'] = [
                '确保家庭成员安全',
                '协调家庭出行计划',
                '指定集合地点'
            ]
        
        return strategy
    
    def _generate_emergency_fallback_strategy(self, path: MovementPath, agent_profile: Dict[str, Any]) -> Dict[str, Any]:
        """生成紧急后备策略"""
        
        return {
            'triggers': [
                '路径完全阻断',
                '交通工具故障',
                '天气条件恶化',
                '安全威胁增加'
            ],
            'fallback_actions': [
                '寻找最近避难所',
                '联系紧急服务',
                '改变交通方式',
                '等待救援'
            ],
            'emergency_contacts': {
                'family': agent_profile.get('emergency_contact', 'unknown'),
                'authorities': '911',
                'local_emergency': 'local_emergency_number'
            },
            'survival_priorities': [
                '确保人身安全',
                '寻找临时庇护',
                '保持通信联系',
                '等待专业救援'
            ]
        }
    
    def _suggest_optimal_departure_time(self, current_hour: int) -> int:
        """建议最优出发时间"""
        peak_hours = self.congestion_params['peak_hours']
        
        # 如果当前是高峰时段，建议延后
        if current_hour in peak_hours:
            # 找到下一个非高峰时段
            for hour in range(current_hour + 1, 24):
                if hour not in peak_hours:
                    return hour
            # 如果当天没有合适时间，建议第二天早上
            return 6
        
        return current_hour  # 当前时间合适
    
    def update_real_time_conditions(
        self, location_updates: Dict[int, Dict[str, Any]]
    ) -> None:
        """更新实时条件"""
        
        for aoi_id, conditions in location_updates.items():
            if aoi_id in self.location_cache:
                # 更新拥堵情况
                if 'congestion_level' in conditions:
                    self.location_cache[aoi_id].congestion_level = conditions['congestion_level']
                
                # 更新安全状况
                if 'safety_level' in conditions:
                    self.location_cache[aoi_id].safety_level = conditions['safety_level']
                
                # 更新可达性
                if 'accessibility' in conditions:
                    self.location_cache[aoi_id].accessibility = conditions['accessibility']
        
        # 清空路径缓存以强制重新计算
        self.route_cache.clear()
        
        logger.info(f"更新了 {len(location_updates)} 个位置的实时条件")
    
    def get_movement_statistics(self) -> Dict[str, Any]:
        """获取移动统计信息"""
        
        return {
            'total_locations_cached': len(self.location_cache),
            'total_routes_cached': len(self.route_cache),
            'congestion_params': self.congestion_params,
            'safety_params': self.safety_params,
            'grid_configuration': {
                'grid_size': self.grid_size,
                'cell_size': self.cell_size
            }
        }