"""基于差序格局的洪灾智能体实现
基于差序格局理论的洪水灾害ABM智能体
集成社交网络、互助策略和疏散决策
"""

import asyncio
import random
import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from agentsociety.agent import CitizenAgentBase
from agentsociety.message import Message
from pycityproto.city.person.v2.motion_pb2 import Status

# 导入基类
import sys
import os
sys.path.append('/Users/jamie_502/Desktop/30_agent城市智能体/AS/AgentSociety-main/packages/agentsociety-benchmark')
from agentsociety_benchmark.benchmarks.HurricaneMobility.template_agent import HurricaneMobilityAgent

logger = logging.getLogger(__name__)

class FloodAgentWithGuanxi(HurricaneMobilityAgent):
    """基于差序格局的洪灾智能体
    
    核心特征：
    1. 差序格局社交网络：血缘、地缘、业缘、学缘关系
    2. 互助决策算法：基于关系强度的资源分配
    3. 洪水风险评估：水位、流速、地形因素
    4. 疏散路径优化：考虑社交网络和安全性
    """
    
    def __init__(self, id: int, name: str, toolbox: Any, memory: Any, **kwargs):
        super().__init__(id=id, name=name, toolbox=toolbox, memory=memory, **kwargs)
        
        # === 基础属性 ===
        self.age = random.randint(18, 80)
        self.gender = random.choice(['male', 'female'])
        self.education_level = random.choice(['primary', 'secondary', 'tertiary'])
        self.income_level = random.choice(['low', 'medium', 'high'])
        self.family_size = random.randint(1, 6)
        
        # === 洪水风险相关 ===
        self.flood_risk_threshold = random.uniform(0.3, 0.8)  # 洪水风险阈值
        self.evacuation_threshold = random.uniform(0.4, 0.9)  # 疏散决策阈值
        self.water_depth_tolerance = random.uniform(0.2, 1.0)  # 水深容忍度(米)
        
        # === 差序格局社交网络 ===
        self.guanxi_network = {
            'family': {},      # 血缘关系 (最强)
            'neighbor': {},    # 地缘关系
            'colleague': {},   # 业缘关系
            'classmate': {}    # 学缘关系
        }
        
        # === 个性化特征 ===
        self.altruism_level = random.uniform(0.2, 0.9)  # 利他主义程度
        self.risk_aversion = random.uniform(0.3, 0.9)   # 风险厌恶
        self.social_trust = random.uniform(0.1, 0.8)    # 社会信任度
        self.mobility_capability = random.uniform(0.2, 1.0)  # 移动能力
        
        # === 资源状态 ===
        self.resources = {
            'food': random.uniform(1.0, 10.0),      # 食物(天)
            'water': random.uniform(1.0, 10.0),     # 饮用水(天)
            'medicine': random.uniform(0.0, 5.0),   # 药品
            'transportation': random.choice([True, False])  # 交通工具
        }
        
        # === 状态记录 ===
        self.has_evacuated = False
        self.current_location = None
        self.evacuation_destination = None
        self.flood_exposure_time = 0  # 洪水暴露时间
        
        # === 行为历史 ===
        self.movement_history = []
        self.mutual_aid_history = []  # 互助行为记录
        self.decision_history = []
        self.flood_risk_history = []
        
        # 初始化社交网络
        self._initialize_guanxi_network()
        
        logger.info(f"初始化洪灾智能体 {self.name} - 风险阈值: {self.flood_risk_threshold:.2f}")
    
    def _initialize_guanxi_network(self):
        """初始化差序格局社交网络"""
        # 模拟生成社交关系（实际应用中从数据加载）
        network_sizes = {
            'family': random.randint(2, 8),      # 家庭成员
            'neighbor': random.randint(3, 15),   # 邻居
            'colleague': random.randint(5, 20),  # 同事
            'classmate': random.randint(3, 12)   # 同学
        }
        
        for relation_type, size in network_sizes.items():
            for i in range(size):
                agent_id = f"{relation_type}_{self.id}_{i}"
                # 关系强度计算：血缘 > 地缘 > 业缘 > 学缘
                base_strength = {
                    'family': random.uniform(0.7, 1.0),
                    'neighbor': random.uniform(0.4, 0.8),
                    'colleague': random.uniform(0.3, 0.7),
                    'classmate': random.uniform(0.2, 0.6)
                }[relation_type]
                
                self.guanxi_network[relation_type][agent_id] = {
                    'strength': base_strength,
                    'distance': random.uniform(0.1, 10.0),  # 物理距离(km)
                    'contact_frequency': random.uniform(0.1, 1.0),  # 联系频率
                    'mutual_aid_history': 0  # 历史互助次数
                }
    
    def calculate_guanxi_strength(self, target_agent_id: str) -> float:
        """计算与目标智能体的关系强度
        
        差序格局关系强度 = 基础关系强度 × 距离衰减 × 互动频率 × 互助历史加成
        """
        for relation_type, relations in self.guanxi_network.items():
            if target_agent_id in relations:
                relation = relations[target_agent_id]
                
                # 基础强度
                base_strength = relation['strength']
                
                # 距离衰减（距离越近，关系越强）
                distance_decay = 1.0 / (1.0 + relation['distance'] * 0.1)
                
                # 互动频率加成
                frequency_bonus = 1.0 + relation['contact_frequency'] * 0.2
                
                # 互助历史加成
                aid_bonus = 1.0 + relation['mutual_aid_history'] * 0.1
                
                # 综合关系强度
                total_strength = base_strength * distance_decay * frequency_bonus * aid_bonus
                
                return min(total_strength, 1.0)
        
        return 0.0  # 无关系
    
    async def get_flood_conditions(self) -> Dict[str, Any]:
        """获取洪水条件信息"""
        try:
            assert self.environment is not None
            # 适配现有weather接口，解释为洪水条件
            weather_data = await self.environment.sense("weather")
            
            if weather_data:
                # 将天气数据转换为洪水参数
                precipitation = weather_data.get('precipitation', 0)
                wind_speed = weather_data.get('wind_speed', 0)
                
                # 模拟洪水参数
                water_level = min(precipitation / 50.0, 3.0)  # 水位(米)
                flow_velocity = min(wind_speed / 10.0, 5.0)   # 流速(m/s)
                flood_duration = random.uniform(1.0, 24.0)    # 持续时间(小时)
                
                return {
                    'water_level': water_level,
                    'flow_velocity': flow_velocity,
                    'flood_duration': flood_duration,
                    'precipitation': precipitation,
                    'affected_area_ratio': min(water_level / 2.0, 1.0)
                }
            
            return {
                'water_level': 0.0,
                'flow_velocity': 0.0,
                'flood_duration': 0.0,
                'precipitation': 0.0,
                'affected_area_ratio': 0.0
            }
            
        except Exception as e:
            logger.error(f"获取洪水条件失败: {e}")
            return {'water_level': 0.0, 'flow_velocity': 0.0, 'flood_duration': 0.0}
    
    def assess_flood_risk(self, flood_data: Dict[str, Any]) -> float:
        """评估洪水风险
        
        综合考虑水位、流速、持续时间和个人脆弱性
        """
        try:
            water_level = flood_data.get('water_level', 0.0)
            flow_velocity = flood_data.get('flow_velocity', 0.0)
            duration = flood_data.get('flood_duration', 0.0)
            
            # 基础风险计算
            water_risk = min(water_level / 2.0, 1.0)  # 2米为极危险水位
            velocity_risk = min(flow_velocity / 3.0, 1.0)  # 3m/s为极危险流速
            duration_risk = min(duration / 12.0, 1.0)  # 12小时为长时间暴露
            
            # 综合风险
            base_risk = water_risk * 0.5 + velocity_risk * 0.3 + duration_risk * 0.2
            
            # 个人脆弱性调整
            vulnerability_factor = 1.0
            if self.age > 65 or self.age < 18:
                vulnerability_factor += 0.3  # 老人和儿童更脆弱
            if self.family_size > 4:
                vulnerability_factor += 0.2  # 大家庭疏散困难
            if not self.resources['transportation']:
                vulnerability_factor += 0.4  # 无交通工具
            
            # 个人风险感知
            perceived_risk = base_risk * vulnerability_factor * self.risk_aversion
            
            # 记录风险历史
            self.flood_risk_history.append({
                'timestamp': len(self.flood_risk_history),
                'base_risk': base_risk,
                'perceived_risk': min(perceived_risk, 1.0),
                'flood_conditions': flood_data,
                'vulnerability_factor': vulnerability_factor
            })
            
            return min(perceived_risk, 1.0)
            
        except Exception as e:
            logger.error(f"洪水风险评估失败: {e}")
            return 0.5
    
    def decide_mutual_aid(self, requester_id: str, aid_type: str, amount: float) -> bool:
        """基于差序格局的互助决策
        
        决策因素：
        1. 关系强度（差序格局核心）
        2. 自身资源状况
        3. 利他主义程度
        4. 风险情况
        """
        try:
            # 计算关系强度
            guanxi_strength = self.calculate_guanxi_strength(requester_id)
            
            # 检查自身资源
            available_resource = self.resources.get(aid_type, 0.0)
            resource_ratio = amount / max(available_resource, 0.1)
            
            # 决策概率计算
            # P(aid) = guanxi_strength × altruism × resource_availability × (1 - current_risk)
            current_risk = self.flood_risk_history[-1]['perceived_risk'] if self.flood_risk_history else 0.0
            
            aid_probability = (
                guanxi_strength * 0.4 +           # 关系强度权重最高
                self.altruism_level * 0.3 +       # 利他主义
                (1.0 - resource_ratio) * 0.2 +    # 资源充裕度
                (1.0 - current_risk) * 0.1        # 自身安全状况
            )
            
            # 随机决策
            decision = random.random() < aid_probability
            
            # 记录互助决策
            self.mutual_aid_history.append({
                'timestamp': len(self.mutual_aid_history),
                'requester': requester_id,
                'aid_type': aid_type,
                'amount': amount,
                'guanxi_strength': guanxi_strength,
                'decision': decision,
                'aid_probability': aid_probability
            })
            
            # 如果同意互助，更新资源和关系
            if decision and available_resource >= amount:
                self.resources[aid_type] -= amount
                # 增强关系（互助会加深关系）
                self._update_guanxi_strength(requester_id, 0.1)
            
            return decision
            
        except Exception as e:
            logger.error(f"互助决策失败: {e}")
            return False
    
    def _update_guanxi_strength(self, target_agent_id: str, increment: float):
        """更新关系强度"""
        for relation_type, relations in self.guanxi_network.items():
            if target_agent_id in relations:
                relations[target_agent_id]['mutual_aid_history'] += 1
                relations[target_agent_id]['strength'] = min(
                    relations[target_agent_id]['strength'] + increment, 1.0
                )
                break
    
    def decide_evacuation(self, flood_risk: float, social_info: Dict[str, Any]) -> bool:
        """疏散决策
        
        考虑因素：
        1. 洪水风险评估
        2. 社交网络影响
        3. 家庭责任
        4. 资源状况
        """
        if self.has_evacuated:
            return False
        
        try:
            # 基础风险决策
            risk_decision = flood_risk > self.evacuation_threshold
            
            # 社交网络影响
            network_evacuation_rate = social_info.get('network_evacuation_rate', 0.0)
            social_pressure = network_evacuation_rate * self.social_trust
            social_decision = social_pressure > 0.6
            
            # 家庭责任考虑
            family_factor = 1.0
            if self.family_size > 3:
                family_factor = 0.8  # 大家庭疏散更困难
            
            # 资源状况
            resource_factor = 1.0
            if not self.resources['transportation']:
                resource_factor = 0.6  # 无交通工具降低疏散意愿
            
            # 综合决策
            final_probability = (
                (risk_decision * 0.4 + social_decision * 0.2) * 
                family_factor * resource_factor
            )
            
            final_decision = random.random() < final_probability
            
            # 记录决策
            self.decision_history.append({
                'timestamp': len(self.decision_history),
                'flood_risk': flood_risk,
                'social_pressure': social_pressure,
                'family_factor': family_factor,
                'resource_factor': resource_factor,
                'final_decision': final_decision
            })
            
            return final_decision
            
        except Exception as e:
            logger.error(f"疏散决策失败: {e}")
            return False
    
    async def select_evacuation_destination(self) -> Optional[int]:
        """选择疏散目的地
        
        优先级：
        1. 亲属所在安全区域
        2. 邻居聚集的安全区域
        3. 官方指定避难所
        """
        try:
            assert self.environment is not None
            aoi_ids = self.environment.get_aoi_ids()
            
            if not aoi_ids:
                return None
            
            # 计算每个AOI的吸引力
            aoi_scores = {}
            
            for aoi_id in aoi_ids:
                if aoi_id == self.current_location:
                    continue
                
                score = 0.0
                
                # 社交网络因素（模拟）
                family_members_count = random.randint(0, 3)
                neighbor_count = random.randint(0, 5)
                
                score += family_members_count * 0.5  # 家人权重最高
                score += neighbor_count * 0.2       # 邻居次之
                
                # 距离因素（越近越好）
                distance = random.uniform(1.0, 20.0)  # 模拟距离
                score += max(0, (20.0 - distance) / 20.0) * 0.3
                
                aoi_scores[aoi_id] = score
            
            # 选择得分最高的目的地
            if aoi_scores:
                best_destination = max(aoi_scores.items(), key=lambda x: x[1])[0]
                self.evacuation_destination = best_destination
                return best_destination
            
            return None
            
        except Exception as e:
            logger.error(f"选择疏散目的地失败: {e}")
            return None
    
    async def execute_evacuation(self, destination: int):
        """执行疏散"""
        try:
            await self.go_to_aoi(destination)
            self.has_evacuated = True
            self.current_location = destination
            
            # 记录疏散行为
            self.movement_history.append({
                'timestamp': len(self.movement_history),
                'action': 'flood_evacuation',
                'from': self.current_location,
                'to': destination,
                'reason': 'flood_emergency',
                'resources_carried': self.resources.copy()
            })
            
            logger.info(f"智能体 {self.name} 成功疏散到 AOI {destination}")
            
        except Exception as e:
            logger.error(f"疏散执行失败: {e}")
    
    async def normal_behavior(self):
        """正常情况下的行为"""
        try:
            # 模拟日常活动和社交互动
            if random.random() < 0.3:  # 30%概率进行移动
                assert self.environment is not None
                aoi_ids = self.environment.get_aoi_ids()
                
                if aoi_ids:
                    destination = random.choice(aoi_ids)
                    await self.go_to_aoi(destination)
                    self.current_location = destination
                    
                    self.movement_history.append({
                        'timestamp': len(self.movement_history),
                        'action': 'daily_activity',
                        'to': destination,
                        'reason': 'routine'
                    })
            
            # 模拟资源消耗
            self.resources['food'] = max(0, self.resources['food'] - random.uniform(0.1, 0.3))
            self.resources['water'] = max(0, self.resources['water'] - random.uniform(0.1, 0.3))
            
        except Exception as e:
            logger.error(f"正常行为执行失败: {e}")
    
    async def forward(self):
        """主要行为逻辑"""
        try:
            # 1. 获取洪水条件
            flood_data = await self.get_flood_conditions()
            
            # 2. 评估洪水风险
            flood_risk = self.assess_flood_risk(flood_data)
            
            # 3. 获取社交网络信息
            social_info = {
                'network_evacuation_rate': random.uniform(0.0, 1.0),  # 模拟网络疏散率
                'mutual_aid_requests': random.randint(0, 3)           # 模拟互助请求
            }
            
            # 4. 处理互助请求（模拟）
            for _ in range(social_info['mutual_aid_requests']):
                requester_id = f"agent_{random.randint(1, 100)}"
                aid_type = random.choice(['food', 'water', 'medicine'])
                amount = random.uniform(0.5, 2.0)
                self.decide_mutual_aid(requester_id, aid_type, amount)
            
            # 5. 疏散决策
            should_evacuate = self.decide_evacuation(flood_risk, social_info)
            
            # 6. 执行行为
            if should_evacuate and not self.has_evacuated:
                destination = await self.select_evacuation_destination()
                if destination:
                    await self.execute_evacuation(destination)
                    logger.info(f"智能体 {self.name} 决定疏散 - 洪水风险: {flood_risk:.2f}")
            else:
                await self.normal_behavior()
            
            # 7. 更新洪水暴露时间
            if flood_data['water_level'] > 0.1 and not self.has_evacuated:
                self.flood_exposure_time += 1
            
        except Exception as e:
            logger.error(f"智能体 {self.name} 执行forward失败: {e}")
    
    async def reset(self):
        """重置智能体状态"""
        self.has_evacuated = False
        self.current_location = None
        self.evacuation_destination = None
        self.flood_exposure_time = 0
        
        # 重置资源
        self.resources = {
            'food': random.uniform(1.0, 10.0),
            'water': random.uniform(1.0, 10.0),
            'medicine': random.uniform(0.0, 5.0),
            'transportation': random.choice([True, False])
        }
        
        # 清空历史记录
        self.movement_history.clear()
        self.mutual_aid_history.clear()
        self.decision_history.clear()
        self.flood_risk_history.clear()
        
        logger.info(f"洪灾智能体 {self.name} 状态已重置")
    
    def get_behavior_summary(self) -> Dict[str, Any]:
        """获取行为总结"""
        return {
            'agent_id': self.id,
            'agent_name': self.name,
            'demographics': {
                'age': self.age,
                'gender': self.gender,
                'education': self.education_level,
                'income': self.income_level,
                'family_size': self.family_size
            },
            'evacuation_status': {
                'has_evacuated': self.has_evacuated,
                'evacuation_destination': self.evacuation_destination,
                'flood_exposure_time': self.flood_exposure_time
            },
            'social_network': {
                'total_connections': sum(len(relations) for relations in self.guanxi_network.values()),
                'family_connections': len(self.guanxi_network['family']),
                'neighbor_connections': len(self.guanxi_network['neighbor'])
            },
            'mutual_aid': {
                'aid_provided_count': len([aid for aid in self.mutual_aid_history if aid['decision']]),
                'aid_requested_count': len(self.mutual_aid_history),
                'total_aid_value': sum(aid['amount'] for aid in self.mutual_aid_history if aid['decision'])
            },
            'resources': self.resources.copy(),
            'behavior_counts': {
                'movements': len(self.movement_history),
                'decisions': len(self.decision_history),
                'risk_assessments': len(self.flood_risk_history)
            },
            'personality': {
                'altruism_level': self.altruism_level,
                'risk_aversion': self.risk_aversion,
                'social_trust': self.social_trust,
                'mobility_capability': self.mobility_capability
            }
        }
    
    def get_guanxi_network_summary(self) -> Dict[str, Any]:
        """获取差序格局网络总结"""
        network_summary = {}
        
        for relation_type, relations in self.guanxi_network.items():
            if relations:
                strengths = [rel['strength'] for rel in relations.values()]
                distances = [rel['distance'] for rel in relations.values()]
                aid_counts = [rel['mutual_aid_history'] for rel in relations.values()]
                
                network_summary[relation_type] = {
                    'count': len(relations),
                    'avg_strength': sum(strengths) / len(strengths),
                    'max_strength': max(strengths),
                    'avg_distance': sum(distances) / len(distances),
                    'total_aid_history': sum(aid_counts)
                }
            else:
                network_summary[relation_type] = {
                    'count': 0,
                    'avg_strength': 0.0,
                    'max_strength': 0.0,
                    'avg_distance': 0.0,
                    'total_aid_history': 0
                }
        
        return network_summary