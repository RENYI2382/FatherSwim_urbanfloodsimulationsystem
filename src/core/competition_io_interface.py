"""比赛输入输出接口模块

实现决赛参赛手册要求的标准化输入输出接口，包括：
输入：事件时序、风险热度图、避难点/关键设施、道路通行能力、个体风险感知、社交影响、预算与资源约束
输出：撤离路径与到达时刻、拥堵时长、服务恢复曲线、合规率与脆弱群体指标

作者: 城市智能体项目组
日期: 2024-12-30
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    FLOOD_WARNING = "flood_warning"  # 洪水预警
    FLOOD_ONSET = "flood_onset"  # 洪水开始
    FLOOD_PEAK = "flood_peak"  # 洪水峰值
    FLOOD_RECESSION = "flood_recession"  # 洪水退去
    EVACUATION_ORDER = "evacuation_order"  # 疏散命令
    SHELTER_OPEN = "shelter_open"  # 避难所开放
    ROAD_CLOSURE = "road_closure"  # 道路封闭
    UTILITY_FAILURE = "utility_failure"  # 公用设施故障
    RESCUE_OPERATION = "rescue_operation"  # 救援行动
    SERVICE_RESTORATION = "service_restoration"  # 服务恢复


class RiskLevel(Enum):
    """风险等级枚举"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME = 6


class FacilityType(Enum):
    """设施类型枚举"""
    SHELTER = "shelter"  # 避难所
    HOSPITAL = "hospital"  # 医院
    SCHOOL = "school"  # 学校
    FIRE_STATION = "fire_station"  # 消防站
    POLICE_STATION = "police_station"  # 警察局
    COMMUNITY_CENTER = "community_center"  # 社区中心
    SUPPLY_DEPOT = "supply_depot"  # 物资储备点
    TRANSPORT_HUB = "transport_hub"  # 交通枢纽


class RoadStatus(Enum):
    """道路状态枚举"""
    NORMAL = "normal"  # 正常通行
    CONGESTED = "congested"  # 拥堵
    RESTRICTED = "restricted"  # 限制通行
    CLOSED = "closed"  # 封闭
    FLOODED = "flooded"  # 被淹
    DAMAGED = "damaged"  # 损坏


@dataclass
class DisasterEvent:
    """灾害事件数据结构"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    location: Tuple[float, float]  # (经度, 纬度)
    severity: float  # 0-1，严重程度
    duration: Optional[timedelta] = None
    affected_area: Optional[List[Tuple[float, float]]] = None  # 影响区域多边形
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskHeatMap:
    """风险热度图数据结构"""
    timestamp: datetime
    grid_size: Tuple[int, int]  # (行数, 列数)
    bounds: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    risk_matrix: np.ndarray  # 风险矩阵，值为0-1
    risk_categories: Dict[str, np.ndarray] = field(default_factory=dict)  # 分类风险
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_risk_at_location(self, lon: float, lat: float) -> float:
        """获取指定位置的风险值"""
        min_lon, min_lat, max_lon, max_lat = self.bounds
        
        # 将经纬度转换为网格坐标
        col = int((lon - min_lon) / (max_lon - min_lon) * self.grid_size[1])
        row = int((lat - min_lat) / (max_lat - min_lat) * self.grid_size[0])
        
        # 边界检查
        col = max(0, min(col, self.grid_size[1] - 1))
        row = max(0, min(row, self.grid_size[0] - 1))
        
        return float(self.risk_matrix[row, col])


@dataclass
class CriticalFacility:
    """关键设施数据结构"""
    facility_id: str
    facility_type: FacilityType
    name: str
    location: Tuple[float, float]  # (经度, 纬度)
    capacity: int  # 容量
    current_occupancy: int = 0  # 当前占用
    operational_status: bool = True  # 运营状态
    accessibility: float = 1.0  # 可达性，0-1
    services: List[str] = field(default_factory=list)  # 提供的服务
    contact_info: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def available_capacity(self) -> int:
        """可用容量"""
        return max(0, self.capacity - self.current_occupancy)
    
    @property
    def occupancy_rate(self) -> float:
        """占用率"""
        return self.current_occupancy / max(1, self.capacity)


@dataclass
class RoadSegment:
    """道路段数据结构"""
    segment_id: str
    start_location: Tuple[float, float]
    end_location: Tuple[float, float]
    road_type: str  # 道路类型：高速公路、主干道、次干道、支路
    length: float  # 长度（公里）
    lanes: int  # 车道数
    speed_limit: float  # 限速（公里/小时）
    current_speed: float  # 当前通行速度
    capacity: int  # 通行能力（车辆/小时）
    current_flow: int  # 当前流量
    status: RoadStatus = RoadStatus.NORMAL
    flood_risk: float = 0.0  # 洪水风险，0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def congestion_level(self) -> float:
        """拥堵程度"""
        return self.current_flow / max(1, self.capacity)
    
    @property
    def travel_time(self) -> float:
        """通行时间（小时）"""
        if self.current_speed <= 0:
            return float('inf')
        return self.length / self.current_speed


@dataclass
class IndividualRiskProfile:
    """个体风险档案"""
    agent_id: str
    risk_perception: Dict[str, float]  # 各类风险感知
    vulnerability_factors: Dict[str, float]  # 脆弱性因子
    coping_capacity: Dict[str, float]  # 应对能力
    social_influence_susceptibility: float  # 社会影响敏感性
    trust_in_authority: float  # 对权威的信任
    past_experience: Dict[str, Any]  # 过往经历
    current_location: Tuple[float, float]  # 当前位置
    mobility_constraints: Dict[str, float]  # 移动约束


@dataclass
class SocialInfluenceNetwork:
    """社会影响网络"""
    network_id: str
    influence_matrix: np.ndarray  # 影响矩阵
    agent_ids: List[str]  # 智能体ID列表
    influence_decay: float = 0.1  # 影响衰减率
    update_frequency: float = 1.0  # 更新频率（小时）
    
    def get_influence_score(self, influencer_id: str, influenced_id: str) -> float:
        """获取影响分数"""
        try:
            i_idx = self.agent_ids.index(influencer_id)
            j_idx = self.agent_ids.index(influenced_id)
            return float(self.influence_matrix[i_idx, j_idx])
        except ValueError:
            return 0.0


@dataclass
class ResourceConstraints:
    """资源约束"""
    agent_id: str
    financial_budget: float  # 财务预算
    time_budget: float  # 时间预算（小时）
    transport_options: Dict[str, bool]  # 交通选项可用性
    physical_capacity: float  # 体力容量
    information_access: Dict[str, float]  # 信息获取能力
    social_support: Dict[str, float]  # 社会支持
    emergency_supplies: Dict[str, float]  # 应急物资


# 输出数据结构

@dataclass
class EvacuationPath:
    """疏散路径"""
    agent_id: str
    path_id: str
    waypoints: List[Tuple[float, float]]  # 路径点
    road_segments: List[str]  # 经过的道路段ID
    total_distance: float  # 总距离（公里）
    estimated_travel_time: float  # 预计通行时间（小时）
    risk_exposure: float  # 风险暴露度
    departure_time: datetime  # 出发时间
    arrival_time: datetime  # 到达时间
    destination_facility_id: str  # 目的地设施ID
    path_quality: float  # 路径质量评分
    alternative_paths: List[str] = field(default_factory=list)  # 备选路径


@dataclass
class CongestionMetrics:
    """拥堵指标"""
    timestamp: datetime
    road_segment_id: str
    congestion_level: float  # 拥堵程度，0-1
    average_speed: float  # 平均速度
    queue_length: float  # 排队长度（公里）
    delay_time: float  # 延误时间（小时）
    throughput: int  # 通过量（车辆/小时）
    bottleneck_severity: float  # 瓶颈严重程度


@dataclass
class ServiceRecoveryPoint:
    """服务恢复点"""
    timestamp: datetime
    service_type: str  # 服务类型
    facility_id: str
    recovery_percentage: float  # 恢复百分比，0-1
    capacity_restored: int  # 恢复的容量
    estimated_full_recovery: Optional[datetime] = None  # 预计完全恢复时间


@dataclass
class ComplianceMetrics:
    """合规率指标"""
    timestamp: datetime
    total_population: int
    evacuated_population: int
    compliance_rate: float  # 总体合规率
    vulnerable_group_compliance: Dict[str, float]  # 脆弱群体合规率
    non_compliance_reasons: Dict[str, int]  # 不合规原因统计
    spatial_compliance_distribution: Dict[str, float]  # 空间合规分布


@dataclass
class VulnerableGroupIndicators:
    """脆弱群体指标"""
    timestamp: datetime
    group_definitions: Dict[str, Dict[str, Any]]  # 群体定义
    group_populations: Dict[str, int]  # 各群体人数
    evacuation_rates: Dict[str, float]  # 各群体疏散率
    assistance_needs: Dict[str, Dict[str, float]]  # 援助需求
    outcome_indicators: Dict[str, Dict[str, float]]  # 结果指标
    intervention_effectiveness: Dict[str, float]  # 干预措施有效性


class CompetitionIOInterface:
    """比赛输入输出接口"""
    
    def __init__(self, data_dir: str = "competition_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 输入数据存储
        self.disaster_events: List[DisasterEvent] = []
        self.risk_heatmaps: Dict[datetime, RiskHeatMap] = {}
        self.critical_facilities: Dict[str, CriticalFacility] = {}
        self.road_network: Dict[str, RoadSegment] = {}
        self.individual_risk_profiles: Dict[str, IndividualRiskProfile] = {}
        self.social_influence_networks: Dict[str, SocialInfluenceNetwork] = {}
        self.resource_constraints: Dict[str, ResourceConstraints] = {}
        
        # 输出数据存储
        self.evacuation_paths: Dict[str, List[EvacuationPath]] = {}
        self.congestion_metrics: List[CongestionMetrics] = []
        self.service_recovery_curves: Dict[str, List[ServiceRecoveryPoint]] = {}
        self.compliance_metrics: List[ComplianceMetrics] = []
        self.vulnerable_group_indicators: List[VulnerableGroupIndicators] = []
        
        # 网络图
        self.road_graph: Optional[nx.Graph] = None
        
        logger.info(f"Competition IO Interface initialized with data directory: {self.data_dir}")
    
    # 输入数据加载方法
    
    def load_disaster_events(self, filepath: str) -> List[DisasterEvent]:
        """加载灾害事件时序数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            events = []
            for event_data in data:
                event = DisasterEvent(
                    event_id=event_data['event_id'],
                    event_type=EventType(event_data['event_type']),
                    timestamp=datetime.fromisoformat(event_data['timestamp']),
                    location=tuple(event_data['location']),
                    severity=event_data['severity'],
                    duration=timedelta(hours=event_data.get('duration_hours', 0)) if event_data.get('duration_hours') else None,
                    affected_area=event_data.get('affected_area'),
                    description=event_data.get('description', ''),
                    metadata=event_data.get('metadata', {})
                )
                events.append(event)
            
            self.disaster_events = events
            logger.info(f"Loaded {len(events)} disaster events")
            return events
            
        except Exception as e:
            logger.error(f"Error loading disaster events: {e}")
            return []
    
    def load_risk_heatmap(self, filepath: str, timestamp: datetime) -> Optional[RiskHeatMap]:
        """加载风险热度图"""
        try:
            data = np.load(filepath, allow_pickle=True).item()
            
            heatmap = RiskHeatMap(
                timestamp=timestamp,
                grid_size=tuple(data['grid_size']),
                bounds=tuple(data['bounds']),
                risk_matrix=data['risk_matrix'],
                risk_categories=data.get('risk_categories', {}),
                metadata=data.get('metadata', {})
            )
            
            self.risk_heatmaps[timestamp] = heatmap
            logger.info(f"Loaded risk heatmap for {timestamp}")
            return heatmap
            
        except Exception as e:
            logger.error(f"Error loading risk heatmap: {e}")
            return None
    
    def load_critical_facilities(self, filepath: str) -> Dict[str, CriticalFacility]:
        """加载关键设施数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            facilities = {}
            for facility_data in data:
                facility = CriticalFacility(
                    facility_id=facility_data['facility_id'],
                    facility_type=FacilityType(facility_data['facility_type']),
                    name=facility_data['name'],
                    location=tuple(facility_data['location']),
                    capacity=facility_data['capacity'],
                    current_occupancy=facility_data.get('current_occupancy', 0),
                    operational_status=facility_data.get('operational_status', True),
                    accessibility=facility_data.get('accessibility', 1.0),
                    services=facility_data.get('services', []),
                    contact_info=facility_data.get('contact_info', {}),
                    metadata=facility_data.get('metadata', {})
                )
                facilities[facility.facility_id] = facility
            
            self.critical_facilities = facilities
            logger.info(f"Loaded {len(facilities)} critical facilities")
            return facilities
            
        except Exception as e:
            logger.error(f"Error loading critical facilities: {e}")
            return {}
    
    def load_road_network(self, filepath: str) -> Dict[str, RoadSegment]:
        """加载道路网络数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            road_segments = {}
            for segment_data in data:
                segment = RoadSegment(
                    segment_id=segment_data['segment_id'],
                    start_location=tuple(segment_data['start_location']),
                    end_location=tuple(segment_data['end_location']),
                    road_type=segment_data['road_type'],
                    length=segment_data['length'],
                    lanes=segment_data['lanes'],
                    speed_limit=segment_data['speed_limit'],
                    current_speed=segment_data.get('current_speed', segment_data['speed_limit']),
                    capacity=segment_data['capacity'],
                    current_flow=segment_data.get('current_flow', 0),
                    status=RoadStatus(segment_data.get('status', 'normal')),
                    flood_risk=segment_data.get('flood_risk', 0.0),
                    metadata=segment_data.get('metadata', {})
                )
                road_segments[segment.segment_id] = segment
            
            self.road_network = road_segments
            self._build_road_graph()
            logger.info(f"Loaded {len(road_segments)} road segments")
            return road_segments
            
        except Exception as e:
            logger.error(f"Error loading road network: {e}")
            return {}
    
    def _build_road_graph(self):
        """构建道路网络图"""
        self.road_graph = nx.Graph()
        
        for segment_id, segment in self.road_network.items():
            start_node = f"{segment.start_location[0]:.6f},{segment.start_location[1]:.6f}"
            end_node = f"{segment.end_location[0]:.6f},{segment.end_location[1]:.6f}"
            
            # 添加边，权重为通行时间
            weight = segment.travel_time if segment.travel_time != float('inf') else 999999
            
            self.road_graph.add_edge(
                start_node, end_node,
                segment_id=segment_id,
                weight=weight,
                length=segment.length,
                capacity=segment.capacity,
                status=segment.status.value
            )
    
    def load_individual_risk_profiles(self, filepath: str) -> Dict[str, IndividualRiskProfile]:
        """加载个体风险档案"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            profiles = {}
            for profile_data in data:
                profile = IndividualRiskProfile(
                    agent_id=profile_data['agent_id'],
                    risk_perception=profile_data['risk_perception'],
                    vulnerability_factors=profile_data['vulnerability_factors'],
                    coping_capacity=profile_data['coping_capacity'],
                    social_influence_susceptibility=profile_data['social_influence_susceptibility'],
                    trust_in_authority=profile_data['trust_in_authority'],
                    past_experience=profile_data.get('past_experience', {}),
                    current_location=tuple(profile_data['current_location']),
                    mobility_constraints=profile_data.get('mobility_constraints', {})
                )
                profiles[profile.agent_id] = profile
            
            self.individual_risk_profiles = profiles
            logger.info(f"Loaded {len(profiles)} individual risk profiles")
            return profiles
            
        except Exception as e:
            logger.error(f"Error loading individual risk profiles: {e}")
            return {}
    
    def load_social_influence_network(self, filepath: str) -> Optional[SocialInfluenceNetwork]:
        """加载社会影响网络"""
        try:
            data = np.load(filepath, allow_pickle=True).item()
            
            network = SocialInfluenceNetwork(
                network_id=data['network_id'],
                influence_matrix=data['influence_matrix'],
                agent_ids=data['agent_ids'],
                influence_decay=data.get('influence_decay', 0.1),
                update_frequency=data.get('update_frequency', 1.0)
            )
            
            self.social_influence_networks[network.network_id] = network
            logger.info(f"Loaded social influence network: {network.network_id}")
            return network
            
        except Exception as e:
            logger.error(f"Error loading social influence network: {e}")
            return None
    
    def load_resource_constraints(self, filepath: str) -> Dict[str, ResourceConstraints]:
        """加载资源约束数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            constraints = {}
            for constraint_data in data:
                constraint = ResourceConstraints(
                    agent_id=constraint_data['agent_id'],
                    financial_budget=constraint_data['financial_budget'],
                    time_budget=constraint_data['time_budget'],
                    transport_options=constraint_data['transport_options'],
                    physical_capacity=constraint_data['physical_capacity'],
                    information_access=constraint_data.get('information_access', {}),
                    social_support=constraint_data.get('social_support', {}),
                    emergency_supplies=constraint_data.get('emergency_supplies', {})
                )
                constraints[constraint.agent_id] = constraint
            
            self.resource_constraints = constraints
            logger.info(f"Loaded {len(constraints)} resource constraint profiles")
            return constraints
            
        except Exception as e:
            logger.error(f"Error loading resource constraints: {e}")
            return {}
    
    # 输出数据生成方法
    
    def generate_evacuation_path(self, agent_id: str, start_location: Tuple[float, float],
                               destination_facility_id: str, departure_time: datetime) -> Optional[EvacuationPath]:
        """生成疏散路径"""
        try:
            if not self.road_graph:
                logger.error("Road graph not available")
                return None
            
            # 获取目标设施
            if destination_facility_id not in self.critical_facilities:
                logger.error(f"Destination facility {destination_facility_id} not found")
                return None
            
            destination = self.critical_facilities[destination_facility_id]
            
            # 找到最近的道路节点
            start_node = self._find_nearest_road_node(start_location)
            end_node = self._find_nearest_road_node(destination.location)
            
            if not start_node or not end_node:
                logger.error("Could not find road nodes for start or end location")
                return None
            
            # 使用Dijkstra算法找最短路径
            try:
                path_nodes = nx.shortest_path(self.road_graph, start_node, end_node, weight='weight')
                path_length = nx.shortest_path_length(self.road_graph, start_node, end_node, weight='weight')
            except nx.NetworkXNoPath:
                logger.error(f"No path found from {start_node} to {end_node}")
                return None
            
            # 构建路径信息
            waypoints = []
            road_segments = []
            total_distance = 0.0
            risk_exposure = 0.0
            
            for i in range(len(path_nodes) - 1):
                current_node = path_nodes[i]
                next_node = path_nodes[i + 1]
                
                # 获取边信息
                edge_data = self.road_graph[current_node][next_node]
                segment_id = edge_data['segment_id']
                segment = self.road_network[segment_id]
                
                waypoints.append(segment.start_location)
                road_segments.append(segment_id)
                total_distance += segment.length
                
                # 计算风险暴露
                risk_exposure += segment.flood_risk * segment.length
            
            # 添加终点
            waypoints.append(destination.location)
            
            # 计算到达时间
            estimated_travel_time = path_length  # 已经是时间权重
            arrival_time = departure_time + timedelta(hours=estimated_travel_time)
            
            # 计算路径质量（基于风险、距离、时间等因素）
            path_quality = self._calculate_path_quality(total_distance, estimated_travel_time, risk_exposure)
            
            evacuation_path = EvacuationPath(
                agent_id=agent_id,
                path_id=f"{agent_id}_{departure_time.isoformat()}",
                waypoints=waypoints,
                road_segments=road_segments,
                total_distance=total_distance,
                estimated_travel_time=estimated_travel_time,
                risk_exposure=risk_exposure / max(total_distance, 1),  # 标准化风险暴露
                departure_time=departure_time,
                arrival_time=arrival_time,
                destination_facility_id=destination_facility_id,
                path_quality=path_quality
            )
            
            # 存储路径
            if agent_id not in self.evacuation_paths:
                self.evacuation_paths[agent_id] = []
            self.evacuation_paths[agent_id].append(evacuation_path)
            
            return evacuation_path
            
        except Exception as e:
            logger.error(f"Error generating evacuation path: {e}")
            return None
    
    def _find_nearest_road_node(self, location: Tuple[float, float]) -> Optional[str]:
        """找到最近的道路节点"""
        if not self.road_graph:
            return None
        
        min_distance = float('inf')
        nearest_node = None
        
        for node in self.road_graph.nodes():
            node_coords = [float(x) for x in node.split(',')]
            distance = ((location[0] - node_coords[0]) ** 2 + (location[1] - node_coords[1]) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        
        return nearest_node
    
    def _calculate_path_quality(self, distance: float, travel_time: float, risk_exposure: float) -> float:
        """计算路径质量评分"""
        # 标准化各项指标（假设合理的最大值）
        max_distance = 50.0  # 最大距离50公里
        max_time = 3.0  # 最大时间3小时
        max_risk = 1.0  # 最大风险1.0
        
        distance_score = 1.0 - min(distance / max_distance, 1.0)
        time_score = 1.0 - min(travel_time / max_time, 1.0)
        risk_score = 1.0 - min(risk_exposure / max_risk, 1.0)
        
        # 加权平均（可调整权重）
        quality = 0.3 * distance_score + 0.3 * time_score + 0.4 * risk_score
        return max(0.0, min(1.0, quality))
    
    def update_congestion_metrics(self, timestamp: datetime):
        """更新拥堵指标"""
        for segment_id, segment in self.road_network.items():
            congestion = CongestionMetrics(
                timestamp=timestamp,
                road_segment_id=segment_id,
                congestion_level=segment.congestion_level,
                average_speed=segment.current_speed,
                queue_length=max(0, segment.length * (segment.congestion_level - 0.5) * 2),
                delay_time=max(0, segment.length / segment.current_speed - segment.length / segment.speed_limit),
                throughput=segment.current_flow,
                bottleneck_severity=min(1.0, segment.congestion_level * 1.5)
            )
            self.congestion_metrics.append(congestion)
    
    def update_service_recovery_curve(self, facility_id: str, timestamp: datetime, 
                                    recovery_percentage: float):
        """更新服务恢复曲线"""
        if facility_id not in self.critical_facilities:
            return
        
        facility = self.critical_facilities[facility_id]
        
        recovery_point = ServiceRecoveryPoint(
            timestamp=timestamp,
            service_type=facility.facility_type.value,
            facility_id=facility_id,
            recovery_percentage=recovery_percentage,
            capacity_restored=int(facility.capacity * recovery_percentage)
        )
        
        if facility_id not in self.service_recovery_curves:
            self.service_recovery_curves[facility_id] = []
        
        self.service_recovery_curves[facility_id].append(recovery_point)
    
    def calculate_compliance_metrics(self, timestamp: datetime, 
                                   agent_evacuation_status: Dict[str, bool]) -> ComplianceMetrics:
        """计算合规率指标"""
        total_population = len(agent_evacuation_status)
        evacuated_population = sum(agent_evacuation_status.values())
        compliance_rate = evacuated_population / max(total_population, 1)
        
        # 计算脆弱群体合规率（需要根据智能体属性分类）
        vulnerable_group_compliance = self._calculate_vulnerable_group_compliance(agent_evacuation_status)
        
        # 统计不合规原因（简化版本）
        non_compliance_reasons = {
            "资源不足": int(total_population * 0.1),
            "信息不足": int(total_population * 0.05),
            "社会压力": int(total_population * 0.03),
            "其他": total_population - evacuated_population - int(total_population * 0.18)
        }
        
        compliance = ComplianceMetrics(
            timestamp=timestamp,
            total_population=total_population,
            evacuated_population=evacuated_population,
            compliance_rate=compliance_rate,
            vulnerable_group_compliance=vulnerable_group_compliance,
            non_compliance_reasons=non_compliance_reasons,
            spatial_compliance_distribution={}  # 需要根据空间位置计算
        )
        
        self.compliance_metrics.append(compliance)
        return compliance
    
    def _calculate_vulnerable_group_compliance(self, agent_evacuation_status: Dict[str, bool]) -> Dict[str, float]:
        """计算脆弱群体合规率"""
        # 简化版本，实际需要根据智能体属性分类
        return {
            "老年人": 0.75,
            "儿童": 0.85,
            "残疾人": 0.65,
            "低收入群体": 0.70,
            "外来人口": 0.60
        }
    
    def generate_vulnerable_group_indicators(self, timestamp: datetime) -> VulnerableGroupIndicators:
        """生成脆弱群体指标"""
        # 定义脆弱群体
        group_definitions = {
            "elderly": {"age_min": 65, "description": "65岁及以上老年人"},
            "children": {"age_max": 12, "description": "12岁及以下儿童"},
            "disabled": {"mobility_impaired": True, "description": "行动不便人群"},
            "low_income": {"income_percentile": 20, "description": "收入最低20%人群"},
            "migrants": {"local_residence_years": 2, "description": "本地居住不足2年人群"}
        }
        
        # 模拟群体人数和指标
        indicators = VulnerableGroupIndicators(
            timestamp=timestamp,
            group_definitions=group_definitions,
            group_populations={
                "elderly": 1500,
                "children": 800,
                "disabled": 300,
                "low_income": 2000,
                "migrants": 1200
            },
            evacuation_rates={
                "elderly": 0.75,
                "children": 0.85,
                "disabled": 0.65,
                "low_income": 0.70,
                "migrants": 0.60
            },
            assistance_needs={
                "elderly": {"transportation": 0.8, "medical": 0.6, "information": 0.7},
                "children": {"supervision": 0.9, "transportation": 0.7, "comfort": 0.8},
                "disabled": {"accessibility": 0.9, "assistance": 0.8, "equipment": 0.6},
                "low_income": {"financial": 0.8, "transportation": 0.6, "shelter": 0.7},
                "migrants": {"information": 0.9, "language": 0.5, "social_support": 0.7}
            },
            outcome_indicators={
                "elderly": {"injury_rate": 0.05, "mortality_rate": 0.02, "recovery_time": 30},
                "children": {"injury_rate": 0.02, "trauma_rate": 0.15, "recovery_time": 14},
                "disabled": {"injury_rate": 0.08, "equipment_loss": 0.3, "recovery_time": 45},
                "low_income": {"property_loss": 0.6, "employment_impact": 0.4, "recovery_time": 60},
                "migrants": {"displacement_rate": 0.4, "social_isolation": 0.3, "recovery_time": 90}
            },
            intervention_effectiveness={
                "targeted_assistance": 0.8,
                "community_support": 0.7,
                "government_programs": 0.6,
                "ngo_services": 0.75
            }
        )
        
        self.vulnerable_group_indicators.append(indicators)
        return indicators
    
    # 数据导出方法
    
    def export_evacuation_paths(self, filepath: str):
        """导出疏散路径数据"""
        export_data = []
        for agent_id, paths in self.evacuation_paths.items():
            for path in paths:
                export_data.append(asdict(path))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Exported {len(export_data)} evacuation paths to {filepath}")
    
    def export_congestion_metrics(self, filepath: str):
        """导出拥堵指标数据"""
        export_data = [asdict(metric) for metric in self.congestion_metrics]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Exported {len(export_data)} congestion metrics to {filepath}")
    
    def export_service_recovery_curves(self, filepath: str):
        """导出服务恢复曲线数据"""
        export_data = {}
        for facility_id, recovery_points in self.service_recovery_curves.items():
            export_data[facility_id] = [asdict(point) for point in recovery_points]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Exported service recovery curves for {len(export_data)} facilities to {filepath}")
    
    def export_compliance_metrics(self, filepath: str):
        """导出合规率指标数据"""
        export_data = [asdict(metric) for metric in self.compliance_metrics]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Exported {len(export_data)} compliance metrics to {filepath}")
    
    def export_vulnerable_group_indicators(self, filepath: str):
        """导出脆弱群体指标数据"""
        export_data = [asdict(indicator) for indicator in self.vulnerable_group_indicators]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Exported {len(export_data)} vulnerable group indicators to {filepath}")
    
    def export_all_outputs(self, output_dir: str):
        """导出所有输出数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        self.export_evacuation_paths(output_path / "evacuation_paths.json")
        self.export_congestion_metrics(output_path / "congestion_metrics.json")
        self.export_service_recovery_curves(output_path / "service_recovery_curves.json")
        self.export_compliance_metrics(output_path / "compliance_metrics.json")
        self.export_vulnerable_group_indicators(output_path / "vulnerable_group_indicators.json")
        
        logger.info(f"All output data exported to {output_dir}")


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== 比赛输入输出接口测试 ===")
    
    # 创建接口实例
    io_interface = CompetitionIOInterface()
    
    # 创建测试数据
    print("\n=== 创建测试数据 ===")
    
    # 测试灾害事件
    test_events = [
        {
            "event_id": "flood_001",
            "event_type": "flood_warning",
            "timestamp": "2024-07-15T08:00:00",
            "location": [113.2644, 23.1291],
            "severity": 0.7,
            "description": "广州市区洪水预警"
        }
    ]
    
    with open("test_events.json", "w", encoding="utf-8") as f:
        json.dump(test_events, f, ensure_ascii=False, indent=2)
    
    events = io_interface.load_disaster_events("test_events.json")
    print(f"加载了 {len(events)} 个灾害事件")
    
    # 测试关键设施
    test_facilities = [
        {
            "facility_id": "shelter_001",
            "facility_type": "shelter",
            "name": "广州体育馆避难所",
            "location": [113.2744, 23.1391],
            "capacity": 5000,
            "services": ["住宿", "餐饮", "医疗"]
        }
    ]
    
    with open("test_facilities.json", "w", encoding="utf-8") as f:
        json.dump(test_facilities, f, ensure_ascii=False, indent=2)
    
    facilities = io_interface.load_critical_facilities("test_facilities.json")
    print(f"加载了 {len(facilities)} 个关键设施")
    
    # 测试道路网络
    test_roads = [
        {
            "segment_id": "road_001",
            "start_location": [113.2644, 23.1291],
            "end_location": [113.2744, 23.1391],
            "road_type": "主干道",
            "length": 2.5,
            "lanes": 4,
            "speed_limit": 60,
            "capacity": 2000
        }
    ]
    
    with open("test_roads.json", "w", encoding="utf-8") as f:
        json.dump(test_roads, f, ensure_ascii=False, indent=2)
    
    roads = io_interface.load_road_network("test_roads.json")
    print(f"加载了 {len(roads)} 个道路段")
    
    # 测试疏散路径生成
    print("\n=== 测试疏散路径生成 ===")
    evacuation_path = io_interface.generate_evacuation_path(
        agent_id="test_agent_001",
        start_location=(113.2644, 23.1291),
        destination_facility_id="shelter_001",
        departure_time=datetime.now()
    )
    
    if evacuation_path:
        print(f"生成疏散路径: {evacuation_path.path_id}")
        print(f"总距离: {evacuation_path.total_distance:.2f} 公里")
        print(f"预计时间: {evacuation_path.estimated_travel_time:.2f} 小时")
    
    # 测试指标计算
    print("\n=== 测试指标计算 ===")
    
    # 更新拥堵指标
    io_interface.update_congestion_metrics(datetime.now())
    print(f"生成了 {len(io_interface.congestion_metrics)} 个拥堵指标")
    
    # 计算合规率
    test_evacuation_status = {f"agent_{i:03d}": i % 3 != 0 for i in range(100)}
    compliance = io_interface.calculate_compliance_metrics(datetime.now(), test_evacuation_status)
    print(f"合规率: {compliance.compliance_rate:.2%}")
    
    # 生成脆弱群体指标
    vulnerable_indicators = io_interface.generate_vulnerable_group_indicators(datetime.now())
    print(f"脆弱群体数量: {len(vulnerable_indicators.group_populations)}")
    
    # 导出所有数据
    print("\n=== 导出测试数据 ===")
    io_interface.export_all_outputs("test_output")
    
    print("\n=== 比赛输入输出接口测试完成 ===")
    
    # 清理测试文件
    import os
    for file in ["test_events.json", "test_facilities.json", "test_roads.json"]:
        if os.path.exists(file):
            os.remove(file)