"""洪灾场景配置文件
基于差序格局的洪水灾害ABM仿真配置
"""

import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class FloodScenarioConfig:
    """洪灾场景配置类"""
    
    # === 基础仿真参数 ===
    simulation_steps: int = 100  # 仿真步数
    time_step_hours: float = 1.0  # 每步代表的小时数
    agent_count: int = 200  # 智能体数量
    random_seed: int = 42  # 随机种子
    
    # === 洪水参数 ===
    flood_start_step: int = 10  # 洪水开始步数
    flood_peak_step: int = 30   # 洪水峰值步数
    flood_end_step: int = 80    # 洪水结束步数
    max_water_level: float = 2.5  # 最大水位(米)
    max_flow_velocity: float = 4.0  # 最大流速(m/s)
    affected_area_ratio: float = 0.6  # 受影响区域比例
    
    # === 地理环境参数 ===
    total_aois: int = 20  # 总AOI数量
    safe_aois: List[int] = None  # 安全区域AOI列表
    high_risk_aois: List[int] = None  # 高风险区域AOI列表
    evacuation_centers: List[int] = None  # 疏散中心AOI列表
    
    # === 智能体人口分布 ===
    age_distribution: Dict[str, float] = None  # 年龄分布
    education_distribution: Dict[str, float] = None  # 教育分布
    income_distribution: Dict[str, float] = None  # 收入分布
    family_size_distribution: Dict[int, float] = None  # 家庭规模分布
    
    # === 差序格局网络参数 ===
    network_density: Dict[str, float] = None  # 各类关系网络密度
    relationship_strength_ranges: Dict[str, tuple] = None  # 关系强度范围
    spatial_clustering: float = 0.7  # 空间聚集度
    
    # === 资源配置 ===
    initial_resource_ranges: Dict[str, tuple] = None  # 初始资源范围
    resource_consumption_rates: Dict[str, float] = None  # 资源消耗率
    transportation_ownership_rate: float = 0.6  # 交通工具拥有率
    
    # === 行为参数 ===
    risk_threshold_range: tuple = (0.3, 0.8)  # 风险阈值范围
    evacuation_threshold_range: tuple = (0.4, 0.9)  # 疏散阈值范围
    altruism_range: tuple = (0.2, 0.9)  # 利他主义范围
    social_trust_range: tuple = (0.1, 0.8)  # 社会信任范围
    
    def __post_init__(self):
        """初始化后处理"""
        if self.safe_aois is None:
            self.safe_aois = list(range(15, 20))  # AOI 15-19为安全区域
        
        if self.high_risk_aois is None:
            self.high_risk_aois = list(range(0, 5))  # AOI 0-4为高风险区域
        
        if self.evacuation_centers is None:
            self.evacuation_centers = [16, 17, 18]  # 疏散中心
        
        if self.age_distribution is None:
            self.age_distribution = {
                'young': 0.25,    # 18-35岁
                'middle': 0.45,   # 36-55岁
                'senior': 0.30    # 56岁以上
            }
        
        if self.education_distribution is None:
            self.education_distribution = {
                'primary': 0.3,
                'secondary': 0.5,
                'tertiary': 0.2
            }
        
        if self.income_distribution is None:
            self.income_distribution = {
                'low': 0.4,
                'medium': 0.4,
                'high': 0.2
            }
        
        if self.family_size_distribution is None:
            self.family_size_distribution = {
                1: 0.15,
                2: 0.25,
                3: 0.25,
                4: 0.20,
                5: 0.10,
                6: 0.05
            }
        
        if self.network_density is None:
            self.network_density = {
                'family': 0.8,     # 家庭网络密度高
                'neighbor': 0.4,   # 邻里网络中等
                'colleague': 0.3,  # 同事网络较低
                'classmate': 0.2   # 同学网络最低
            }
        
        if self.relationship_strength_ranges is None:
            self.relationship_strength_ranges = {
                'family': (0.7, 1.0),
                'neighbor': (0.4, 0.8),
                'colleague': (0.3, 0.7),
                'classmate': (0.2, 0.6)
            }
        
        if self.initial_resource_ranges is None:
            self.initial_resource_ranges = {
                'food': (2.0, 10.0),      # 食物储备(天)
                'water': (1.0, 8.0),      # 饮用水储备(天)
                'medicine': (0.0, 5.0),   # 药品储备
                'money': (100.0, 5000.0)  # 现金储备
            }
        
        if self.resource_consumption_rates is None:
            self.resource_consumption_rates = {
                'food': 0.2,     # 每步消耗食物
                'water': 0.25,   # 每步消耗水
                'medicine': 0.05 # 每步消耗药品
            }

class FloodScenarioGenerator:
    """洪灾场景生成器"""
    
    def __init__(self, config: FloodScenarioConfig):
        self.config = config
        random.seed(config.random_seed)
    
    def generate_flood_timeline(self) -> Dict[int, Dict[str, float]]:
        """生成洪水时间线"""
        timeline = {}
        
        for step in range(self.config.simulation_steps):
            if step < self.config.flood_start_step:
                # 洪水前期
                water_level = 0.0
                flow_velocity = 0.0
                precipitation = random.uniform(0, 10)
            
            elif step <= self.config.flood_peak_step:
                # 洪水上升期
                progress = (step - self.config.flood_start_step) / (self.config.flood_peak_step - self.config.flood_start_step)
                water_level = progress * self.config.max_water_level
                flow_velocity = progress * self.config.max_flow_velocity
                precipitation = random.uniform(20, 80)
            
            elif step <= self.config.flood_end_step:
                # 洪水消退期
                progress = 1.0 - (step - self.config.flood_peak_step) / (self.config.flood_end_step - self.config.flood_peak_step)
                water_level = progress * self.config.max_water_level
                flow_velocity = progress * self.config.max_flow_velocity
                precipitation = random.uniform(10, 40)
            
            else:
                # 洪水后期
                water_level = 0.0
                flow_velocity = 0.0
                precipitation = random.uniform(0, 15)
            
            timeline[step] = {
                'water_level': water_level,
                'flow_velocity': flow_velocity,
                'precipitation': precipitation,
                'affected_area_ratio': min(water_level / self.config.max_water_level * self.config.affected_area_ratio, 1.0)
            }
        
        return timeline
    
    def generate_agent_demographics(self) -> List[Dict[str, Any]]:
        """生成智能体人口统计特征"""
        demographics = []
        
        for i in range(self.config.agent_count):
            # 年龄生成
            age_category = self._sample_from_distribution(self.config.age_distribution)
            if age_category == 'young':
                age = random.randint(18, 35)
            elif age_category == 'middle':
                age = random.randint(36, 55)
            else:  # senior
                age = random.randint(56, 80)
            
            # 其他属性
            education = self._sample_from_distribution(self.config.education_distribution)
            income = self._sample_from_distribution(self.config.income_distribution)
            family_size = self._sample_from_distribution(self.config.family_size_distribution)
            
            demographics.append({
                'agent_id': i,
                'age': age,
                'age_category': age_category,
                'gender': random.choice(['male', 'female']),
                'education': education,
                'income': income,
                'family_size': family_size,
                'has_transportation': random.random() < self.config.transportation_ownership_rate
            })
        
        return demographics
    
    def generate_social_network(self, demographics: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """生成差序格局社交网络"""
        networks = {}
        
        for demo in demographics:
            agent_id = demo['agent_id']
            networks[agent_id] = {
                'family': {},
                'neighbor': {},
                'colleague': {},
                'classmate': {}
            }
            
            # 为每种关系类型生成连接
            for relation_type, density in self.config.network_density.items():
                connection_count = int(random.uniform(2, 15) * density)
                strength_range = self.config.relationship_strength_ranges[relation_type]
                
                for _ in range(connection_count):
                    # 选择连接对象（避免自连接）
                    target_candidates = [d['agent_id'] for d in demographics if d['agent_id'] != agent_id]
                    if target_candidates:
                        target_id = random.choice(target_candidates)
                        
                        # 生成关系属性
                        base_strength = random.uniform(*strength_range)
                        distance = random.uniform(0.1, 10.0)  # 物理距离
                        contact_frequency = random.uniform(0.1, 1.0)
                        
                        networks[agent_id][relation_type][f"{relation_type}_{target_id}"] = {
                            'target_id': target_id,
                            'strength': base_strength,
                            'distance': distance,
                            'contact_frequency': contact_frequency,
                            'mutual_aid_history': 0
                        }
        
        return networks
    
    def generate_initial_resources(self, demographics: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """生成初始资源配置"""
        resources = {}
        
        for demo in demographics:
            agent_id = demo['agent_id']
            
            # 根据收入水平调整资源
            income_multiplier = {
                'low': 0.7,
                'medium': 1.0,
                'high': 1.5
            }[demo['income']]
            
            # 根据家庭规模调整资源
            family_multiplier = 1.0 + (demo['family_size'] - 1) * 0.2
            
            agent_resources = {}
            for resource_type, (min_val, max_val) in self.config.initial_resource_ranges.items():
                base_amount = random.uniform(min_val, max_val)
                adjusted_amount = base_amount * income_multiplier * family_multiplier
                agent_resources[resource_type] = adjusted_amount
            
            # 交通工具
            agent_resources['transportation'] = demo['has_transportation']
            
            resources[agent_id] = agent_resources
        
        return resources
    
    def generate_spatial_distribution(self, demographics: List[Dict[str, Any]]) -> Dict[int, int]:
        """生成智能体空间分布"""
        spatial_distribution = {}
        
        # 考虑空间聚集效应
        for demo in demographics:
            agent_id = demo['agent_id']
            
            # 根据收入水平分配初始位置
            if demo['income'] == 'high':
                # 高收入倾向于安全区域
                initial_aoi = random.choice(self.config.safe_aois + list(range(10, 15)))
            elif demo['income'] == 'low':
                # 低收入倾向于高风险区域
                initial_aoi = random.choice(self.config.high_risk_aois + list(range(5, 10)))
            else:
                # 中等收入随机分布
                initial_aoi = random.randint(0, self.config.total_aois - 1)
            
            spatial_distribution[agent_id] = initial_aoi
        
        return spatial_distribution
    
    def _sample_from_distribution(self, distribution: Dict) -> Any:
        """从分布中采样"""
        rand_val = random.random()
        cumulative = 0.0
        
        for key, prob in distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                return key
        
        # 如果没有匹配，返回最后一个键
        return list(distribution.keys())[-1]
    
    def generate_complete_scenario(self) -> Dict[str, Any]:
        """生成完整的洪灾场景"""
        print("正在生成洪灾场景...")
        
        # 生成各个组件
        flood_timeline = self.generate_flood_timeline()
        demographics = self.generate_agent_demographics()
        social_networks = self.generate_social_network(demographics)
        initial_resources = self.generate_initial_resources(demographics)
        spatial_distribution = self.generate_spatial_distribution(demographics)
        
        scenario = {
            'config': self.config,
            'flood_timeline': flood_timeline,
            'agent_demographics': demographics,
            'social_networks': social_networks,
            'initial_resources': initial_resources,
            'spatial_distribution': spatial_distribution,
            'metadata': {
                'generation_timestamp': None,  # 可以添加时间戳
                'total_agents': len(demographics),
                'total_steps': self.config.simulation_steps,
                'flood_duration': self.config.flood_end_step - self.config.flood_start_step
            }
        }
        
        print(f"场景生成完成：{len(demographics)}个智能体，{self.config.simulation_steps}个时间步")
        return scenario

# 预定义场景配置
DEFAULT_FLOOD_CONFIG = FloodScenarioConfig(
    simulation_steps=100,
    agent_count=200,
    flood_start_step=10,
    flood_peak_step=30,
    flood_end_step=80,
    max_water_level=2.5,
    max_flow_velocity=4.0
)

SMALL_SCALE_CONFIG = FloodScenarioConfig(
    simulation_steps=50,
    agent_count=50,
    flood_start_step=5,
    flood_peak_step=15,
    flood_end_step=40,
    max_water_level=2.0,
    max_flow_velocity=3.0
)

LARGE_SCALE_CONFIG = FloodScenarioConfig(
    simulation_steps=200,
    agent_count=500,
    flood_start_step=20,
    flood_peak_step=60,
    flood_end_step=160,
    max_water_level=3.0,
    max_flow_velocity=5.0
)

# 使用示例
if __name__ == "__main__":
    # 生成默认场景
    generator = FloodScenarioGenerator(DEFAULT_FLOOD_CONFIG)
    scenario = generator.generate_complete_scenario()
    
    print("\n=== 场景统计 ===")
    print(f"智能体数量: {scenario['metadata']['total_agents']}")
    print(f"仿真步数: {scenario['metadata']['total_steps']}")
    print(f"洪水持续时间: {scenario['metadata']['flood_duration']}步")
    
    # 统计人口分布
    age_stats = {}
    for demo in scenario['agent_demographics']:
        age_cat = demo['age_category']
        age_stats[age_cat] = age_stats.get(age_cat, 0) + 1
    
    print("\n年龄分布:")
    for age_cat, count in age_stats.items():
        print(f"  {age_cat}: {count} ({count/len(scenario['agent_demographics'])*100:.1f}%)")
    
    # 统计网络连接
    total_connections = 0
    for agent_id, networks in scenario['social_networks'].items():
        for relation_type, connections in networks.items():
            total_connections += len(connections)
    
    print(f"\n总社交连接数: {total_connections}")
    print(f"平均每个智能体连接数: {total_connections/len(scenario['agent_demographics']):.1f}")