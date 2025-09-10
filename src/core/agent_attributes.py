"""智能体属性系统模块

整合个人、社交、经济三类属性，支持差序格局理论的智能体建模。
基于决赛项目方案中的属性设计，实现城市洪灾场景下的智能体特征。

作者: 城市智能体项目组
日期: 2024-12-30
"""

import logging
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Gender(Enum):
    """性别枚举"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class EducationLevel(Enum):
    """教育水平枚举"""
    PRIMARY = "primary"  # 小学
    MIDDLE = "middle"  # 初中
    HIGH = "high"  # 高中
    COLLEGE = "college"  # 大专
    BACHELOR = "bachelor"  # 本科
    MASTER = "master"  # 硕士
    DOCTOR = "doctor"  # 博士


class OccupationType(Enum):
    """职业类型枚举"""
    STUDENT = "student"  # 学生
    WORKER = "worker"  # 工人
    FARMER = "farmer"  # 农民
    TEACHER = "teacher"  # 教师
    DOCTOR = "doctor"  # 医生
    ENGINEER = "engineer"  # 工程师
    MANAGER = "manager"  # 管理人员
    ENTREPRENEUR = "entrepreneur"  # 企业家
    CIVIL_SERVANT = "civil_servant"  # 公务员
    RETIRED = "retired"  # 退休人员
    UNEMPLOYED = "unemployed"  # 失业
    OTHER = "other"  # 其他


class HealthStatus(Enum):
    """健康状态枚举"""
    EXCELLENT = "excellent"  # 优秀
    GOOD = "good"  # 良好
    FAIR = "fair"  # 一般
    POOR = "poor"  # 较差
    CRITICAL = "critical"  # 危重


class MobilityLevel(Enum):
    """行动能力枚举"""
    HIGH = "high"  # 高行动力
    MEDIUM = "medium"  # 中等行动力
    LOW = "low"  # 低行动力
    DISABLED = "disabled"  # 行动不便


class IncomeLevel(Enum):
    """收入水平枚举"""
    VERY_LOW = "very_low"  # 极低收入
    LOW = "low"  # 低收入
    MEDIUM_LOW = "medium_low"  # 中低收入
    MEDIUM = "medium"  # 中等收入
    MEDIUM_HIGH = "medium_high"  # 中高收入
    HIGH = "high"  # 高收入
    VERY_HIGH = "very_high"  # 极高收入


class HousingType(Enum):
    """住房类型枚举"""
    RENT = "rent"  # 租房
    OWN = "own"  # 自有住房
    FAMILY = "family"  # 家庭住房
    DORMITORY = "dormitory"  # 宿舍
    TEMPORARY = "temporary"  # 临时住所


class TransportMode(Enum):
    """交通方式枚举"""
    WALK = "walk"  # 步行
    BICYCLE = "bicycle"  # 自行车
    MOTORCYCLE = "motorcycle"  # 摩托车
    CAR = "car"  # 汽车
    PUBLIC_TRANSPORT = "public_transport"  # 公共交通
    TAXI = "taxi"  # 出租车


@dataclass
class PersonalAttributes:
    """个人属性类"""
    # 基本信息
    agent_id: str
    age: int
    gender: Gender
    education_level: EducationLevel
    occupation: OccupationType
    
    # 身体状况
    health_status: HealthStatus
    mobility_level: MobilityLevel
    has_disability: bool = False
    chronic_diseases: List[str] = field(default_factory=list)
    
    # 心理特征
    risk_tolerance: float = 0.5  # 风险容忍度 0-1
    anxiety_level: float = 0.5  # 焦虑水平 0-1
    optimism_level: float = 0.5  # 乐观程度 0-1
    trust_in_authority: float = 0.5  # 对权威的信任 0-1
    
    # 认知能力
    information_processing_ability: float = 0.5  # 信息处理能力 0-1
    decision_making_speed: float = 0.5  # 决策速度 0-1
    learning_ability: float = 0.5  # 学习能力 0-1
    
    # 经验相关
    disaster_experience: int = 0  # 灾害经历次数
    evacuation_experience: int = 0  # 疏散经历次数
    local_knowledge: float = 0.5  # 本地知识 0-1
    
    def get_vulnerability_score(self) -> float:
        """计算脆弱性评分"""
        vulnerability = 0.0
        
        # 年龄因素
        if self.age < 18 or self.age > 65:
            vulnerability += 0.2
        
        # 健康状况
        health_weights = {
            HealthStatus.EXCELLENT: 0.0,
            HealthStatus.GOOD: 0.1,
            HealthStatus.FAIR: 0.2,
            HealthStatus.POOR: 0.3,
            HealthStatus.CRITICAL: 0.4
        }
        vulnerability += health_weights.get(self.health_status, 0.2)
        
        # 行动能力
        mobility_weights = {
            MobilityLevel.HIGH: 0.0,
            MobilityLevel.MEDIUM: 0.1,
            MobilityLevel.LOW: 0.2,
            MobilityLevel.DISABLED: 0.3
        }
        vulnerability += mobility_weights.get(self.mobility_level, 0.1)
        
        # 残疾状况
        if self.has_disability:
            vulnerability += 0.1
        
        # 慢性疾病
        vulnerability += min(0.2, len(self.chronic_diseases) * 0.05)
        
        return min(1.0, vulnerability)
    
    def get_adaptability_score(self) -> float:
        """计算适应性评分"""
        adaptability = 0.0
        
        # 教育水平
        education_weights = {
            EducationLevel.PRIMARY: 0.1,
            EducationLevel.MIDDLE: 0.2,
            EducationLevel.HIGH: 0.3,
            EducationLevel.COLLEGE: 0.4,
            EducationLevel.BACHELOR: 0.5,
            EducationLevel.MASTER: 0.6,
            EducationLevel.DOCTOR: 0.7
        }
        adaptability += education_weights.get(self.education_level, 0.3)
        
        # 认知能力
        adaptability += (self.information_processing_ability + 
                        self.decision_making_speed + 
                        self.learning_ability) / 3 * 0.3
        
        # 经验
        experience_score = min(1.0, (self.disaster_experience * 0.1 + 
                                   self.evacuation_experience * 0.1 + 
                                   self.local_knowledge) / 3)
        adaptability += experience_score * 0.2
        
        return min(1.0, adaptability)


@dataclass
class SocialRelationship:
    """社会关系数据结构"""
    target_id: str
    relationship_type: str  # 关系类型
    strength: float  # 关系强度 0-1
    trust_level: float  # 信任程度 0-1
    contact_frequency: float  # 联系频率 0-1
    mutual_support: float  # 互助程度 0-1
    geographic_distance: float  # 地理距离（公里）
    relationship_duration: int  # 关系持续时间（年）
    
    def get_relationship_value(self) -> float:
        """计算关系价值"""
        return (self.strength * 0.3 + 
                self.trust_level * 0.3 + 
                self.mutual_support * 0.2 + 
                self.contact_frequency * 0.2)


@dataclass
class SocialAttributes:
    """社交属性类"""
    # 社会网络
    relationships: Dict[str, SocialRelationship] = field(default_factory=dict)
    
    # 社会地位
    social_status: float = 0.5  # 社会地位 0-1
    community_influence: float = 0.5  # 社区影响力 0-1
    leadership_ability: float = 0.5  # 领导能力 0-1
    
    # 社交特征
    extroversion: float = 0.5  # 外向性 0-1
    social_skills: float = 0.5  # 社交技能 0-1
    empathy_level: float = 0.5  # 同理心 0-1
    cooperation_tendency: float = 0.5  # 合作倾向 0-1
    
    # 群体归属
    community_attachment: float = 0.5  # 社区依恋 0-1
    cultural_identity: float = 0.5  # 文化认同 0-1
    group_memberships: List[str] = field(default_factory=list)  # 群体成员身份
    
    # 社会支持
    social_support_received: float = 0.5  # 接受的社会支持 0-1
    social_support_provided: float = 0.5  # 提供的社会支持 0-1
    
    def add_relationship(self, relationship: SocialRelationship):
        """添加社会关系"""
        self.relationships[relationship.target_id] = relationship
    
    def get_relationship(self, target_id: str) -> Optional[SocialRelationship]:
        """获取特定关系"""
        return self.relationships.get(target_id)
    
    def get_close_relationships(self, threshold: float = 0.7) -> List[SocialRelationship]:
        """获取亲密关系列表"""
        return [rel for rel in self.relationships.values() 
                if rel.get_relationship_value() >= threshold]
    
    def get_network_size(self) -> int:
        """获取社会网络规模"""
        return len(self.relationships)
    
    def get_network_density(self) -> float:
        """计算网络密度（简化版）"""
        if len(self.relationships) < 2:
            return 0.0
        
        # 简化计算：基于关系强度的平均值
        total_strength = sum(rel.strength for rel in self.relationships.values())
        return total_strength / len(self.relationships)
    
    def get_social_capital(self) -> float:
        """计算社会资本"""
        if not self.relationships:
            return 0.0
        
        # 基于关系质量和数量的综合评分
        quality_score = sum(rel.get_relationship_value() 
                          for rel in self.relationships.values()) / len(self.relationships)
        quantity_score = min(1.0, len(self.relationships) / 20)  # 假设20个关系为满分
        
        return (quality_score * 0.7 + quantity_score * 0.3) * \
               (self.social_status * 0.3 + self.community_influence * 0.3 + 
                self.leadership_ability * 0.2 + self.social_skills * 0.2)


@dataclass
class EconomicAttributes:
    """经济属性类"""
    # 收入状况
    income_level: IncomeLevel
    monthly_income: float  # 月收入（元）
    income_stability: float = 0.5  # 收入稳定性 0-1
    
    # 资产状况
    total_assets: float = 0.0  # 总资产（元）
    liquid_assets: float = 0.0  # 流动资产（元）
    real_estate_value: float = 0.0  # 房产价值（元）
    
    # 住房状况
    housing_type: HousingType
    housing_cost_ratio: float = 0.3  # 住房成本占收入比例
    housing_quality: float = 0.5  # 住房质量 0-1
    
    # 交通状况
    primary_transport: TransportMode
    transport_budget: float = 0.0  # 交通预算（月）
    vehicle_ownership: List[str] = field(default_factory=list)  # 拥有的交通工具
    
    # 保险状况
    has_health_insurance: bool = False
    has_property_insurance: bool = False
    has_life_insurance: bool = False
    insurance_coverage: float = 0.0  # 保险覆盖额度
    
    # 债务状况
    total_debt: float = 0.0  # 总债务
    debt_to_income_ratio: float = 0.0  # 债务收入比
    
    # 储蓄状况
    savings_rate: float = 0.1  # 储蓄率
    emergency_fund: float = 0.0  # 应急资金
    
    def get_economic_vulnerability(self) -> float:
        """计算经济脆弱性"""
        vulnerability = 0.0
        
        # 收入水平
        income_weights = {
            IncomeLevel.VERY_LOW: 0.4,
            IncomeLevel.LOW: 0.3,
            IncomeLevel.MEDIUM_LOW: 0.2,
            IncomeLevel.MEDIUM: 0.1,
            IncomeLevel.MEDIUM_HIGH: 0.05,
            IncomeLevel.HIGH: 0.0,
            IncomeLevel.VERY_HIGH: 0.0
        }
        vulnerability += income_weights.get(self.income_level, 0.2)
        
        # 收入稳定性
        vulnerability += (1 - self.income_stability) * 0.2
        
        # 债务状况
        if self.debt_to_income_ratio > 0.5:
            vulnerability += 0.2
        elif self.debt_to_income_ratio > 0.3:
            vulnerability += 0.1
        
        # 应急资金
        if self.emergency_fund < self.monthly_income * 3:
            vulnerability += 0.1
        
        # 住房成本
        if self.housing_cost_ratio > 0.5:
            vulnerability += 0.1
        
        return min(1.0, vulnerability)
    
    def get_economic_resilience(self) -> float:
        """计算经济韧性"""
        resilience = 0.0
        
        # 资产状况
        if self.total_assets > self.monthly_income * 12:
            resilience += 0.3
        elif self.total_assets > self.monthly_income * 6:
            resilience += 0.2
        elif self.total_assets > self.monthly_income * 3:
            resilience += 0.1
        
        # 流动性
        liquidity_ratio = self.liquid_assets / max(1, self.monthly_income)
        resilience += min(0.2, liquidity_ratio * 0.05)
        
        # 保险覆盖
        insurance_score = 0
        if self.has_health_insurance:
            insurance_score += 1
        if self.has_property_insurance:
            insurance_score += 1
        if self.has_life_insurance:
            insurance_score += 1
        resilience += insurance_score / 3 * 0.2
        
        # 储蓄能力
        resilience += min(0.2, self.savings_rate * 2)
        
        # 收入稳定性
        resilience += self.income_stability * 0.1
        
        return min(1.0, resilience)
    
    def can_afford_evacuation(self, cost: float) -> bool:
        """判断是否能承担疏散成本"""
        available_funds = self.liquid_assets + self.emergency_fund
        return available_funds >= cost
    
    def get_transport_options(self) -> List[TransportMode]:
        """获取可用交通方式"""
        options = [TransportMode.WALK]  # 步行总是可用的
        
        # 基于拥有的交通工具
        if "bicycle" in self.vehicle_ownership:
            options.append(TransportMode.BICYCLE)
        if "motorcycle" in self.vehicle_ownership:
            options.append(TransportMode.MOTORCYCLE)
        if "car" in self.vehicle_ownership:
            options.append(TransportMode.CAR)
        
        # 基于经济能力
        if self.monthly_income > 3000:  # 假设阈值
            options.extend([TransportMode.PUBLIC_TRANSPORT, TransportMode.TAXI])
        elif self.monthly_income > 1500:
            options.append(TransportMode.PUBLIC_TRANSPORT)
        
        return list(set(options))  # 去重


@dataclass
class IntegratedAgentAttributes:
    """整合的智能体属性类"""
    personal: PersonalAttributes
    social: SocialAttributes
    economic: EconomicAttributes
    
    # 差序格局相关
    strategy_phenotype: str = "moderate"  # 策略表型：strict/moderate/universalist
    risk_perception: float = 0.5  # 风险感知 0-1
    
    # 动态属性
    current_location: Tuple[float, float] = (0.0, 0.0)  # 当前位置（经纬度）
    current_stress_level: float = 0.5  # 当前压力水平 0-1
    current_resources: Dict[str, float] = field(default_factory=dict)  # 当前资源
    
    # 决策历史
    decision_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        # 初始化当前资源
        if not self.current_resources:
            self.current_resources = {
                'food': 100.0,
                'water': 100.0,
                'money': self.economic.liquid_assets,
                'energy': 100.0
            }
    
    def get_overall_vulnerability(self) -> float:
        """计算综合脆弱性"""
        personal_vuln = self.personal.get_vulnerability_score()
        economic_vuln = self.economic.get_economic_vulnerability()
        
        # 社会脆弱性（基于社会支持的缺乏）
        social_vuln = 1 - self.social.social_support_received
        
        # 加权平均
        return (personal_vuln * 0.4 + 
                economic_vuln * 0.4 + 
                social_vuln * 0.2)
    
    def get_overall_resilience(self) -> float:
        """计算综合韧性"""
        personal_resilience = self.personal.get_adaptability_score()
        economic_resilience = self.economic.get_economic_resilience()
        social_resilience = self.social.get_social_capital()
        
        # 加权平均
        return (personal_resilience * 0.3 + 
                economic_resilience * 0.4 + 
                social_resilience * 0.3)
    
    def update_stress_level(self, event_severity: float, social_support: float = 0.0):
        """更新压力水平"""
        # 基础压力增加
        stress_increase = event_severity * (1 - self.personal.risk_tolerance)
        
        # 社会支持缓解
        stress_relief = social_support * self.social.social_support_received
        
        # 个人特征调节
        anxiety_factor = 1 + self.personal.anxiety_level * 0.5
        optimism_factor = 1 - self.personal.optimism_level * 0.3
        
        # 更新压力水平
        new_stress = self.current_stress_level + \
                    (stress_increase - stress_relief) * anxiety_factor * optimism_factor
        
        self.current_stress_level = max(0.0, min(1.0, new_stress))
    
    def make_evacuation_decision(self, risk_level: float, 
                               evacuation_cost: float,
                               social_influence: float = 0.0) -> Dict[str, Any]:
        """做出疏散决策"""
        # 风险感知调节
        perceived_risk = risk_level * self.risk_perception
        
        # 经济约束
        can_afford = self.economic.can_afford_evacuation(evacuation_cost)
        
        # 社会影响
        social_pressure = social_influence * (1 - self.personal.trust_in_authority)
        
        # 个人因素
        personal_factor = (self.personal.get_vulnerability_score() + 
                          self.current_stress_level) / 2
        
        # 综合决策评分
        decision_score = (perceived_risk * 0.4 + 
                         personal_factor * 0.3 + 
                         social_pressure * 0.2 + 
                         (1 if can_afford else 0) * 0.1)
        
        # 决策阈值（基于策略表型）
        thresholds = {
            'strict': 0.7,  # 保守，需要高确定性
            'moderate': 0.5,  # 中等
            'universalist': 0.4  # 相对激进
        }
        threshold = thresholds.get(self.strategy_phenotype, 0.5)
        
        will_evacuate = decision_score >= threshold
        
        decision = {
            'will_evacuate': will_evacuate,
            'decision_score': decision_score,
            'threshold': threshold,
            'factors': {
                'perceived_risk': perceived_risk,
                'can_afford': can_afford,
                'social_influence': social_influence,
                'personal_factor': personal_factor
            },
            'timestamp': datetime.now()
        }
        
        self.decision_history.append(decision)
        return decision
    
    def get_preferred_destinations(self, available_destinations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取偏好目的地排序"""
        scored_destinations = []
        
        for dest in available_destinations:
            score = 0.0
            
            # 安全性权重
            score += dest.get('safety_level', 0.5) * 0.3
            
            # 成本考虑
            cost = dest.get('cost', 0)
            if self.economic.can_afford_evacuation(cost):
                score += (1 - min(1.0, cost / self.economic.monthly_income)) * 0.2
            
            # 距离考虑（越近越好，但要考虑交通能力）
            distance = dest.get('distance', 0)
            transport_options = self.economic.get_transport_options()
            if TransportMode.CAR in transport_options:
                distance_penalty = min(0.2, distance / 100)  # 100km为参考
            else:
                distance_penalty = min(0.3, distance / 50)  # 50km为参考
            score += (0.2 - distance_penalty)
            
            # 社会网络考虑
            social_connections = dest.get('social_connections', 0)
            score += min(0.2, social_connections / 10) * 0.15
            
            # 设施完善度
            score += dest.get('facility_level', 0.5) * 0.15
            
            scored_destinations.append({
                **dest,
                'preference_score': score
            })
        
        # 按偏好评分排序
        return sorted(scored_destinations, key=lambda x: x['preference_score'], reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'agent_id': self.personal.agent_id,
            'personal': {
                'age': self.personal.age,
                'gender': self.personal.gender.value,
                'education': self.personal.education_level.value,
                'occupation': self.personal.occupation.value,
                'health_status': self.personal.health_status.value,
                'vulnerability_score': self.personal.get_vulnerability_score(),
                'adaptability_score': self.personal.get_adaptability_score()
            },
            'social': {
                'network_size': self.social.get_network_size(),
                'network_density': self.social.get_network_density(),
                'social_capital': self.social.get_social_capital(),
                'social_status': self.social.social_status
            },
            'economic': {
                'income_level': self.economic.income_level.value,
                'monthly_income': self.economic.monthly_income,
                'economic_vulnerability': self.economic.get_economic_vulnerability(),
                'economic_resilience': self.economic.get_economic_resilience()
            },
            'integrated': {
                'strategy_phenotype': self.strategy_phenotype,
                'overall_vulnerability': self.get_overall_vulnerability(),
                'overall_resilience': self.get_overall_resilience(),
                'current_stress_level': self.current_stress_level
            }
        }


class AttributeGenerator:
    """属性生成器类"""
    
    @staticmethod
    def generate_random_personal_attributes(agent_id: str) -> PersonalAttributes:
        """生成随机个人属性"""
        return PersonalAttributes(
            agent_id=agent_id,
            age=random.randint(18, 80),
            gender=random.choice(list(Gender)),
            education_level=random.choice(list(EducationLevel)),
            occupation=random.choice(list(OccupationType)),
            health_status=random.choice(list(HealthStatus)),
            mobility_level=random.choice(list(MobilityLevel)),
            has_disability=random.random() < 0.1,
            chronic_diseases=random.sample(['高血压', '糖尿病', '心脏病', '哮喘'], 
                                         random.randint(0, 2)),
            risk_tolerance=random.random(),
            anxiety_level=random.random(),
            optimism_level=random.random(),
            trust_in_authority=random.random(),
            information_processing_ability=random.random(),
            decision_making_speed=random.random(),
            learning_ability=random.random(),
            disaster_experience=random.randint(0, 5),
            evacuation_experience=random.randint(0, 3),
            local_knowledge=random.random()
        )
    
    @staticmethod
    def generate_random_social_attributes() -> SocialAttributes:
        """生成随机社交属性"""
        return SocialAttributes(
            social_status=random.random(),
            community_influence=random.random(),
            leadership_ability=random.random(),
            extroversion=random.random(),
            social_skills=random.random(),
            empathy_level=random.random(),
            cooperation_tendency=random.random(),
            community_attachment=random.random(),
            cultural_identity=random.random(),
            group_memberships=random.sample(['社区委员会', '业主委员会', '志愿者组织', '宗教团体'], 
                                          random.randint(0, 2)),
            social_support_received=random.random(),
            social_support_provided=random.random()
        )
    
    @staticmethod
    def generate_random_economic_attributes() -> EconomicAttributes:
        """生成随机经济属性"""
        income_level = random.choice(list(IncomeLevel))
        
        # 基于收入水平设置月收入
        income_ranges = {
            IncomeLevel.VERY_LOW: (1000, 2000),
            IncomeLevel.LOW: (2000, 4000),
            IncomeLevel.MEDIUM_LOW: (4000, 6000),
            IncomeLevel.MEDIUM: (6000, 10000),
            IncomeLevel.MEDIUM_HIGH: (10000, 15000),
            IncomeLevel.HIGH: (15000, 25000),
            IncomeLevel.VERY_HIGH: (25000, 50000)
        }
        
        min_income, max_income = income_ranges[income_level]
        monthly_income = random.uniform(min_income, max_income)
        
        return EconomicAttributes(
            income_level=income_level,
            monthly_income=monthly_income,
            income_stability=random.random(),
            total_assets=monthly_income * random.uniform(12, 60),
            liquid_assets=monthly_income * random.uniform(1, 6),
            real_estate_value=monthly_income * random.uniform(50, 200) if random.random() > 0.3 else 0,
            housing_type=random.choice(list(HousingType)),
            housing_cost_ratio=random.uniform(0.2, 0.6),
            housing_quality=random.random(),
            primary_transport=random.choice(list(TransportMode)),
            transport_budget=monthly_income * random.uniform(0.05, 0.2),
            vehicle_ownership=random.sample(['bicycle', 'motorcycle', 'car'], 
                                          random.randint(0, 2)),
            has_health_insurance=random.random() > 0.3,
            has_property_insurance=random.random() > 0.5,
            has_life_insurance=random.random() > 0.6,
            insurance_coverage=monthly_income * random.uniform(50, 200),
            total_debt=monthly_income * random.uniform(0, 20),
            debt_to_income_ratio=random.uniform(0, 0.8),
            savings_rate=random.uniform(0, 0.3),
            emergency_fund=monthly_income * random.uniform(0, 6)
        )
    
    @staticmethod
    def generate_integrated_attributes(agent_id: str) -> IntegratedAgentAttributes:
        """生成完整的智能体属性"""
        personal = AttributeGenerator.generate_random_personal_attributes(agent_id)
        social = AttributeGenerator.generate_random_social_attributes()
        economic = AttributeGenerator.generate_random_economic_attributes()
        
        # 基于个人特征确定策略表型
        if personal.trust_in_authority > 0.7 and social.community_attachment > 0.7:
            strategy_phenotype = "strict"
        elif personal.empathy_level > 0.6 and social.cooperation_tendency > 0.6:
            strategy_phenotype = "universalist"
        else:
            strategy_phenotype = "moderate"
        
        return IntegratedAgentAttributes(
            personal=personal,
            social=social,
            economic=economic,
            strategy_phenotype=strategy_phenotype,
            risk_perception=random.random(),
            current_location=(random.uniform(39.8, 40.0), random.uniform(116.3, 116.5))  # 北京范围
        )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 生成测试智能体
    agent = AttributeGenerator.generate_integrated_attributes("test_agent_001")
    
    print("=== 智能体属性测试 ===")
    print(f"智能体ID: {agent.personal.agent_id}")
    print(f"策略表型: {agent.strategy_phenotype}")
    print(f"综合脆弱性: {agent.get_overall_vulnerability():.3f}")
    print(f"综合韧性: {agent.get_overall_resilience():.3f}")
    
    # 测试疏散决策
    decision = agent.make_evacuation_decision(
        risk_level=0.8,
        evacuation_cost=5000,
        social_influence=0.6
    )
    
    print(f"\n=== 疏散决策测试 ===")
    print(f"是否疏散: {decision['will_evacuate']}")
    print(f"决策评分: {decision['decision_score']:.3f}")
    print(f"决策阈值: {decision['threshold']:.3f}")
    
    # 输出属性字典
    print(f"\n=== 属性字典 ===")
    attr_dict = agent.to_dict()
    for key, value in attr_dict.items():
        print(f"{key}: {value}")