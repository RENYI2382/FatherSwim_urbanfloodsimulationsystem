"""三层数据驱动系统模块

整合广州统计年鉴(宏观层)、CFPS 2020/2022(微观层)、CGSS 2021(认知层)数据，
为差序格局理论的智能体建模提供数据支撑。

作者: 城市智能体项目组
日期: 2024-12-30
"""

import logging
import pandas as pd
import numpy as np
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import random
from datetime import datetime, timedelta

from .agent_attributes import (
    IntegratedAgentAttributes, PersonalAttributes, SocialAttributes, EconomicAttributes,
    Gender, EducationLevel, OccupationType, HealthStatus, IncomeLevel, HousingType
)

logger = logging.getLogger(__name__)


class DataLayer(Enum):
    """数据层次枚举"""
    MACRO = "macro"  # 宏观层：统计年鉴数据
    MICRO = "micro"  # 微观层：CFPS个体数据
    COGNITIVE = "cognitive"  # 认知层：CGSS认知数据


class DataSource(Enum):
    """数据源枚举"""
    GUANGZHOU_YEARBOOK = "guangzhou_yearbook"  # 广州统计年鉴
    CFPS_2020 = "cfps_2020"  # 中国家庭追踪调查2020
    CFPS_2022 = "cfps_2022"  # 中国家庭追踪调查2022
    CGSS_2021 = "cgss_2021"  # 中国综合社会调查2021


@dataclass
class MacroData:
    """宏观层数据结构"""
    # 人口统计
    total_population: int = 0
    population_density: float = 0.0
    age_distribution: Dict[str, float] = field(default_factory=dict)
    gender_ratio: float = 0.5
    
    # 经济指标
    gdp_per_capita: float = 0.0
    average_income: float = 0.0
    income_distribution: Dict[str, float] = field(default_factory=dict)
    unemployment_rate: float = 0.0
    
    # 社会指标
    education_levels: Dict[str, float] = field(default_factory=dict)
    healthcare_coverage: float = 0.0
    social_security_coverage: float = 0.0
    
    # 住房指标
    housing_prices: Dict[str, float] = field(default_factory=dict)
    homeownership_rate: float = 0.0
    housing_types: Dict[str, float] = field(default_factory=dict)
    
    # 交通指标
    public_transport_coverage: float = 0.0
    car_ownership_rate: float = 0.0
    transport_modes: Dict[str, float] = field(default_factory=dict)
    
    # 灾害相关
    disaster_frequency: Dict[str, int] = field(default_factory=dict)
    emergency_facilities: Dict[str, int] = field(default_factory=dict)
    evacuation_capacity: int = 0


@dataclass
class MicroData:
    """微观层数据结构（基于CFPS）"""
    # 个体基本信息
    individual_id: str = ""
    family_id: str = ""
    age: int = 0
    gender: str = ""
    education: str = ""
    occupation: str = ""
    marital_status: str = ""
    
    # 经济状况
    personal_income: float = 0.0
    family_income: float = 0.0
    assets: Dict[str, float] = field(default_factory=dict)
    debts: float = 0.0
    
    # 健康状况
    health_status: str = ""
    chronic_diseases: List[str] = field(default_factory=list)
    healthcare_access: bool = False
    
    # 社会关系
    family_size: int = 0
    social_network_size: int = 0
    community_participation: float = 0.0
    
    # 居住状况
    housing_type: str = ""
    housing_area: float = 0.0
    housing_value: float = 0.0
    location_type: str = ""  # 城市/农村
    
    # 移动性
    migration_history: List[str] = field(default_factory=list)
    travel_frequency: float = 0.0
    transport_access: Dict[str, bool] = field(default_factory=dict)


@dataclass
class CognitiveData:
    """认知层数据结构（基于CGSS）"""
    # 风险认知
    risk_perception: Dict[str, float] = field(default_factory=dict)
    disaster_awareness: float = 0.0
    preparedness_level: float = 0.0
    
    # 社会态度
    trust_in_government: float = 0.0
    trust_in_neighbors: float = 0.0
    social_cohesion: float = 0.0
    collective_efficacy: float = 0.0
    
    # 价值观念
    individualism_collectivism: float = 0.0  # -1到1，-1极端集体主义，1极端个人主义
    traditional_modern: float = 0.0  # -1到1，-1极端传统，1极端现代
    hierarchy_egalitarian: float = 0.0  # -1到1，-1极端等级，1极端平等
    
    # 差序格局相关
    family_orientation: float = 0.0  # 家庭取向
    in_group_favoritism: float = 0.0  # 内群体偏好
    reciprocity_norm: float = 0.0  # 互惠规范
    guanxi_importance: float = 0.0  # 关系重要性
    
    # 应对策略
    coping_strategies: Dict[str, float] = field(default_factory=dict)
    help_seeking_behavior: Dict[str, float] = field(default_factory=dict)
    information_sources: Dict[str, float] = field(default_factory=dict)


class DataIntegrationSystem:
    """数据整合系统"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 数据存储
        self.macro_data: Optional[MacroData] = None
        self.micro_data: List[MicroData] = []
        self.cognitive_data: List[CognitiveData] = []
        
        # 数据库连接
        self.db_path = self.data_dir / "integrated_data.db"
        self._init_database()
        
        # 数据映射表
        self.education_mapping = {
            "小学及以下": EducationLevel.PRIMARY,
            "初中": EducationLevel.MIDDLE,
            "高中/中专": EducationLevel.HIGH,
            "大专": EducationLevel.COLLEGE,
            "本科": EducationLevel.BACHELOR,
            "硕士": EducationLevel.MASTER,
            "博士": EducationLevel.DOCTOR
        }
        
        self.occupation_mapping = {
            "学生": OccupationType.STUDENT,
            "工人": OccupationType.WORKER,
            "农民": OccupationType.FARMER,
            "教师": OccupationType.TEACHER,
            "医生": OccupationType.DOCTOR,
            "工程师": OccupationType.ENGINEER,
            "管理人员": OccupationType.MANAGER,
            "企业家": OccupationType.ENTREPRENEUR,
            "公务员": OccupationType.CIVIL_SERVANT,
            "退休": OccupationType.RETIRED,
            "失业": OccupationType.UNEMPLOYED
        }
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建表结构
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_data (
                id INTEGER PRIMARY KEY,
                data_type TEXT,
                value REAL,
                category TEXT,
                timestamp TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS micro_data (
                individual_id TEXT PRIMARY KEY,
                family_id TEXT,
                age INTEGER,
                gender TEXT,
                education TEXT,
                occupation TEXT,
                personal_income REAL,
                family_income REAL,
                health_status TEXT,
                housing_type TEXT,
                location_type TEXT,
                data_source TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_data (
                id INTEGER PRIMARY KEY,
                individual_id TEXT,
                risk_perception REAL,
                trust_in_government REAL,
                trust_in_neighbors REAL,
                family_orientation REAL,
                in_group_favoritism REAL,
                reciprocity_norm REAL,
                data_source TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_macro_data(self, source: DataSource = DataSource.GUANGZHOU_YEARBOOK) -> MacroData:
        """加载宏观层数据"""
        if source == DataSource.GUANGZHOU_YEARBOOK:
            # 模拟广州统计年鉴数据
            macro_data = MacroData(
                total_population=15760000,  # 广州常住人口
                population_density=2059.0,  # 人/平方公里
                age_distribution={
                    "0-14": 0.12,
                    "15-59": 0.73,
                    "60+": 0.15
                },
                gender_ratio=0.52,  # 男性比例
                gdp_per_capita=156427.0,  # 元
                average_income=68162.0,  # 城镇居民人均可支配收入
                income_distribution={
                    "低收入": 0.20,
                    "中低收入": 0.25,
                    "中等收入": 0.30,
                    "中高收入": 0.15,
                    "高收入": 0.10
                },
                unemployment_rate=0.024,
                education_levels={
                    "小学及以下": 0.15,
                    "初中": 0.25,
                    "高中/中专": 0.30,
                    "大专": 0.15,
                    "本科": 0.12,
                    "硕士及以上": 0.03
                },
                healthcare_coverage=0.98,
                social_security_coverage=0.95,
                housing_prices={
                    "平均房价": 33000.0,  # 元/平方米
                    "租金水平": 45.0  # 元/平方米/月
                },
                homeownership_rate=0.72,
                housing_types={
                    "商品房": 0.60,
                    "保障房": 0.15,
                    "租赁房": 0.25
                },
                public_transport_coverage=0.85,
                car_ownership_rate=0.35,
                transport_modes={
                    "步行": 0.25,
                    "自行车": 0.10,
                    "公共交通": 0.45,
                    "私家车": 0.20
                },
                disaster_frequency={
                    "洪涝": 2,  # 年均次数
                    "台风": 3,
                    "暴雨": 15
                },
                emergency_facilities={
                    "避难场所": 1200,
                    "医院": 450,
                    "消防站": 180
                },
                evacuation_capacity=500000
            )
            
            self.macro_data = macro_data
            self._save_macro_data_to_db(macro_data, source)
            return macro_data
        
        return MacroData()
    
    def load_micro_data(self, source: DataSource, sample_size: int = 1000) -> List[MicroData]:
        """加载微观层数据"""
        micro_data_list = []
        
        if source in [DataSource.CFPS_2020, DataSource.CFPS_2022]:
            # 模拟CFPS数据
            for i in range(sample_size):
                family_id = f"family_{i // 3}"  # 平均每个家庭3个成员
                
                micro_data = MicroData(
                    individual_id=f"cfps_{source.value}_{i:04d}",
                    family_id=family_id,
                    age=random.randint(18, 80),
                    gender=random.choice(["男", "女"]),
                    education=random.choice(list(self.education_mapping.keys())),
                    occupation=random.choice(list(self.occupation_mapping.keys())),
                    marital_status=random.choice(["未婚", "已婚", "离异", "丧偶"]),
                    personal_income=random.lognormal(10.5, 0.8),  # 对数正态分布
                    family_income=random.lognormal(11.2, 0.9),
                    assets={
                        "房产": random.lognormal(12.0, 1.2),
                        "金融资产": random.lognormal(9.5, 1.5),
                        "其他资产": random.lognormal(8.0, 1.0)
                    },
                    debts=random.lognormal(9.0, 1.5) if random.random() > 0.3 else 0.0,
                    health_status=random.choice(["很好", "好", "一般", "不好", "很不好"]),
                    chronic_diseases=random.sample(["高血压", "糖尿病", "心脏病", "关节炎"], 
                                                 random.randint(0, 2)),
                    healthcare_access=random.random() > 0.1,
                    family_size=random.randint(1, 6),
                    social_network_size=random.randint(5, 50),
                    community_participation=random.random(),
                    housing_type=random.choice(["自有住房", "租赁住房", "其他"]),
                    housing_area=random.normal(80, 30),
                    housing_value=random.lognormal(12.5, 0.8),
                    location_type=random.choice(["城市", "农村"]),
                    migration_history=random.sample(["广州", "深圳", "北京", "上海", "其他"], 
                                                  random.randint(0, 3)),
                    travel_frequency=random.random(),
                    transport_access={
                        "公共交通": random.random() > 0.2,
                        "私家车": random.random() > 0.6,
                        "自行车": random.random() > 0.3
                    }
                )
                
                micro_data_list.append(micro_data)
            
            self.micro_data.extend(micro_data_list)
            self._save_micro_data_to_db(micro_data_list, source)
        
        return micro_data_list
    
    def load_cognitive_data(self, source: DataSource = DataSource.CGSS_2021, 
                          sample_size: int = 800) -> List[CognitiveData]:
        """加载认知层数据"""
        cognitive_data_list = []
        
        if source == DataSource.CGSS_2021:
            # 模拟CGSS数据
            for i in range(sample_size):
                cognitive_data = CognitiveData(
                    risk_perception={
                        "自然灾害": random.uniform(0.3, 0.9),
                        "经济风险": random.uniform(0.4, 0.8),
                        "健康风险": random.uniform(0.5, 0.9),
                        "社会风险": random.uniform(0.2, 0.7)
                    },
                    disaster_awareness=random.uniform(0.3, 0.8),
                    preparedness_level=random.uniform(0.2, 0.7),
                    trust_in_government=random.uniform(0.4, 0.9),
                    trust_in_neighbors=random.uniform(0.3, 0.8),
                    social_cohesion=random.uniform(0.3, 0.8),
                    collective_efficacy=random.uniform(0.4, 0.8),
                    individualism_collectivism=random.uniform(-0.5, 0.3),  # 偏向集体主义
                    traditional_modern=random.uniform(-0.2, 0.6),  # 传统与现代并存
                    hierarchy_egalitarian=random.uniform(-0.3, 0.4),  # 略偏等级
                    family_orientation=random.uniform(0.6, 0.95),  # 高家庭取向
                    in_group_favoritism=random.uniform(0.4, 0.85),  # 内群体偏好
                    reciprocity_norm=random.uniform(0.5, 0.9),  # 互惠规范
                    guanxi_importance=random.uniform(0.6, 0.95),  # 关系重要性
                    coping_strategies={
                        "寻求家庭帮助": random.uniform(0.7, 0.95),
                        "寻求朋友帮助": random.uniform(0.5, 0.8),
                        "寻求政府帮助": random.uniform(0.3, 0.7),
                        "自力更生": random.uniform(0.4, 0.8)
                    },
                    help_seeking_behavior={
                        "家人": random.uniform(0.8, 0.98),
                        "亲戚": random.uniform(0.6, 0.85),
                        "朋友": random.uniform(0.5, 0.8),
                        "邻居": random.uniform(0.3, 0.7),
                        "同事": random.uniform(0.2, 0.6),
                        "陌生人": random.uniform(0.05, 0.2)
                    },
                    information_sources={
                        "家人朋友": random.uniform(0.6, 0.9),
                        "官方媒体": random.uniform(0.4, 0.8),
                        "社交媒体": random.uniform(0.5, 0.85),
                        "专业机构": random.uniform(0.3, 0.7)
                    }
                )
                
                cognitive_data_list.append(cognitive_data)
            
            self.cognitive_data.extend(cognitive_data_list)
            self._save_cognitive_data_to_db(cognitive_data_list, source)
        
        return cognitive_data_list
    
    def integrate_agent_attributes(self, agent_id: str, 
                                 micro_data: Optional[MicroData] = None,
                                 cognitive_data: Optional[CognitiveData] = None) -> IntegratedAgentAttributes:
        """整合生成智能体属性"""
        # 如果没有提供具体数据，随机选择
        if micro_data is None and self.micro_data:
            micro_data = random.choice(self.micro_data)
        
        if cognitive_data is None and self.cognitive_data:
            cognitive_data = random.choice(self.cognitive_data)
        
        # 生成个人属性
        personal_attrs = self._generate_personal_attributes(agent_id, micro_data)
        
        # 生成社交属性
        social_attrs = self._generate_social_attributes(micro_data, cognitive_data)
        
        # 生成经济属性
        economic_attrs = self._generate_economic_attributes(micro_data)
        
        # 确定策略表型
        strategy_phenotype = self._determine_strategy_phenotype(cognitive_data)
        
        # 计算风险感知
        risk_perception = self._calculate_risk_perception(cognitive_data)
        
        return IntegratedAgentAttributes(
            personal=personal_attrs,
            social=social_attrs,
            economic=economic_attrs,
            strategy_phenotype=strategy_phenotype,
            risk_perception=risk_perception
        )
    
    def _generate_personal_attributes(self, agent_id: str, 
                                    micro_data: Optional[MicroData]) -> PersonalAttributes:
        """生成个人属性"""
        if micro_data:
            age = micro_data.age
            gender = Gender.MALE if micro_data.gender == "男" else Gender.FEMALE
            education = self.education_mapping.get(micro_data.education, EducationLevel.HIGH)
            occupation = self.occupation_mapping.get(micro_data.occupation, OccupationType.OTHER)
            
            # 基于健康状况映射
            health_mapping = {
                "很好": HealthStatus.EXCELLENT,
                "好": HealthStatus.GOOD,
                "一般": HealthStatus.FAIR,
                "不好": HealthStatus.POOR,
                "很不好": HealthStatus.CRITICAL
            }
            health_status = health_mapping.get(micro_data.health_status, HealthStatus.FAIR)
            
            chronic_diseases = micro_data.chronic_diseases
        else:
            # 使用宏观数据分布生成
            age = self._sample_from_age_distribution()
            gender = Gender.MALE if random.random() < 0.52 else Gender.FEMALE
            education = self._sample_from_education_distribution()
            occupation = random.choice(list(OccupationType))
            health_status = random.choice(list(HealthStatus))
            chronic_diseases = []
        
        return PersonalAttributes(
            agent_id=agent_id,
            age=age,
            gender=gender,
            education_level=education,
            occupation=occupation,
            health_status=health_status,
            chronic_diseases=chronic_diseases,
            risk_tolerance=random.uniform(0.2, 0.8),
            anxiety_level=random.uniform(0.3, 0.7),
            optimism_level=random.uniform(0.3, 0.8),
            trust_in_authority=random.uniform(0.4, 0.8),
            information_processing_ability=random.uniform(0.3, 0.9),
            decision_making_speed=random.uniform(0.3, 0.8),
            learning_ability=random.uniform(0.4, 0.9),
            disaster_experience=random.randint(0, 5),
            evacuation_experience=random.randint(0, 3),
            local_knowledge=random.uniform(0.3, 0.9)
        )
    
    def _generate_social_attributes(self, micro_data: Optional[MicroData],
                                  cognitive_data: Optional[CognitiveData]) -> SocialAttributes:
        """生成社交属性"""
        if cognitive_data:
            social_support_received = cognitive_data.help_seeking_behavior.get("家人", 0.8)
            social_support_provided = cognitive_data.reciprocity_norm
            community_attachment = cognitive_data.family_orientation
            cooperation_tendency = 1 - cognitive_data.individualism_collectivism
        else:
            social_support_received = random.uniform(0.4, 0.9)
            social_support_provided = random.uniform(0.3, 0.8)
            community_attachment = random.uniform(0.5, 0.9)
            cooperation_tendency = random.uniform(0.4, 0.8)
        
        if micro_data:
            network_size_factor = min(1.0, micro_data.social_network_size / 30.0)
        else:
            network_size_factor = random.uniform(0.3, 0.8)
        
        return SocialAttributes(
            social_status=random.uniform(0.3, 0.8),
            community_influence=random.uniform(0.2, 0.7),
            leadership_ability=random.uniform(0.2, 0.7),
            extroversion=random.uniform(0.3, 0.8),
            social_skills=network_size_factor,
            empathy_level=random.uniform(0.4, 0.8),
            cooperation_tendency=cooperation_tendency,
            community_attachment=community_attachment,
            cultural_identity=random.uniform(0.6, 0.9),
            social_support_received=social_support_received,
            social_support_provided=social_support_provided
        )
    
    def _generate_economic_attributes(self, micro_data: Optional[MicroData]) -> EconomicAttributes:
        """生成经济属性"""
        if micro_data:
            monthly_income = micro_data.personal_income / 12
            total_assets = sum(micro_data.assets.values())
            liquid_assets = micro_data.assets.get("金融资产", 0)
            real_estate_value = micro_data.assets.get("房产", 0)
            total_debt = micro_data.debts
            
            # 基于收入确定收入等级
            if monthly_income < 3000:
                income_level = IncomeLevel.LOW
            elif monthly_income < 6000:
                income_level = IncomeLevel.MEDIUM_LOW
            elif monthly_income < 10000:
                income_level = IncomeLevel.MEDIUM
            elif monthly_income < 15000:
                income_level = IncomeLevel.MEDIUM_HIGH
            else:
                income_level = IncomeLevel.HIGH
            
            # 住房类型映射
            housing_mapping = {
                "自有住房": HousingType.OWN,
                "租赁住房": HousingType.RENT,
                "其他": HousingType.FAMILY
            }
            housing_type = housing_mapping.get(micro_data.housing_type, HousingType.RENT)
        else:
            # 基于宏观数据分布生成
            income_level = self._sample_from_income_distribution()
            monthly_income = self._get_income_from_level(income_level)
            total_assets = monthly_income * random.uniform(20, 100)
            liquid_assets = total_assets * random.uniform(0.1, 0.3)
            real_estate_value = total_assets * random.uniform(0.5, 0.8)
            total_debt = monthly_income * random.uniform(0, 15)
            housing_type = random.choice(list(HousingType))
        
        return EconomicAttributes(
            income_level=income_level,
            monthly_income=monthly_income,
            income_stability=random.uniform(0.5, 0.9),
            total_assets=total_assets,
            liquid_assets=liquid_assets,
            real_estate_value=real_estate_value,
            housing_type=housing_type,
            housing_cost_ratio=random.uniform(0.2, 0.5),
            housing_quality=random.uniform(0.4, 0.9),
            transport_budget=monthly_income * random.uniform(0.05, 0.15),
            has_health_insurance=random.random() > 0.2,
            has_property_insurance=random.random() > 0.4,
            has_life_insurance=random.random() > 0.5,
            total_debt=total_debt,
            debt_to_income_ratio=total_debt / max(monthly_income * 12, 1),
            savings_rate=random.uniform(0.05, 0.25),
            emergency_fund=monthly_income * random.uniform(1, 6)
        )
    
    def _determine_strategy_phenotype(self, cognitive_data: Optional[CognitiveData]) -> str:
        """确定策略表型"""
        if not cognitive_data:
            return random.choice(["strict", "moderate", "universalist"])
        
        # 基于认知数据确定策略
        family_orientation = cognitive_data.family_orientation
        in_group_favoritism = cognitive_data.in_group_favoritism
        individualism_score = cognitive_data.individualism_collectivism
        
        # 强差序格局：高家庭取向 + 高内群体偏好 + 低个人主义
        if (family_orientation > 0.8 and 
            in_group_favoritism > 0.7 and 
            individualism_score < -0.2):
            return "strict"
        
        # 普遍主义：低内群体偏好 + 高个人主义
        elif (in_group_favoritism < 0.5 and 
              individualism_score > 0.3):
            return "universalist"
        
        # 弱差序格局：中等水平
        else:
            return "moderate"
    
    def _calculate_risk_perception(self, cognitive_data: Optional[CognitiveData]) -> float:
        """计算风险感知"""
        if not cognitive_data:
            return random.uniform(0.3, 0.8)
        
        # 综合各类风险感知
        risk_scores = list(cognitive_data.risk_perception.values())
        return sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
    
    def _sample_from_age_distribution(self) -> int:
        """从年龄分布中采样"""
        if not self.macro_data:
            return random.randint(18, 80)
        
        rand = random.random()
        if rand < self.macro_data.age_distribution.get("0-14", 0.12):
            return random.randint(18, 25)  # 调整为成年人
        elif rand < self.macro_data.age_distribution.get("15-59", 0.73):
            return random.randint(25, 59)
        else:
            return random.randint(60, 80)
    
    def _sample_from_education_distribution(self) -> EducationLevel:
        """从教育分布中采样"""
        if not self.macro_data:
            return random.choice(list(EducationLevel))
        
        rand = random.random()
        cumulative = 0.0
        
        for edu_str, prob in self.macro_data.education_levels.items():
            cumulative += prob
            if rand < cumulative:
                return self.education_mapping.get(edu_str, EducationLevel.HIGH)
        
        return EducationLevel.HIGH
    
    def _sample_from_income_distribution(self) -> IncomeLevel:
        """从收入分布中采样"""
        if not self.macro_data:
            return random.choice(list(IncomeLevel))
        
        rand = random.random()
        cumulative = 0.0
        
        income_mapping = {
            "低收入": IncomeLevel.LOW,
            "中低收入": IncomeLevel.MEDIUM_LOW,
            "中等收入": IncomeLevel.MEDIUM,
            "中高收入": IncomeLevel.MEDIUM_HIGH,
            "高收入": IncomeLevel.HIGH
        }
        
        for income_str, prob in self.macro_data.income_distribution.items():
            cumulative += prob
            if rand < cumulative:
                return income_mapping.get(income_str, IncomeLevel.MEDIUM)
        
        return IncomeLevel.MEDIUM
    
    def _get_income_from_level(self, income_level: IncomeLevel) -> float:
        """根据收入等级获取具体收入"""
        income_ranges = {
            IncomeLevel.VERY_LOW: (1000, 2000),
            IncomeLevel.LOW: (2000, 4000),
            IncomeLevel.MEDIUM_LOW: (4000, 6000),
            IncomeLevel.MEDIUM: (6000, 10000),
            IncomeLevel.MEDIUM_HIGH: (10000, 15000),
            IncomeLevel.HIGH: (15000, 25000),
            IncomeLevel.VERY_HIGH: (25000, 50000)
        }
        
        min_income, max_income = income_ranges.get(income_level, (5000, 8000))
        return random.uniform(min_income, max_income)
    
    def _save_macro_data_to_db(self, macro_data: MacroData, source: DataSource):
        """保存宏观数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # 保存各类宏观指标
        data_entries = [
            ("population", macro_data.total_population, "demographic", timestamp),
            ("population_density", macro_data.population_density, "demographic", timestamp),
            ("gdp_per_capita", macro_data.gdp_per_capita, "economic", timestamp),
            ("average_income", macro_data.average_income, "economic", timestamp),
            ("unemployment_rate", macro_data.unemployment_rate, "economic", timestamp),
            ("healthcare_coverage", macro_data.healthcare_coverage, "social", timestamp),
            ("homeownership_rate", macro_data.homeownership_rate, "housing", timestamp)
        ]
        
        cursor.executemany(
            "INSERT INTO macro_data (data_type, value, category, timestamp) VALUES (?, ?, ?, ?)",
            data_entries
        )
        
        conn.commit()
        conn.close()
    
    def _save_micro_data_to_db(self, micro_data_list: List[MicroData], source: DataSource):
        """保存微观数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_entries = []
        for micro_data in micro_data_list:
            data_entries.append((
                micro_data.individual_id,
                micro_data.family_id,
                micro_data.age,
                micro_data.gender,
                micro_data.education,
                micro_data.occupation,
                micro_data.personal_income,
                micro_data.family_income,
                micro_data.health_status,
                micro_data.housing_type,
                micro_data.location_type,
                source.value
            ))
        
        cursor.executemany(
            """INSERT OR REPLACE INTO micro_data 
               (individual_id, family_id, age, gender, education, occupation, 
                personal_income, family_income, health_status, housing_type, 
                location_type, data_source) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data_entries
        )
        
        conn.commit()
        conn.close()
    
    def _save_cognitive_data_to_db(self, cognitive_data_list: List[CognitiveData], source: DataSource):
        """保存认知数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        data_entries = []
        for i, cognitive_data in enumerate(cognitive_data_list):
            data_entries.append((
                f"{source.value}_{i:04d}",
                cognitive_data.risk_perception.get("自然灾害", 0.5),
                cognitive_data.trust_in_government,
                cognitive_data.trust_in_neighbors,
                cognitive_data.family_orientation,
                cognitive_data.in_group_favoritism,
                cognitive_data.reciprocity_norm,
                source.value
            ))
        
        cursor.executemany(
            """INSERT OR REPLACE INTO cognitive_data 
               (individual_id, risk_perception, trust_in_government, trust_in_neighbors, 
                family_orientation, in_group_favoritism, reciprocity_norm, data_source) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            data_entries
        )
        
        conn.commit()
        conn.close()
    
    def generate_agent_population(self, size: int = 1000) -> List[IntegratedAgentAttributes]:
        """生成智能体群体"""
        # 确保有足够的数据
        if not self.macro_data:
            self.load_macro_data()
        
        if len(self.micro_data) < size:
            self.load_micro_data(DataSource.CFPS_2020, size)
        
        if len(self.cognitive_data) < size:
            self.load_cognitive_data(DataSource.CGSS_2021, size)
        
        # 生成智能体群体
        agent_population = []
        for i in range(size):
            agent_id = f"agent_{i:04d}"
            agent_attrs = self.integrate_agent_attributes(agent_id)
            agent_population.append(agent_attrs)
        
        return agent_population
    
    def export_integrated_data(self, filepath: str, agent_population: List[IntegratedAgentAttributes]):
        """导出整合数据"""
        export_data = {
            'macro_data': self.macro_data.__dict__ if self.macro_data else {},
            'agent_population': [agent.to_dict() for agent in agent_population],
            'data_sources': {
                'macro': 'guangzhou_yearbook',
                'micro': 'cfps_2020_2022',
                'cognitive': 'cgss_2021'
            },
            'generation_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据整合系统
    data_system = DataIntegrationSystem()
    
    print("=== 数据整合系统测试 ===")
    
    # 加载各层数据
    macro_data = data_system.load_macro_data()
    print(f"宏观数据加载完成: 人口{macro_data.total_population}, GDP人均{macro_data.gdp_per_capita}")
    
    micro_data = data_system.load_micro_data(DataSource.CFPS_2020, 100)
    print(f"微观数据加载完成: {len(micro_data)}个样本")
    
    cognitive_data = data_system.load_cognitive_data(DataSource.CGSS_2021, 80)
    print(f"认知数据加载完成: {len(cognitive_data)}个样本")
    
    # 生成智能体群体
    agent_population = data_system.generate_agent_population(50)
    print(f"\n=== 智能体群体生成 ===")
    print(f"生成智能体数量: {len(agent_population)}")
    
    # 统计策略表型分布
    strategy_counts = {}
    for agent in agent_population:
        strategy = agent.strategy_phenotype
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"策略表型分布: {strategy_counts}")
    
    # 统计脆弱性分布
    vulnerabilities = [agent.get_overall_vulnerability() for agent in agent_population]
    print(f"平均脆弱性: {np.mean(vulnerabilities):.3f}")
    print(f"脆弱性标准差: {np.std(vulnerabilities):.3f}")
    
    # 导出数据
    data_system.export_integrated_data("integrated_agent_data.json", agent_population)
    print(f"\n数据已导出到 integrated_agent_data.json")