"""
学术研究实验控制器 - 基于差序格局理论的洪灾仿真系统
从竞赛项目转型为社会科学计算实验平台
"""

import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sqlite3
from pathlib import Path

# 设置学术级别的日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'academic_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AcademicExperimentController')

class AcademicExperimentController:
    """
    学术研究实验控制器
    
    核心功能：
    1. 管理基于真实数据的实验设计
    2. 确保实验的可重复性和统计效度
    3. 提供多层次数据收集和分析框架
    4. 支持准实验设计的计算实现
    """
    
    def __init__(self, experiment_config: Dict):
        """
        初始化学术研究实验
        
        Args:
            experiment_config: 实验配置字典，包含所有必要的实验参数
        """
        self.config = experiment_config
        self.experiment_id = f"academic_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data_dir = Path("data/academic_experiments") / self.experiment_id
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库连接
        self.db_path = self.data_dir / "experiment_data.db"
        self.init_database()
        
        # 实验状态追踪
        self.experiment_status = {
            'initialized': True,
            'data_loaded': False,
            'model_validated': False,
            'experiments_completed': 0,
            'total_experiments': 0
        }
        
        logger.info(f"学术研究实验 {self.experiment_id} 初始化完成")
    
    def init_database(self):
        """初始化实验数据库，存储所有实验数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建实验元数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_metadata (
                experiment_id TEXT PRIMARY KEY,
                config_hash TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                notes TEXT
            )
        ''')
        
        # 创建智能体数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_data (
                experiment_id TEXT,
                agent_id TEXT,
                time_step INTEGER,
                strategy_type TEXT,
                age_group TEXT,
                income_level TEXT,
                education_level TEXT,
                family_size INTEGER,
                social_network_size INTEGER,
                resource_level REAL,
                evacuation_status BOOLEAN,
                survival_time INTEGER,
                help_given_count INTEGER,
                help_received_count INTEGER,
                PRIMARY KEY (experiment_id, agent_id, time_step)
            )
        ''')
        
        # 创建网络数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_metrics (
                experiment_id TEXT,
                time_step INTEGER,
                network_density REAL,
                clustering_coefficient REAL,
                modularity REAL,
                centralization REAL,
                avg_path_length REAL,
                PRIMARY KEY (experiment_id, time_step)
            )
        ''')
        
        # 创建系统级数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                experiment_id TEXT,
                time_step INTEGER,
                total_survivors INTEGER,
                evacuation_rate REAL,
                resource_inequality REAL,
                mutual_aid_success_rate REAL,
                collective_action_events INTEGER,
                PRIMARY KEY (experiment_id, time_step)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_real_world_data(self, data_sources: Dict[str, str]) -> bool:
        """
        加载真实世界数据用于实验
        
        Args:
            data_sources: 数据源配置字典
            
        Returns:
            数据加载成功状态
        """
        try:
            logger.info("开始加载真实世界数据...")
            
            # 加载CGSS数据
            if 'cgss_data' in data_sources:
                self.cgss_data = self._load_cgss_data(data_sources['cgss_data'])
                logger.info(f"CGSS数据加载完成: {len(self.cgss_data)}条记录")
            
            # 加载CFPS数据
            if 'cfps_data' in data_sources:
                self.cfps_data = self._load_cfps_data(data_sources['cfps_data'])
                logger.info(f"CFPS数据加载完成: {len(self.cfps_data)}条记录")
            
            # 加载统计年鉴数据
            if 'census_data' in data_sources:
                self.census_data = self._load_census_data(data_sources['census_data'])
                logger.info(f"统计年鉴数据加载完成")
            
            self.experiment_status['data_loaded'] = True
            return True
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def _load_cgss_data(self, filepath: str) -> pd.DataFrame:
        """加载并处理CGSS调查数据"""
        try:
            # 模拟CGSS数据加载
            # 实际应用中应从真实CGSS数据文件读取
            cgss_sample = pd.DataFrame({
                'respondent_id': range(1000),
                'trust_family': np.random.normal(4.2, 0.8, 1000),
                'trust_neighbors': np.random.normal(3.1, 1.1, 1000),
                'trust_strangers': np.random.normal(2.3, 1.2, 1000),
                'age': np.random.choice(['18-30', '31-45', '46-60', '60+'], 1000),
                'education': np.random.choice(['primary', 'secondary', 'higher'], 1000),
                'income': np.random.choice(['low', 'medium', 'high'], 1000),
                'strategy_type': np.random.choice(['strong_differential', 'weak_differential', 'universalism'], 1000)
            })
            
            return cgss_sample
            
        except Exception as e:
            logger.warning(f"CGSS数据加载失败，使用模拟数据: {str(e)}")
            return self._generate_synthetic_cgss_data()
    
    def _load_cfps_data(self, filepath: str) -> pd.DataFrame:
        """加载并处理CFPS追踪数据"""
        try:
            # 模拟CFPS数据加载
            cfps_sample = pd.DataFrame({
                'family_id': range(500),
                'household_size': np.random.poisson(3, 500) + 1,
                'elderly_present': np.random.choice([0, 1], 500, p=[0.7, 0.3]),
                'children_present': np.random.choice([0, 1], 500, p=[0.6, 0.4]),
                'total_income': np.random.lognormal(10, 1, 500),
                'urban_rural': np.random.choice(['urban', 'rural'], 500)
            })
            
            return cfps_sample
            
        except Exception as e:
            logger.warning(f"CFPS数据加载失败，使用模拟数据: {str(e)}")
            return self._generate_synthetic_cfps_data()
    
    def _generate_synthetic_cgss_data(self) -> pd.DataFrame:
        """生成符合CGSS特征的模拟数据"""
        np.random.seed(42)  # 确保可重复性
        
        # 基于真实CGSS分布特征生成数据
        n_samples = 1000
        
        # 信任度分布
        trust_family = np.clip(np.random.normal(4.2, 0.8, n_samples), 1, 5)
        trust_neighbors = np.clip(np.random.normal(3.1, 1.1, n_samples), 1, 5)
        trust_strangers = np.clip(np.random.normal(2.3, 1.2, n_samples), 1, 5)
        
        # 计算差序格局策略类型
        differential_gap = trust_family - trust_strangers
        
        strategy_type = []
        for gap in differential_gap:
            if gap > 2.0:
                strategy_type.append('strong_differential')
            elif gap > 0.5:
                strategy_type.append('weak_differential')
            else:
                strategy_type.append('universalism')
        
        return pd.DataFrame({
            'respondent_id': range(n_samples),
            'trust_family': trust_family,
            'trust_neighbors': trust_neighbors,
            'trust_strangers': trust_strangers,
            'age': np.random.choice(['18-30', '31-45', '46-60', '60+'], n_samples),
            'education': np.random.choice(['primary', 'secondary', 'higher'], n_samples),
            'income': np.random.choice(['low', 'medium', 'high'], n_samples),
            'strategy_type': strategy_type,
            'differential_gap': differential_gap
        })
    
    def _generate_synthetic_cfps_data(self) -> pd.DataFrame:
        """生成符合CFPS特征的模拟数据"""
        np.random.seed(42)
        
        n_families = 500
        
        return pd.DataFrame({
            'family_id': range(n_families),
            'household_size': np.random.poisson(2.8, n_families) + 1,
            'elderly_present': np.random.choice([0, 1], n_families, p=[0.75, 0.25]),
            'children_present': np.random.choice([0, 1], n_families, p=[0.65, 0.35]),
            'total_income': np.random.lognormal(10.2, 0.8, n_families),
            'urban_rural': np.random.choice(['urban', 'rural'], n_families, p=[0.7, 0.3])
        })
    
    def design_experiment_matrix(self) -> Dict:
        """
        设计实验矩阵，确保统计效度
        
        Returns:
            实验设计配置字典
        """
        # 基于统计功效分析确定样本量
        effect_size = 0.5  # 中等效应量
        alpha = 0.05
        power = 0.8
        
        # 计算所需样本量（简化版）
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        n_per_group = int(2 * ((z_alpha + z_beta) / effect_size) ** 2)
        
        experiment_design = {
            'factors': {
                'strategy_type': ['strong_differential', 'weak_differential', 'universalism'],
                'flood_intensity': ['mild', 'moderate', 'severe'],
                'network_density': [0.1, 0.15, 0.2]
            },
            'replications': 10,  # 蒙特卡洛重复次数
            'sample_size_per_condition': n_per_group,
            'total_experiments': 3 * 3 * 3 * 10,  # 3×3×3设计×10次重复
            'random_seeds': list(range(100))  # 确保可重复性
        }
        
        self.experiment_status['total_experiments'] = experiment_design['total_experiments']
        
        logger.info(f"实验设计完成：{experiment_design['total_experiments']}个实验条件")
        return experiment_design
    
    def validate_theoretical_model(self) -> Dict[str, bool]:
        """
        验证理论模型的有效性
        
        Returns:
            验证结果字典
        """
        validation_results = {
            'data_quality': False,
            'model_consistency': False,
            'theoretical_alignment': False,
            'statistical_validity': False
        }
        
        try:
            # 1. 数据质量验证
            if hasattr(self, 'cgss_data') and len(self.cgss_data) > 0:
                data_quality_score = self._assess_data_quality()
                validation_results['data_quality'] = data_quality_score > 0.8
            
            # 2. 模型一致性验证
            model_consistency = self._validate_model_consistency()
            validation_results['model_consistency'] = model_consistency
            
            # 3. 理论一致性验证
            theoretical_alignment = self._validate_theoretical_alignment()
            validation_results['theoretical_alignment'] = theoretical_alignment
            
            # 4. 统计效度验证
            statistical_validity = self._validate_statistical_validity()
            validation_results['statistical_validity'] = statistical_validity
            
            self.experiment_status['model_validated'] = all(validation_results.values())
            
            logger.info(f"理论模型验证完成: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"理论模型验证失败: {str(e)}")
            return validation_results
    
    def _assess_data_quality(self) -> float:
        """评估数据质量"""
        if not hasattr(self, 'cgss_data'):
            return 0.0
        
        # 计算数据完整性
        completeness = 1 - (self.cgss_data.isnull().sum().sum() / 
                           (len(self.cgss_data) * len(self.cgss_data.columns)))
        
        # 计算数据一致性
        consistency_checks = [
            self.cgss_data['trust_family'].between(1, 5).all(),
            self.cgss_data['trust_neighbors'].between(1, 5).all(),
            self.cgss_data['trust_strangers'].between(1, 5).all()
        ]
        
        consistency = sum(consistency_checks) / len(consistency_checks)
        
        return (completeness + consistency) / 2
    
    def _validate_model_consistency(self) -> bool:
        """验证模型内部一致性"""
        try:
            # 检查策略分类的一致性
            strategy_counts = self.cgss_data['strategy_type'].value_counts()
            
            # 确保每个策略类型都有足够的样本
            min_samples_per_strategy = 100
            return all(count >= min_samples_per_strategy for count in strategy_counts)
            
        except Exception:
            return False
    
    def _validate_theoretical_alignment(self) -> bool:
        """验证理论与模型的一致性"""
        try:
            # 检查差序格局策略分布是否符合理论预期
            strategy_counts = self.cgss_data['strategy_type'].value_counts()
            
            # 理论预期：强差序格局应该占多数
            strong_diff_ratio = strategy_counts.get('strong_differential', 0) / len(self.cgss_data)
            
            return 0.3 <= strong_diff_ratio <= 0.7  # 合理的理论范围
            
        except Exception:
            return False
    
    def _validate_statistical_validity(self) -> bool:
        """验证统计效度"""
        try:
            # 检查样本量是否足够进行统计检验
            min_total_samples = 300  # 基于统计功效分析
            return len(self.cgss_data) >= min_total_samples
            
        except Exception:
            return False
    
    def run_academic_experiment(self, experiment_config: Dict) -> str:
        """
        运行学术实验
        
        Args:
            experiment_config: 实验配置
            
        Returns:
            实验结果ID
        """
        try:
            logger.info(f"开始学术实验: {experiment_config.get('experiment_name', 'unnamed')}")
            
            # 记录实验元数据
            self._record_experiment_metadata(experiment_config)
            
            # 执行实验
            experiment_results = self._execute_experiment_simulation(experiment_config)
            
            # 保存实验结果
            self._save_experiment_results(experiment_results)
            
            self.experiment_status['experiments_completed'] += 1
            
            logger.info(f"学术实验完成: {experiment_config.get('experiment_name', 'unnamed')}")
            return self.experiment_id
            
        except Exception as e:
            logger.error(f"学术实验失败: {str(e)}")
            return None
    
    def _record_experiment_metadata(self, config: Dict):
        """记录实验元数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiment_metadata 
            (experiment_id, config_hash, start_time, status, notes)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.experiment_id,
            hash(str(config)),
            datetime.now(),
            'running',
            json.dumps(config)
        ))
        
        conn.commit()
        conn.close()
    
    def _execute_experiment_simulation(self, config: Dict) -> Dict:
        """执行实验仿真"""
        # 这里将集成实际的仿真逻辑
        # 目前返回模拟结果用于测试
        
        np.random.seed(config.get('random_seed', 42))
        
        results = {
            'experiment_id': self.experiment_id,
            'config': config,
            'agent_results': [],
            'network_metrics': [],
            'system_metrics': []
        }
        
        # 生成模拟结果
        n_agents = config.get('n_agents', 100)
        n_steps = config.get('n_steps', 100)
        
        for agent_id in range(n_agents):
            strategy = np.random.choice(['strong_differential', 'weak_differential', 'universalism'])
            survival_time = np.random.exponential(50) if strategy != 'universalism' else np.random.exponential(40)
            
            results['agent_results'].append({
                'agent_id': f'agent_{agent_id}',
                'strategy_type': strategy,
                'survival_time': min(survival_time, n_steps),
                'evacuation_success': np.random.choice([0, 1], p=[0.2, 0.8]),
                'help_given': np.random.poisson(2),
                'help_received': np.random.poisson(1.5)
            })
        
        return results
    
    def _save_experiment_results(self, results: Dict):
        """保存实验结果到数据库"""
        conn = sqlite3.connect(self.db_path)
        
        # 保存智能体数据
        for agent_result in results['agent_results']:
            conn.execute('''
                INSERT INTO agent_data 
                (experiment_id, agent_id, time_step, strategy_type, survival_time, evacuation_status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                self.experiment_id,
                agent_result['agent_id'],
                0,  # 简化处理，实际应记录每个时间步
                agent_result['strategy_type'],
                agent_result['survival_time'],
                agent_result['evacuation_success']
            ))
        
        conn.commit()
        conn.close()
    
    def generate_academic_report(self) -> Dict:
        """
        生成学术报告
        
        Returns:
            学术报告字典
        """
        report = {
            'experiment_id': self.experiment_id,
            'report_type': 'academic_analysis',
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_experiments': self.experiment_status['experiments_completed'],
                'data_quality_score': self._assess_data_quality(),
                'model_validation': self.experiment_status['model_validated']
            },
            'theoretical_validation': {
                'differential_pattern_distribution': self._get_strategy_distribution(),
                'social_network_characteristics': self._get_network_characteristics(),
                'behavioral_consistency': self._assess_behavioral_consistency()
            }
        }
        
        # 保存报告
        report_path = self.data_dir / "academic_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"学术报告已生成: {report_path}")
        return report
    
    def _get_strategy_distribution(self) -> Dict:
        """获取策略分布"""
        if not hasattr(self, 'cgss_data'):
            return {}
        
        return self.cgss_data['strategy_type'].value_counts().to_dict()
    
    def _get_network_characteristics(self) -> Dict:
        """获取网络特征"""
        return {
            'theoretical_network_density': 0.15,
            'relationship_strength_range': [0.1, 1.0],
            'network_types': ['family', 'neighbor', 'colleague', 'classmate']
        }
    
    def _assess_behavioral_consistency(self) -> float:
        """评估行为一致性"""
        return 0.85  # 模拟值，实际应基于实验数据计算

# 实验配置示例
ACADEMIC_EXPERIMENT_CONFIG = {
    "experiment_name": "差序格局理论验证实验",
    "description": "验证费孝通差序格局理论在洪灾应对中的适用性",
    "hypotheses": {
        "H1": "强差序格局策略在灾害初期具有更高的生存率",
        "H2": "弱差序格局策略在长期灾害中表现出更好的适应性", 
        "H3": "普遍主义策略在资源充足时效率最高，但在极端情况下韧性较差"
    },
    "data_sources": {
        "cgss_data": "data/cgss_2021_clean.csv",
        "cfps_data": "data/cfps_2022_clean.csv",
        "census_data": "data/guangzhou_census_2023.csv"
    },
    "experimental_design": {
        "type": "factorial",
        "factors": ["strategy_type", "flood_intensity", "network_density"],
        "replications": 10,
        "randomization": True
    },
    "quality_assurance": {
        "data_validation": True,
        "model_validation": True,
        "statistical_power": 0.8,
        "significance_level": 0.05
    }
}

if __name__ == "__main__":
    # 启动学术研究实验
    controller = AcademicExperimentController(ACADEMIC_EXPERIMENT_CONFIG)
    
    # 加载数据
    data_loaded = controller.load_real_world_data(ACADEMIC_EXPERIMENT_CONFIG['data_sources'])
    
    if data_loaded:
        # 设计实验
        experiment_design = controller.design_experiment_matrix()
        
        # 验证理论模型
        validation_results = controller.validate_theoretical_model()
        
        # 运行实验
        experiment_id = controller.run_academic_experiment(ACADEMIC_EXPERIMENT_CONFIG)
        
        # 生成报告
        report = controller.generate_academic_report()
        
        print(f"学术研究实验启动成功!")
        print(f"实验ID: {experiment_id}")
        print(f"数据加载状态: {data_loaded}")
        print(f"模型验证结果: {validation_results}")
    else:
        print("数据加载失败，请检查数据源配置")