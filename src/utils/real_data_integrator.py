"""
真实数据集成模块
用于集成和分析tsinghua-fib-lab/hurricane-mobility-generation-benchmark数据集
包括columbia.pb文件分析和profiles.json数据处理
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import struct
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RealDataProfile:
    """真实数据用户配置文件"""
    id: int
    home: int
    work: int
    gender: str
    race: str
    education: str
    income: int
    consumption: str
    age: int
    
    def get_income_level(self) -> str:
        """根据收入获取收入等级"""
        if self.income < 35000:
            return "low_income"
        elif self.income > 70000:
            return "high_income"
        else:
            return "middle_income"
    
    def get_age_group(self) -> str:
        """根据年龄获取年龄组"""
        if self.age < 35:
            return "young_adult"
        elif self.age > 65:
            return "elderly"
        else:
            return "middle_age"
    
    def has_vehicle(self) -> bool:
        """根据收入和年龄推断是否有车"""
        # 基于统计数据的简单推断
        if self.age < 18:
            return False
        if self.income > 50000:
            return True
        if self.age > 75:
            return False
        return self.income > 30000

@dataclass
class LocationNode:
    """地理位置节点"""
    id: int
    lat: float
    lon: float
    type: str = "unknown"

class RealDataIntegrator:
    """真实数据集成器"""
    
    def __init__(self, dataset_path: str = None):
        """初始化数据集成器"""
        if dataset_path is None:
            self.dataset_path = Path(".agentsociety-benchmark/datasets/HurricaneMobility")
        else:
            self.dataset_path = Path(dataset_path)
        
        self.profiles_data = None
        self.location_data = None
        self.mobility_patterns = None
        
        logger.info(f"初始化真实数据集成器，数据路径: {self.dataset_path}")
    
    def load_profiles(self) -> List[RealDataProfile]:
        """加载用户配置文件"""
        profiles_file = self.dataset_path / "profiles.json"
        
        if not profiles_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {profiles_file}")
        
        logger.info(f"加载用户配置文件: {profiles_file}")
        
        with open(profiles_file, 'r', encoding='utf-8') as f:
            profiles_raw = json.load(f)
        
        profiles = []
        for profile_data in profiles_raw:
            profile = RealDataProfile(
                id=profile_data['id'],
                home=profile_data['home'],
                work=profile_data['work'],
                gender=profile_data['gender'],
                race=profile_data['race'],
                education=profile_data['education'],
                income=profile_data['income'],
                consumption=profile_data['consumption'],
                age=profile_data['age']
            )
            profiles.append(profile)
        
        self.profiles_data = profiles
        logger.info(f"成功加载 {len(profiles)} 个用户配置文件")
        return profiles
    
    def analyze_columbia_pb(self) -> Dict[str, Any]:
        """分析columbia.pb文件"""
        pb_file = self.dataset_path / "columbia.pb"
        
        if not pb_file.exists():
            raise FileNotFoundError(f"地图文件不存在: {pb_file}")
        
        logger.info(f"分析地图文件: {pb_file}")
        
        try:
            # 尝试读取protobuf文件
            with open(pb_file, 'rb') as f:
                data = f.read()
            
            # 基本文件信息
            file_size = len(data)
            
            # 尝试解析基本结构
            analysis = {
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "data_type": "protobuf",
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
            
            # 尝试识别数据模式
            # 查找重复的数据模式，可能表示节点或边
            chunk_size = 1000
            patterns = defaultdict(int)
            
            for i in range(0, min(len(data), 100000), chunk_size):
                chunk = data[i:i+chunk_size]
                # 查找可能的ID模式（4字节整数）
                for j in range(0, len(chunk)-4, 4):
                    try:
                        value = struct.unpack('<I', chunk[j:j+4])[0]
                        if 500000000 <= value <= 600000000:  # 基于profiles中的ID范围
                            patterns[value] += 1
                    except:
                        continue
            
            # 统计可能的节点数量
            potential_nodes = len([k for k, v in patterns.items() if v >= 2])
            analysis["potential_nodes"] = potential_nodes
            analysis["id_patterns_found"] = len(patterns)
            
            logger.info(f"地图文件分析完成: {analysis['file_size_mb']}MB, 潜在节点数: {potential_nodes}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析columbia.pb文件时出错: {e}")
            return {
                "error": str(e),
                "file_size": file_size,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            }
    
    def analyze_demographic_patterns(self) -> Dict[str, Any]:
        """分析人口统计学模式"""
        if self.profiles_data is None:
            self.load_profiles()
        
        logger.info("分析人口统计学模式")
        
        df = pd.DataFrame([
            {
                'id': p.id,
                'age': p.age,
                'income': p.income,
                'gender': p.gender,
                'race': p.race,
                'education': p.education,
                'consumption': p.consumption,
                'age_group': p.get_age_group(),
                'income_level': p.get_income_level(),
                'has_vehicle': p.has_vehicle()
            }
            for p in self.profiles_data
        ])
        
        analysis = {
            "total_profiles": len(df),
            "age_distribution": {
                "mean": df['age'].mean(),
                "std": df['age'].std(),
                "min": df['age'].min(),
                "max": df['age'].max(),
                "groups": df['age_group'].value_counts().to_dict()
            },
            "income_distribution": {
                "mean": df['income'].mean(),
                "std": df['income'].std(),
                "min": df['income'].min(),
                "max": df['income'].max(),
                "levels": df['income_level'].value_counts().to_dict()
            },
            "gender_distribution": df['gender'].value_counts().to_dict(),
            "race_distribution": df['race'].value_counts().to_dict(),
            "education_distribution": df['education'].value_counts().to_dict(),
            "consumption_distribution": df['consumption'].value_counts().to_dict(),
            "vehicle_ownership": df['has_vehicle'].value_counts().to_dict()
        }
        
        # 计算相关性
        numeric_cols = ['age', 'income']
        correlation_matrix = df[numeric_cols].corr().to_dict()
        analysis["correlations"] = correlation_matrix
        
        logger.info(f"人口统计学分析完成，总计 {len(df)} 个配置文件")
        
        return analysis
    
    def extract_mobility_patterns(self) -> Dict[str, Any]:
        """提取移动模式"""
        if self.profiles_data is None:
            self.load_profiles()
        
        logger.info("提取移动模式")
        
        # 分析家庭-工作地点分布
        home_locations = [p.home for p in self.profiles_data]
        work_locations = [p.work for p in self.profiles_data]
        
        # 计算距离分布（基于ID差异的简单估计）
        distances = []
        for p in self.profiles_data:
            # 简单的距离估计（实际应该使用地理坐标）
            distance_proxy = abs(p.home - p.work)
            distances.append(distance_proxy)
        
        patterns = {
            "home_work_analysis": {
                "unique_home_locations": len(set(home_locations)),
                "unique_work_locations": len(set(work_locations)),
                "avg_distance_proxy": np.mean(distances),
                "std_distance_proxy": np.std(distances)
            },
            "location_clusters": {
                "home_id_range": {
                    "min": min(home_locations),
                    "max": max(home_locations)
                },
                "work_id_range": {
                    "min": min(work_locations),
                    "max": max(work_locations)
                }
            }
        }
        
        # 按人口群体分析移动模式
        by_age_group = defaultdict(list)
        by_income_level = defaultdict(list)
        
        for p in self.profiles_data:
            distance = abs(p.home - p.work)
            by_age_group[p.get_age_group()].append(distance)
            by_income_level[p.get_income_level()].append(distance)
        
        patterns["by_demographics"] = {
            "by_age_group": {
                group: {
                    "count": len(distances),
                    "avg_distance": np.mean(distances),
                    "std_distance": np.std(distances)
                }
                for group, distances in by_age_group.items()
            },
            "by_income_level": {
                level: {
                    "count": len(distances),
                    "avg_distance": np.mean(distances),
                    "std_distance": np.std(distances)
                }
                for level, distances in by_income_level.items()
            }
        }
        
        self.mobility_patterns = patterns
        logger.info("移动模式提取完成")
        
        return patterns
    
    def generate_calibration_data(self) -> Dict[str, Any]:
        """生成校准数据"""
        if self.profiles_data is None:
            self.load_profiles()
        
        logger.info("生成校准数据")
        
        # 基于真实数据生成校准参数
        demographic_analysis = self.analyze_demographic_patterns()
        mobility_patterns = self.extract_mobility_patterns()
        
        # 生成年龄组的移动倍数
        age_multipliers = {}
        age_groups = demographic_analysis["age_distribution"]["groups"]
        
        for group, count in age_groups.items():
            if group == "young_adult":
                # 年轻人更活跃
                base_multiplier = 1.2
            elif group == "elderly":
                # 老年人较保守
                base_multiplier = 0.8
            else:
                # 中年人适中
                base_multiplier = 1.0
            
            age_multipliers[group] = {
                "base_multiplier": base_multiplier,
                "population_ratio": count / demographic_analysis["total_profiles"],
                "sample_size": count
            }
        
        # 生成收入组的移动倍数
        income_multipliers = {}
        income_levels = demographic_analysis["income_distribution"]["levels"]
        
        for level, count in income_levels.items():
            if level == "high_income":
                base_multiplier = 1.3
            elif level == "low_income":
                base_multiplier = 0.7
            else:
                base_multiplier = 1.0
            
            income_multipliers[level] = {
                "base_multiplier": base_multiplier,
                "population_ratio": count / demographic_analysis["total_profiles"],
                "sample_size": count
            }
        
        calibration_data = {
            "metadata": {
                "source": "tsinghua-fib-lab/hurricane-mobility-generation-benchmark",
                "total_profiles": demographic_analysis["total_profiles"],
                "generation_timestamp": pd.Timestamp.now().isoformat()
            },
            "demographic_multipliers": {
                "age_groups": age_multipliers,
                "income_levels": income_multipliers
            },
            "baseline_patterns": {
                "vehicle_ownership_rate": demographic_analysis.get("vehicle_ownership", {}).get(True, 0) / demographic_analysis["total_profiles"],
                "avg_age": demographic_analysis["age_distribution"]["mean"],
                "avg_income": demographic_analysis["income_distribution"]["mean"]
            },
            "mobility_characteristics": mobility_patterns
        }
        
        logger.info("校准数据生成完成")
        
        return calibration_data
    
    def save_analysis_results(self, output_dir: str = "results") -> str:
        """保存分析结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 执行所有分析
        pb_analysis = self.analyze_columbia_pb()
        demographic_analysis = self.analyze_demographic_patterns()
        mobility_patterns = self.extract_mobility_patterns()
        calibration_data = self.generate_calibration_data()
        
        # 合并所有结果
        complete_analysis = {
            "columbia_pb_analysis": pb_analysis,
            "demographic_analysis": demographic_analysis,
            "mobility_patterns": mobility_patterns,
            "calibration_data": calibration_data,
            "analysis_metadata": {
                "dataset_path": str(self.dataset_path),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "version": "1.0.0"
            }
        }
        
        # 保存结果
        output_file = output_path / "real_data_integration_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(complete_analysis, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"分析结果已保存到: {output_file}")
        
        return str(output_file)

def main():
    """主函数，用于测试数据集成功能"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 初始化数据集成器
        integrator = RealDataIntegrator()
        
        # 执行完整分析
        result_file = integrator.save_analysis_results()
        
        print(f"✅ 真实数据集成分析完成")
        print(f"📊 结果文件: {result_file}")
        
        # 显示基本统计信息
        if integrator.profiles_data:
            print(f"👥 用户配置文件数量: {len(integrator.profiles_data)}")
            
        demographic_analysis = integrator.analyze_demographic_patterns()
        print(f"📈 年龄分布: {demographic_analysis['age_distribution']['groups']}")
        print(f"💰 收入分布: {demographic_analysis['income_distribution']['levels']}")
        
    except Exception as e:
        logger.error(f"数据集成过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()