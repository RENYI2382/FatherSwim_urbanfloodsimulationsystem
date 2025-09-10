"""
Hurricane Mobility 数据处理模块
用于处理飓风数据和生成符合评估要求的输出格式
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class HurricaneDataProcessor:
    """飓风数据处理器"""
    
    def __init__(self, data_dir: str = ".agentsociety-benchmark/datasets/HurricaneMobility"):
        """
        初始化数据处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.profiles_data = None
        self.mobility_data = None
        
        logger.info(f"初始化数据处理器，数据目录: {data_dir}")
    
    def load_profiles(self) -> Dict[str, Any]:
        """
        加载用户档案数据
        
        Returns:
            Dict: 用户档案数据
        """
        try:
            profiles_path = self.data_dir / "profiles.json"
            with open(profiles_path, 'r', encoding='utf-8') as f:
                self.profiles_data = json.load(f)
            
            logger.info(f"成功加载 {len(self.profiles_data)} 个用户档案")
            return self.profiles_data
            
        except Exception as e:
            logger.error(f"加载用户档案失败: {e}")
            return {}
    
    def extract_user_features(self, user_id: str) -> Dict[str, Any]:
        """
        提取用户特征
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 用户特征
        """
        if not self.profiles_data:
            self.load_profiles()
        
        user_profile = self.profiles_data.get(user_id, {})
        
        # 提取关键特征
        features = {
            "age": user_profile.get("age", 35),
            "income": user_profile.get("income_level", "medium"),
            "family_size": user_profile.get("household_size", 2),
            "has_vehicle": user_profile.get("vehicle_access", True),
            "education": user_profile.get("education_level", "college"),
            "employment": user_profile.get("employment_status", "employed")
        }
        
        return features
    
    def calculate_baseline_mobility(self, user_id: str) -> Dict[str, float]:
        """
        计算用户基线移动性指标
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 基线移动性指标
        """
        # 这里应该从历史数据中计算，暂时使用模拟数据
        features = self.extract_user_features(user_id)
        
        # 基于用户特征估算基线指标
        base_daily_trips = 2.5
        base_travel_time = 45.0  # 分钟
        
        # 根据特征调整
        if features["employment"] == "employed":
            base_daily_trips += 1.0
            base_travel_time += 15.0
        
        if features["family_size"] > 2:
            base_daily_trips += 0.5 * (features["family_size"] - 2)
        
        if not features["has_vehicle"]:
            base_travel_time *= 1.5
        
        return {
            "daily_trips": base_daily_trips,
            "total_travel_time": base_travel_time,
            "average_trip_duration": base_travel_time / max(base_daily_trips, 1)
        }


class MobilityMetricsCalculator:
    """移动性指标计算器"""
    
    @staticmethod
    def calculate_change_rate(
        pre_hurricane: float,
        during_hurricane: float,
        post_hurricane: float
    ) -> Dict[str, float]:
        """
        计算变化率
        
        Args:
            pre_hurricane: 飓风前数值
            during_hurricane: 飓风中数值
            post_hurricane: 飓风后数值
            
        Returns:
            Dict: 变化率指标
        """
        # 避免除零错误
        pre_value = max(pre_hurricane, 0.001)
        
        during_change = (during_hurricane - pre_hurricane) / pre_value
        post_change = (post_hurricane - pre_hurricane) / pre_value
        
        return {
            "during_change_rate": during_change,
            "post_change_rate": post_change,
            "recovery_rate": (post_hurricane - during_hurricane) / max(abs(during_hurricane - pre_hurricane), 0.001)
        }
    
    @staticmethod
    def calculate_distribution_similarity(
        distribution1: List[float],
        distribution2: List[float]
    ) -> float:
        """
        计算分布相似度（余弦相似度）
        
        Args:
            distribution1: 分布1
            distribution2: 分布2
            
        Returns:
            float: 相似度 (0-1)
        """
        try:
            # 转换为numpy数组
            dist1 = np.array(distribution1)
            dist2 = np.array(distribution2)
            
            # 计算余弦相似度
            dot_product = np.dot(dist1, dist2)
            norm1 = np.linalg.norm(dist1)
            norm2 = np.linalg.norm(dist2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # 确保非负
            
        except Exception as e:
            logger.error(f"计算分布相似度失败: {e}")
            return 0.0
    
    @staticmethod
    def calculate_mape(actual: List[float], predicted: List[float]) -> float:
        """
        计算平均绝对百分比误差 (MAPE)
        
        Args:
            actual: 实际值
            predicted: 预测值
            
        Returns:
            float: MAPE值
        """
        try:
            actual_arr = np.array(actual)
            predicted_arr = np.array(predicted)
            
            # 避免除零
            actual_arr = np.where(actual_arr == 0, 0.001, actual_arr)
            
            mape = np.mean(np.abs((actual_arr - predicted_arr) / actual_arr)) * 100
            return mape
            
        except Exception as e:
            logger.error(f"计算MAPE失败: {e}")
            return 100.0  # 返回最大误差


class ResultFormatter:
    """结果格式化器"""
    
    @staticmethod
    def format_mobility_results(
        user_results: Dict[str, Any],
        phase: str
    ) -> Dict[str, Any]:
        """
        格式化移动性结果
        
        Args:
            user_results: 用户结果数据
            phase: 飓风阶段
            
        Returns:
            Dict: 格式化后的结果
        """
        formatted_result = {
            "user_id": user_results.get("user_id"),
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            "mobility_metrics": {
                "total_travel_time": user_results.get("total_travel_time", 0.0),
                "trip_count": user_results.get("trip_count", 0),
                "average_trip_duration": user_results.get("average_trip_duration", 0.0),
                "max_trip_duration": user_results.get("max_trip_duration", 0.0)
            },
            "hourly_distribution": user_results.get("hourly_distribution", [0.0] * 24),
            "risk_assessment": {
                "risk_score": user_results.get("risk_score", 0.0),
                "evacuation_decision": user_results.get("evacuation_decision", False),
                "decision_reason": user_results.get("decision_reason", "")
            }
        }
        
        return formatted_result
    
    @staticmethod
    def aggregate_results(
        individual_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        聚合个体结果
        
        Args:
            individual_results: 个体结果列表
            
        Returns:
            Dict: 聚合结果
        """
        if not individual_results:
            return {}
        
        # 按阶段分组
        phase_groups = {}
        for result in individual_results:
            phase = result.get("phase", "unknown")
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(result)
        
        # 计算聚合指标
        aggregated = {}
        for phase, results in phase_groups.items():
            total_travel_times = [r["mobility_metrics"]["total_travel_time"] for r in results]
            hourly_distributions = [r["hourly_distribution"] for r in results]
            
            # 计算平均值
            avg_travel_time = np.mean(total_travel_times) if total_travel_times else 0.0
            avg_hourly_dist = np.mean(hourly_distributions, axis=0).tolist() if hourly_distributions else [0.0] * 24
            
            aggregated[phase] = {
                "user_count": len(results),
                "average_travel_time": avg_travel_time,
                "total_travel_time": sum(total_travel_times),
                "average_hourly_distribution": avg_hourly_dist,
                "evacuation_rate": sum(1 for r in results if r["risk_assessment"]["evacuation_decision"]) / len(results)
            }
        
        return aggregated
    
    @staticmethod
    def save_results(
        results: Dict[str, Any],
        output_path: str,
        format_type: str = "json"
    ) -> bool:
        """
        保存结果到文件
        
        Args:
            results: 结果数据
            output_path: 输出路径
            format_type: 格式类型 ("json" 或 "csv")
            
        Returns:
            bool: 保存是否成功
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format_type.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            elif format_type.lower() == "csv":
                # 将结果转换为DataFrame并保存为CSV
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False, encoding='utf-8')
            else:
                logger.error(f"不支持的格式类型: {format_type}")
                return False
            
            logger.info(f"结果已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            return False


class ValidationHelper:
    """验证辅助工具"""
    
    @staticmethod
    def validate_mobility_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证移动性数据
        
        Args:
            data: 移动性数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查必要字段
        required_fields = ["user_id", "phase", "mobility_metrics", "hourly_distribution"]
        for field in required_fields:
            if field not in data:
                errors.append(f"缺少必要字段: {field}")
        
        # 检查数值范围
        if "mobility_metrics" in data:
            metrics = data["mobility_metrics"]
            if metrics.get("total_travel_time", 0) < 0:
                errors.append("总旅行时间不能为负数")
            if metrics.get("trip_count", 0) < 0:
                errors.append("出行次数不能为负数")
        
        # 检查小时分布
        if "hourly_distribution" in data:
            dist = data["hourly_distribution"]
            if len(dist) != 24:
                errors.append("小时分布必须包含24个值")
            if any(x < 0 for x in dist):
                errors.append("小时分布值不能为负数")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_results_format(results: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证结果格式
        
        Args:
            results: 结果数据
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 检查结果结构
        if not isinstance(results, dict):
            errors.append("结果必须是字典格式")
            return False, errors
        
        # 检查阶段数据
        expected_phases = ["pre", "during", "post"]
        for phase in expected_phases:
            if phase not in results:
                errors.append(f"缺少阶段数据: {phase}")
        
        return len(errors) == 0, errors