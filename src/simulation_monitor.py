"""
仿真监测模块 - 负责监控和管理仿真系统的运行状态
"""

import time
import json
import logging
import threading
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局监测器实例
_monitor_instance = None

class SimulationMonitor:
    """仿真监测类，用于跟踪和管理仿真系统的运行状态"""
    
    def __init__(self):
        """初始化仿真监测器"""
        self.simulation_id = f"sim_{int(time.time())}"
        self.status = "未开始"  # 仿真状态：未开始、运行中、已暂停、已停止、已完成
        self.current_step = 0   # 当前步骤
        self.total_steps = 100  # 总步骤数
        self.start_time = None  # 开始时间
        self.elapsed_time = 0   # 已运行时间（秒）
        self.agent_count = 0    # 智能体数量
        self.statistics = {     # 统计数据
            "water_level": {
                "max": 0.0,
                "avg": 0.0,
                "min": 0.0,
                "change_rate": 0.0
            },
            "affected_area": 0.0,
            "evacuation_rate": 0.0,
            "social_network": {
                "total_aid_actions": 0,
                "density": 0.0,
                "avg_strength": 0.0,
                "family_aid_ratio": 0.0
            }
        }
        self.simulation_data = {  # 仿真数据
            "steps": [],
            "agent_stats": [],
            "flood_data": [],
            "network_data": []
        }
        self.simulation_thread = None  # 仿真线程
        self.stop_flag = False  # 停止标志
        
        # 初始化模拟数据
        self._initialize_mock_data()
        
        # 初始化智能体统计数据
        self.agent_stats = {
            "evacuated": 0,
            "trapped": 0,
            "safe": 0,
            "in_danger": 0
        }
        
        # 初始化洪水数据
        self.flood_data = {
            "max_water_level": 0.0,
            "avg_water_level": 0.0,
            "affected_area_km2": 0.0
        }
        
        # 初始化社交网络数据
        self.social_network_data = {
            "total_aid_actions": 0,
            "family_aid_actions": 0,
            "neighbor_aid_actions": 0,
            "colleague_aid_actions": 0,
            "classmate_aid_actions": 0,
            "network_density": 0.0
        }
        
        # 历史数据
        self.history = {
            "agent_stats": [],
            "flood_data": [],
            "social_network_data": []
        }
        
    def _initialize_mock_data(self):
        """初始化模拟数据（用于演示）"""
        # 模拟智能体数量
        self.agent_count = 500
        
        # 模拟步骤数据
        for i in range(self.total_steps + 1):
            progress = i / self.total_steps
            
            # 水位数据
            max_water = 0.5 + 2.5 * self._sigmoid(progress * 2 - 1)
            avg_water = max_water * 0.6
            min_water = max_water * 0.3
            
            # 疏散率
            evacuation_rate = 100 * self._sigmoid(progress * 3 - 1.5)
            
            # 影响面积
            affected_area = 5 + 35 * self._sigmoid(progress * 2 - 1)
            
            # 互助行为
            family_aid = int(100 * self._sigmoid(progress * 3 - 1))
            neighbor_aid = int(80 * self._sigmoid(progress * 3 - 1))
            colleague_aid = int(60 * self._sigmoid(progress * 3 - 1))
            classmate_aid = int(50 * self._sigmoid(progress * 3 - 1))
            
            # 智能体状态
            normal_agents = int(self.agent_count * (1 - self._sigmoid(progress * 2)))
            evacuating_agents = int(self.agent_count * self._sigmoid(progress * 2) * 0.6)
            trapped_agents = int(self.agent_count * self._sigmoid(progress * 2) * 0.3)
            rescued_agents = int(self.agent_count * self._sigmoid(progress * 2) * 0.1)
            
            # 确保总数一致
            total = normal_agents + evacuating_agents + trapped_agents + rescued_agents
            if total != self.agent_count:
                normal_agents += (self.agent_count - total)
            
            # 添加到数据集
            self.simulation_data["steps"].append({
                "step": i,
                "water_level": {
                    "max": max_water,
                    "avg": avg_water,
                    "min": min_water,
                    "change_rate": 0.2 if i > 0 else 0.0
                },
                "affected_area": affected_area,
                "evacuation_rate": evacuation_rate,
                "agent_status": {
                    "normal": normal_agents,
                    "evacuating": evacuating_agents,
                    "trapped": trapped_agents,
                    "rescued": rescued_agents
                },
                "aid_actions": {
                    "family": family_aid,
                    "neighbor": neighbor_aid,
                    "colleague": colleague_aid,
                    "classmate": classmate_aid
                }
            })
    
    def _sigmoid(self, x):
        """S形函数，用于生成平滑的模拟数据"""
        return 1 / (1 + pow(2.71828, -5 * x))
    
    def start_simulation(self, total_steps=None) -> bool:
        """
        启动仿真
        
        Args:
            total_steps: 总步骤数，如果提供则更新总步骤数
            
        Returns:
            bool: 是否成功启动
        """
        if self.status == "运行中":
            logger.warning("仿真已经在运行中")
            return False
        
        # 如果是从暂停状态恢复，不重置步骤
        if self.status != "已暂停":
            self.current_step = 0
            self.start_time = time.time()
            
            # 如果提供了总步骤数，则更新
            if total_steps is not None:
                self.total_steps = total_steps
                
            # 重置历史数据
            self.history = {
                "agent_stats": [],
                "flood_data": [],
                "social_network_data": []
            }
        else:
            # 更新开始时间，考虑已经过的时间
            self.start_time = time.time() - self.elapsed_time
        
        self.status = "运行中"
        self.stop_flag = False
        
        # 启动仿真线程
        if self.simulation_thread is None or not self.simulation_thread.is_alive():
            self.simulation_thread = threading.Thread(target=self._run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
        
        logger.info("仿真已启动")
        return True
    
    def pause_simulation(self) -> bool:
        """
        暂停仿真
        
        Returns:
            bool: 是否成功暂停
        """
        if self.status != "运行中":
            logger.warning(f"仿真当前状态为{self.status}，无法暂停")
            return False
        
        self.status = "已暂停"
        # 记录已运行时间
        if self.start_time:
            self.elapsed_time = time.time() - self.start_time
        
        logger.info("仿真已暂停")
        return True
    
    def resume_simulation(self) -> bool:
        """
        恢复仿真
        
        Returns:
            bool: 是否成功恢复
        """
        if self.status != "已暂停":
            logger.warning(f"仿真当前状态为{self.status}，无法恢复")
            return False
        
        # 更新开始时间，考虑已经过的时间
        self.start_time = time.time() - self.elapsed_time
        self.status = "运行中"
        
        logger.info("仿真已恢复")
        return True
    
    def stop_simulation(self) -> bool:
        """
        停止仿真
        
        Returns:
            bool: 是否成功停止
        """
        if self.status not in ["运行中", "已暂停"]:
            logger.warning(f"仿真当前状态为{self.status}，无法停止")
            return False
        
        self.status = "已停止"
        self.stop_flag = True
        self.elapsed_time = 0
        
        logger.info("仿真已停止")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取仿真状态
        
        Returns:
            Dict[str, Any]: 包含仿真状态的字典
        """
        # 计算已运行时间
        if self.start_time and self.status == "运行中":
            self.elapsed_time = time.time() - self.start_time
        
        # 获取当前步骤的统计数据
        if 0 <= self.current_step < len(self.simulation_data["steps"]):
            step_data = self.simulation_data["steps"][self.current_step]
            self.statistics = {
                "water_level": step_data["water_level"],
                "affected_area": step_data["affected_area"],
                "evacuation_rate": step_data["evacuation_rate"],
                "social_network": {
                    "total_aid_actions": sum(step_data["aid_actions"].values()),
                    "density": 0.4 + 0.3 * (self.current_step / self.total_steps),
                    "avg_strength": 0.5 + 0.3 * (self.current_step / self.total_steps),
                    "family_aid_ratio": step_data["aid_actions"]["family"] / 
                                       max(1, sum(step_data["aid_actions"].values()))
                }
            }
        
        return {
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "elapsed_time": int(self.elapsed_time),
            "agent_count": self.agent_count,
            "statistics": self.statistics,
            "agent_stats": self.agent_stats,
            "flood_data": self.flood_data,
            "social_network_data": self.social_network_data
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取详细统计数据
        
        Returns:
            Dict[str, Any]: 包含详细统计数据的字典
        """
        # 更新统计数据
        status = self.get_status()
        
        # 获取时间序列数据
        time_series = {
            "steps": list(range(self.current_step + 1)),
            "water_level": {
                "max": [self.simulation_data["steps"][i]["water_level"]["max"] 
                        for i in range(self.current_step + 1)],
                "avg": [self.simulation_data["steps"][i]["water_level"]["avg"] 
                        for i in range(self.current_step + 1)],
                "min": [self.simulation_data["steps"][i]["water_level"]["min"] 
                        for i in range(self.current_step + 1)]
            },
            "affected_area": [self.simulation_data["steps"][i]["affected_area"] 
                             for i in range(self.current_step + 1)],
            "evacuation_rate": [self.simulation_data["steps"][i]["evacuation_rate"] 
                               for i in range(self.current_step + 1)],
            "agent_status": {
                "normal": [self.simulation_data["steps"][i]["agent_status"]["normal"] 
                          for i in range(self.current_step + 1)],
                "evacuating": [self.simulation_data["steps"][i]["agent_status"]["evacuating"] 
                              for i in range(self.current_step + 1)],
                "trapped": [self.simulation_data["steps"][i]["agent_status"]["trapped"] 
                           for i in range(self.current_step + 1)],
                "rescued": [self.simulation_data["steps"][i]["agent_status"]["rescued"] 
                           for i in range(self.current_step + 1)]
            },
            "aid_actions": {
                "family": [self.simulation_data["steps"][i]["aid_actions"]["family"] 
                          for i in range(self.current_step + 1)],
                "neighbor": [self.simulation_data["steps"][i]["aid_actions"]["neighbor"] 
                            for i in range(self.current_step + 1)],
                "colleague": [self.simulation_data["steps"][i]["aid_actions"]["colleague"] 
                             for i in range(self.current_step + 1)],
                "classmate": [self.simulation_data["steps"][i]["aid_actions"]["classmate"] 
                             for i in range(self.current_step + 1)]
            }
        }
        
        return {
            "success": True,
            "statistics": status["statistics"],
            "time_series": time_series
        }
    
    def get_history(self, data_type=None, start_step=None, end_step=None) -> Dict[str, Any]:
        """
        获取历史数据
        
        Args:
            data_type: 数据类型（agent_stats, flood_data, social_network_data）
            start_step: 起始步骤
            end_step: 结束步骤
            
        Returns:
            Dict[str, Any]: 包含历史数据的字典
        """
        # 如果没有指定数据类型，返回所有类型的数据
        if data_type is None:
            result = self.history
        else:
            # 如果指定了数据类型，只返回该类型的数据
            if data_type in self.history:
                result = {data_type: self.history[data_type]}
            else:
                return {"success": False, "message": f"不支持的数据类型: {data_type}"}
        
        # 如果指定了步骤范围，进行过滤
        if start_step is not None or end_step is not None:
            filtered_result = {}
            
            for key, value in result.items():
                if start_step is None:
                    start_step = 0
                if end_step is None:
                    end_step = len(value) - 1
                
                # 确保索引在有效范围内
                start_step = max(0, min(start_step, len(value) - 1))
                end_step = max(0, min(end_step, len(value) - 1))
                
                filtered_result[key] = value[start_step:end_step+1]
            
            result = filtered_result
        
        return {"success": True, "data": result}
    
    def update_step(self, step: int) -> bool:
        """
        更新当前步骤
        
        Args:
            step: 当前步骤
            
        Returns:
            bool: 是否成功更新
        """
        if step < 0 or step > self.total_steps:
            logger.warning(f"步骤 {step} 超出有效范围 [0, {self.total_steps}]")
            return False
        
        self.current_step = step
        logger.info(f"更新当前步骤: {step}")
        return True
    
    def update_agent_stats(self, stats: Dict[str, int]) -> bool:
        """
        更新智能体统计数据
        
        Args:
            stats: 智能体统计数据
            
        Returns:
            bool: 是否成功更新
        """
        self.agent_stats.update(stats)
        self.history["agent_stats"].append(self.agent_stats.copy())
        logger.info(f"更新智能体统计数据: {stats}")
        return True
    
    def update_flood_data(self, data: Dict[str, float]) -> bool:
        """
        更新洪水数据
        
        Args:
            data: 洪水数据
            
        Returns:
            bool: 是否成功更新
        """
        self.flood_data.update(data)
        self.history["flood_data"].append(self.flood_data.copy())
        logger.info(f"更新洪水数据: {data}")
        return True
    
    def update_social_network_data(self, data: Dict[str, Any]) -> bool:
        """
        更新社交网络数据
        
        Args:
            data: 社交网络数据
            
        Returns:
            bool: 是否成功更新
        """
        self.social_network_data.update(data)
        self.history["social_network_data"].append(self.social_network_data.copy())
        logger.info(f"更新社交网络数据: {data}")
        return True
    
    def _run_simulation(self):
        """仿真运行线程"""
        while self.current_step < self.total_steps and not self.stop_flag:
            if self.status == "运行中":
                # 更新当前步骤
                self.current_step += 1
                
                # 模拟仿真计算时间
                time.sleep(1)
                
                logger.info(f"仿真步骤: {self.current_step}/{self.total_steps}")
                
                # 如果达到最大步骤，标记为完成
                if self.current_step >= self.total_steps:
                    self.status = "已完成"
                    logger.info("仿真已完成")
                    break
            else:
                # 如果不是运行状态，暂停线程
                time.sleep(0.5)
        
        # 如果是因为停止标志而退出，确保状态正确
        if self.stop_flag:
            self.status = "已停止"


def get_monitor_instance() -> SimulationMonitor:
    """
    获取仿真监测器实例（单例模式）
    
    Returns:
        SimulationMonitor: 仿真监测器实例
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SimulationMonitor()
    return _monitor_instance