"""
数据收集器模块 - 负责收集和处理仿真数据
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

# 导入仿真监测器
from simulation_monitor import get_monitor_instance

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataCollector:
    """数据收集器类，用于收集和处理仿真数据"""
    
    def __init__(self):
        """初始化数据收集器"""
        self.monitor = get_monitor_instance()
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 初始化数据存储
        self.collected_data = {
            "agent_data": [],
            "flood_data": [],
            "network_data": [],
            "event_data": []
        }
        
        # 初始化地理数据
        self.geo_data = self._load_geo_data()
        
    def _load_geo_data(self) -> Dict[str, Any]:
        """
        加载地理数据
        
        Returns:
            Dict[str, Any]: 地理数据
        """
        # 这里应该从实际的地理数据文件加载数据
        # 由于没有实际的地理数据文件，这里使用模拟数据
        
        # 广州市中心坐标
        center_lat = 23.13
        center_lng = 113.26
        
        # 生成网格点
        grid_size = 20
        grid_step = 0.01
        
        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                lat = center_lat - grid_step * grid_size / 2 + i * grid_step
                lng = center_lng - grid_step * grid_size / 2 + j * grid_step
                grid_points.append({
                    "id": i * grid_size + j,
                    "lat": lat,
                    "lng": lng,
                    "type": "grid_point"
                })
        
        # 生成区域
        areas = []
        for i in range(5):
            area_lat = center_lat - 0.05 + i * 0.025
            area_lng = center_lng - 0.05 + i * 0.025
            areas.append({
                "id": i,
                "name": f"区域{i+1}",
                "lat": area_lat,
                "lng": area_lng,
                "radius": 0.03,
                "type": "area"
            })
        
        # 生成疏散点
        evacuation_points = []
        for i in range(3):
            evac_lat = center_lat - 0.08 + i * 0.08
            evac_lng = center_lng - 0.08 + i * 0.08
            evacuation_points.append({
                "id": i,
                "name": f"疏散点{i+1}",
                "lat": evac_lat,
                "lng": evac_lng,
                "capacity": 1000,
                "type": "evacuation_point"
            })
        
        return {
            "grid_points": grid_points,
            "areas": areas,
            "evacuation_points": evacuation_points,
            "center": {"lat": center_lat, "lng": center_lng}
        }
    
    def collect_simulation_data(self) -> Dict[str, Any]:
        """
        收集仿真数据
        
        Returns:
            Dict[str, Any]: 收集到的仿真数据
        """
        # 获取仿真状态
        status = self.monitor.get_status()
        
        # 获取详细统计数据
        statistics = self.monitor.get_statistics()
        
        # 收集智能体数据
        agent_data = self._collect_agent_data(status)
        
        # 收集洪水数据
        flood_data = self._collect_flood_data(status)
        
        # 收集社交网络数据
        network_data = self._collect_network_data(status)
        
        # 收集事件数据
        event_data = self._collect_event_data(status)
        
        # 更新数据存储
        self.collected_data["agent_data"].append(agent_data)
        self.collected_data["flood_data"].append(flood_data)
        self.collected_data["network_data"].append(network_data)
        self.collected_data["event_data"].append(event_data)
        
        # 返回收集到的数据
        return {
            "status": status,
            "statistics": statistics,
            "agent_data": agent_data,
            "flood_data": flood_data,
            "network_data": network_data,
            "event_data": event_data,
            "geo_data": self.geo_data
        }
    
    def _collect_agent_data(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集智能体数据
        
        Args:
            status: 仿真状态
            
        Returns:
            Dict[str, Any]: 智能体数据
        """
        # 获取当前步骤
        current_step = status["current_step"]
        
        # 如果当前步骤超出了模拟数据范围，返回空数据
        if current_step >= len(self.monitor.simulation_data["steps"]):
            return {
                "step": current_step,
                "timestamp": datetime.now().isoformat(),
                "agents": []
            }
        
        # 获取当前步骤的智能体状态
        step_data = self.monitor.simulation_data["steps"][current_step]
        agent_status = step_data["agent_status"]
        
        # 生成智能体数据
        agents = []
        agent_id = 0
        
        # 正常状态的智能体
        for _ in range(agent_status["normal"]):
            # 随机生成位置（在广州市中心附近）
            lat = self.geo_data["center"]["lat"] + (np.random.random() - 0.5) * 0.1
            lng = self.geo_data["center"]["lng"] + (np.random.random() - 0.5) * 0.1
            
            agents.append({
                "id": agent_id,
                "status": "normal",
                "position": {"lat": lat, "lng": lng},
                "risk_level": np.random.randint(0, 3),  # 0-2的风险等级
                "has_vehicle": np.random.random() > 0.5,
                "family_size": np.random.randint(1, 6),
                "age": np.random.randint(18, 80),
                "evacuation_willingness": np.random.random()
            })
            agent_id += 1
        
        # 疏散中的智能体
        for _ in range(agent_status["evacuating"]):
            # 随机生成位置（向疏散点移动）
            evac_point = np.random.choice(self.geo_data["evacuation_points"])
            direction = np.random.random()  # 0-1之间的移动进度
            
            # 起点（随机位置）
            start_lat = self.geo_data["center"]["lat"] + (np.random.random() - 0.5) * 0.1
            start_lng = self.geo_data["center"]["lng"] + (np.random.random() - 0.5) * 0.1
            
            # 计算当前位置（起点和疏散点之间）
            lat = start_lat + (evac_point["lat"] - start_lat) * direction
            lng = start_lng + (evac_point["lng"] - start_lng) * direction
            
            agents.append({
                "id": agent_id,
                "status": "evacuating",
                "position": {"lat": lat, "lng": lng},
                "risk_level": np.random.randint(2, 5),  # 2-4的风险等级
                "has_vehicle": np.random.random() > 0.3,
                "family_size": np.random.randint(1, 6),
                "age": np.random.randint(18, 80),
                "evacuation_willingness": 0.7 + np.random.random() * 0.3,  # 较高的疏散意愿
                "destination": {
                    "id": evac_point["id"],
                    "name": evac_point["name"],
                    "lat": evac_point["lat"],
                    "lng": evac_point["lng"]
                }
            })
            agent_id += 1
        
        # 受困的智能体
        for _ in range(agent_status["trapped"]):
            # 随机生成位置（在高风险区域）
            area = np.random.choice(self.geo_data["areas"])
            
            # 在区域内随机生成位置
            angle = np.random.random() * 2 * np.pi
            distance = np.random.random() * area["radius"]
            lat = area["lat"] + distance * np.cos(angle)
            lng = area["lng"] + distance * np.sin(angle)
            
            agents.append({
                "id": agent_id,
                "status": "trapped",
                "position": {"lat": lat, "lng": lng},
                "risk_level": np.random.randint(4, 6),  # 4-5的风险等级（最高）
                "has_vehicle": np.random.random() > 0.7,  # 大多数没有车辆
                "family_size": np.random.randint(1, 6),
                "age": np.random.randint(18, 80),
                "evacuation_willingness": np.random.random() * 0.5,  # 较低的疏散意愿
                "trapped_time": np.random.randint(1, current_step + 1)  # 被困时间
            })
            agent_id += 1
        
        # 已救援的智能体
        for _ in range(agent_status["rescued"]):
            # 随机选择一个疏散点
            evac_point = np.random.choice(self.geo_data["evacuation_points"])
            
            # 在疏散点附近随机生成位置
            lat = evac_point["lat"] + (np.random.random() - 0.5) * 0.01
            lng = evac_point["lng"] + (np.random.random() - 0.5) * 0.01
            
            agents.append({
                "id": agent_id,
                "status": "rescued",
                "position": {"lat": lat, "lng": lng},
                "risk_level": np.random.randint(0, 3),  # 0-2的风险等级（已安全）
                "has_vehicle": np.random.random() > 0.5,
                "family_size": np.random.randint(1, 6),
                "age": np.random.randint(18, 80),
                "evacuation_willingness": 0.8 + np.random.random() * 0.2,  # 高疏散意愿
                "rescue_time": np.random.randint(1, current_step + 1)  # 被救援时间
            })
            agent_id += 1
        
        return {
            "step": current_step,
            "timestamp": datetime.now().isoformat(),
            "agents": agents
        }
    
    def _collect_flood_data(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集洪水数据
        
        Args:
            status: 仿真状态
            
        Returns:
            Dict[str, Any]: 洪水数据
        """
        # 获取当前步骤
        current_step = status["current_step"]
        
        # 如果当前步骤超出了模拟数据范围，返回空数据
        if current_step >= len(self.monitor.simulation_data["steps"]):
            return {
                "step": current_step,
                "timestamp": datetime.now().isoformat(),
                "water_levels": []
            }
        
        # 获取当前步骤的水位数据
        step_data = self.monitor.simulation_data["steps"][current_step]
        water_level = step_data["water_level"]
        
        # 生成网格点的水位数据
        water_levels = []
        
        for point in self.geo_data["grid_points"]:
            # 基础水位（使用当前步骤的平均水位）
            base_level = water_level["avg"]
            
            # 添加随机波动
            variation = (np.random.random() - 0.5) * 0.5
            
            # 根据位置调整水位（距离中心越远，水位越低）
            distance = np.sqrt(
                (point["lat"] - self.geo_data["center"]["lat"]) ** 2 +
                (point["lng"] - self.geo_data["center"]["lng"]) ** 2
            )
            position_factor = max(0, 1 - distance * 10)  # 距离因子
            
            # 计算最终水位
            final_level = base_level * position_factor + variation
            final_level = max(0, final_level)  # 确保水位非负
            
            water_levels.append({
                "grid_id": point["id"],
                "lat": point["lat"],
                "lng": point["lng"],
                "water_level": final_level,
                "flow_speed": np.random.random() * 2,  # 0-2 m/s的流速
                "flow_direction": np.random.randint(0, 360)  # 0-359度的流向
            })
        
        return {
            "step": current_step,
            "timestamp": datetime.now().isoformat(),
            "max_water_level": water_level["max"],
            "avg_water_level": water_level["avg"],
            "min_water_level": water_level["min"],
            "change_rate": water_level["change_rate"],
            "affected_area": step_data["affected_area"],
            "water_levels": water_levels
        }
    
    def _collect_network_data(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集社交网络数据
        
        Args:
            status: 仿真状态
            
        Returns:
            Dict[str, Any]: 社交网络数据
        """
        # 获取当前步骤
        current_step = status["current_step"]
        
        # 如果当前步骤超出了模拟数据范围，返回空数据
        if current_step >= len(self.monitor.simulation_data["steps"]):
            return {
                "step": current_step,
                "timestamp": datetime.now().isoformat(),
                "nodes": [],
                "links": []
            }
        
        # 获取当前步骤的互助行为数据
        step_data = self.monitor.simulation_data["steps"][current_step]
        aid_actions = step_data["aid_actions"]
        
        # 生成节点数据
        nodes = []
        node_count = 50  # 节点数量
        
        for i in range(node_count):
            # 随机分配群体类型
            group_type = np.random.choice(["family", "neighbor", "colleague", "classmate"])
            
            # 根据群体类型设置属性
            if group_type == "family":
                size = np.random.randint(2, 6)
                strength = 0.7 + np.random.random() * 0.3  # 较高的关系强度
            elif group_type == "neighbor":
                size = np.random.randint(1, 4)
                strength = 0.4 + np.random.random() * 0.4
            elif group_type == "colleague":
                size = np.random.randint(1, 3)
                strength = 0.3 + np.random.random() * 0.4
            else:  # classmate
                size = np.random.randint(1, 3)
                strength = 0.2 + np.random.random() * 0.5
            
            nodes.append({
                "id": i,
                "group": group_type,
                "size": size,
                "strength": strength,
                "aid_given": np.random.randint(0, 10),
                "aid_received": np.random.randint(0, 10)
            })
        
        # 生成连接数据
        links = []
        link_count = int(node_count * 1.5)  # 连接数量
        
        for _ in range(link_count):
            # 随机选择两个节点
            source = np.random.randint(0, node_count)
            target = np.random.randint(0, node_count)
            
            # 避免自连接
            while target == source:
                target = np.random.randint(0, node_count)
            
            # 确定关系类型
            if np.random.random() < 0.4:
                # 40%的概率使用相同的群体类型
                relationship = nodes[source]["group"]
            else:
                # 60%的概率随机选择关系类型
                relationship = np.random.choice(["family", "neighbor", "colleague", "classmate"])
            
            # 根据关系类型设置强度
            if relationship == "family":
                strength = 0.7 + np.random.random() * 0.3
            elif relationship == "neighbor":
                strength = 0.4 + np.random.random() * 0.4
            elif relationship == "colleague":
                strength = 0.3 + np.random.random() * 0.4
            else:  # classmate
                strength = 0.2 + np.random.random() * 0.5
            
            links.append({
                "source": source,
                "target": target,
                "relationship": relationship,
                "strength": strength,
                "aid_count": np.random.randint(0, 5)
            })
        
        return {
            "step": current_step,
            "timestamp": datetime.now().isoformat(),
            "nodes": nodes,
            "links": links,
            "aid_actions": {
                "family": aid_actions["family"],
                "neighbor": aid_actions["neighbor"],
                "colleague": aid_actions["colleague"],
                "classmate": aid_actions["classmate"],
                "total": sum(aid_actions.values())
            },
            "network_density": 0.4 + 0.3 * (current_step / status["total_steps"]),
            "avg_relationship_strength": 0.5 + 0.3 * (current_step / status["total_steps"])
        }
    
    def _collect_event_data(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """
        收集事件数据
        
        Args:
            status: 仿真状态
            
        Returns:
            Dict[str, Any]: 事件数据
        """
        # 获取当前步骤
        current_step = status["current_step"]
        total_steps = status["total_steps"]
        
        # 预定义的关键事件
        key_events = [
            {"step": 0, "time": "0小时", "description": "模拟开始，初始水位设置", "type": "simulation"},
            {"step": int(total_steps * 0.1), "time": "2小时", "description": "洪水开始上涨，首批预警发出", "type": "warning"},
            {"step": int(total_steps * 0.2), "time": "4小时", "description": "部分低洼地区开始积水", "type": "flood"},
            {"step": int(total_steps * 0.3), "time": "6小时", "description": "首批居民开始自发疏散", "type": "evacuation"},
            {"step": int(total_steps * 0.4), "time": "8小时", "description": "洪水蔓延至居民区，政府启动应急预案", "type": "emergency"},
            {"step": int(total_steps * 0.5), "time": "10小时", "description": "洪水达到峰值，大规模疏散行动", "type": "flood_peak"},
            {"step": int(total_steps * 0.6), "time": "12小时", "description": "互助网络形成，社区自救行动开始", "type": "social_network"},
            {"step": int(total_steps * 0.7), "time": "14小时", "description": "救援队伍到达，开始救援被困人员", "type": "rescue"},
            {"step": int(total_steps * 0.8), "time": "16小时", "description": "洪水开始退去，部分地区恢复通行", "type": "recovery"},
            {"step": int(total_steps * 0.9), "time": "18小时", "description": "大部分居民安全疏散，救援工作继续", "type": "evacuation_complete"},
            {"step": total_steps, "time": "20小时", "description": "模拟结束，统计疏散和救援结果", "type": "simulation_end"}
        ]
        
        # 筛选当前步骤之前（含当前步骤）的事件
        current_events = [event for event in key_events if event["step"] <= current_step]
        
        # 生成随机事件（每10步生成一个）
        random_events = []
        for step in range(0, current_step + 1, 10):
            if step % 10 == 0 and step > 0:
                # 随机事件类型
                event_types = ["flood", "evacuation", "rescue", "social_aid"]
                event_type = np.random.choice(event_types)
                
                # 根据事件类型生成描述
                if event_type == "flood":
                    descriptions = [
                        "某区域水位突然上涨",
                        "暴雨加剧，洪水蔓延速度加快",
                        "某处堤坝出现裂缝，加固工作开始"
                    ]
                elif event_type == "evacuation":
                    descriptions = [
                        "某小区居民集体疏散",
                        "交通拥堵导致疏散延迟",
                        "新的疏散路线开通"
                    ]
                elif event_type == "rescue":
                    descriptions = [
                        "救援队成功解救被困群众",
                        "直升机空投救援物资",
                        "志愿者组织开展救援行动"
                    ]
                else:  # social_aid
                    descriptions = [
                        "邻里互助网络形成，共享资源",
                        "家庭成员互相帮助，共同疏散",
                        "同事之间提供临时住所"
                    ]
                
                description = np.random.choice(descriptions)
                time = f"{int(step / total_steps * 20)}小时"
                
                random_events.append({
                    "step": step,
                    "time": time,
                    "description": description,
                    "type": event_type
                })
        
        # 合并关键事件和随机事件
        all_events = current_events + random_events
        
        # 按步骤排序
        all_events.sort(key=lambda x: x["step"])
        
        return {
            "step": current_step,
            "timestamp": datetime.now().isoformat(),
            "events": all_events
        }
    
    def export_data(self, data_type: str, format_type: str) -> Dict[str, Any]:
        """
        导出数据
        
        Args:
            data_type: 数据类型（agent_data, flood_data, network_data, event_data）
            format_type: 导出格式（csv, json）
            
        Returns:
            Dict[str, Any]: 导出结果
        """
        # 检查数据类型是否有效
        if data_type not in self.collected_data:
            return {"success": False, "message": f"不支持的数据类型: {data_type}"}
        
        # 检查是否有数据
        if not self.collected_data[data_type]:
            return {"success": False, "message": f"没有可导出的{data_type}数据"}
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{data_type}_{timestamp}"
        
        # 根据格式类型导出数据
        if format_type == "json":
            filepath = os.path.join(self.data_dir, f"{filename}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.collected_data[data_type], f, ensure_ascii=False, indent=2)
        elif format_type == "csv":
            filepath = os.path.join(self.data_dir, f"{filename}.csv")
            
            # 将数据转换为DataFrame
            if data_type == "agent_data":
                # 展平智能体数据
                rows = []
                for step_data in self.collected_data[data_type]:
                    step = step_data["step"]
                    timestamp = step_data["timestamp"]
                    for agent in step_data["agents"]:
                        row = {
                            "step": step,
                            "timestamp": timestamp,
                            "agent_id": agent["id"],
                            "status": agent["status"],
                            "lat": agent["position"]["lat"],
                            "lng": agent["position"]["lng"],
                            "risk_level": agent["risk_level"],
                            "has_vehicle": agent["has_vehicle"],
                            "family_size": agent["family_size"],
                            "age": agent["age"],
                            "evacuation_willingness": agent["evacuation_willingness"]
                        }
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)
            elif data_type == "flood_data":
                # 展平洪水数据
                rows = []
                for step_data in self.collected_data[data_type]:
                    step = step_data["step"]
                    timestamp = step_data["timestamp"]
                    for water_level in step_data["water_levels"]:
                        row = {
                            "step": step,
                            "timestamp": timestamp,
                            "grid_id": water_level["grid_id"],
                            "lat": water_level["lat"],
                            "lng": water_level["lng"],
                            "water_level": water_level["water_level"],
                            "flow_speed": water_level["flow_speed"],
                            "flow_direction": water_level["flow_direction"],
                            "max_water_level": step_data["max_water_level"],
                            "avg_water_level": step_data["avg_water_level"],
                            "min_water_level": step_data["min_water_level"],
                            "affected_area": step_data["affected_area"]
                        }
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)
            elif data_type == "network_data":
                # 展平社交网络数据
                rows = []
                for step_data in self.collected_data[data_type]:
                    step = step_data["step"]
                    timestamp = step_data["timestamp"]
                    for link in step_data["links"]:
                        row = {
                            "step": step,
                            "timestamp": timestamp,
                            "source": link["source"],
                            "target": link["target"],
                            "relationship": link["relationship"],
                            "strength": link["strength"],
                            "aid_count": link["aid_count"],
                            "family_aid": step_data["aid_actions"]["family"],
                            "neighbor_aid": step_data["aid_actions"]["neighbor"],
                            "colleague_aid": step_data["aid_actions"]["colleague"],
                            "classmate_aid": step_data["aid_actions"]["classmate"],
                            "total_aid": step_data["aid_actions"]["total"],
                            "network_density": step_data["network_density"],
                            "avg_relationship_strength": step_data["avg_relationship_strength"]
                        }
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)
            elif data_type == "event_data":
                # 展平事件数据
                rows = []
                for step_data in self.collected_data[data_type]:
                    for event in step_data["events"]:
                        row = {
                            "step": event["step"],
                            "time": event["time"],
                            "description": event["description"],
                            "type": event["type"],
                            "timestamp": step_data["timestamp"]
                        }
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(filepath, index=False)
        else:
            return {"success": False, "message": f"不支持的导出格式: {format_type}"}
        
        return {
            "success": True,
            "message": f"数据已导出到: {filepath}",
            "filepath": filepath
        }
    
    def clear_data(self) -> Dict[str, Any]:
        """
        清除收集的数据
        
        Returns:
            Dict[str, Any]: 清除结果
        """
        self.collected_data = {
            "agent_data": [],
            "flood_data": [],
            "network_data": [],
            "event_data": []
        }
        
        return {"success": True, "message": "数据已清除"}


# 获取数据收集器实例
_data_collector_instance = None

def get_data_collector_instance() -> DataCollector:
    """
    获取数据收集器实例（单例模式）
    
    Returns:
        DataCollector: 数据收集器实例
    """
    global _data_collector_instance
    if _data_collector_instance is None:
        _data_collector_instance = DataCollector()
    return _data_collector_instance


if __name__ == "__main__":
    # 测试数据收集器
    collector = DataCollector()
    
    # 从模拟监测器获取状态
    monitor = get_monitor_instance()
    status = monitor.get_status()
    
    # 收集数据
    data = collector.collect_simulation_data()
    
    # 打印收集到的数据
    print(f"收集到的数据: {len(data['agent_data']['agents'])} 个智能体, "
          f"{len(data['flood_data']['water_levels'])} 个水位点, "
          f"{len(data['network_data']['nodes'])} 个网络节点, "
          f"{len(data['event_data']['events'])} 个事件")
    
    # 导出数据
    export_result = collector.export_data("agent_data", "json")
    print(f"导出结果: {export_result}")
