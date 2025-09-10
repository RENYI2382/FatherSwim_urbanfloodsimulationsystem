#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真数据管理器 - 负责仿真数据的保存、读取和管理
实现每次仿真结果的保存-输出-可视化-记录留存存档功能
"""

import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimulationDataManager:
    """仿真数据管理器"""
    
    def __init__(self, base_dir: str = None):
        """初始化数据管理器"""
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'simulation_results')
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.raw_data_dir = self.base_dir / 'raw_data'
        self.processed_data_dir = self.base_dir / 'processed_data'
        self.visualizations_dir = self.base_dir / 'visualizations'
        self.archives_dir = self.base_dir / 'archives'
        
        for dir_path in [self.raw_data_dir, self.processed_data_dir, 
                        self.visualizations_dir, self.archives_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 当前仿真数据
        self.current_simulation_data = {
            'simulation_id': None,
            'start_time': None,
            'end_time': None,
            'parameters': {},
            'steps_data': [],
            'agent_trajectories': {},
            'network_evolution': [],
            'statistics': {}
        }
        
        logger.info(f"仿真数据管理器初始化完成，数据目录: {self.base_dir}")
    
    def start_new_simulation(self, parameters: Dict[str, Any]) -> str:
        """开始新的仿真记录"""
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_simulation_data = {
            'simulation_id': simulation_id,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'parameters': parameters.copy(),
            'steps_data': [],
            'agent_trajectories': {},
            'network_evolution': [],
            'statistics': {
                'total_steps': 0,
                'agent_count': parameters.get('agent_count', 0),
                'evacuation_rate': 0.0,
                'help_actions_total': 0,
                'network_metrics': []
            }
        }
        
        logger.info(f"开始新仿真记录: {simulation_id}")
        return simulation_id
    
    def record_step_data(self, step: int, step_data: Dict[str, Any]):
        """记录单步仿真数据"""
        if self.current_simulation_data['simulation_id'] is None:
            logger.warning("未开始仿真记录，无法记录步骤数据")
            return
        
        # 记录步骤数据
        step_record = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'agents_data': step_data.get('agents', []),
            'behavior_stats': step_data.get('behaviorStats', {}),
            'risk_stats': step_data.get('riskStats', {}),
            'network_stats': step_data.get('networkStats', {})
        }
        
        self.current_simulation_data['steps_data'].append(step_record)
        
        # 更新智能体轨迹
        for agent in step_data.get('agents', []):
            agent_id = agent.get('id')
            if agent_id:
                if agent_id not in self.current_simulation_data['agent_trajectories']:
                    self.current_simulation_data['agent_trajectories'][agent_id] = []
                
                self.current_simulation_data['agent_trajectories'][agent_id].append({
                    'step': step,
                    'position': agent.get('position', {}),
                    'status': agent.get('status', 'unknown'),
                    'risk': agent.get('risk', 0.0),
                    'help_count': agent.get('help_count', 0)
                })
        
        # 记录网络演化数据
        if 'networkStats' in step_data:
            network_record = {
                'step': step,
                'density': step_data['networkStats'].get('density', 0.5),
                'clustering_coefficient': step_data['networkStats'].get('clustering_coefficient', 0.5),
                'centralization': step_data['networkStats'].get('centralization', 0.5),
                'modularity': step_data['networkStats'].get('modularity', 0.5)
            }
            self.current_simulation_data['network_evolution'].append(network_record)
        
        # 更新统计信息
        self.current_simulation_data['statistics']['total_steps'] = step
        if 'behaviorStats' in step_data:
            self.current_simulation_data['statistics']['help_actions_total'] += step_data['behaviorStats'].get('helpActions', 0)
    
    def end_simulation(self) -> Optional[str]:
        """结束仿真并保存数据"""
        if self.current_simulation_data['simulation_id'] is None:
            logger.warning("没有进行中的仿真记录")
            return None
        
        self.current_simulation_data['end_time'] = datetime.now().isoformat()
        
        # 计算最终统计
        self._calculate_final_statistics()
        
        # 保存原始数据
        simulation_id = self.current_simulation_data['simulation_id']
        raw_file = self.raw_data_dir / f"{simulation_id}_raw.json"
        
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_simulation_data, f, indent=2, ensure_ascii=False)
        
        # 处理数据并保存
        processed_data = self._process_simulation_data()
        processed_file = self.processed_data_dir / f"{simulation_id}_processed.json"
        
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"仿真数据已保存: {simulation_id}")
        return simulation_id
    
    def _calculate_final_statistics(self):
        """计算最终统计数据"""
        if not self.current_simulation_data['steps_data']:
            return
        
        # 计算疏散率
        final_step = self.current_simulation_data['steps_data'][-1]
        total_agents = len(final_step.get('agents_data', []))
        evacuated_agents = sum(1 for agent in final_step.get('agents_data', []) 
                             if agent.get('evacuated', False))
        
        if total_agents > 0:
            self.current_simulation_data['statistics']['evacuation_rate'] = evacuated_agents / total_agents
        
        # 计算网络指标平均值
        if self.current_simulation_data['network_evolution']:
            network_data = self.current_simulation_data['network_evolution']
            avg_density = np.mean([d['density'] for d in network_data])
            avg_clustering = np.mean([d['clustering_coefficient'] for d in network_data])
            avg_centralization = np.mean([d['centralization'] for d in network_data])
            avg_modularity = np.mean([d['modularity'] for d in network_data])
            
            self.current_simulation_data['statistics']['network_metrics'] = {
                'avg_density': avg_density,
                'avg_clustering_coefficient': avg_clustering,
                'avg_centralization': avg_centralization,
                'avg_modularity': avg_modularity
            }
    
    def _process_simulation_data(self) -> Dict[str, Any]:
        """处理仿真数据，生成可视化所需的格式"""
        processed = {
            'simulation_id': self.current_simulation_data['simulation_id'],
            'metadata': {
                'start_time': self.current_simulation_data['start_time'],
                'end_time': self.current_simulation_data['end_time'],
                'parameters': self.current_simulation_data['parameters'],
                'statistics': self.current_simulation_data['statistics']
            },
            'time_series': {
                'timestamps': [],
                'agent_positions': {},
                'survival_rates': {},
                'network_metrics': [],
                'resource_distributions': {}
            }
        }
        
        # 处理时间序列数据
        start_time = datetime.fromisoformat(self.current_simulation_data['start_time'])
        
        for step_data in self.current_simulation_data['steps_data']:
            step = step_data['step']
            timestamp = start_time + timedelta(minutes=step * 5)  # 假设每步5分钟
            processed['time_series']['timestamps'].append(timestamp.isoformat())
            
            # 处理智能体位置
            for agent in step_data.get('agents_data', []):
                agent_id = agent.get('id')
                if agent_id:
                    if agent_id not in processed['time_series']['agent_positions']:
                        processed['time_series']['agent_positions'][agent_id] = []
                    
                    pos = agent.get('position', {})
                    processed['time_series']['agent_positions'][agent_id].append(
                        (pos.get('lat', 0), pos.get('lng', 0))
                    )
        
        # 处理网络指标
        processed['time_series']['network_metrics'] = self.current_simulation_data['network_evolution']
        
        return processed
    
    def get_latest_simulation_data(self) -> Optional[Dict[str, Any]]:
        """获取最新的仿真数据"""
        processed_files = list(self.processed_data_dir.glob("*_processed.json"))
        if not processed_files:
            return None
        
        # 按修改时间排序，获取最新的
        latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_simulation_data(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """获取指定仿真的数据"""
        processed_file = self.processed_data_dir / f"{simulation_id}_processed.json"
        
        if not processed_file.exists():
            return None
        
        with open(processed_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_simulations(self) -> List[Dict[str, Any]]:
        """列出所有仿真记录"""
        simulations = []
        
        for processed_file in self.processed_data_dir.glob("*_processed.json"):
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    simulations.append({
                        'simulation_id': data['simulation_id'],
                        'start_time': data['metadata']['start_time'],
                        'end_time': data['metadata']['end_time'],
                        'parameters': data['metadata']['parameters'],
                        'file_path': str(processed_file)
                    })
            except Exception as e:
                logger.error(f"读取仿真文件失败 {processed_file}: {e}")
        
        # 按开始时间排序
        simulations.sort(key=lambda x: x['start_time'], reverse=True)
        return simulations
    
    def archive_simulation(self, simulation_id: str) -> bool:
        """归档仿真数据"""
        try:
            raw_file = self.raw_data_dir / f"{simulation_id}_raw.json"
            processed_file = self.processed_data_dir / f"{simulation_id}_processed.json"
            
            if raw_file.exists():
                archive_raw = self.archives_dir / f"{simulation_id}_raw.json"
                raw_file.rename(archive_raw)
            
            if processed_file.exists():
                archive_processed = self.archives_dir / f"{simulation_id}_processed.json"
                processed_file.rename(archive_processed)
            
            logger.info(f"仿真数据已归档: {simulation_id}")
            return True
        
        except Exception as e:
            logger.error(f"归档仿真数据失败 {simulation_id}: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 30):
        """清理旧数据"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        for data_dir in [self.raw_data_dir, self.processed_data_dir]:
            for file_path in data_dir.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    try:
                        # 先归档再删除
                        simulation_id = file_path.stem.replace('_raw', '').replace('_processed', '')
                        self.archive_simulation(simulation_id)
                    except Exception as e:
                        logger.error(f"清理旧数据失败 {file_path}: {e}")

# 全局数据管理器实例
_data_manager_instance = None

def get_data_manager() -> SimulationDataManager:
    """获取数据管理器实例"""
    global _data_manager_instance
    if _data_manager_instance is None:
        _data_manager_instance = SimulationDataManager()
    return _data_manager_instance