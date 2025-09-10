#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强数据记录器 - 支持OLS回归悖论分析
基于差序格局理论的智能体行为数据收集系统
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyGroup(Enum):
    """社会行为策略类型"""
    STRICT = "strict"        # 强差序格局
    MODERATE = "moderate"    # 弱差序格局
    UNIVERSALIST = "universalist"  # 普遍主义

class ActionType(Enum):
    """互助行为类型"""
    C_CONSERVATIVE = "C"  # 保守型救助
    D_DAILY = "D"         # 日常型分享
    S_SACRIFICE = "S"     # 犧牲型奉献

@dataclass
class AgentRecord:
    """智能体记录数据结构"""
    # 基础标识信息
    agent_id: str
    strategy_group: StrategyGroup
    timestep: int
    
    # 资源状态
    current_resources: float
    initial_resources: float
    survival_status: int  # 0=死亡, 1=存活
    
    # 行为计数 (当前步骤)
    c_help_given: int = 0
    c_help_received: int = 0
    d_share_given: int = 0
    d_share_received: int = 0
    s_sacrifice_given: int = 0
    s_sacrifice_received: int = 0
    
    # 累积统计
    total_c_given: int = 0
    total_c_received: int = 0
    total_d_given: int = 0
    total_d_received: int = 0
    total_s_given: int = 0
    total_s_received: int = 0
    
    # 网络属性
    network_degree: int = 0
    strong_ties_count: int = 0
    weak_ties_count: int = 0
    
    # 动态过程变量
    poverty_duration: int = 0  # 贫困状态持续时间(<50)
    critical_duration: int = 0  # 危险状态持续时间(<20)
    resource_transitions: int = 0  # 资源状态转换次数
    help_request_count: int = 0  # 求助请求次数
    help_success_rate: float = 0.0  # 求助成功率
    
    # 位置信息
    position_lat: float = 0.0
    position_lng: float = 0.0
    
    # 其他属性
    age: int = 30
    has_vehicle: bool = False
    family_size: int = 1
    risk_level: int = 0

@dataclass
class StateTransition:
    """状态转换记录"""
    agent_id: str
    timestep: int
    from_state: str
    to_state: str
    trigger_event: str
    resource_before: float
    resource_after: float

class EnhancedAgentDataLogger:
    """增强智能体数据记录器"""
    
    def __init__(self, output_dir: str = "results/regression_data"):
        """初始化数据记录器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据存储
        self.agent_records: List[AgentRecord] = []
        self.state_transitions: List[StateTransition] = []
        self.interaction_log: List[Dict[str, Any]] = []
        
        # 状态追踪
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.resource_thresholds = {
            'poverty_line': 50.0,
            'critical_line': 20.0,
            'death_line': 0.0
        }
        
        logger.info(f"增强数据记录器初始化完成，输出目录: {output_dir}")
    
    def initialize_agent(self, agent_id: str, family_group: str, 
                        initial_resources: float, **kwargs) -> None:
        """初始化智能体记录"""
        # 转换策略群体类型（保持参数名兼容性）
        try:
            group_enum = StrategyGroup(family_group)
        except ValueError:
            group_enum = StrategyGroup.UNIVERSALIST
            logger.warning(f"未知策略群体类型: {family_group}, 设置为普遍主义")
        
        # 初始化状态追踪
        self.agent_states[agent_id] = {
            'strategy_group': group_enum,
            'initial_resources': initial_resources,
            'current_resources': initial_resources,
            'survival_status': 1,
            'last_state': self._get_resource_state(initial_resources),
            'state_durations': {'rich': 0, 'poor': 0, 'critical': 0, 'dead': 0},
            'cumulative_actions': {
                'c_given': 0, 'c_received': 0,
                'd_given': 0, 'd_received': 0,
                's_given': 0, 's_received': 0
            },
            'help_requests': {'total': 0, 'successful': 0},
            'transition_count': 0,
            **kwargs
        }
        
        logger.debug(f"智能体 {agent_id} ({family_group}) 初始化完成")
    
    def record_timestep(self, agent_id: str, timestep: int, 
                       current_resources: float, position: Tuple[float, float],
                       **kwargs) -> None:
        """记录时间步数据"""
        if agent_id not in self.agent_states:
            logger.error(f"智能体 {agent_id} 未初始化")
            return
        
        agent_state = self.agent_states[agent_id]
        
        # 更新资源状态
        old_resources = agent_state['current_resources']
        agent_state['current_resources'] = current_resources
        
        # 检查状态转换
        old_state = agent_state['last_state']
        new_state = self._get_resource_state(current_resources)
        
        if old_state != new_state:
            self._record_state_transition(
                agent_id, timestep, old_state, new_state,
                f"资源变化: {old_resources:.1f} -> {current_resources:.1f}",
                old_resources, current_resources
            )
            agent_state['last_state'] = new_state
            agent_state['transition_count'] += 1
        
        # 更新状态持续时间
        agent_state['state_durations'][new_state] += 1
        
        # 更新生存状态
        survival_status = 1 if current_resources > 0 else 0
        agent_state['survival_status'] = survival_status
        
        # 创建记录
        record = AgentRecord(
            agent_id=agent_id,
            strategy_group=agent_state['strategy_group'],
            timestep=timestep,
            current_resources=current_resources,
            initial_resources=agent_state['initial_resources'],
            survival_status=survival_status,
            
            # 累积统计
            total_c_given=agent_state['cumulative_actions']['c_given'],
            total_c_received=agent_state['cumulative_actions']['c_received'],
            total_d_given=agent_state['cumulative_actions']['d_given'],
            total_d_received=agent_state['cumulative_actions']['d_received'],
            total_s_given=agent_state['cumulative_actions']['s_given'],
            total_s_received=agent_state['cumulative_actions']['s_received'],
            
            # 动态过程变量
            poverty_duration=agent_state['state_durations']['poor'],
            critical_duration=agent_state['state_durations']['critical'],
            resource_transitions=agent_state['transition_count'],
            help_request_count=agent_state['help_requests']['total'],
            help_success_rate=(
                agent_state['help_requests']['successful'] / 
                max(1, agent_state['help_requests']['total'])
            ),
            
            # 位置信息
            position_lat=position[0],
            position_lng=position[1],
            
            # 其他属性
            **{k: v for k, v in kwargs.items() if hasattr(AgentRecord, k)}
        )
        
        self.agent_records.append(record)
    
    def record_interaction(self, giver_id: str, receiver_id: str, 
                          action_type: str, amount: float, timestep: int,
                          success: bool = True, **kwargs) -> None:
        """记录互助交互"""
        # 验证行为类型
        try:
            action_enum = ActionType(action_type)
        except ValueError:
            logger.error(f"未知行为类型: {action_type}")
            return
        
        # 更新给予者统计
        if giver_id in self.agent_states and success:
            giver_state = self.agent_states[giver_id]
            action_key = f"{action_type.lower()}_given"
            giver_state['cumulative_actions'][action_key] += 1
        
        # 更新接受者统计
        if receiver_id in self.agent_states and success:
            receiver_state = self.agent_states[receiver_id]
            action_key = f"{action_type.lower()}_received"
            receiver_state['cumulative_actions'][action_key] += 1
        
        # 记录交互日志
        interaction = {
            'timestep': timestep,
            'giver_id': giver_id,
            'receiver_id': receiver_id,
            'action_type': action_type,
            'amount': amount,
            'success': success,
            'giver_group': self.agent_states.get(giver_id, {}).get('strategy_group', 'unknown'),
            'receiver_group': self.agent_states.get(receiver_id, {}).get('strategy_group', 'unknown'),
            **kwargs
        }
        
        self.interaction_log.append(interaction)
        
        logger.debug(f"记录交互: {giver_id} -> {receiver_id}, {action_type}, {amount}")
    
    def record_help_request(self, requester_id: str, timestep: int, 
                           success: bool, **kwargs) -> None:
        """记录求助请求"""
        if requester_id in self.agent_states:
            state = self.agent_states[requester_id]
            state['help_requests']['total'] += 1
            if success:
                state['help_requests']['successful'] += 1
    
    def _get_resource_state(self, resources: float) -> str:
        """获取资源状态"""
        if resources <= self.resource_thresholds['death_line']:
            return 'dead'
        elif resources <= self.resource_thresholds['critical_line']:
            return 'critical'
        elif resources <= self.resource_thresholds['poverty_line']:
            return 'poor'
        else:
            return 'rich'
    
    def _record_state_transition(self, agent_id: str, timestep: int,
                                from_state: str, to_state: str,
                                trigger_event: str, resource_before: float,
                                resource_after: float) -> None:
        """记录状态转换"""
        transition = StateTransition(
            agent_id=agent_id,
            timestep=timestep,
            from_state=from_state,
            to_state=to_state,
            trigger_event=trigger_event,
            resource_before=resource_before,
            resource_after=resource_after
        )
        
        self.state_transitions.append(transition)
        logger.debug(f"状态转换: {agent_id} {from_state} -> {to_state} @ {timestep}")
    
    def get_regression_dataset(self) -> pd.DataFrame:
        """获取回归分析数据集"""
        if not self.agent_records:
            logger.warning("没有可用的智能体记录")
            return pd.DataFrame()
        
        # 转换为DataFrame
        records_dict = [asdict(record) for record in self.agent_records]
        df = pd.DataFrame(records_dict)
        
        # 转换枚举类型为字符串
        df['strategy_group'] = df['strategy_group'].apply(
            lambda x: x.value if hasattr(x, 'value') else str(x)
        )
        
        # 计算衍生变量
        df['survival_time'] = df.groupby('agent_id')['timestep'].transform('max') + 1
        df['final_resources'] = df.groupby('agent_id')['current_resources'].transform('last')
        df['avg_resources'] = df['final_resources'] / df['survival_time']
        df['resource_change_rate'] = (
            (df['current_resources'] - df['initial_resources']) / 
            df['initial_resources'].replace(0, 1)
        )
        
        # 添加二元变量
        df['received_c_help'] = (df['total_c_received'] > 0).astype(int)
        df['received_d_help'] = (df['total_d_received'] > 0).astype(int)
        df['gave_c_help'] = (df['total_c_given'] > 0).astype(int)
        df['gave_d_help'] = (df['total_d_given'] > 0).astype(int)
        
        logger.info(f"生成回归数据集: {len(df)} 条记录, {df['agent_id'].nunique()} 个智能体")
        return df
    
    def get_final_agent_summary(self) -> pd.DataFrame:
        """获取智能体最终状态摘要"""
        df = self.get_regression_dataset()
        if df.empty:
            return pd.DataFrame()
        
        # 获取每个智能体的最终状态
        final_df = df.groupby('agent_id').last().reset_index()
        
        # 添加额外统计
        interaction_stats = self._get_interaction_statistics()
        if interaction_stats is not None and not interaction_stats.empty:
            final_df = final_df.merge(
                interaction_stats, on='agent_id', how='left'
            ).fillna(0)
        
        logger.info(f"生成最终摘要: {len(final_df)} 个智能体")
        return final_df
    
    def _get_interaction_statistics(self) -> Optional[pd.DataFrame]:
        """获取交互统计"""
        if not self.interaction_log:
            return None
        
        df = pd.DataFrame(self.interaction_log)
        
        # 按智能体统计给予行为
        giver_stats = df.groupby('giver_id').agg({
            'amount': ['sum', 'mean', 'count'],
            'success': 'mean'
        }).round(3)
        giver_stats.columns = ['total_given_amount', 'avg_given_amount', 
                              'total_given_count', 'give_success_rate']
        giver_stats = giver_stats.reset_index().rename(columns={'giver_id': 'agent_id'})
        
        # 按智能体统计接受行为
        receiver_stats = df.groupby('receiver_id').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(3)
        receiver_stats.columns = ['total_received_amount', 'avg_received_amount', 
                                 'total_received_count']
        receiver_stats = receiver_stats.reset_index().rename(columns={'receiver_id': 'agent_id'})
        
        # 合并统计
        stats = giver_stats.merge(receiver_stats, on='agent_id', how='outer').fillna(0)
        
        return stats
    
    def export_data(self, filename_prefix: str = "regression_data") -> Dict[str, str]:
        """导出数据到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出回归数据集
        regression_df = self.get_regression_dataset()
        regression_file = os.path.join(
            self.output_dir, f"{filename_prefix}_full_{timestamp}.csv"
        )
        regression_df.to_csv(regression_file, index=False, encoding='utf-8')
        
        # 导出最终摘要
        summary_df = self.get_final_agent_summary()
        summary_file = os.path.join(
            self.output_dir, f"{filename_prefix}_summary_{timestamp}.csv"
        )
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        
        # 导出状态转换数据
        if self.state_transitions:
            transitions_df = pd.DataFrame([asdict(t) for t in self.state_transitions])
            transitions_file = os.path.join(
                self.output_dir, f"{filename_prefix}_transitions_{timestamp}.csv"
            )
            transitions_df.to_csv(transitions_file, index=False, encoding='utf-8')
        
        # 导出交互日志
        if self.interaction_log:
            interactions_df = pd.DataFrame(self.interaction_log)
            interactions_file = os.path.join(
                self.output_dir, f"{filename_prefix}_interactions_{timestamp}.csv"
            )
            interactions_df.to_csv(interactions_file, index=False, encoding='utf-8')
        
        result = {
            'regression_data': regression_file,
            'summary_data': summary_file,
            'transitions_data': transitions_file if self.state_transitions else None,
            'interactions_data': interactions_file if self.interaction_log else None,
            'total_records': len(self.agent_records),
            'total_agents': len(self.agent_states),
            'export_timestamp': timestamp
        }
        
        logger.info(f"数据导出完成: {result}")
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        df = self.get_regression_dataset()
        
        if df.empty:
            return {'error': '没有可用数据'}
        
        stats = {
            'basic_info': {
                'total_records': len(df),
                'unique_agents': df['agent_id'].nunique(),
                'max_timestep': df['timestep'].max(),
                'strategy_groups': df['strategy_group'].value_counts().to_dict()
            },
            'survival_stats': {
                'survival_rate': df.groupby('agent_id')['survival_status'].last().mean(),
                'avg_survival_time': df.groupby('agent_id')['timestep'].max().mean(),
                'survival_by_group': df.groupby(['agent_id', 'strategy_group'])['survival_status'].last().groupby('strategy_group').mean().to_dict()
            },
            'resource_stats': {
                'initial_resources': {
                    'mean': df['initial_resources'].mean(),
                    'std': df['initial_resources'].std(),
                    'min': df['initial_resources'].min(),
                    'max': df['initial_resources'].max()
                },
                'final_resources': {
                    'mean': df.groupby('agent_id')['current_resources'].last().mean(),
                    'std': df.groupby('agent_id')['current_resources'].last().std()
                }
            },
            'behavior_stats': {
                'total_c_actions': df['total_c_given'].sum() + df['total_c_received'].sum(),
                'total_d_actions': df['total_d_given'].sum() + df['total_d_received'].sum(),
                'total_s_actions': df['total_s_given'].sum() + df['total_s_received'].sum(),
                'avg_actions_per_agent': {
                    'c_given': df.groupby('agent_id')['total_c_given'].last().mean(),
                    'd_given': df.groupby('agent_id')['total_d_given'].last().mean(),
                    's_given': df.groupby('agent_id')['total_s_given'].last().mean()
                }
            }
        }
        
        return stats

# 全局实例
_enhanced_logger_instance = None

def get_enhanced_logger_instance(output_dir: str = "results/regression_data") -> EnhancedAgentDataLogger:
    """获取增强数据记录器实例"""
    global _enhanced_logger_instance
    if _enhanced_logger_instance is None:
        _enhanced_logger_instance = EnhancedAgentDataLogger(output_dir)
    return _enhanced_logger_instance

if __name__ == "__main__":
    # 测试代码
    logger = get_enhanced_logger_instance()
    
    # 模拟数据记录
    logger.initialize_agent("agent_001", "strict", 100.0, age=35, has_vehicle=True)
    logger.initialize_agent("agent_002", "moderate", 80.0, age=28, has_vehicle=False)
    
    # 模拟时间步记录
    for t in range(10):
        logger.record_timestep("agent_001", t, 100 - t*5, (23.13, 113.26))
        logger.record_timestep("agent_002", t, 80 - t*3, (23.14, 113.27))
        
        # 模拟交互
        if t % 3 == 0:
            logger.record_interaction("agent_001", "agent_002", "D", 10.0, t)
    
    # 获取数据
    df = logger.get_regression_dataset()
    print(f"生成数据集: {len(df)} 条记录")
    print(df.head())
    
    # 导出数据
    result = logger.export_data("test")
    print(f"导出结果: {result}")
    
    # 获取统计信息
    stats = logger.get_statistics()
    print(f"统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")