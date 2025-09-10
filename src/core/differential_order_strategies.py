"""差序格局理论策略实现模块

基于费孝通先生的"差序格局"理论，实现三种社会行为策略：
1. 强差序格局 (Strict Differential Order) - 内外分明，亲疏有别
2. 弱差序格局 (Moderate Differential Order) - 内外有别，兼容并蓄  
3. 普遍主义 (Universalist) - 灾难面前，人人平等

作者: 城市智能体项目组
日期: 2024-12-30
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    """社会关系类型枚举"""
    FAMILY = "family"  # 家人
    CLOSE_FRIEND = "close_friend"  # 密友
    NEIGHBOR = "neighbor"  # 邻居
    ACQUAINTANCE = "acquaintance"  # 熟人
    STRANGER = "stranger"  # 陌生人
    OUTSIDER = "outsider"  # 外族人


class ResourceType(Enum):
    """资源类型枚举"""
    FOOD = "food"  # 食物
    MONEY = "money"  # 金钱
    SHELTER = "shelter"  # 住所
    INFORMATION = "information"  # 信息
    TRANSPORTATION = "transportation"  # 交通工具
    MEDICAL = "medical"  # 医疗资源


@dataclass
class HelpRequest:
    """求助请求数据结构"""
    requester_id: str
    relationship_type: RelationshipType
    relationship_strength: float  # 0-1之间，1表示最亲密
    resource_type: ResourceType
    urgency_level: int  # 1-10，10表示最紧急
    resource_amount: float
    requester_current_resources: float
    requester_health_status: str
    context: str  # 求助背景描述


@dataclass
class HelpDecision:
    """帮助决策结果"""
    will_help: bool
    help_amount: float
    justification: str  # 决策理由
    strategy_type: str  # 使用的策略类型
    decision_factors: Dict[str, float]  # 决策因子权重


class DifferentialOrderStrategy(ABC):
    """差序格局策略抽象基类"""
    
    def __init__(self, agent_id: str, strategy_name: str):
        self.agent_id = agent_id
        self.strategy_name = strategy_name
        self.decision_history = []
        
    @abstractmethod
    def make_help_decision(self, request: HelpRequest, agent_resources: float) -> HelpDecision:
        """做出帮助决策"""
        pass
    
    @abstractmethod
    def evaluate_relationship_priority(self, relationship_type: RelationshipType, 
                                     relationship_strength: float) -> float:
        """评估关系优先级"""
        pass
    
    def record_decision(self, decision: HelpDecision, request: HelpRequest):
        """记录决策历史"""
        self.decision_history.append({
            'timestamp': len(self.decision_history),
            'decision': decision,
            'request': request
        })
        
    def get_decision_statistics(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        if not self.decision_history:
            return {}
            
        total_decisions = len(self.decision_history)
        help_decisions = sum(1 for d in self.decision_history if d['decision'].will_help)
        
        # 按关系类型统计
        relationship_stats = {}
        for rel_type in RelationshipType:
            rel_decisions = [d for d in self.decision_history 
                           if d['request'].relationship_type == rel_type]
            if rel_decisions:
                help_rate = sum(1 for d in rel_decisions if d['decision'].will_help) / len(rel_decisions)
                relationship_stats[rel_type.value] = {
                    'total': len(rel_decisions),
                    'help_rate': help_rate
                }
        
        return {
            'strategy_name': self.strategy_name,
            'total_decisions': total_decisions,
            'overall_help_rate': help_decisions / total_decisions if total_decisions > 0 else 0,
            'relationship_stats': relationship_stats
        }


class StrictDifferentialOrderStrategy(DifferentialOrderStrategy):
    """强差序格局策略：内外分明，亲疏有别
    
    特征：
    - 严格区分内圈（家人、密友）和外圈（其他人）
    - 资源和帮助仅在内圈流动
    - 对外圈持排斥和不信任态度
    - 优先保护内圈成员的利益
    """
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "强差序格局")
        # 定义内圈关系类型
        self.inner_circle = {RelationshipType.FAMILY, RelationshipType.CLOSE_FRIEND}
        # 内圈帮助阈值（更宽松）
        self.inner_help_threshold = 0.3
        # 外圈帮助阈值（极严格）
        self.outer_help_threshold = 0.9
        # 资源保留比例（为内圈预留）
        self.resource_reserve_ratio = 0.7
        
    def make_help_decision(self, request: HelpRequest, agent_resources: float) -> HelpDecision:
        """基于强差序格局原则做出帮助决策"""
        
        # 判断是否为内圈成员
        is_inner_circle = request.relationship_type in self.inner_circle
        
        # 计算关系优先级
        relationship_priority = self.evaluate_relationship_priority(
            request.relationship_type, request.relationship_strength
        )
        
        # 计算可用资源（为内圈预留资源）
        if is_inner_circle:
            available_resources = agent_resources
        else:
            available_resources = agent_resources * (1 - self.resource_reserve_ratio)
        
        # 决策逻辑
        will_help = False
        help_amount = 0.0
        justification = ""
        
        if is_inner_circle:
            # 内圈成员：优先帮助
            if relationship_priority >= self.inner_help_threshold:
                help_ratio = min(0.8, relationship_priority)  # 最多帮助80%
                help_amount = min(request.resource_amount, 
                                available_resources * help_ratio)
                will_help = help_amount > 0
                justification = f"内圈成员{request.requester_id}，关系亲密度{relationship_priority:.2f}，" \
                              f"基于血缘/友情纽带，应当优先帮助。内外有别，亲疏有度。"
            else:
                justification = f"虽为内圈成员，但关系强度{relationship_priority:.2f}不足，" \
                              f"且需为更亲密的内圈成员保留资源。"
        else:
            # 外圈成员：极少帮助
            if (relationship_priority >= self.outer_help_threshold and 
                request.urgency_level >= 9 and 
                available_resources > agent_resources * 0.8):  # 只有在资源充裕时
                help_amount = min(request.resource_amount * 0.1, 
                                available_resources * 0.1)  # 最多帮助10%
                will_help = help_amount > 0
                justification = f"外圈成员{request.requester_id}，虽非内圈，但情况极其紧急，" \
                              f"在保证内圈利益前提下给予有限帮助。"
            else:
                justification = f"外圈成员{request.requester_id}，关系疏远，" \
                              f"应优先保障内圈成员利益。内外分明是根本原则。"
        
        decision = HelpDecision(
            will_help=will_help,
            help_amount=help_amount,
            justification=justification,
            strategy_type=self.strategy_name,
            decision_factors={
                'relationship_priority': relationship_priority,
                'is_inner_circle': 1.0 if is_inner_circle else 0.0,
                'urgency_level': request.urgency_level / 10.0,
                'resource_availability': available_resources / agent_resources if agent_resources > 0 else 0
            }
        )
        
        self.record_decision(decision, request)
        return decision
    
    def evaluate_relationship_priority(self, relationship_type: RelationshipType, 
                                     relationship_strength: float) -> float:
        """评估关系优先级（强差序格局视角）"""
        # 基础权重
        base_weights = {
            RelationshipType.FAMILY: 1.0,
            RelationshipType.CLOSE_FRIEND: 0.8,
            RelationshipType.NEIGHBOR: 0.3,
            RelationshipType.ACQUAINTANCE: 0.1,
            RelationshipType.STRANGER: 0.05,
            RelationshipType.OUTSIDER: 0.01
        }
        
        base_weight = base_weights.get(relationship_type, 0.01)
        # 关系强度调节
        priority = base_weight * relationship_strength
        
        # 内圈成员额外加权
        if relationship_type in self.inner_circle:
            priority *= 1.5
            
        return min(priority, 1.0)


class ModerateDifferentialOrderStrategy(DifferentialOrderStrategy):
    """弱差序格局策略：内外有别，兼容并蓄
    
    特征：
    - 内圈享有绝对优先权
    - 满足内圈需求后，愿意有限帮助外圈
    - 相对开放和包容
    - 平衡内外关系
    """
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "弱差序格局")
        # 内圈定义更宽泛
        self.inner_circle = {RelationshipType.FAMILY, RelationshipType.CLOSE_FRIEND, 
                           RelationshipType.NEIGHBOR}
        # 帮助阈值相对宽松
        self.inner_help_threshold = 0.2
        self.outer_help_threshold = 0.5
        # 资源保留比例适中
        self.resource_reserve_ratio = 0.4
        
    def make_help_decision(self, request: HelpRequest, agent_resources: float) -> HelpDecision:
        """基于弱差序格局原则做出帮助决策"""
        
        is_inner_circle = request.relationship_type in self.inner_circle
        relationship_priority = self.evaluate_relationship_priority(
            request.relationship_type, request.relationship_strength
        )
        
        # 动态调整可用资源
        if is_inner_circle:
            available_resources = agent_resources * 0.9  # 为自己保留10%
        else:
            available_resources = agent_resources * (1 - self.resource_reserve_ratio)
        
        will_help = False
        help_amount = 0.0
        justification = ""
        
        if is_inner_circle:
            # 内圈成员：积极帮助
            if relationship_priority >= self.inner_help_threshold:
                help_ratio = min(0.6, relationship_priority * 1.2)
                help_amount = min(request.resource_amount, 
                                available_resources * help_ratio)
                will_help = help_amount > 0
                justification = f"内圈成员{request.requester_id}，关系强度{relationship_priority:.2f}，" \
                              f"应当积极帮助。内圈优先，但也要为外圈留有余地。"
            else:
                justification = f"内圈成员{request.requester_id}，但关系强度{relationship_priority:.2f}较弱，" \
                              f"需要综合考虑其他因素。"
        else:
            # 外圈成员：有条件帮助
            urgency_factor = request.urgency_level / 10.0
            combined_score = relationship_priority * 0.6 + urgency_factor * 0.4
            
            if combined_score >= self.outer_help_threshold:
                help_ratio = min(0.3, combined_score * 0.5)
                help_amount = min(request.resource_amount, 
                                available_resources * help_ratio)
                will_help = help_amount > 0
                justification = f"外圈成员{request.requester_id}，虽非内圈，但综合评分{combined_score:.2f}，" \
                              f"在满足内圈需求后给予适当帮助。兼容并蓄。"
            else:
                justification = f"外圈成员{request.requester_id}，综合评分{combined_score:.2f}不足，" \
                              f"优先保障内圈成员利益。"
        
        decision = HelpDecision(
            will_help=will_help,
            help_amount=help_amount,
            justification=justification,
            strategy_type=self.strategy_name,
            decision_factors={
                'relationship_priority': relationship_priority,
                'is_inner_circle': 1.0 if is_inner_circle else 0.0,
                'urgency_level': request.urgency_level / 10.0,
                'resource_availability': available_resources / agent_resources if agent_resources > 0 else 0,
                'combined_score': relationship_priority * 0.6 + (request.urgency_level / 10.0) * 0.4
            }
        )
        
        self.record_decision(decision, request)
        return decision
    
    def evaluate_relationship_priority(self, relationship_type: RelationshipType, 
                                     relationship_strength: float) -> float:
        """评估关系优先级（弱差序格局视角）"""
        base_weights = {
            RelationshipType.FAMILY: 1.0,
            RelationshipType.CLOSE_FRIEND: 0.8,
            RelationshipType.NEIGHBOR: 0.6,
            RelationshipType.ACQUAINTANCE: 0.4,
            RelationshipType.STRANGER: 0.2,
            RelationshipType.OUTSIDER: 0.1
        }
        
        base_weight = base_weights.get(relationship_type, 0.1)
        priority = base_weight * relationship_strength
        
        # 内圈成员适度加权
        if relationship_type in self.inner_circle:
            priority *= 1.2
            
        return min(priority, 1.0)


class UniversalistStrategy(DifferentialOrderStrategy):
    """普遍主义策略：灾难面前，人人平等
    
    特征：
    - 基于需求紧迫性和人道原则决策
    - 不区分关系亲疏
    - 资源相对平均分配
    - 强调普世价值
    """
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "普遍主义")
        # 统一的帮助阈值
        self.help_threshold = 0.4
        # 较低的资源保留比例
        self.resource_reserve_ratio = 0.2
        
    def make_help_decision(self, request: HelpRequest, agent_resources: float) -> HelpDecision:
        """基于普遍主义原则做出帮助决策"""
        
        # 可用资源（为自己保留少量）
        available_resources = agent_resources * (1 - self.resource_reserve_ratio)
        
        # 主要基于紧急程度和需求合理性
        urgency_factor = request.urgency_level / 10.0
        need_factor = min(1.0, request.resource_amount / (request.requester_current_resources + 1))
        health_factor = {'Poor': 1.0, 'Fair': 0.7, 'Good': 0.5}.get(request.requester_health_status, 0.5)
        
        # 综合评分（关系因素权重很低）
        relationship_priority = self.evaluate_relationship_priority(
            request.relationship_type, request.relationship_strength
        )
        
        combined_score = (urgency_factor * 0.4 + 
                         need_factor * 0.3 + 
                         health_factor * 0.2 + 
                         relationship_priority * 0.1)
        
        will_help = False
        help_amount = 0.0
        justification = ""
        
        if combined_score >= self.help_threshold:
            # 基于需求程度决定帮助比例
            help_ratio = min(0.5, combined_score * 0.8)
            help_amount = min(request.resource_amount, 
                            available_resources * help_ratio)
            will_help = help_amount > 0
            justification = f"求助者{request.requester_id}，综合需求评分{combined_score:.2f}，" \
                          f"基于人道主义原则，灾难面前人人平等，应当给予帮助。"
        else:
            justification = f"求助者{request.requester_id}，综合需求评分{combined_score:.2f}不足，" \
                          f"需要优先帮助更紧急的求助者。"
        
        decision = HelpDecision(
            will_help=will_help,
            help_amount=help_amount,
            justification=justification,
            strategy_type=self.strategy_name,
            decision_factors={
                'urgency_factor': urgency_factor,
                'need_factor': need_factor,
                'health_factor': health_factor,
                'relationship_priority': relationship_priority,
                'combined_score': combined_score,
                'resource_availability': available_resources / agent_resources if agent_resources > 0 else 0
            }
        )
        
        self.record_decision(decision, request)
        return decision
    
    def evaluate_relationship_priority(self, relationship_type: RelationshipType, 
                                     relationship_strength: float) -> float:
        """评估关系优先级（普遍主义视角）"""
        # 所有关系类型权重相对平等
        base_weights = {
            RelationshipType.FAMILY: 0.8,
            RelationshipType.CLOSE_FRIEND: 0.7,
            RelationshipType.NEIGHBOR: 0.6,
            RelationshipType.ACQUAINTANCE: 0.5,
            RelationshipType.STRANGER: 0.4,
            RelationshipType.OUTSIDER: 0.3
        }
        
        base_weight = base_weights.get(relationship_type, 0.3)
        priority = base_weight * relationship_strength * 0.5  # 降低关系因素影响
        
        return min(priority, 1.0)


class StrategyFactory:
    """策略工厂类"""
    
    @staticmethod
    def create_strategy(strategy_type: str, agent_id: str) -> DifferentialOrderStrategy:
        """创建策略实例"""
        strategy_map = {
            'strict': StrictDifferentialOrderStrategy,
            'moderate': ModerateDifferentialOrderStrategy,
            'universalist': UniversalistStrategy
        }
        
        strategy_class = strategy_map.get(strategy_type.lower())
        if not strategy_class:
            raise ValueError(f"未知的策略类型: {strategy_type}")
            
        return strategy_class(agent_id)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """获取可用策略列表"""
        return ['strict', 'moderate', 'universalist']


def simulate_help_scenario(strategies: List[DifferentialOrderStrategy], 
                          requests: List[HelpRequest],
                          agent_resources: Dict[str, float]) -> Dict[str, Any]:
    """模拟帮助场景，比较不同策略的表现"""
    
    results = {}
    
    for strategy in strategies:
        strategy_results = []
        current_resources = agent_resources.get(strategy.agent_id, 100.0)
        
        for request in requests:
            decision = strategy.make_help_decision(request, current_resources)
            if decision.will_help:
                current_resources -= decision.help_amount
            
            strategy_results.append({
                'request': request,
                'decision': decision,
                'remaining_resources': current_resources
            })
        
        results[strategy.strategy_name] = {
            'decisions': strategy_results,
            'final_resources': current_resources,
            'statistics': strategy.get_decision_statistics()
        }
    
    return results


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试策略
    strategies = [
        StrategyFactory.create_strategy('strict', 'agent_001'),
        StrategyFactory.create_strategy('moderate', 'agent_002'),
        StrategyFactory.create_strategy('universalist', 'agent_003')
    ]
    
    # 创建测试请求
    test_requests = [
        HelpRequest(
            requester_id='家庭成员A',
            relationship_type=RelationshipType.FAMILY,
            relationship_strength=0.9,
            resource_type=ResourceType.FOOD,
            urgency_level=8,
            resource_amount=20.0,
            requester_current_resources=5.0,
            requester_health_status='Fair',
            context='家人急需食物'
        ),
        HelpRequest(
            requester_id='邻居B',
            relationship_type=RelationshipType.NEIGHBOR,
            relationship_strength=0.6,
            resource_type=ResourceType.MONEY,
            urgency_level=6,
            resource_amount=50.0,
            requester_current_resources=10.0,
            requester_health_status='Good',
            context='邻居需要资金'
        ),
        HelpRequest(
            requester_id='陌生人C',
            relationship_type=RelationshipType.STRANGER,
            relationship_strength=0.1,
            resource_type=ResourceType.SHELTER,
            urgency_level=9,
            resource_amount=30.0,
            requester_current_resources=0.0,
            requester_health_status='Poor',
            context='陌生人急需住所'
        ),
        HelpRequest(
            requester_id='outsider_d',
            relationship_type=RelationshipType.OUTSIDER,
            relationship_strength=0.05,
            resource_type=ResourceType.MEDICAL,
            urgency_level=10,
            resource_amount=40.0,
            requester_current_resources=0.0,
            requester_health_status='Poor',
            context='外部人员急需医疗资源'
        )
    ]
    
    # 模拟场景
    agent_resources = {'agent_001': 100.0, 'agent_002': 100.0, 'agent_003': 100.0}
    results = simulate_help_scenario(strategies, test_requests, agent_resources)
    
    # 输出结果
    for strategy_name, result in results.items():
        print(f"\n=== {strategy_name} 策略结果 ===")
        print(f"最终剩余资源: {result['final_resources']:.2f}")
        print(f"总体帮助率: {result['statistics']['overall_help_rate']:.2%}")
        
        for i, decision_result in enumerate(result['decisions']):
            request = decision_result['request']
            decision = decision_result['decision']
            print(f"\n请求{i+1}: {request.requester_id}({request.relationship_type.value})")
            print(f"决策: {'帮助' if decision.will_help else '不帮助'}")
            if decision.will_help:
                print(f"帮助金额: {decision.help_amount:.2f}")
            print(f"理由: {decision.justification}")