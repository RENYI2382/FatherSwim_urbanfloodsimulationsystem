#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理增强智能体
集成推理模型的城市智能体，提供高级决策能力
"""

import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# 导入推理模型客户端
sys.path.append('..')
from ..reasoning_model_integration import ReasoningModelClient, ReasoningResult

logger = logging.getLogger(__name__)

@dataclass
class AgentDecision:
    """智能体决策结果"""
    decision_type: str  # 决策类型：evacuation, mobility, risk_assessment
    decision: Any       # 具体决策
    confidence: float   # 置信度
    reasoning_steps: List[str]  # 推理步骤
    reasoning_content: str      # 完整推理内容
    timestamp: float           # 决策时间戳
    agent_id: str             # 智能体ID
    
@dataclass
class AgentProfile:
    """智能体档案"""
    agent_id: str
    age: int
    gender: str
    income_level: str  # low, medium, high
    education_level: str  # primary, secondary, tertiary
    family_size: int
    has_vehicle: bool
    health_status: str  # poor, fair, good, excellent
    social_network_size: int
    risk_tolerance: float  # 0.0-1.0
    behavioral_type: str   # cohesive, moderate, universalistic
    
class ReasoningEnhancedAgent:
    """推理增强智能体
    
    特性：
    - 集成推理模型进行高级决策
    - 支持多步推理和自修正
    - 提供详细的决策解释
    - 适应不同的应急场景
    """
    
    def __init__(self, agent_profile: AgentProfile, config_path: str = None):
        self.profile = agent_profile
        self.reasoning_client = ReasoningModelClient(config_path)
        self.decision_history: List[AgentDecision] = []
        self.current_location = None
        self.current_risk_level = 0.0
        self.social_influences = {}
        
        # 行为表型配置
        self.behavioral_configs = {
            'cohesive': {
                'family_priority_weight': 0.9,
                'stranger_help_probability': 0.2,
                'risk_aversion': 0.8,
                'social_conformity': 0.6
            },
            'moderate': {
                'family_priority_weight': 0.7,
                'stranger_help_probability': 0.5,
                'risk_aversion': 0.5,
                'social_conformity': 0.7
            },
            'universalistic': {
                'family_priority_weight': 0.5,
                'stranger_help_probability': 0.8,
                'risk_aversion': 0.3,
                'social_conformity': 0.4
            }
        }
        
        logger.info(f"推理增强智能体初始化完成: {agent_profile.agent_id}")
        
    async def make_evacuation_decision(
        self, 
        weather_data: Dict[str, Any],
        location_data: Dict[str, Any],
        social_influence: float = 0.0,
        official_recommendation: Optional[str] = None
    ) -> AgentDecision:
        """进行疏散决策
        
        Args:
            weather_data: 天气数据
            location_data: 位置数据
            social_influence: 社会影响因子
            official_recommendation: 官方建议
            
        Returns:
            AgentDecision: 决策结果
        """
        try:
            # 准备推理输入
            reasoning_input = self._prepare_evacuation_input(
                weather_data, location_data, social_influence, official_recommendation
            )
            
            # 调用推理模型
            reasoning_result = await self.reasoning_client.hurricane_evacuation_reasoning(
                weather_data=weather_data,
                agent_profile=asdict(self.profile),
                current_location=self.current_location or 0,
                social_influence=social_influence
            )
            
            # 解析决策结果
            decision = self._parse_evacuation_decision(reasoning_result)
            
            # 应用行为表型调整
            decision = self._apply_behavioral_adjustment(decision, 'evacuation')
            
            # 创建决策记录
            agent_decision = AgentDecision(
                decision_type="evacuation",
                decision=decision,
                confidence=reasoning_result.confidence,
                reasoning_steps=reasoning_result.reasoning_steps,
                reasoning_content=reasoning_result.reasoning_content,
                timestamp=time.time(),
                agent_id=self.profile.agent_id
            )
            
            # 记录决策历史
            self.decision_history.append(agent_decision)
            
            logger.info(f"智能体 {self.profile.agent_id} 完成疏散决策: {decision.get('should_evacuate', False)}")
            
            return agent_decision
            
        except Exception as e:
            logger.error(f"疏散决策失败: {e}")
            # 返回默认决策
            return self._create_fallback_decision("evacuation", str(e))
            
    async def assess_risk(
        self,
        weather_data: Dict[str, Any],
        location_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None
    ) -> AgentDecision:
        """进行风险评估
        
        Args:
            weather_data: 天气数据
            location_data: 位置数据
            historical_data: 历史数据
            
        Returns:
            AgentDecision: 风险评估结果
        """
        try:
            # 调用推理模型进行风险评估
            reasoning_result = await self.reasoning_client.risk_assessment_reasoning(
                weather_data=weather_data,
                location_data=location_data
            )
            
            # 解析风险评估结果
            risk_assessment = self._parse_risk_assessment(reasoning_result)
            
            # 应用个人风险偏好调整
            risk_assessment = self._apply_risk_tolerance_adjustment(risk_assessment)
            
            # 更新当前风险水平
            self.current_risk_level = risk_assessment.get('risk_score', 0.5)
            
            # 创建决策记录
            agent_decision = AgentDecision(
                decision_type="risk_assessment",
                decision=risk_assessment,
                confidence=reasoning_result.confidence,
                reasoning_steps=reasoning_result.reasoning_steps,
                reasoning_content=reasoning_result.reasoning_content,
                timestamp=time.time(),
                agent_id=self.profile.agent_id
            )
            
            self.decision_history.append(agent_decision)
            
            logger.info(f"智能体 {self.profile.agent_id} 完成风险评估: {risk_assessment.get('risk_level', 'unknown')}")
            
            return agent_decision
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return self._create_fallback_decision("risk_assessment", str(e))
            
    async def plan_mobility(
        self,
        current_location: int,
        available_destinations: List[int],
        evacuation_decision: bool,
        constraints: Optional[Dict[str, Any]] = None
    ) -> AgentDecision:
        """规划移动路径
        
        Args:
            current_location: 当前位置
            available_destinations: 可用目的地
            evacuation_decision: 疏散决策
            constraints: 约束条件
            
        Returns:
            AgentDecision: 移动规划结果
        """
        try:
            # 构建移动规划提示
            system_prompt = self._build_mobility_planning_prompt()
            user_prompt = self._build_mobility_user_prompt(
                current_location, available_destinations, evacuation_decision, constraints
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用推理模型
            reasoning_result = await self.reasoning_client.reasoning_chat_completion(
                messages, "qwen-reasoning"  # 使用较快的模型进行路径规划
            )
            
            # 解析移动规划结果
            mobility_plan = self._parse_mobility_plan(reasoning_result)
            
            # 应用行为表型调整
            mobility_plan = self._apply_behavioral_adjustment(mobility_plan, 'mobility')
            
            # 更新当前位置
            self.current_location = current_location
            
            # 创建决策记录
            agent_decision = AgentDecision(
                decision_type="mobility_planning",
                decision=mobility_plan,
                confidence=reasoning_result.confidence,
                reasoning_steps=reasoning_result.reasoning_steps,
                reasoning_content=reasoning_result.reasoning_content,
                timestamp=time.time(),
                agent_id=self.profile.agent_id
            )
            
            self.decision_history.append(agent_decision)
            
            logger.info(f"智能体 {self.profile.agent_id} 完成移动规划")
            
            return agent_decision
            
        except Exception as e:
            logger.error(f"移动规划失败: {e}")
            return self._create_fallback_decision("mobility_planning", str(e))
            
    def _prepare_evacuation_input(
        self, 
        weather_data: Dict[str, Any],
        location_data: Dict[str, Any],
        social_influence: float,
        official_recommendation: Optional[str]
    ) -> Dict[str, Any]:
        """准备疏散决策输入"""
        return {
            "weather_data": weather_data,
            "location_data": location_data,
            "agent_profile": asdict(self.profile),
            "social_influence": social_influence,
            "official_recommendation": official_recommendation,
            "behavioral_config": self.behavioral_configs.get(self.profile.behavioral_type, {})
        }
        
    def _parse_evacuation_decision(self, reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """解析疏散决策结果"""
        try:
            # 尝试从推理结果中提取JSON
            if reasoning_result.decision:
                return reasoning_result.decision
                
            # 从内容中解析
            content = reasoning_result.content.lower()
            
            decision = {
                "should_evacuate": False,
                "confidence": reasoning_result.confidence,
                "reasoning": reasoning_result.content
            }
            
            # 简单的关键词匹配
            if any(keyword in content for keyword in ['应该疏散', 'should evacuate', '建议撤离']):
                decision["should_evacuate"] = True
            elif any(keyword in content for keyword in ['不需要疏散', 'should not evacuate', '无需撤离']):
                decision["should_evacuate"] = False
            else:
                # 基于风险评分决策
                if 'risk_score' in content:
                    # 尝试提取风险评分
                    import re
                    risk_match = re.search(r'risk_score["\']?\s*:\s*([0-9.]+)', content)
                    if risk_match:
                        risk_score = float(risk_match.group(1))
                        decision["should_evacuate"] = risk_score > 0.6
                        
            return decision
            
        except Exception as e:
            logger.error(f"解析疏散决策失败: {e}")
            return {
                "should_evacuate": False,
                "confidence": 0.3,
                "reasoning": f"解析失败: {str(e)}"
            }
            
    def _parse_risk_assessment(self, reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """解析风险评估结果"""
        try:
            if reasoning_result.decision:
                return reasoning_result.decision
                
            # 默认风险评估
            return {
                "risk_score": 0.5,
                "risk_level": "medium",
                "reasoning": reasoning_result.content
            }
            
        except Exception as e:
            logger.error(f"解析风险评估失败: {e}")
            return {
                "risk_score": 0.5,
                "risk_level": "unknown",
                "reasoning": f"解析失败: {str(e)}"
            }
            
    def _parse_mobility_plan(self, reasoning_result: ReasoningResult) -> Dict[str, Any]:
        """解析移动规划结果"""
        try:
            if reasoning_result.decision:
                return reasoning_result.decision
                
            # 默认移动规划
            return {
                "destination": None,
                "route": [],
                "estimated_time": 0,
                "reasoning": reasoning_result.content
            }
            
        except Exception as e:
            logger.error(f"解析移动规划失败: {e}")
            return {
                "destination": None,
                "route": [],
                "estimated_time": 0,
                "reasoning": f"解析失败: {str(e)}"
            }
            
    def _apply_behavioral_adjustment(
        self, 
        decision: Dict[str, Any], 
        decision_type: str
    ) -> Dict[str, Any]:
        """应用行为表型调整"""
        behavioral_config = self.behavioral_configs.get(self.profile.behavioral_type, {})
        
        if decision_type == "evacuation":
            # 根据行为表型调整疏散决策
            if self.profile.behavioral_type == "cohesive":
                # 内聚型更保守，更容易疏散
                if decision.get("should_evacuate") == False and self.current_risk_level > 0.4:
                    decision["should_evacuate"] = True
                    decision["reasoning"] += " [行为调整: 内聚型倾向于保守决策]"
                    
            elif self.profile.behavioral_type == "universalistic":
                # 普遍主义型更理性，基于客观风险
                pass  # 保持原决策
                
        return decision
        
    def _apply_risk_tolerance_adjustment(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """应用风险承受能力调整"""
        original_risk = risk_assessment.get("risk_score", 0.5)
        
        # 根据个人风险承受能力调整
        risk_tolerance = self.profile.risk_tolerance
        
        if risk_tolerance < 0.3:  # 低风险承受能力
            adjusted_risk = min(original_risk * 1.2, 1.0)
        elif risk_tolerance > 0.7:  # 高风险承受能力
            adjusted_risk = original_risk * 0.8
        else:
            adjusted_risk = original_risk
            
        risk_assessment["risk_score"] = adjusted_risk
        risk_assessment["original_risk_score"] = original_risk
        risk_assessment["risk_tolerance_applied"] = True
        
        return risk_assessment
        
    def _build_mobility_planning_prompt(self) -> str:
        """构建移动规划系统提示"""
        return """
你是一个城市交通规划专家，需要为智能体规划最优的移动路径。

请考虑以下因素：
1. 当前交通状况
2. 目的地的安全性和可达性
3. 智能体的个人条件（年龄、健康状况、交通工具等）
4. 时间约束和紧急程度
5. 社会因素（拥堵、其他人的行为等）

请给出详细的移动规划，包括目的地选择、路径规划和时间估算。
"""
        
    def _build_mobility_user_prompt(
        self,
        current_location: int,
        available_destinations: List[int],
        evacuation_decision: bool,
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """构建移动规划用户提示"""
        return f"""
请为以下智能体规划移动路径：

智能体信息：
- ID: {self.profile.agent_id}
- 年龄: {self.profile.age}
- 健康状况: {self.profile.health_status}
- 是否有车: {self.profile.has_vehicle}
- 家庭规模: {self.profile.family_size}

移动信息：
- 当前位置: {current_location}
- 可用目的地: {available_destinations}
- 是否疏散: {evacuation_decision}
- 约束条件: {constraints or {}}

请提供最优的移动规划方案。
"""
        
    def _create_fallback_decision(self, decision_type: str, error_msg: str) -> AgentDecision:
        """创建回退决策"""
        # 基于智能体属性的智能回退决策
        if decision_type == "evacuation":
            # 基于风险容忍度、健康状况、家庭规模等因素决策
            risk_factors = [
                self.current_risk_level > 0.5,
                self.profile.risk_tolerance < 0.4,
                self.profile.health_status in ['poor', 'fair'],
                self.profile.family_size > 3
            ]
            should_evacuate = sum(risk_factors) >= 2
            confidence = 0.7 if sum(risk_factors) >= 3 else 0.6
            
            decision = {
                "should_evacuate": should_evacuate,
                "confidence": confidence,
                "reasoning": f"基于智能体属性的回退决策: 风险等级{self.current_risk_level:.2f}, 风险容忍度{self.profile.risk_tolerance:.2f}"
            }
            
        elif decision_type == "risk_assessment":
            # 基于教育水平和社会网络规模评估风险
            education_factor = {'primary': 0.3, 'secondary': 0.5, 'tertiary': 0.7}.get(self.profile.education_level, 0.5)
            social_factor = min(self.profile.social_network_size / 10.0, 1.0)
            risk_score = (education_factor + social_factor) / 2
            
            decision = {
                "risk_score": risk_score,
                "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low",
                "reasoning": f"基于教育水平({self.profile.education_level})和社会网络({self.profile.social_network_size})的风险评估"
            }
            confidence = 0.6
            
        elif decision_type == "mobility_planning":
            # 基于交通工具和收入水平规划移动
            has_transport = self.profile.has_vehicle
            income_mobility = {'low': 0.3, 'medium': 0.6, 'high': 0.9}.get(self.profile.income_level, 0.5)
            
            decision = {
                "destination": "safe_zone" if has_transport else "nearby_shelter",
                "route": ["direct_route"] if has_transport else ["walking_route"],
                "estimated_time": 30 if has_transport else 90,
                "reasoning": f"基于交通工具({has_transport})和收入水平({self.profile.income_level})的移动规划"
            }
            confidence = 0.7 if has_transport else 0.5
            
        else:
            decision = {"reasoning": f"未知决策类型的回退: {error_msg}"}
            confidence = 0.4
        
        return AgentDecision(
            decision_type=decision_type,
            decision=decision,
            confidence=confidence,
            reasoning_steps=["API调用失败", "启用智能回退机制", "基于智能体属性推理"],
            reasoning_content=f"网络连接问题导致API调用失败，使用基于智能体属性的回退决策: {error_msg}",
            timestamp=time.time(),
            agent_id=self.profile.agent_id
        )
        
    def get_decision_history(self, decision_type: Optional[str] = None) -> List[AgentDecision]:
        """获取决策历史"""
        if decision_type:
            return [d for d in self.decision_history if d.decision_type == decision_type]
        return self.decision_history.copy()
        
    def get_latest_decision(self, decision_type: str) -> Optional[AgentDecision]:
        """获取最新决策"""
        decisions = self.get_decision_history(decision_type)
        return decisions[-1] if decisions else None
        
    def clear_decision_history(self):
        """清空决策历史"""
        self.decision_history.clear()
        logger.info(f"智能体 {self.profile.agent_id} 决策历史已清空")
        
# 使用示例和测试
async def test_reasoning_enhanced_agent():
    """测试推理增强智能体"""
    print("=== 推理增强智能体测试 ===")
    
    # 创建智能体档案
    profile = AgentProfile(
        agent_id="test_agent_001",
        age=35,
        gender="female",
        income_level="medium",
        education_level="tertiary",
        family_size=3,
        has_vehicle=True,
        health_status="good",
        social_network_size=50,
        risk_tolerance=0.6,
        behavioral_type="moderate"
    )
    
    # 创建智能体
    agent = ReasoningEnhancedAgent(profile)
    
    # 测试数据
    weather_data = {
        "wind_speed": 140,
        "pressure": 945,
        "temperature": 27,
        "rainfall": 30
    }
    
    location_data = {
        "elevation": 3,
        "distance_to_coast": 1.5,
        "flood_zone": "A",
        "building_type": "residential"
    }
    
    try:
        # 测试风险评估
        print("\n1. 风险评估测试:")
        risk_decision = await agent.assess_risk(weather_data, location_data)
        print(f"风险评估结果: {risk_decision.decision}")
        print(f"置信度: {risk_decision.confidence}")
        print(f"推理步骤数: {len(risk_decision.reasoning_steps)}")
        
        # 测试疏散决策
        print("\n2. 疏散决策测试:")
        evacuation_decision = await agent.make_evacuation_decision(
            weather_data, location_data, social_influence=0.3
        )
        print(f"疏散决策: {evacuation_decision.decision}")
        print(f"置信度: {evacuation_decision.confidence}")
        
        # 测试移动规划
        print("\n3. 移动规划测试:")
        mobility_decision = await agent.plan_mobility(
            current_location=100,
            available_destinations=[101, 102, 103],
            evacuation_decision=evacuation_decision.decision.get("should_evacuate", False)
        )
        print(f"移动规划: {mobility_decision.decision}")
        
        # 显示决策历史
        print("\n4. 决策历史:")
        history = agent.get_decision_history()
        for i, decision in enumerate(history, 1):
            print(f"  {i}. {decision.decision_type}: 置信度 {decision.confidence:.2f}")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    import sys
    asyncio.run(test_reasoning_enhanced_agent())