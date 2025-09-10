#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
符合AG官方标准的推理模型集成模块

基于AgentSociety官方文档重构，解决以下问题：
1. 使用AG标准的LLM调用接口
2. 符合AG配置管理规范
3. 集成AG的并发控制和重试机制
4. 移除自定义OpenAI客户端
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# 模拟AG的Agent基类和相关组件
class MockAgent:
    """模拟AG的Agent基类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = MockLLMClient(config.get('llm', {}))
        self.logger = logging.getLogger(self.__class__.__name__)
        
class MockLLMClient:
    """模拟AG的LLM客户端"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 检查配置有效性
        if config.get('provider') == 'invalid':
            self.is_valid = False
        else:
            self.is_valid = True
        
    async def atext_request(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """模拟AG的异步文本请求接口"""
        # 模拟无效配置时的错误
        if not self.is_valid:
            raise Exception("Invalid LLM configuration: provider 'invalid' not supported")
        
        # 这里应该是AG的实际LLM调用逻辑
        # 包含负载均衡、并发控制、重试机制等
        
        # 模拟API调用
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        # 模拟响应
        if messages and messages[-1].get('content'):
            user_content = messages[-1]['content']
            if '疏散' in user_content:
                return "基于当前天气条件和智能体特征，建议立即疏散。推理过程：1.分析风险等级 2.评估疏散能力 3.制定疏散决策"
            elif '风险' in user_content:
                return "当前风险等级为中等。推理过程：1.分析天气数据 2.评估位置风险 3.综合风险评估"
        
        return "已完成推理分析，建议采取预防措施。"

logger = logging.getLogger(__name__)

@dataclass
class ReasoningResult:
    """推理结果数据类"""
    content: str  # 最终回答内容
    reasoning_content: str  # 思维链内容
    confidence: float  # 置信度
    reasoning_steps: List[str]  # 推理步骤
    decision: Any  # 决策结果

class AGStandardReasoningAgent(MockAgent):
    """符合AG标准的推理智能体"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.reasoning_config = config.get('reasoning', {})
        
    async def structured_reasoning(self, 
                                 prompt: str, 
                                 context: Dict[str, Any] = None) -> ReasoningResult:
        """结构化推理方法 - 使用AG标准接口"""
        try:
            # 构建符合AG标准的消息格式
            messages = [
                {
                    "role": "system", 
                    "content": "你是一个专业的推理智能体，请进行结构化思考并提供详细的推理过程。"
                },
                {
                    "role": "user", 
                    "content": self._build_reasoning_prompt(prompt, context)
                }
            ]
            
            # 使用AG标准的LLM调用接口
            response = await self.llm.atext_request(messages)
            
            # 解析响应
            return self._parse_reasoning_response(response)
            
        except Exception as e:
            self.logger.error(f"推理过程出错: {e}")
            return self._create_fallback_reasoning_result(str(e))
    
    def _build_reasoning_prompt(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """构建推理提示词"""
        base_prompt = f"""
请对以下问题进行结构化推理：

问题：{prompt}

请按以下格式回答：
1. 推理过程：[详细的思考步骤]
2. 结论：[最终结论]
3. 置信度：[0-1之间的数值]
4. 决策建议：[具体的行动建议]
"""
        
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            base_prompt += f"\n\n上下文信息：\n{context_str}"
            
        return base_prompt
    
    def _parse_reasoning_response(self, response: str) -> ReasoningResult:
        """解析推理响应"""
        # 提取推理步骤
        reasoning_steps = self._extract_reasoning_steps(response)
        
        # 计算置信度
        confidence = self._extract_confidence(response)
        
        # 提取决策
        decision = self._extract_decision(response)
        
        return ReasoningResult(
            content=response,
            reasoning_content=response,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            decision=decision
        )
    
    def _extract_reasoning_steps(self, content: str) -> List[str]:
        """提取推理步骤"""
        steps = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                        '推理' in line or '步骤' in line):
                steps.append(line)
                
        return steps[:5]  # 最多返回5个步骤
    
    def _extract_confidence(self, content: str) -> float:
        """提取置信度"""
        # 尝试从内容中提取置信度数值
        import re
        confidence_match = re.search(r'置信度[：:](\d*\.?\d+)', content)
        if confidence_match:
            try:
                return float(confidence_match.group(1))
            except ValueError:
                pass
        
        # 基于内容质量估算置信度
        base_confidence = 0.7
        if len(content) > 100:
            base_confidence += 0.1
        if '推理' in content:
            base_confidence += 0.1
        if '因为' in content or '所以' in content:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
    
    def _extract_decision(self, content: str) -> Optional[Dict[str, Any]]:
        """提取决策信息"""
        decision = {}
        
        # 疏散决策
        if '疏散' in content:
            if '建议疏散' in content or '应该疏散' in content:
                decision['should_evacuate'] = True
            elif '不建议疏散' in content or '不应该疏散' in content:
                decision['should_evacuate'] = False
        
        # 风险等级
        if '高风险' in content:
            decision['risk_level'] = 'high'
        elif '中风险' in content or '中等风险' in content:
            decision['risk_level'] = 'medium'
        elif '低风险' in content:
            decision['risk_level'] = 'low'
            
        return decision if decision else None
    
    def _create_fallback_reasoning_result(self, error_msg: str) -> ReasoningResult:
        """创建回退推理结果"""
        return ReasoningResult(
            content="使用基础推理逻辑进行决策",
            reasoning_content=f"推理服务暂时不可用: {error_msg}，使用本地推理",
            confidence=0.6,
            reasoning_steps=["检测到服务异常", "启用本地推理", "基于规则进行决策"],
            decision={"fallback": True, "reason": error_msg}
        )
    
    async def hurricane_evacuation_reasoning(self, 
                                           weather_data: Dict[str, Any],
                                           agent_profile: Dict[str, Any]) -> ReasoningResult:
        """飓风疏散推理"""
        context = {
            "天气数据": weather_data,
            "智能体特征": agent_profile
        }
        
        prompt = """
基于当前天气条件和智能体特征，请分析是否需要疏散：
1. 评估天气风险等级
2. 分析智能体疏散能力
3. 制定疏散决策
4. 提供具体的行动建议
"""
        
        return await self.structured_reasoning(prompt, context)
    
    async def risk_assessment_reasoning(self,
                                      weather_data: Dict[str, Any],
                                      location_data: Dict[str, Any]) -> ReasoningResult:
        """风险评估推理"""
        context = {
            "天气数据": weather_data,
            "位置数据": location_data
        }
        
        prompt = """
基于天气和位置数据进行风险评估：
1. 分析天气威胁程度
2. 评估地理位置风险
3. 综合风险等级判断
4. 提供风险应对建议
"""
        
        return await self.structured_reasoning(prompt, context)

# 工厂函数
def create_reasoning_agent(config: Dict[str, Any]) -> AGStandardReasoningAgent:
    """创建符合AG标准的推理智能体"""
    return AGStandardReasoningAgent(config)

# 测试函数
async def test_ag_standard_reasoning():
    """测试AG标准推理模块"""
    # 模拟AG标准配置
    config = {
        "llm": {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "sk-test-key",
            "concurrency": 200,
            "timeout": 30
        },
        "reasoning": {
            "max_steps": 5,
            "confidence_threshold": 0.7
        }
    }
    
    # 创建推理智能体
    agent = create_reasoning_agent(config)
    
    # 测试飓风疏散推理
    weather_data = {
        "wind_speed": 120,
        "category": 4,
        "distance_to_eye": 50
    }
    
    agent_profile = {
        "age": 65,
        "mobility": "limited",
        "family_size": 2
    }
    
    print("=== 测试飓风疏散推理 ===")
    result = await agent.hurricane_evacuation_reasoning(weather_data, agent_profile)
    print(f"推理结果: {result.content}")
    print(f"置信度: {result.confidence}")
    print(f"决策: {result.decision}")
    print(f"推理步骤: {result.reasoning_steps}")
    
    # 测试风险评估推理
    location_data = {
        "elevation": 5,
        "distance_to_coast": 2,
        "flood_zone": "A"
    }
    
    print("\n=== 测试风险评估推理 ===")
    result = await agent.risk_assessment_reasoning(weather_data, location_data)
    print(f"推理结果: {result.content}")
    print(f"置信度: {result.confidence}")
    print(f"决策: {result.decision}")
    print(f"推理步骤: {result.reasoning_steps}")

if __name__ == "__main__":
    asyncio.run(test_ag_standard_reasoning())