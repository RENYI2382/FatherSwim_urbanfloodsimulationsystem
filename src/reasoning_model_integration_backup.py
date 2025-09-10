#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模型集成模块
基于Silicon Flow推理模型手册，集成DeepSeek-R1等推理模型
提供结构化思维、多步推理和自修正机制
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI
import yaml
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class ReasoningResult:
    """推理结果数据类"""
    content: str  # 最终回答内容
    reasoning_content: str  # 思维链内容
    confidence: float  # 置信度
    reasoning_steps: List[str]  # 推理步骤
    decision: Any  # 决策结果
    
class ReasoningModelClient:
    """推理模型客户端
    
    特性：
    - 支持DeepSeek-R1等推理模型
    - 思维链（Chain-of-Thought）推理
    - 结构化决策过程
    - 自修正机制
    - 多步推理验证
    """
    
    def __init__(self, config_path: str = "config/reasoning_model_config.yaml"):
        self.config = self._load_config(config_path)
        self.client = None
        self.model_configs = {
            "deepseek-r1": {
                "model_name": "deepseek-ai/DeepSeek-R1",  # 使用非Pro版本，支持赠送余额
                "temperature": 0.6,  # 推荐值
                "top_p": 0.95,
                "max_tokens": 4096,
                "thinking_budget": 2048,  # 思维链token预算
                "supports_reasoning": True
            },
            "qwen-reasoning": {
                "model_name": "Qwen/Qwen2.5-72B-Instruct",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 4096,
                "thinking_budget": 1024,
                "supports_reasoning": False
            }
        }
        self._setup_client()
        self._last_network_check = 0
        self._network_check_interval = 300  # 5分钟检查一次
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件未找到: {config_path}，使用默认配置")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "silicon_flow": {
                "base_url": "https://api.siliconflow.cn/v1/",
                "api_key": "sk-your-api-key",
                "timeout": 60
            },
            "default_model": "deepseek-r1",
            "reasoning": {
                "max_reasoning_steps": 5,
                "confidence_threshold": 0.7,
                "enable_self_correction": True
            }
        }
        
    def _setup_client(self):
        """设置OpenAI客户端"""
        try:
            silicon_config = self.config.get("silicon_flow", {})
            self.client = AsyncOpenAI(
                base_url=silicon_config.get("base_url", "https://api.siliconflow.cn/v1/"),
                api_key=silicon_config.get("api_key", "sk-your-api-key"),
                timeout=silicon_config.get("timeout", 30)  # 使用更短的超时时间
            )
            self.network_available = True  # 网络可用性标志
            logger.info("推理模型客户端初始化成功")
        except Exception as e:
            logger.error(f"推理模型客户端初始化失败: {e}")
            self.network_available = False
    
    async def _check_network_status(self) -> bool:
        """检查网络连接状态"""
        current_time = time.time()
        
        # 如果距离上次检查时间不足间隔，直接返回当前状态
        if current_time - self._last_network_check < self._network_check_interval:
            return self.network_available
            
        try:
            # 尝试连接Silicon Flow API
            silicon_config = self.config.get("silicon_flow", {})
            base_url = silicon_config.get("base_url", "https://api.siliconflow.cn/v1/")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(base_url.rstrip('/v1/') + '/health', 
                                     headers={'User-Agent': 'ReasoningModelClient/1.0'}) as response:
                    # 如果能够连接（即使返回404等错误），说明网络是通的
                    self.network_available = True
                    logger.debug("网络连接检查成功")
                    
        except Exception as e:
            logger.warning(f"网络连接检查失败: {e}")
            self.network_available = False
            
        self._last_network_check = current_time
        return self.network_available
            
    async def reasoning_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model_type: str = "deepseek-r1",
        stream: bool = False
    ) -> ReasoningResult:
        """推理模型聊天完成
        
        Args:
            messages: 消息列表
            model_type: 模型类型
            stream: 是否流式输出
            
        Returns:
            ReasoningResult: 推理结果
        """
        # 检查网络状态
        await self._check_network_status()
        
        # 如果网络不可用，直接使用回退机制
        if not self.network_available:
            logger.warning("网络不可用，直接使用回退决策")
            return self._create_fallback_reasoning_result("网络连接不可用")
            
        model_config = self.model_configs.get(model_type, self.model_configs["deepseek-r1"])
        
        try:
            # 构建请求参数
            request_params = {
                "model": model_config["model_name"],
                "messages": messages,
                "temperature": model_config["temperature"],
                "top_p": model_config["top_p"],
                "max_tokens": model_config["max_tokens"],
                "stream": stream
            }
            
            # 如果模型支持推理，添加thinking_budget
            if model_config["supports_reasoning"]:
                request_params["extra_body"] = {
                    "thinking_budget": model_config["thinking_budget"]
                }
            
            # 发送请求
            if stream:
                return await self._handle_stream_response(request_params)
            else:
                return await self._handle_non_stream_response(request_params)
                
        except Exception as e:
            logger.error(f"推理模型调用失败: {e}")
            # 标记网络不可用
            self.network_available = False
            # 提供更好的回退结果
            return self._create_fallback_reasoning_result(str(e))
            
    async def _handle_non_stream_response(self, request_params: Dict) -> ReasoningResult:
        """处理非流式响应"""
        max_retries = 2  # 减少重试次数
        retry_delay = 1.0  # 重试延迟
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.chat.completions.create(**request_params)
                
                message = response.choices[0].message
                content = message.content or ""
                reasoning_content = getattr(message, 'reasoning_content', '') or ""
                
                # 解析推理步骤
                reasoning_steps = self._extract_reasoning_steps(reasoning_content)
                
                # 计算置信度
                confidence = self._calculate_confidence(content, reasoning_content)
                
                return ReasoningResult(
                    content=content,
                    reasoning_content=reasoning_content,
                    confidence=confidence,
                    reasoning_steps=reasoning_steps,
                    decision=self._extract_decision(content)
                )
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"推理模型调用失败，第{attempt + 1}次重试: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5  # 指数退避
                else:
                    logger.error(f"推理模型调用最终失败: {e}")
                    raise e
        
    async def _handle_stream_response(self, request_params: Dict) -> ReasoningResult:
        """处理流式响应"""
        content = ""
        reasoning_content = ""
        
        response = await self.client.chat.completions.create(**request_params)
        
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
                
        # 解析推理步骤
        reasoning_steps = self._extract_reasoning_steps(reasoning_content)
        
        # 计算置信度
        confidence = self._calculate_confidence(content, reasoning_content)
        
        return ReasoningResult(
            content=content,
            reasoning_content=reasoning_content,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            decision=self._extract_decision(content)
        )
        
    def _extract_reasoning_steps(self, reasoning_content: str) -> List[str]:
        """从推理内容中提取推理步骤"""
        if not reasoning_content:
            return []
            
        # 简单的步骤提取逻辑
        steps = []
        lines = reasoning_content.split('\n')
        current_step = ""
        
        for line in lines:
            line = line.strip()
            if line:
                if line.startswith(('1.', '2.', '3.', '4.', '5.', '步骤', 'Step')):
                    if current_step:
                        steps.append(current_step)
                    current_step = line
                else:
                    current_step += " " + line
                    
        if current_step:
            steps.append(current_step)
            
        return steps[:5]  # 最多返回5个步骤
        
    def _calculate_confidence(self, content: str, reasoning_content: str) -> float:
        """计算置信度"""
        base_confidence = 0.5
        
        # 基于内容长度的置信度调整
        if len(content) > 50:
            base_confidence += 0.1
            
        # 基于推理内容的置信度调整
        if reasoning_content:
            if len(reasoning_content) > 100:
                base_confidence += 0.2
            if '因为' in reasoning_content or 'because' in reasoning_content.lower():
                base_confidence += 0.1
            if '所以' in reasoning_content or 'therefore' in reasoning_content.lower():
                base_confidence += 0.1
                
        # 基于决策明确性的置信度调整
        decision_keywords = ['是', '否', '应该', '不应该', 'yes', 'no', 'should', 'shouldn\'t']
        if any(keyword in content.lower() for keyword in decision_keywords):
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)
        
    def _extract_decision(self, content: str) -> Optional[Dict[str, Any]]:
        """从内容中提取决策"""
        try:
            # 尝试解析JSON格式的决策
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                json_str = content[start:end]
                return json.loads(json_str)
        except:
            pass
            
        # 简单的决策提取
        decision = {}
        
        # 疏散决策
        if '疏散' in content or 'evacuate' in content.lower():
            if '应该疏散' in content or 'should evacuate' in content.lower():
                decision['should_evacuate'] = True
            elif '不应该疏散' in content or 'should not evacuate' in content.lower():
                decision['should_evacuate'] = False
                
        # 风险评估
        if '风险' in content or 'risk' in content.lower():
            if '高风险' in content or 'high risk' in content.lower():
                decision['risk_level'] = 'high'
            elif '中风险' in content or 'medium risk' in content.lower():
                decision['risk_level'] = 'medium'
            elif '低风险' in content or 'low risk' in content.lower():
                decision['risk_level'] = 'low'
                
        return decision if decision else None
        
    def _create_fallback_reasoning_result(self, error_msg: str) -> ReasoningResult:
        """创建回退推理结果"""
        return ReasoningResult(
            content="由于网络连接问题，使用基础决策逻辑",
            reasoning_content=f"API调用失败: {error_msg}，启用回退机制",
            confidence=0.6,  # 提高回退决策的置信度
            reasoning_steps=["检测到API连接问题", "启用本地回退决策", "基于智能体属性进行基础推理"],
            decision={"fallback": True, "reason": error_msg}
        )
        
    async def hurricane_evacuation_reasoning(
        self, 
        weather_data: Dict[str, Any],
        agent_profile: Dict[str, Any],
        current_location: int,
        social_influence: float = 0.0
    ) -> ReasoningResult:
        """飓风疏散推理决策
        
        Args:
            weather_data: 天气数据
            agent_profile: 智能体档案
            current_location: 当前位置
            social_influence: 社会影响
            
        Returns:
            ReasoningResult: 推理结果
        """
        # 构建推理提示词
        system_prompt = """
你是一个专业的应急管理专家，需要基于给定信息进行飓风疏散决策推理。

请按照以下步骤进行推理：
1. 分析天气风险等级
2. 评估个人和家庭情况
3. 考虑社会影响因素
4. 综合判断是否需要疏散
5. 给出最终决策和理由

请逐步推理，并将最终答案以JSON格式输出：
{
  "should_evacuate": true/false,
  "confidence": 0.0-1.0,
  "risk_score": 0.0-1.0,
  "reasoning": "详细推理过程"
}
"""
        
        user_prompt = f"""
请基于以下信息进行飓风疏散决策推理：

天气数据：
- 风速：{weather_data.get('wind_speed', 0)} km/h
- 气压：{weather_data.get('pressure', 1013)} hPa
- 温度：{weather_data.get('temperature', 25)}°C
- 降雨量：{weather_data.get('rainfall', 0)} mm

个人档案：
- 年龄：{agent_profile.get('age', 35)}岁
- 收入水平：{agent_profile.get('income_level', 'medium')}
- 家庭规模：{agent_profile.get('family_size', 1)}人
- 是否有车：{agent_profile.get('has_vehicle', False)}
- 健康状况：{agent_profile.get('health_status', 'good')}

位置信息：
- 当前位置ID：{current_location}
- 社会影响因子：{social_influence}

请进行详细的推理分析，并给出最终决策。
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await self.reasoning_chat_completion(messages, "deepseek-r1")
        
    async def risk_assessment_reasoning(
        self,
        weather_data: Dict[str, Any],
        location_data: Dict[str, Any]
    ) -> ReasoningResult:
        """风险评估推理
        
        Args:
            weather_data: 天气数据
            location_data: 位置数据
            
        Returns:
            ReasoningResult: 推理结果
        """
        system_prompt = """
你是一个气象风险评估专家，需要基于天气和地理数据评估飓风风险。

请按照以下步骤进行推理：
1. 分析天气参数的危险程度
2. 评估地理位置的脆弱性
3. 计算综合风险评分
4. 给出风险等级和建议

请逐步推理，并将最终答案以JSON格式输出：
{
  "risk_score": 0.0-1.0,
  "risk_level": "low/medium/high/extreme",
  "reasoning": "详细推理过程"
}
"""
        
        user_prompt = f"""
请基于以下数据进行风险评估：

天气数据：
{json.dumps(weather_data, indent=2, ensure_ascii=False)}

位置数据：
{json.dumps(location_data, indent=2, ensure_ascii=False)}

请进行详细的风险评估推理。
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await self.reasoning_chat_completion(messages, "deepseek-r1")
        
# 使用示例和测试
async def test_reasoning_model():
    """测试推理模型"""
    print("=== 推理模型集成测试 ===")
    
    client = ReasoningModelClient()
    
    # 测试数据
    weather_data = {
        "wind_speed": 150,
        "pressure": 940,
        "temperature": 28,
        "rainfall": 50
    }
    
    agent_profile = {
        "age": 45,
        "income_level": "medium",
        "family_size": 3,
        "has_vehicle": True,
        "health_status": "good"
    }
    
    try:
        # 测试疏散决策推理
        print("\n1. 疏散决策推理测试:")
        evacuation_result = await client.hurricane_evacuation_reasoning(
            weather_data, agent_profile, 100, 0.3
        )
        
        print(f"决策内容: {evacuation_result.content}")
        print(f"推理过程: {evacuation_result.reasoning_content[:200]}...")
        print(f"置信度: {evacuation_result.confidence}")
        print(f"推理步骤数: {len(evacuation_result.reasoning_steps)}")
        
        # 测试风险评估推理
        print("\n2. 风险评估推理测试:")
        location_data = {"elevation": 5, "distance_to_coast": 2, "flood_zone": "A"}
        risk_result = await client.risk_assessment_reasoning(weather_data, location_data)
        
        print(f"风险评估: {risk_result.content}")
        print(f"推理步骤: {risk_result.reasoning_steps}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    asyncio.run(test_reasoning_model())