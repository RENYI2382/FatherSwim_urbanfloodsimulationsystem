"""
增强的LLM客户端 - 修复版本
解决API调用失败、参数错误、返回值解析等问题
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import aiohttp
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class EnhancedLLMClient:
    """增强的LLM客户端 - 修复版本"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vllm_client = None
        self.zhipu_client = None
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_breaker_open = False
        self._setup_clients()
    
    def _setup_clients(self):
        """设置客户端"""
        try:
            # vLLM客户端
            vllm_config = self.config['llm']['vllm_server']
            self.vllm_client = AsyncOpenAI(
                base_url=vllm_config['base_url'],
                api_key=vllm_config['api_key'],
                timeout=self.config['llm']['api']['timeout']
            )
            
            # 智普客户端（备选）
            if 'zhipu' in self.config['llm']:
                zhipu_config = self.config['llm']['zhipu']
                self.zhipu_client = AsyncOpenAI(
                    base_url=zhipu_config['base_url'],
                    api_key=zhipu_config['api_key'],
                    timeout=zhipu_config.get('timeout', 30)
                )
            
            logger.info("增强LLM客户端初始化成功")
            
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
    
    def _is_circuit_breaker_open(self) -> bool:
        """检查熔断器状态"""
        threshold = self.config['llm']['api'].get('circuit_breaker_threshold', 10)
        if self.failure_count >= threshold:
            # 检查是否应该重置
            if time.time() - self.last_failure_time > 300:  # 5分钟后重置
                self.failure_count = 0
                self.circuit_breaker_open = False
                logger.info("熔断器已重置")
            else:
                self.circuit_breaker_open = True
        return self.circuit_breaker_open
    
    async def chat_completion_with_retry(
        self, 
        messages: List[Dict[str, str]], 
        use_zhipu: bool = False,
        max_retries: int = None
    ) -> Optional[str]:
        """带重试的聊天完成"""
        if self._is_circuit_breaker_open():
            logger.warning("熔断器开启，跳过LLM调用")
            return None
        
        max_retries = max_retries or self.config['llm']['api']['max_retries']
        retry_delay = self.config['llm']['api'].get('retry_delay', 1.0)
        
        for attempt in range(max_retries + 1):
            try:
                # 选择客户端
                if use_zhipu and self.zhipu_client:
                    client = self.zhipu_client
                    model = self.config['llm']['zhipu']['model']
                else:
                    client = self.vllm_client
                    model = self.config['llm']['model']['name']
                
                if not client:
                    raise Exception("客户端未初始化")
                
                # 发送请求
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.config['llm']['api']['temperature'],
                    max_tokens=self.config['llm']['api']['max_tokens']
                )
                
                # 重置失败计数
                self.failure_count = 0
                self.circuit_breaker_open = False
                
                return response.choices[0].message.content
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                logger.warning(f"LLM请求失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    # 如果vLLM失败，尝试智普
                    if not use_zhipu and self.zhipu_client and attempt == 0:
                        logger.info("尝试使用智普API作为备选")
                        return await self.chat_completion_with_retry(messages, use_zhipu=True, max_retries=2)
                    
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # 指数退避
                else:
                    logger.error(f"LLM请求最终失败: {e}")
                    return None
        
        return None

class FixedFloodAgentLLM:
    """修复版Flood Agent LLM"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.client = EnhancedLLMClient(self.config)
        self.prompts = self.config.get('agent_llm', {}).get('decision_prompts', {})
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        import yaml
        config_path = config_path or "config/llm_config.yaml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return {}
    
    def _safe_json_parse(self, response: str, default_value: Any) -> Any:
        """安全的JSON解析"""
        if not response:
            return default_value
        
        try:
            # 尝试直接解析JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # 返回默认值
            logger.warning(f"JSON解析失败，使用默认值: {response[:100]}...")
            return default_value
    
    async def assess_risk_with_llm(
        self, 
        weather_data: Dict[str, Any],
        agent_profile: Dict[str, Any],
        current_location: int
    ) -> Dict[str, Any]:
        """修复版风险评估"""
        try:
            prompt = self.prompts.get('risk_assessment', '').format(
                weather_data=json.dumps(weather_data, ensure_ascii=False),
                current_location=current_location,
                age=agent_profile.get('age', 35),
                income_level=agent_profile.get('income_level', 'medium'),
                family_size=agent_profile.get('family_size', 2),
                has_vehicle=agent_profile.get('has_vehicle', True),
                risk_aversion=agent_profile.get('risk_aversion', 0.5)
            )
            
            messages = [
                {"role": "system", "content": "你是专业的风险评估专家，请严格按照JSON格式回复。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat_completion_with_retry(messages)
            
            if response:
                result = self._safe_json_parse(response, {
                    "risk_score": 0.5, 
                    "reasoning": "解析失败，使用默认值"
                })
                
                # 验证风险分数
                risk_score = result.get('risk_score', 0.5)
                if not isinstance(risk_score, (int, float)) or risk_score < 0 or risk_score > 1:
                    risk_score = 0.5
                
                return {
                    "risk_score": risk_score,
                    "reasoning": result.get('reasoning', 'LLM风险评估')
                }
            
            return {"risk_score": 0.5, "reasoning": "LLM调用失败，使用默认值"}
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return {"risk_score": 0.5, "reasoning": f"评估异常: {e}"}
    
    async def make_evacuation_decision(
        self,
        risk_score: float,
        agent_profile: Dict[str, Any],
        current_location: int,
        home_location: int = None,
        work_location: int = None,
        social_influence: float = 0.0  # 修复：添加默认参数
    ) -> Dict[str, Any]:
        """修复版疏散决策"""
        try:
            # 修复：确保所有参数都有默认值
            home_location = home_location or current_location
            work_location = work_location or current_location
            
            prompt = self.prompts.get('evacuation_decision', '').format(
                risk_score=risk_score,
                current_location=current_location,
                agent_profile=json.dumps(agent_profile, ensure_ascii=False),
                social_influence=social_influence  # 修复：添加社会影响参数
            )
            
            messages = [
                {"role": "system", "content": "你是专业的疏散决策专家，请严格按照JSON格式回复。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat_completion_with_retry(messages)
            
            if response:
                result = self._safe_json_parse(response, {
                    "should_evacuate": risk_score > 0.7,
                    "confidence": 0.5,
                    "reasoning": "解析失败，使用默认决策"
                })
                
                return {
                    "should_evacuate": bool(result.get('should_evacuate', risk_score > 0.7)),
                    "confidence": float(result.get('confidence', 0.5)),
                    "reasoning": result.get('reasoning', 'LLM疏散决策')
                }
            
            return {
                "should_evacuate": risk_score > 0.7,
                "confidence": 0.5,
                "reasoning": "LLM调用失败，使用阈值决策"
            }
            
        except Exception as e:
            logger.error(f"疏散决策失败: {e}")
            return {
                "should_evacuate": risk_score > 0.7,
                "confidence": 0.5,
                "reasoning": f"决策异常: {e}"
            }
    
    async def plan_mobility_for_agent(
        self,
        weather_data: Dict[str, Any],
        agent_profile: Dict[str, Any],
        available_aois: List[int],
        current_location: int,
        should_evacuate: bool
    ) -> Dict[str, Any]:
        """修复版移动规划"""
        try:
            prompt = self.prompts.get('mobility_planning', '').format(
                current_location=current_location,
                available_aois=available_aois,
                should_evacuate=should_evacuate,
                weather_data=json.dumps(weather_data, ensure_ascii=False)
            )
            
            messages = [
                {"role": "system", "content": "你是专业的移动规划专家，请严格按照JSON格式回复。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.client.chat_completion_with_retry(messages)
            
            if response:
                result = self._safe_json_parse(response, {
                    "destination_aoi": None,
                    "reasoning": "解析失败"
                })
                
                destination = result.get('destination_aoi')
                
                # 验证目的地
                if destination and destination in available_aois:
                    return result
                elif should_evacuate and available_aois:
                    # 选择安全位置
                    safe_aoi = max(available_aois)
                    return {
                        "destination_aoi": safe_aoi,
                        "reasoning": f"LLM选择无效，自动选择安全位置{safe_aoi}"
                    }
                else:
                    return {"destination_aoi": None, "reasoning": "无需移动"}
            
            # 默认逻辑
            if should_evacuate and available_aois:
                safe_aoi = max(available_aois)
                return {
                    "destination_aoi": safe_aoi,
                    "reasoning": "LLM失败，使用默认疏散逻辑"
                }
            
            return {"destination_aoi": None, "reasoning": "LLM失败，无需移动"}
            
        except Exception as e:
            logger.error(f"移动规划失败: {e}")
            if should_evacuate and available_aois:
                safe_aoi = max(available_aois)
                return {
                    "destination_aoi": safe_aoi,
                    "reasoning": f"规划异常: {e}，使用默认疏散逻辑"
                }
            return {"destination_aoi": None, "reasoning": f"规划异常: {e}"}

# 工厂函数
def get_fixed_llm_client(config_path: str = None) -> FixedFloodAgentLLM:
    """获取修复版LLM客户端"""
    return FixedFloodAgentLLM(config_path)
