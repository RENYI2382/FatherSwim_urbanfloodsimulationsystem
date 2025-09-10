"""
LLM集成模块 - 为Hurricane Mobility Agent提供LLM推理能力
支持vLLM和智普GLM-4-air API
"""

import asyncio
import json
import logging
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import aiohttp
import openai
from openai import AsyncOpenAI

from ..utils.error_handler import retry_with_backoff, circuit_breaker, RetryConfig, error_handler
from ..utils.performance_monitor import monitor_performance, track_calls, metrics_collector
from ..utils.cache_manager import async_cached, cache_manager

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """LLM配置类"""
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30
    max_retries: int = 3

class LLMClient:
    """LLM客户端 - 统一的LLM API接口"""
    
    def __init__(self, config_path: str = None):
        """初始化LLM客户端"""
        self.config = self._load_config(config_path)
        self.client = None
        self.zhipu_client = None
        self.client_type = None
        self._setup_clients()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path is None:
            config_path = "config/llm_config.yaml"
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载LLM配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'llm': {
                'vllm_server': {
                    'base_url': 'http://localhost:8000/v1',
                    'api_key': 'default-key',
                },
                'model': {
                    'name': 'Qwen/Qwen2.5-7B-Instruct'
                },
                'api': {
                    'temperature': 0.7,
                    'max_tokens': 1024,
                    'timeout': 30,
                    'max_retries': 3
                }
            }
        }
    
    def _setup_clients(self):
        """设置LLM客户端"""
        try:
            # vLLM客户端
            vllm_config = self.config['llm']['vllm_server']
            self.client = AsyncOpenAI(
                base_url=vllm_config['base_url'],
                api_key=vllm_config['api_key']
            )
            
            # 智普客户端（备选）
            if 'zhipu' in self.config['llm']:
                zhipu_config = self.config['llm']['zhipu']
                self.zhipu_client = AsyncOpenAI(
                    base_url=zhipu_config['base_url'],
                    api_key=zhipu_config['api_key']
                )
                
            logger.info("LLM客户端初始化成功")
            
        except Exception as e:
            logger.error(f"LLM客户端初始化失败: {e}")
    
    @retry_with_backoff(
        retry_config=RetryConfig(max_retries=3, base_delay=1.0),
        exceptions=(Exception,),
        fallback_operation="llm_call"
    )
    @circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
    @monitor_performance(name="llm.chat_completion", threshold=10.0)
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        use_zhipu: bool = False
    ) -> str:
        """
        发送聊天完成请求
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            use_zhipu: 是否使用智普API
            
        Returns:
            LLM响应文本
        """
        try:
            # 选择客户端和配置
            if use_zhipu and self.zhipu_client:
                client = self.zhipu_client
                model = model or self.config['llm']['zhipu']['model']
            else:
                client = self.client
                model = model or self.config['llm']['model']['name']
            
            # 设置参数
            api_config = self.config['llm']['api']
            temperature = temperature or api_config['temperature']
            max_tokens = max_tokens or api_config['max_tokens']
            
            # 发送请求
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=api_config['timeout']
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"LLM请求失败: {e}")
            # 如果vLLM失败，尝试智普API
            if not use_zhipu and self.zhipu_client:
                logger.info("尝试使用智普API作为备选")
                return await self.chat_completion(
                    messages, model, temperature, max_tokens, use_zhipu=True
                )
            return None

class FloodAgentLLM:
    """洪水智能体LLM集成类"""
    
    def __init__(self, config_path: str = None):
        """初始化Flood Agent LLM"""
        self.llm_client = LLMClient(config_path)
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict[str, str]:
        """加载提示模板"""
        try:
            config = self.llm_client.config
            return config.get('agent_llm', {}).get('decision_prompts', {})
        except:
            return self._get_default_prompts()
    
    def _get_default_prompts(self) -> Dict[str, str]:
        """获取默认提示模板"""
        return {
            'risk_assessment': """你是一个在洪水期间进行风险评估的智能体。
当前情况：
- 天气条件：{weather_data}
- 当前位置：{current_location}
- 个人信息：年龄{age}，收入水平{income_level}，家庭规模{family_size}
- 是否有车：{has_vehicle}
- 风险厌恶程度：{risk_aversion}

注意：洪水期间应大幅提高风险评分以促进疏散，目标是减少75-80%的日常出行。
请评估当前的风险等级（0-1，0为无风险，1为极高风险）并说明原因。
请以JSON格式回复：{"risk_score": 0.0-1.0, "reasoning": "原因说明"}""",
            
            'evacuation_decision': """你是一个需要决定是否疏散的智能体。
当前情况：
- 风险评估：{risk_score}
- 当前位置：{current_location}
- 家庭位置：{home_location}
- 工作位置：{work_location}
- 社会影响：{social_influence}
- 个人特征：{agent_profile}

重要目标：洪水期间应实现75-80%的出行减少率，请积极考虑疏散以减少不必要出行。
请决定是否需要疏散并说明原因。
请以JSON格式回复：{"should_evacuate": true/false, "destination": "目的地ID或null", "reasoning": "原因说明"}""",
            
            'mobility_planning': """你是一个规划移动路径的智能体。
当前情况：
- 当前位置：{current_location}
- 目标位置：{target_location}
- 交通工具：{transportation}
- 天气条件：{weather_conditions}

目标：减少75-80%的日常出行，优先选择安全的疏散目的地，避免不必要的移动。
请规划最佳的移动路径和时间安排。
请以JSON格式回复：{"next_location": "下一个位置ID", "transportation": "交通方式", "reasoning": "原因说明"}"""
        }
    
    @async_cached(ttl=300, namespace="llm_risk_assessment")  # 5分钟缓存
    @track_calls(name="llm.risk_assessment")
    @monitor_performance(name="llm.assess_risk", threshold=5.0)
    async def assess_risk_with_llm(
        self, 
        weather_data: Dict[str, Any],
        agent_profile: Dict[str, Any],
        current_location: int
    ) -> Dict[str, Any]:
        """使用LLM进行风险评估"""
        try:
            prompt = self.prompts['risk_assessment'].format(
                weather_data=json.dumps(weather_data, ensure_ascii=False),
                current_location=current_location,
                age=agent_profile.get('age', 35),
                income_level=agent_profile.get('income_level', 'medium'),
                family_size=agent_profile.get('family_size', 2),
                has_vehicle=agent_profile.get('has_vehicle', True),
                risk_aversion=agent_profile.get('risk_aversion', 0.5)
            )
            
            messages = [
                {"role": "system", "content": "你是一个专业的风险评估专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_client.chat_completion(messages)
            
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # 如果不是JSON格式，尝试提取数值
                    import re
                    risk_match = re.search(r'(\d+\.?\d*)', response)
                    if risk_match:
                        risk_score = float(risk_match.group(1))
                        if risk_score > 1:
                            risk_score = risk_score / 10  # 假设是0-10范围
                        return {"risk_score": risk_score, "reasoning": response}
            
            return {"risk_score": 0.5, "reasoning": "LLM评估失败，使用默认值"}
            
        except Exception as e:
            logger.error(f"LLM风险评估失败: {e}")
            return {"risk_score": 0.5, "reasoning": f"评估失败: {e}"}
    
    @async_cached(ttl=180, namespace="llm_evacuation")  # 3分钟缓存
    @track_calls(name="llm.evacuation_decision")
    @monitor_performance(name="llm.decide_evacuation", threshold=5.0)
    async def make_evacuation_decision(
        self,
        risk_score: float,
        agent_profile: Dict[str, Any],
        current_location: int,
        home_location: int,
        work_location: int,
        social_influence: float = 0.0  # 添加social_influence参数
    ) -> Dict[str, Any]:
        """使用LLM进行疏散决策"""
        try:
            prompt = self.prompts['evacuation_decision'].format(
                risk_score=risk_score,
                current_location=current_location,
                home_location=home_location,
                work_location=work_location,
                social_influence=social_influence,  # 添加到prompt中
                agent_profile=json.dumps(agent_profile, ensure_ascii=False)
            )
            
            messages = [
                {"role": "system", "content": "你是一个专业的应急疏散决策专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_client.chat_completion(messages)
            
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    # 简单的文本解析
                    should_evacuate = "疏散" in response or "evacuate" in response.lower()
                    return {
                        "should_evacuate": should_evacuate,
                        "destination": None,
                        "reasoning": response
                    }
            
            return {
                "should_evacuate": risk_score > 0.7,
                "destination": None,
                "reasoning": "LLM决策失败，使用基于风险阈值的默认决策"
            }
            
        except Exception as e:
            logger.error(f"LLM疏散决策失败: {e}")
            return {
                "should_evacuate": risk_score > 0.7,
                "destination": None,
                "reasoning": f"决策失败: {e}"
            }
    
    @async_cached(ttl=600, namespace="llm_mobility")  # 10分钟缓存
    @track_calls(name="llm.mobility_planning")
    @monitor_performance(name="llm.plan_mobility", threshold=5.0)
    async def plan_mobility_for_agent(
        self,
        weather_data: Dict[str, Any],
        agent_profile: Dict[str, Any],
        available_aois: List[int],
        current_location: int,
        should_evacuate: bool
    ) -> Dict[str, Any]:
        """为智能体规划移动（适配智能体调用）"""
        try:
            # 构建移动规划提示
            prompt = f"""你是一个在洪水期间规划移动的智能体。
当前情况：
- 天气条件：{json.dumps(weather_data, ensure_ascii=False)}
- 当前位置：{current_location}
- 可选目的地：{available_aois}
- 是否需要疏散：{should_evacuate}
- 个人特征：{json.dumps(agent_profile, ensure_ascii=False)}

请选择最佳的目的地AOI ID并说明原因。
如果不需要移动，请返回null。
请以JSON格式回复：{{"destination_aoi": AOI_ID或null, "reasoning": "原因说明"}}"""
            
            messages = [
                {"role": "system", "content": "你是一个专业的移动规划专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_client.chat_completion(messages)
            
            if response:
                try:
                    result = json.loads(response)
                    destination = result.get('destination_aoi')
                    # 验证目的地是否有效
                    if destination and destination in available_aois:
                        return result
                    elif should_evacuate and available_aois:
                        # 如果需要疏散但LLM没有选择有效目的地，选择一个安全的
                        safe_aoi = max(available_aois)  # 假设ID越大越安全
                        return {
                            "destination_aoi": safe_aoi,
                            "reasoning": f"LLM选择无效，自动选择安全位置{safe_aoi}"
                        }
                    else:
                        return {"destination_aoi": None, "reasoning": "无需移动"}
                except json.JSONDecodeError:
                    # 尝试从文本中提取AOI ID
                    import re
                    aoi_match = re.search(r'AOI[:\s]*(\d+)', response)
                    if aoi_match:
                        aoi_id = int(aoi_match.group(1))
                        if aoi_id in available_aois:
                            return {"destination_aoi": aoi_id, "reasoning": response}
                    
                    return {"destination_aoi": None, "reasoning": response}
            
            # 默认逻辑
            if should_evacuate and available_aois:
                safe_aoi = max(available_aois)
                return {
                    "destination_aoi": safe_aoi,
                    "reasoning": "LLM失败，使用默认疏散逻辑"
                }
            
            return {"destination_aoi": None, "reasoning": "LLM失败，无需移动"}
            
        except Exception as e:
            logger.error(f"LLM移动规划失败: {e}")
            # 回退逻辑
            if should_evacuate and available_aois:
                safe_aoi = max(available_aois)
                return {
                    "destination_aoi": safe_aoi,
                    "reasoning": f"规划失败: {e}，使用默认疏散逻辑"
                }
            return {"destination_aoi": None, "reasoning": f"规划失败: {e}"}

    async def plan_mobility(
        self,
        current_location: int,
        target_location: int,
        transportation: str,
        weather_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用LLM进行移动规划"""
        try:
            prompt = self.prompts['mobility_planning'].format(
                current_location=current_location,
                target_location=target_location,
                transportation=transportation,
                weather_conditions=json.dumps(weather_conditions, ensure_ascii=False)
            )
            
            messages = [
                {"role": "system", "content": "你是一个专业的移动路径规划专家。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_client.chat_completion(messages)
            
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    return {
                        "next_location": target_location,
                        "transportation": transportation,
                        "reasoning": response
                    }
            
            return {
                "next_location": target_location,
                "transportation": transportation,
                "reasoning": "LLM规划失败，使用直接路径"
            }
            
        except Exception as e:
            logger.error(f"LLM移动规划失败: {e}")
            return {
                "next_location": target_location,
                "transportation": transportation,
                "reasoning": f"规划失败: {e}"
            }

# 全局LLM实例
_llm_instance = None

def get_llm_client(config_path: str = None) -> FloodAgentLLM:
    """获取全局LLM客户端实例"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = FloodAgentLLM(config_path)
    return _llm_instance