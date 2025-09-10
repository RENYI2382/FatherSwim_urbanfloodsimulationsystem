"""
DeepSeek-R1 LLM集成模块
基于Silicon Flow API的DeepSeek-R1模型接口
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeepSeekConfig:
    """DeepSeek-R1配置类"""
    api_key: str = "sk-jesuodwohxitnklkmpextgqocusbsbpwpaicxejwxemfvbsp"
    base_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    model: str = "deepseek-ai/DeepSeek-R1"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 30

class DeepSeekR1Client:
    """DeepSeek-R1客户端"""
    
    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig()
        self.session = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """发送请求到DeepSeek-R1 API"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "stream": False
        }
        
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                )
            
            async with self.session.post(
                self.config.base_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"DeepSeek API错误 {response.status}: {error_text}")
                    raise Exception(f"API请求失败: {response.status}")
                    
        except Exception as e:
            logger.error(f"DeepSeek API请求异常: {e}")
            raise
    
    async def assess_hurricane_risk(
        self, 
        weather_data: Dict[str, Any], 
        agent_profile: Dict[str, Any],
        current_location: int
    ) -> Dict[str, Any]:
        """使用DeepSeek-R1评估飓风风险"""
        
        system_prompt = """你是一个专业的飓风风险评估专家。请基于以下信息评估个人的风险等级：

评估框架：
1. 风险感知与评估 (Risk Perception & Assessment)
   - 官方预警等级、风暴潮预测、洪水风险
   - 个人经验和社区信息
   
2. 疏散自我效能评估 (Evacuation Self-Efficacy)
   - 交通工具可得性、经济能力
   - 家庭状况、避难所信息、时间预算
   
3. 社会影响与规范 (Social Influence & Norms)
   - 家人意见、邻居行为、社区凝聚力

请返回JSON格式：
{
    "risk_score": 0.0-1.0,
    "risk_level": "low/medium/high",
    "reasoning": "详细分析原因",
    "evacuation_recommendation": "stay/prepare/evacuate",
    "confidence": 0.0-1.0
}"""

        user_prompt = f"""
天气数据：
- 飓风等级: {weather_data.get('hurricane_category', 'N/A')}
- 风速: {weather_data.get('wind_speed', 'N/A')} mph
- 降雨量: {weather_data.get('rainfall', 'N/A')} mm
- 风暴潮高度: {weather_data.get('storm_surge', 'N/A')} ft
- 预警等级: {weather_data.get('warning_level', 'N/A')}

个人档案：
- 年龄: {agent_profile.get('age', 35)}
- 收入水平: {agent_profile.get('income_level', 'medium')}
- 家庭规模: {agent_profile.get('family_size', 2)}
- 拥有车辆: {agent_profile.get('has_vehicle', True)}
- 风险厌恶程度: {agent_profile.get('risk_aversion', 0.5)}
- 当前位置: AOI {current_location}

请基于飓风风险评估框架进行专业分析。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self._make_request(messages, temperature=0.3)
            content = response["choices"][0]["message"]["content"]
            
            # 尝试解析JSON
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                # 如果不是JSON格式，提取关键信息
                logger.warning("DeepSeek返回非JSON格式，尝试解析")
                return self._parse_text_response(content)
                
        except Exception as e:
            logger.error(f"DeepSeek风险评估失败: {e}")
            # 返回默认值
            return {
                "risk_score": 0.5,
                "risk_level": "medium",
                "reasoning": f"API调用失败，使用默认评估: {str(e)}",
                "evacuation_recommendation": "prepare",
                "confidence": 0.3
            }
    
    async def make_evacuation_decision(
        self,
        risk_assessment: Dict[str, Any],
        agent_profile: Dict[str, Any],
        social_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用DeepSeek-R1做出疏散决策"""
        
        system_prompt = """你是一个疏散决策专家。基于风险评估和个人情况，帮助做出疏散决策。

决策逻辑框架：
1. 优先考虑官方强制疏散令
2. 高风险感知 + 高自我效能 + 社会支持 → 疏散
3. 高风险感知 + 低自我效能 → 寻求帮助或延迟疏散
4. 中低风险感知 → 就地避险

返回JSON格式：
{
    "decision": "evacuate/stay/prepare",
    "confidence": 0.0-1.0,
    "reasoning": "决策理由",
    "timing": "immediate/within_hours/monitor",
    "destination_type": "shelter/family/hotel/none"
}"""

        user_prompt = f"""
风险评估结果：
- 风险等级: {risk_assessment.get('risk_level', 'medium')}
- 风险分数: {risk_assessment.get('risk_score', 0.5)}
- 推荐行动: {risk_assessment.get('evacuation_recommendation', 'prepare')}

个人档案：
- 年龄: {agent_profile.get('age', 35)}
- 家庭规模: {agent_profile.get('family_size', 2)}
- 拥有车辆: {agent_profile.get('has_vehicle', True)}
- 经济能力: {agent_profile.get('income_level', 'medium')}

社会环境：
- 邻居疏散比例: {social_context.get('neighbor_evacuation_rate', 0.5)}
- 官方疏散令: {social_context.get('mandatory_evacuation', False)}
- 家人支持: {social_context.get('family_support', True)}

请基于决策逻辑框架做出疏散决策。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self._make_request(messages, temperature=0.2)
            content = response["choices"][0]["message"]["content"]
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                return self._parse_decision_response(content)
                
        except Exception as e:
            logger.error(f"DeepSeek疏散决策失败: {e}")
            return {
                "decision": "prepare",
                "confidence": 0.3,
                "reasoning": f"API调用失败，使用默认决策: {str(e)}",
                "timing": "monitor",
                "destination_type": "none"
            }
    
    async def generate_mobility_pattern(
        self,
        decision: Dict[str, Any],
        agent_profile: Dict[str, Any],
        time_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用DeepSeek-R1生成移动模式"""
        
        system_prompt = """你是一个人类移动行为专家。基于疏散决策和个人情况，生成合理的移动模式。

移动模式考虑因素：
1. 疏散决策（疏散/留守/准备）
2. 时间紧迫性
3. 交通状况
4. 个人能力

返回JSON格式：
{
    "movement_type": "evacuation/shelter_in_place/normal/emergency_prep",
    "destination_aoi": 目标AOI编号或null,
    "travel_mode": "driving/walking/public_transport",
    "urgency_level": 0.0-1.0,
    "estimated_duration": 预计时长(分钟),
    "route_preference": "fastest/safest/familiar"
}"""

        user_prompt = f"""
疏散决策：
- 决策: {decision.get('decision', 'prepare')}
- 时机: {decision.get('timing', 'monitor')}
- 目的地类型: {decision.get('destination_type', 'none')}

个人档案：
- 拥有车辆: {agent_profile.get('has_vehicle', True)}
- 家庭规模: {agent_profile.get('family_size', 2)}
- 年龄: {agent_profile.get('age', 35)}

时间环境：
- 当前时间: {time_context.get('current_hour', 12)}
- 飓风阶段: {time_context.get('hurricane_phase', 'approaching')}
- 交通拥堵程度: {time_context.get('traffic_congestion', 0.5)}

请生成合理的移动模式。
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = await self._make_request(messages, temperature=0.4)
            content = response["choices"][0]["message"]["content"]
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                return self._parse_mobility_response(content)
                
        except Exception as e:
            logger.error(f"DeepSeek移动模式生成失败: {e}")
            return {
                "movement_type": "normal",
                "destination_aoi": None,
                "travel_mode": "walking",
                "urgency_level": 0.3,
                "estimated_duration": 30,
                "route_preference": "familiar"
            }
    
    def _parse_text_response(self, content: str) -> Dict[str, Any]:
        """解析文本响应，提取关键信息"""
        # 简单的文本解析逻辑
        risk_score = 0.5
        if "高风险" in content or "high risk" in content.lower():
            risk_score = 0.8
        elif "低风险" in content or "low risk" in content.lower():
            risk_score = 0.3
        
        return {
            "risk_score": risk_score,
            "risk_level": "medium",
            "reasoning": content[:200],
            "evacuation_recommendation": "prepare",
            "confidence": 0.5
        }
    
    def _parse_decision_response(self, content: str) -> Dict[str, Any]:
        """解析决策响应"""
        decision = "prepare"
        if "疏散" in content or "evacuate" in content.lower():
            decision = "evacuate"
        elif "留守" in content or "stay" in content.lower():
            decision = "stay"
        
        return {
            "decision": decision,
            "confidence": 0.5,
            "reasoning": content[:200],
            "timing": "monitor",
            "destination_type": "none"
        }
    
    def _parse_mobility_response(self, content: str) -> Dict[str, Any]:
        """解析移动模式响应"""
        return {
            "movement_type": "normal",
            "destination_aoi": None,
            "travel_mode": "walking",
            "urgency_level": 0.3,
            "estimated_duration": 30,
            "route_preference": "familiar"
        }

# 全局客户端实例
_deepseek_client = None

async def get_deepseek_client() -> DeepSeekR1Client:
    """获取DeepSeek客户端实例"""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = DeepSeekR1Client()
    return _deepseek_client

async def cleanup_deepseek_client():
    """清理DeepSeek客户端"""
    global _deepseek_client
    if _deepseek_client and _deepseek_client.session:
        await _deepseek_client.session.close()
        _deepseek_client = None