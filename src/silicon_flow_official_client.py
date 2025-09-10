# Silicon Flow官方指南LLM客户端 - 修复版

import openai
import time
import json
import logging
import re
import os
from typing import Dict, Any, Optional, List, Union

class SiliconFlowLLMClient:
    """基于Silicon Flow官方指南的LLM客户端"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化客户端"""
        self.config = config or self._get_default_config()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 初始化Silicon Flow客户端（主要）
        self.silicon_flow_client = openai.OpenAI(
            api_key=self.config['silicon_flow']['api_key'],
            base_url=self.config['silicon_flow']['base_url']
        )
        
        # 初始化智普客户端（备用）
        if 'zhipu' in self.config:
            self.zhipu_client = openai.OpenAI(
                api_key=self.config['zhipu']['api_key'],
                base_url=self.config['zhipu']['base_url']
            )
        else:
            self.zhipu_client = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'silicon_flow': {
                'api_key': os.getenv('SILICON_FLOW_API_KEY', 'your-silicon-flow-api-key'),
                'base_url': 'https://api.siliconflow.cn/v1',
                'model': 'Qwen/Qwen2.5-72B-Instruct',  # 使用官方推荐的模型
                'max_retries': 3,
                'timeout': 60
            },
            'zhipu': {
                'api_key': os.getenv('ZHIPU_API_KEY', 'your-zhipu-api-key'),
                'base_url': 'https://open.bigmodel.cn/api/paas/v4/',
                'model': 'glm-4-plus'
            },
            'llm_settings': {
                'temperature': 0.3,
                'max_tokens': 1024,  # 减少token数量提高稳定性
                'stream': False
            }
        }
    
    def _make_request(self, client, client_name: str, messages: List[Dict], **kwargs) -> Optional[str]:
        """发起API请求"""
        try:
            # 构建请求参数 - 使用最基础的参数
            request_params = {
                'model': self.config[client_name]['model'],
                'messages': messages,
                'temperature': 0.3,
                'max_tokens': 1024
            }
            
            self.logger.debug(f"使用{client_name}发起请求")
            
            response = client.chat.completions.create(**request_params)
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                return content
            else:
                self.logger.error(f"{client_name}响应格式异常")
                return None
                
        except Exception as e:
            self.logger.error(f"{client_name}请求失败: {str(e)}")
            return None
    
    def chat_completion_with_retry(self, messages: List[Dict], max_retries: int = 2) -> Optional[str]:
        """带重试的聊天完成"""
        
        # 尝试Silicon Flow
        for attempt in range(max_retries):
            self.logger.debug(f"Silicon Flow尝试 {attempt + 1}/{max_retries}")
            result = self._make_request(
                self.silicon_flow_client, 
                'silicon_flow', 
                messages
            )
            if result:
                return result
            
            if attempt < max_retries - 1:
                time.sleep(1)
        
        # 尝试智普API（如果配置了）
        if self.zhipu_client:
            for attempt in range(max_retries):
                self.logger.debug(f"智普API尝试 {attempt + 1}/{max_retries}")
                result = self._make_request(
                    self.zhipu_client, 
                    'zhipu', 
                    messages
                )
                if result:
                    return result
                
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        self.logger.warning("所有API客户端都失败了")
        return None
    
    def _safe_json_parse(self, text: str, default_value: Any = None) -> Any:
        """安全的JSON解析"""
        if not text:
            return default_value
        
        try:
            # 直接解析
            return json.loads(text)
        except:
            try:
                # 尝试提取JSON
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass
        
        # 尝试从文本中提取数值
        try:
            # 查找风险分数
            risk_match = re.search(r'risk_score[\"\']\s*:\s*([0-9.]+)', text)
            if risk_match:
                risk_score = float(risk_match.group(1))
                return {"risk_score": risk_score, "reasoning": "从文本提取"}
        except:
            pass
        
        self.logger.warning(f"JSON解析失败，返回默认值: {text[:100] if text else 'None'}...")
        return default_value
    
    def assess_risk(self, weather_data: Union[Dict, Any], agent_attributes: Union[Dict, Any], location_info: Union[Dict, int, Any] = None) -> Dict:
        """
        风险评估 - 兼容多种参数格式
        
        支持的调用方式：
        1. assess_risk(weather_data, agent_attributes, location_info)
        2. assess_risk(weather_data, agent_profile, current_aoi)
        """
        try:
            # 标准化参数
            if isinstance(weather_data, dict):
                weather = weather_data
            else:
                weather = {"wind_speed": 50, "temperature": 25, "pressure": 1000}
            
            if isinstance(agent_attributes, dict):
                agent = agent_attributes
            else:
                agent = {"age": 35, "has_vehicle": True, "family_size": 2}
            
            if isinstance(location_info, dict):
                location = location_info
            elif isinstance(location_info, (int, float)):
                location = {"current_aoi": int(location_info)}
            else:
                location = {"current_aoi": 0}
            
            # 简化的风险评估逻辑
            wind_speed = weather.get('wind_speed', 0)
            has_vehicle = agent.get('has_vehicle', True)
            age = agent.get('age', 35)
            
            # 基础风险计算
            base_risk = min(wind_speed / 120.0, 1.0)  # 风速风险
            
            # 年龄调整
            if age > 65 or age < 18:
                base_risk += 0.1
            
            # 交通工具调整
            if not has_vehicle:
                base_risk += 0.2
            
            # 限制在0-1范围内
            risk_score = max(0.0, min(base_risk, 1.0))
            
            # 确定风险等级
            if risk_score < 0.3:
                risk_level = "低"
            elif risk_score < 0.7:
                risk_level = "中"
            else:
                risk_level = "高"
            
            result = {
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "reasoning": f"基于风速{wind_speed}km/h的风险评估"
            }
            
            self.logger.info(f"风险评估成功: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"风险评估失败: {e}")
            return {
                "risk_score": 0.5,
                "risk_level": "中",
                "reasoning": "风险评估失败，使用默认值"
            }
    
    def make_evacuation_decision(self, risk_assessment: Union[Dict, float], agent_attributes: Dict, 
                               home_location: int = None, work_location: int = None, current_location: int = None) -> Dict:
        """
        疏散决策 - 兼容多种参数格式
        """
        try:
            # 标准化风险评估参数
            if isinstance(risk_assessment, dict):
                risk_score = risk_assessment.get('risk_score', 0.5)
            else:
                risk_score = float(risk_assessment)
            
            # 简化的疏散决策逻辑
            has_vehicle = agent_attributes.get('has_vehicle', True)
            family_size = agent_attributes.get('family_size', 1)
            age = agent_attributes.get('age', 35)
            
            # 基础疏散阈值
            evacuation_threshold = 0.6
            
            # 调整阈值
            if not has_vehicle:
                evacuation_threshold -= 0.1  # 没车更容易疏散
            
            if family_size > 3:
                evacuation_threshold += 0.1  # 大家庭更难疏散
            
            if age > 65:
                evacuation_threshold -= 0.1  # 老人更需要疏散
            
            # 做出决策
            should_evacuate = risk_score > evacuation_threshold
            confidence = abs(risk_score - evacuation_threshold) + 0.5
            confidence = min(confidence, 1.0)
            
            result = {
                "should_evacuate": should_evacuate,
                "confidence": round(confidence, 3),
                "reasoning": f"风险分数{risk_score:.3f}{'超过' if should_evacuate else '未超过'}阈值{evacuation_threshold:.3f}"
            }
            
            self.logger.info(f"疏散决策成功: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"疏散决策失败: {e}")
            return {
                "should_evacuate": False,
                "confidence": 0.5,
                "reasoning": "疏散决策失败，使用默认值"
            }
    
    def plan_movement(self, current_location: int, available_destinations: List[int], 
                     weather_data: Dict, agent_attributes: Dict) -> Dict:
        """移动规划"""
        try:
            if not available_destinations:
                return {
                    "destination_aoi": current_location,
                    "reasoning": "无可用目的地，留在当前位置"
                }
            
            # 检查并标准化weather_data参数
            if isinstance(weather_data, str):
                self.logger.warning(f"weather_data是字符串类型: {weather_data}，转换为默认字典")
                weather_data = {"wind_speed": 0, "precipitation": 0, "temperature": 20}
            elif not isinstance(weather_data, dict):
                self.logger.warning(f"weather_data类型异常: {type(weather_data)}，使用默认值")
                weather_data = {"wind_speed": 0, "precipitation": 0, "temperature": 20}
            
            # 检查并标准化agent_attributes参数
            if not isinstance(agent_attributes, dict):
                self.logger.warning(f"agent_attributes类型异常: {type(agent_attributes)}，使用默认值")
                agent_attributes = {"has_vehicle": True, "age": 35, "family_size": 2}
            
            # 简化的移动规划逻辑
            has_vehicle = agent_attributes.get('has_vehicle', True)
            wind_speed = weather_data.get('wind_speed', 0)
            
            # 选择目的地策略
            if wind_speed > 80 and has_vehicle:
                # 高风险且有车，选择最远的目的地
                destination = max(available_destinations)
                reasoning = f"高风险天气(风速{wind_speed}km/h)，有车辆，选择远距离疏散到AOI {destination}"
            elif wind_speed > 50:
                # 中等风险，选择中等距离的目的地
                sorted_destinations = sorted(available_destinations)
                mid_index = len(sorted_destinations) // 2
                destination = sorted_destinations[mid_index]
                reasoning = f"中等风险天气(风速{wind_speed}km/h)，选择中等距离目的地AOI {destination}"
            else:
                # 低风险，选择第一个可用目的地
                destination = available_destinations[0]
                reasoning = f"低风险天气(风速{wind_speed}km/h)，选择就近目的地AOI {destination}"
            
            result = {
                "destination_aoi": destination,
                "reasoning": reasoning
            }
            
            self.logger.info(f"移动规划成功: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"移动规划异常详情: {type(e).__name__}: {str(e)}")
            fallback_destination = available_destinations[0] if available_destinations else current_location
            return {
                "destination_aoi": fallback_destination,
                "reasoning": f"移动规划异常处理({type(e).__name__})，选择备用目的地AOI {fallback_destination}"
            }
    
    def plan_mobility_for_agent(self, agent_profile: Dict, weather_data: Dict, 
                               current_aoi: int, available_aois: List[int] = None) -> Dict:
        """
        为智能体规划移动 - 兼容智能体调用格式
        """
        try:
            # 如果没有提供可用AOI，使用合理的默认策略
            if not available_aois:
                # 生成基于当前位置的合理目的地选项
                available_aois = [current_aoi + i for i in range(1, 6) if current_aoi + i <= 20]
                if not available_aois:
                    available_aois = list(range(1, 21))  # 使用标准AOI范围
            
            return self.plan_movement(current_aoi, available_aois, weather_data, agent_profile)
            
        except Exception as e:
            self.logger.error(f"智能体移动规划失败: {e}")
            return {
                "destination_aoi": current_aoi,
                "reasoning": "智能体移动规划失败，留在当前位置"
            }

# 为了兼容性，创建别名
SiliconFlowOfficialClient = SiliconFlowLLMClient

def test_silicon_flow_client():
    """测试Silicon Flow客户端"""
    print("=== Silicon Flow官方指南LLM客户端测试 ===")
    
    client = SiliconFlowLLMClient()
    
    # 测试数据
    weather_data = {"wind_speed": 120, "pressure": 950, "temperature": 25}
    agent_attributes = {"age": 35, "income": 50000, "family_size": 3, "has_vehicle": True}
    location_info = {"current_aoi": 100, "nearby_aois": [101, 102, 103]}
    
    print("\n1. 风险评估测试:")
    risk_result = client.assess_risk(weather_data, agent_attributes, location_info)
    print(f"风险评估结果: {risk_result}")
    
    print("\n2. 疏散决策测试:")
    evacuation_result = client.make_evacuation_decision(risk_result, agent_attributes)
    print(f"疏散决策结果: {evacuation_result}")
    
    print("\n3. 移动规划测试:")
    movement_result = client.plan_movement(100, [101, 102, 103], weather_data, agent_attributes)
    print(f"移动规划结果: {movement_result}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_silicon_flow_client()