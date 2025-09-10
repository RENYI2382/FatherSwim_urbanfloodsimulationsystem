"""LLM提示工程模块

基于差序格局理论设计的智能体决策提示系统，注入内外有别、亲疏有度的社会逻辑，
确保智能体在洪灾情境下按照中国传统社会结构进行决策。

作者: 城市智能体项目组
日期: 2024-12-30
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
from datetime import datetime

from .differential_order_strategies import (
    DifferentialOrderStrategy, StrictDifferentialOrder, 
    ModerateDifferentialOrder, Universalism,
    HelpRequest, HelpDecision
)
from .agent_attributes import IntegratedAgentAttributes
from .social_network import SocialNetwork, RelationshipType

logger = logging.getLogger(__name__)


class DecisionContext(Enum):
    """决策情境枚举"""
    EVACUATION_DECISION = "evacuation_decision"  # 疏散决策
    HELP_SEEKING = "help_seeking"  # 求助决策
    HELP_PROVIDING = "help_providing"  # 助人决策
    RESOURCE_ALLOCATION = "resource_allocation"  # 资源分配
    ROUTE_SELECTION = "route_selection"  # 路径选择
    SHELTER_SELECTION = "shelter_selection"  # 避难所选择
    INFORMATION_SHARING = "information_sharing"  # 信息分享
    RISK_ASSESSMENT = "risk_assessment"  # 风险评估


class EmergencyPhase(Enum):
    """应急阶段枚举"""
    PREPARATION = "preparation"  # 准备阶段
    WARNING = "warning"  # 预警阶段
    IMPACT = "impact"  # 冲击阶段
    RESPONSE = "response"  # 响应阶段
    RECOVERY = "recovery"  # 恢复阶段


@dataclass
class DecisionScenario:
    """决策场景"""
    context: DecisionContext
    phase: EmergencyPhase
    urgency_level: float  # 0-1，紧急程度
    risk_level: float  # 0-1，风险等级
    available_options: List[str]  # 可选方案
    social_pressure: float  # 0-1，社会压力
    resource_constraints: Dict[str, float]  # 资源约束
    time_pressure: float  # 0-1，时间压力
    information_completeness: float  # 0-1，信息完整度
    
    # 社会情境
    family_members_present: List[str] = field(default_factory=list)
    friends_present: List[str] = field(default_factory=list)
    neighbors_present: List[str] = field(default_factory=list)
    strangers_present: List[str] = field(default_factory=list)
    authority_present: bool = False
    
    # 环境情境
    location: str = ""
    weather_conditions: str = ""
    infrastructure_status: Dict[str, str] = field(default_factory=dict)
    communication_status: str = "normal"


class DifferentialOrderPromptEngine:
    """差序格局提示工程引擎"""
    
    def __init__(self):
        self.base_prompts = self._load_base_prompts()
        self.strategy_prompts = self._load_strategy_prompts()
        self.context_prompts = self._load_context_prompts()
        self.cultural_principles = self._load_cultural_principles()
    
    def _load_base_prompts(self) -> Dict[str, str]:
        """加载基础提示模板"""
        return {
            "system_role": """
你是一个基于中国传统社会结构"差序格局"理论的智能体，正在面临洪灾等自然灾害的威胁。
你的行为遵循费孝通先生提出的"差序格局"原则：以自我为中心，按照亲疏远近的关系差序来决定行为策略。

核心原则：
1. 内外有别：对内群体（家人、亲戚、朋友）和外群体（陌生人、外族人）采用不同标准
2. 亲疏有度：根据关系远近调整帮助意愿和资源分配
3. 互惠原则：优先帮助那些可能回报的人，维护长期关系
4. 面子考量：在公共场合要维护自己和家族的声誉
5. 权威尊重：对政府官员、长辈、专家等权威人士给予特殊考虑
""",
            
            "decision_framework": """
在做决策时，请按照以下框架思考：

1. 【关系识别】首先识别涉及的人员关系：
   - 家人（最高优先级）
   - 亲戚（高优先级）
   - 朋友/熟人（中等优先级）
   - 邻居/同事（较低优先级）
   - 陌生人（最低优先级）

2. 【利益权衡】评估各方利益：
   - 自身和家庭利益
   - 内群体利益
   - 长期关系维护
   - 社会声誉影响

3. 【资源分配】根据关系远近分配资源：
   - 优先保障核心家庭成员
   - 其次考虑扩展家庭和密友
   - 最后考虑其他人群

4. 【行为选择】选择符合差序格局逻辑的行为
""",
            
            "response_format": """
请按照以下格式回应：

【思考过程】
- 当前情况分析
- 关系网络识别
- 利益冲突评估
- 文化原则考量

【决策结果】
- 具体行动方案
- 资源分配策略
- 预期结果评估

【理由说明】
- 符合差序格局的逻辑
- 文化价值观体现
- 长期关系考虑
"""
        }
    
    def _load_strategy_prompts(self) -> Dict[str, str]:
        """加载策略特定提示"""
        return {
            "strict": """
你是一个"强差序格局"类型的人，具有以下特征：
- 极强的家族观念，家人利益高于一切
- 明确的内外群体界限，对外人保持警惕
- 严格按照关系远近分配资源和注意力
- 重视传统礼仪和等级秩序
- 在危机中优先保护"自己人"

决策时要体现：
1. 家人 > 亲戚 > 朋友 > 邻居 > 陌生人的严格优先级
2. 对陌生人的帮助非常有限，除非有明确回报
3. 重视面子和声誉，避免在熟人面前失面子
4. 倾向于依赖熟人网络解决问题
""",
            
            "moderate": """
你是一个"弱差序格局"类型的人，具有以下特征：
- 有家族观念但不极端，会考虑更广泛的社会责任
- 内外群体界限相对模糊，对外人有一定同情心
- 在关系远近和普遍原则之间寻求平衡
- 既重视传统也接受现代价值观
- 在危机中会适度帮助他人

决策时要体现：
1. 仍有关系远近的考虑，但不绝对
2. 会在能力范围内帮助陌生人，特别是弱势群体
3. 考虑社会道德和公共利益
4. 在传统关系网络和现代制度之间灵活选择
""",
            
            "universalist": """
你是一个"普遍主义"类型的人，具有以下特征：
- 相信人人平等，不因关系远近区别对待
- 按照普遍的道德原则和法律规范行事
- 重视个人权利和社会公正
- 倾向于通过正式制度解决问题
- 在危机中会无私帮助他人

决策时要体现：
1. 按需分配资源，不因关系远近而偏私
2. 优先帮助最需要帮助的人（如老人、儿童、残疾人）
3. 相信并依赖政府和专业机构
4. 重视规则和程序的公正性
"""
        }
    
    def _load_context_prompts(self) -> Dict[str, str]:
        """加载情境特定提示"""
        return {
            "evacuation_decision": """
【疏散决策情境】
你需要决定是否疏散以及如何疏散。考虑因素：
- 家人的安全和意愿
- 熟人的疏散计划
- 对政府疏散建议的信任度
- 财产和生计的保护
- 疏散路径的安全性

请根据你的差序格局类型做出决策。
""",
            
            "help_seeking": """
【求助决策情境】
你遇到困难需要寻求帮助。考虑因素：
- 向谁求助（关系远近）
- 如何求助（直接/间接）
- 能提供什么回报
- 是否会欠人情
- 对方的帮助能力

请根据你的差序格局类型选择求助策略。
""",
            
            "help_providing": """
【助人决策情境】
有人向你求助，你需要决定是否帮助以及帮助程度。考虑因素：
- 求助者与你的关系
- 你的帮助能力
- 预期的回报或互惠
- 其他人的看法
- 帮助的风险和成本

请根据你的差序格局类型做出助人决策。
""",
            
            "resource_allocation": """
【资源分配情境】
你有有限的资源（食物、水、药品、交通工具等）需要分配。考虑因素：
- 家人的需求
- 其他人的需求程度
- 关系远近
- 未来的互惠可能
- 社会道德压力

请根据你的差序格局类型制定分配方案。
""",
            
            "information_sharing": """
【信息分享情境】
你获得了重要的灾害信息，需要决定与谁分享。考虑因素：
- 信息的准确性和重要性
- 分享对象的关系远近
- 信息传播的风险
- 优先通知的人群
- 信息的独占价值

请根据你的差序格局类型决定信息分享策略。
"""
        }
    
    def _load_cultural_principles(self) -> Dict[str, List[str]]:
        """加载文化原则"""
        return {
            "core_values": [
                "家庭至上：家人利益高于个人利益",
                "关系本位：通过关系网络解决问题",
                "互惠原则：施恩图报，维护长期关系",
                "面子文化：维护个人和家族声誉",
                "等级秩序：尊重长幼有序、上下有别",
                "内外有别：区分自己人和外人",
                "中庸之道：避免极端，寻求平衡",
                "集体和谐：维护群体稳定和团结"
            ],
            
            "behavioral_norms": [
                "先家后己：优先考虑家庭成员需求",
                "亲疏有序：按关系远近分配资源",
                "礼尚往来：维护互惠关系网络",
                "给人留面：在公共场合维护他人尊严",
                "尊老爱幼：特别照顾老人和儿童",
                "男主外女主内：性别角色分工",
                "听从权威：尊重政府、专家、长辈意见",
                "避免冲突：通过协商和妥协解决分歧"
            ],
            
            "decision_heuristics": [
                "关系优先：关系越近，优先级越高",
                "互惠考量：评估长期互惠价值",
                "声誉影响：考虑行为对声誉的影响",
                "集体利益：平衡个人与集体利益",
                "风险规避：避免过度冒险",
                "信息依赖：依赖熟人提供的信息",
                "权威服从：在不确定时听从权威",
                "面子维护：避免在熟人面前丢脸"
            ]
        }
    
    def generate_decision_prompt(self, 
                               agent_attrs: IntegratedAgentAttributes,
                               scenario: DecisionScenario,
                               social_network: Optional[SocialNetwork] = None,
                               additional_context: Optional[Dict[str, Any]] = None) -> str:
        """生成决策提示"""
        
        # 构建基础提示
        prompt_parts = []
        
        # 1. 系统角色设定
        prompt_parts.append(self.base_prompts["system_role"])
        
        # 2. 策略特定提示
        strategy_type = agent_attrs.strategy_phenotype
        if strategy_type in self.strategy_prompts:
            prompt_parts.append(self.strategy_prompts[strategy_type])
        
        # 3. 个人属性描述
        personal_desc = self._generate_personal_description(agent_attrs)
        prompt_parts.append(f"\n【你的个人情况】\n{personal_desc}")
        
        # 4. 社会关系描述
        if social_network:
            social_desc = self._generate_social_description(agent_attrs.personal.agent_id, social_network)
            prompt_parts.append(f"\n【你的社会关系】\n{social_desc}")
        
        # 5. 当前情境描述
        scenario_desc = self._generate_scenario_description(scenario)
        prompt_parts.append(f"\n【当前情境】\n{scenario_desc}")
        
        # 6. 情境特定提示
        if scenario.context.value in self.context_prompts:
            prompt_parts.append(f"\n{self.context_prompts[scenario.context.value]}")
        
        # 7. 文化原则提醒
        cultural_reminder = self._generate_cultural_reminder(strategy_type)
        prompt_parts.append(f"\n【文化原则提醒】\n{cultural_reminder}")
        
        # 8. 决策框架
        prompt_parts.append(f"\n{self.base_prompts['decision_framework']}")
        
        # 9. 回应格式
        prompt_parts.append(f"\n{self.base_prompts['response_format']}")
        
        # 10. 附加上下文
        if additional_context:
            context_desc = self._format_additional_context(additional_context)
            prompt_parts.append(f"\n【附加信息】\n{context_desc}")
        
        return "\n\n".join(prompt_parts)
    
    def _generate_personal_description(self, agent_attrs: IntegratedAgentAttributes) -> str:
        """生成个人情况描述"""
        personal = agent_attrs.personal
        social = agent_attrs.social
        economic = agent_attrs.economic
        
        desc_parts = []
        
        # 基本信息
        desc_parts.append(f"年龄：{personal.age}岁，{personal.gender.value}")
        desc_parts.append(f"教育：{personal.education_level.value}")
        desc_parts.append(f"职业：{personal.occupation.value}")
        desc_parts.append(f"健康状况：{personal.health_status.value}")
        
        # 经济状况
        desc_parts.append(f"收入水平：{economic.income_level.value}")
        desc_parts.append(f"住房：{economic.housing_type.value}")
        
        # 心理特征
        desc_parts.append(f"风险容忍度：{personal.risk_tolerance:.2f}")
        desc_parts.append(f"对权威的信任：{personal.trust_in_authority:.2f}")
        desc_parts.append(f"社区归属感：{social.community_attachment:.2f}")
        
        # 灾害经验
        if personal.disaster_experience > 0:
            desc_parts.append(f"有{personal.disaster_experience}次灾害经历")
        
        return "\n".join([f"- {desc}" for desc in desc_parts])
    
    def _generate_social_description(self, agent_id: str, social_network: SocialNetwork) -> str:
        """生成社会关系描述"""
        desc_parts = []
        
        # 获取各类关系
        family_relations = social_network.get_relationships_by_type(agent_id, RelationshipType.FAMILY)
        friend_relations = social_network.get_relationships_by_type(agent_id, RelationshipType.FRIEND)
        neighbor_relations = social_network.get_relationships_by_type(agent_id, RelationshipType.NEIGHBOR)
        colleague_relations = social_network.get_relationships_by_type(agent_id, RelationshipType.COLLEAGUE)
        
        if family_relations:
            desc_parts.append(f"家人：{len(family_relations)}人")
        
        if friend_relations:
            desc_parts.append(f"朋友：{len(friend_relations)}人")
        
        if neighbor_relations:
            desc_parts.append(f"邻居：{len(neighbor_relations)}人")
        
        if colleague_relations:
            desc_parts.append(f"同事：{len(colleague_relations)}人")
        
        # 差序格局层次分析
        differential_layers = social_network.get_differential_order_layers(agent_id)
        if differential_layers:
            desc_parts.append("\n关系层次：")
            for layer, members in differential_layers.items():
                if members:
                    desc_parts.append(f"- {layer}：{len(members)}人")
        
        return "\n".join(desc_parts) if desc_parts else "暂无明确的社会关系网络"
    
    def _generate_scenario_description(self, scenario: DecisionScenario) -> str:
        """生成情境描述"""
        desc_parts = []
        
        # 基本情境
        desc_parts.append(f"决策类型：{scenario.context.value}")
        desc_parts.append(f"应急阶段：{scenario.phase.value}")
        desc_parts.append(f"紧急程度：{scenario.urgency_level:.2f}")
        desc_parts.append(f"风险等级：{scenario.risk_level:.2f}")
        
        # 时间和信息压力
        if scenario.time_pressure > 0.7:
            desc_parts.append("时间非常紧迫")
        elif scenario.time_pressure > 0.4:
            desc_parts.append("时间较为紧迫")
        
        if scenario.information_completeness < 0.5:
            desc_parts.append("信息不完整，存在不确定性")
        
        # 社会情境
        social_context = []
        if scenario.family_members_present:
            social_context.append(f"家人在场：{len(scenario.family_members_present)}人")
        if scenario.friends_present:
            social_context.append(f"朋友在场：{len(scenario.friends_present)}人")
        if scenario.neighbors_present:
            social_context.append(f"邻居在场：{len(scenario.neighbors_present)}人")
        if scenario.strangers_present:
            social_context.append(f"陌生人在场：{len(scenario.strangers_present)}人")
        if scenario.authority_present:
            social_context.append("有政府官员或专业人员在场")
        
        if social_context:
            desc_parts.append("\n在场人员：")
            desc_parts.extend([f"- {ctx}" for ctx in social_context])
        
        # 可选方案
        if scenario.available_options:
            desc_parts.append("\n可选方案：")
            desc_parts.extend([f"- {option}" for option in scenario.available_options])
        
        # 资源约束
        if scenario.resource_constraints:
            desc_parts.append("\n资源约束：")
            for resource, level in scenario.resource_constraints.items():
                desc_parts.append(f"- {resource}：{level:.2f}")
        
        return "\n".join(desc_parts)
    
    def _generate_cultural_reminder(self, strategy_type: str) -> str:
        """生成文化原则提醒"""
        core_values = self.cultural_principles["core_values"]
        behavioral_norms = self.cultural_principles["behavioral_norms"]
        
        if strategy_type == "strict":
            # 强差序格局：强调传统价值观
            selected_values = core_values[:4]  # 家庭至上、关系本位、互惠原则、面子文化
            selected_norms = behavioral_norms[:4]  # 先家后己、亲疏有序、礼尚往来、给人留面
        elif strategy_type == "moderate":
            # 弱差序格局：平衡传统与现代
            selected_values = [core_values[0], core_values[2], core_values[6], core_values[7]]
            selected_norms = [behavioral_norms[0], behavioral_norms[2], behavioral_norms[4], behavioral_norms[7]]
        else:  # universalist
            # 普遍主义：强调普遍价值
            selected_values = [core_values[6], core_values[7]]  # 中庸之道、集体和谐
            selected_norms = [behavioral_norms[4], behavioral_norms[6], behavioral_norms[7]]  # 尊老爱幼、听从权威、避免冲突
        
        reminder_parts = []
        reminder_parts.append("核心价值观：")
        reminder_parts.extend([f"- {value}" for value in selected_values])
        reminder_parts.append("\n行为规范：")
        reminder_parts.extend([f"- {norm}" for norm in selected_norms])
        
        return "\n".join(reminder_parts)
    
    def _format_additional_context(self, context: Dict[str, Any]) -> str:
        """格式化附加上下文"""
        formatted_parts = []
        
        for key, value in context.items():
            if isinstance(value, dict):
                formatted_parts.append(f"{key}：")
                for sub_key, sub_value in value.items():
                    formatted_parts.append(f"  - {sub_key}：{sub_value}")
            elif isinstance(value, list):
                formatted_parts.append(f"{key}：{', '.join(map(str, value))}")
            else:
                formatted_parts.append(f"{key}：{value}")
        
        return "\n".join(formatted_parts)
    
    def generate_help_decision_prompt(self, 
                                    agent_attrs: IntegratedAgentAttributes,
                                    help_request: HelpRequest,
                                    social_network: Optional[SocialNetwork] = None) -> str:
        """生成助人决策提示"""
        
        # 创建助人决策场景
        scenario = DecisionScenario(
            context=DecisionContext.HELP_PROVIDING,
            phase=EmergencyPhase.RESPONSE,
            urgency_level=help_request.urgency,
            risk_level=0.5,  # 默认风险等级
            available_options=["提供帮助", "拒绝帮助", "部分帮助", "转介他人"],
            social_pressure=0.6,
            resource_constraints={"时间": 0.7, "精力": 0.6, "物资": 0.5},
            time_pressure=help_request.urgency,
            information_completeness=0.8
        )
        
        # 添加求助相关信息
        additional_context = {
            "求助者": help_request.requester_id,
            "求助类型": help_request.help_type,
            "求助内容": help_request.description,
            "预期回报": help_request.expected_reciprocity,
            "求助紧急程度": f"{help_request.urgency:.2f}"
        }
        
        # 如果有社会网络，添加关系信息
        if social_network:
            relationship = social_network.get_relationship(agent_attrs.personal.agent_id, help_request.requester_id)
            if relationship:
                additional_context["与求助者关系"] = {
                    "关系类型": relationship.relationship_type.value,
                    "关系强度": f"{relationship.strength:.2f}",
                    "信任度": f"{relationship.trust:.2f}",
                    "互动频率": f"{relationship.interaction_frequency:.2f}"
                }
        
        return self.generate_decision_prompt(agent_attrs, scenario, social_network, additional_context)
    
    def generate_evacuation_prompt(self, 
                                 agent_attrs: IntegratedAgentAttributes,
                                 disaster_info: Dict[str, Any],
                                 social_network: Optional[SocialNetwork] = None) -> str:
        """生成疏散决策提示"""
        
        # 创建疏散决策场景
        scenario = DecisionScenario(
            context=DecisionContext.EVACUATION_DECISION,
            phase=EmergencyPhase.WARNING,
            urgency_level=disaster_info.get("urgency", 0.7),
            risk_level=disaster_info.get("risk_level", 0.8),
            available_options=[
                "立即疏散", "等待观察", "就地避险", "寻求帮助后疏散"
            ],
            social_pressure=0.5,
            resource_constraints={
                "交通工具": disaster_info.get("transport_availability", 0.6),
                "资金": 0.7,
                "时间": disaster_info.get("time_remaining", 0.5)
            },
            time_pressure=disaster_info.get("urgency", 0.7),
            information_completeness=disaster_info.get("info_reliability", 0.6),
            authority_present=disaster_info.get("official_warning", False)
        )
        
        return self.generate_decision_prompt(agent_attrs, scenario, social_network, disaster_info)
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """解析LLM回应"""
        parsed_response = {
            "thinking_process": "",
            "decision_result": "",
            "reasoning": "",
            "confidence": 0.5,
            "cultural_alignment": 0.5
        }
        
        # 简单的文本解析（实际应用中可能需要更复杂的NLP处理）
        sections = response.split("【")
        
        for section in sections:
            if section.startswith("思考过程】"):
                parsed_response["thinking_process"] = section.replace("思考过程】", "").strip()
            elif section.startswith("决策结果】"):
                parsed_response["decision_result"] = section.replace("决策结果】", "").strip()
            elif section.startswith("理由说明】"):
                parsed_response["reasoning"] = section.replace("理由说明】", "").strip()
        
        # 评估文化一致性（简单的关键词匹配）
        cultural_keywords = ["家人", "亲戚", "关系", "面子", "互惠", "传统", "长辈", "权威"]
        cultural_score = sum(1 for keyword in cultural_keywords if keyword in response) / len(cultural_keywords)
        parsed_response["cultural_alignment"] = min(1.0, cultural_score)
        
        return parsed_response


class PromptTemplate:
    """提示模板类"""
    
    def __init__(self, template_name: str, template_content: str, 
                 required_variables: List[str] = None):
        self.name = template_name
        self.content = template_content
        self.required_variables = required_variables or []
    
    def render(self, **kwargs) -> str:
        """渲染模板"""
        # 检查必需变量
        missing_vars = [var for var in self.required_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template variable not provided: {e}")


class PromptLibrary:
    """提示库管理类"""
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """加载默认模板"""
        # 疏散决策模板
        evacuation_template = PromptTemplate(
            "evacuation_decision",
            """
当前洪水风险等级为{risk_level}，政府建议{government_advice}。
你的家庭情况：{family_situation}
可用交通工具：{transport_options}
预计疏散时间：{evacuation_time}

请根据你的差序格局类型（{strategy_type}）做出疏散决策。
""",
            ["risk_level", "government_advice", "family_situation", "transport_options", "evacuation_time", "strategy_type"]
        )
        
        self.add_template(evacuation_template)
    
    def add_template(self, template: PromptTemplate):
        """添加模板"""
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self.templates.get(name)
    
    def render_template(self, name: str, **kwargs) -> str:
        """渲染指定模板"""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        return template.render(**kwargs)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    from .agent_attributes import PersonalAttributes, SocialAttributes, EconomicAttributes
    from .agent_attributes import Gender, EducationLevel, OccupationType, HealthStatus, IncomeLevel, HousingType
    
    print("=== LLM提示工程系统测试 ===")
    
    # 创建提示引擎
    prompt_engine = DifferentialOrderPromptEngine()
    
    # 创建测试智能体属性
    personal_attrs = PersonalAttributes(
        agent_id="test_agent_001",
        age=35,
        gender=Gender.MALE,
        education_level=EducationLevel.BACHELOR,
        occupation=OccupationType.ENGINEER,
        health_status=HealthStatus.GOOD,
        risk_tolerance=0.6,
        trust_in_authority=0.7
    )
    
    social_attrs = SocialAttributes(
        community_attachment=0.8,
        cooperation_tendency=0.7,
        social_support_received=0.6
    )
    
    economic_attrs = EconomicAttributes(
        income_level=IncomeLevel.MEDIUM,
        monthly_income=8000,
        housing_type=HousingType.OWN
    )
    
    agent_attrs = IntegratedAgentAttributes(
        personal=personal_attrs,
        social=social_attrs,
        economic=economic_attrs,
        strategy_phenotype="strict",
        risk_perception=0.7
    )
    
    # 创建测试场景
    scenario = DecisionScenario(
        context=DecisionContext.EVACUATION_DECISION,
        phase=EmergencyPhase.WARNING,
        urgency_level=0.8,
        risk_level=0.9,
        available_options=["立即疏散", "等待观察", "就地避险"],
        social_pressure=0.6,
        resource_constraints={"交通工具": 0.7, "资金": 0.8},
        time_pressure=0.8,
        information_completeness=0.6,
        family_members_present=["妻子", "儿子"],
        authority_present=True
    )
    
    # 生成决策提示
    prompt = prompt_engine.generate_decision_prompt(agent_attrs, scenario)
    
    print("\n=== 生成的决策提示 ===")
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    
    # 测试助人决策提示
    help_request = HelpRequest(
        requester_id="neighbor_001",
        help_type="物资援助",
        description="需要食物和饮用水",
        urgency=0.7,
        expected_reciprocity=0.5
    )
    
    help_prompt = prompt_engine.generate_help_decision_prompt(agent_attrs, help_request)
    
    print("\n=== 助人决策提示示例 ===")
    print(help_prompt[:500] + "..." if len(help_prompt) > 500 else help_prompt)
    
    print("\n=== 提示工程系统测试完成 ===")