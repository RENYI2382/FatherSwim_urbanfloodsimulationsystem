"""
信息信任评估模块 (w₂: 对疏散信息的信任程度)
基于DDABM论文和博弈论模型完善现有实现
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class InformationSource(Enum):
    """信息来源类型"""
    OFFICIAL_GOVERNMENT = "official_government"
    EMERGENCY_SERVICES = "emergency_services"
    MEDIA_NEWS = "media_news"
    SOCIAL_MEDIA = "social_media"
    FAMILY_FRIENDS = "family_friends"
    NEIGHBORS = "neighbors"
    WORKPLACE = "workplace"
    WEATHER_SERVICE = "weather_service"

@dataclass
class TrustComponents:
    """信息信任组件数据类"""
    source_credibility: float = 0.0      # 信息源可信度
    information_consistency: float = 0.0  # 信息一致性
    personal_experience: float = 0.0      # 个人经验影响
    social_validation: float = 0.0        # 社会验证
    timeliness: float = 0.0              # 信息时效性
    clarity: float = 0.0                 # 信息清晰度
    total_trust: float = 0.0             # 总信任度

class InformationTrustAssessment:
    """
    信息信任评估类
    基于DDABM论文的w₂标准：对疏散信息的信任程度
    完善现有的部分实现
    """
    
    def __init__(self):
        # 信息源可信度基础分数
        self.source_credibility = {
            InformationSource.OFFICIAL_GOVERNMENT: 0.85,
            InformationSource.EMERGENCY_SERVICES: 0.90,
            InformationSource.WEATHER_SERVICE: 0.88,
            InformationSource.MEDIA_NEWS: 0.65,
            InformationSource.SOCIAL_MEDIA: 0.35,
            InformationSource.FAMILY_FRIENDS: 0.70,
            InformationSource.NEIGHBORS: 0.60,
            InformationSource.WORKPLACE: 0.55
        }
        
        # 信任组件权重
        self.trust_weights = {
            'source_credibility': 0.30,      # 信息源可信度权重
            'information_consistency': 0.25, # 信息一致性权重
            'personal_experience': 0.20,     # 个人经验权重
            'social_validation': 0.15,       # 社会验证权重
            'timeliness': 0.05,             # 时效性权重
            'clarity': 0.05                 # 清晰度权重
        }
        
        # 人口统计学对信任的影响
        self.demographic_trust_factors = {
            'education': {
                'low': 0.85,      # 低教育水平更信任权威
                'medium': 1.0,
                'high': 1.15      # 高教育水平更理性分析
            },
            'age': {
                'young': 0.90,    # 年轻人更怀疑权威
                'middle': 1.0,
                'elderly': 1.10   # 老年人更信任权威
            },
            'income': {
                'low': 0.95,
                'medium': 1.0,
                'high': 1.05      # 高收入群体信息获取能力强
            }
        }
        
        # 过往经验对信任的影响
        self.experience_factors = {
            'no_experience': 1.0,
            'positive_experience': 1.2,      # 正面经验增强信任
            'negative_experience': 0.6,      # 负面经验降低信任
            'mixed_experience': 0.9
        }
    
    def calculate_information_trust(
        self,
        agent_profile: Dict[str, Any],
        information_sources: List[InformationSource],
        information_content: Dict[str, Any],
        social_context: Dict[str, Any]
    ) -> TrustComponents:
        """
        计算信息信任程度
        
        Args:
            agent_profile: 智能体档案
            information_sources: 信息来源列表
            information_content: 信息内容特征
            social_context: 社会环境上下文
            
        Returns:
            TrustComponents: 信任组件详情
        """
        try:
            # 1. 信息源可信度评估
            source_credibility = self._assess_source_credibility(
                information_sources, agent_profile
            )
            
            # 2. 信息一致性评估
            information_consistency = self._assess_information_consistency(
                information_content, information_sources
            )
            
            # 3. 个人经验影响评估
            personal_experience = self._assess_personal_experience_impact(
                agent_profile, information_content
            )
            
            # 4. 社会验证评估
            social_validation = self._assess_social_validation(
                social_context, agent_profile
            )
            
            # 5. 信息时效性评估
            timeliness = self._assess_information_timeliness(information_content)
            
            # 6. 信息清晰度评估
            clarity = self._assess_information_clarity(information_content)
            
            # 7. 综合信任度计算
            total_trust = (
                source_credibility * self.trust_weights['source_credibility'] +
                information_consistency * self.trust_weights['information_consistency'] +
                personal_experience * self.trust_weights['personal_experience'] +
                social_validation * self.trust_weights['social_validation'] +
                timeliness * self.trust_weights['timeliness'] +
                clarity * self.trust_weights['clarity']
            )
            
            # 8. 应用人口统计学调整
            demographic_adjusted_trust = self._apply_demographic_adjustments(
                total_trust, agent_profile
            )
            
            return TrustComponents(
                source_credibility=source_credibility,
                information_consistency=information_consistency,
                personal_experience=personal_experience,
                social_validation=social_validation,
                timeliness=timeliness,
                clarity=clarity,
                total_trust=min(demographic_adjusted_trust, 1.0)
            )
            
        except Exception as e:
            logger.error(f"信息信任计算失败: {e}")
            return TrustComponents(total_trust=0.6)  # 返回默认中等信任
    
    def _assess_source_credibility(
        self, 
        sources: List[InformationSource], 
        agent_profile: Dict[str, Any]
    ) -> float:
        """
        评估信息源可信度
        """
        if not sources:
            return 0.5
        
        # 计算多源信息的加权可信度
        total_credibility = 0.0
        weight_sum = 0.0
        
        for source in sources:
            base_credibility = self.source_credibility.get(source, 0.5)
            
            # 个人偏好调整
            preference_factor = self._get_source_preference_factor(source, agent_profile)
            adjusted_credibility = base_credibility * preference_factor
            
            # 信息源权重（官方源权重更高）
            source_weight = self._get_source_weight(source)
            
            total_credibility += adjusted_credibility * source_weight
            weight_sum += source_weight
        
        return total_credibility / weight_sum if weight_sum > 0 else 0.5
    
    def _get_source_preference_factor(
        self, 
        source: InformationSource, 
        agent_profile: Dict[str, Any]
    ) -> float:
        """
        获取个人对信息源的偏好因子
        """
        # 政治倾向影响
        political_trust = agent_profile.get('government_trust', 0.7)
        
        # 技术接受度影响
        tech_savvy = agent_profile.get('tech_savvy', 0.5)
        
        # 社交倾向影响
        social_orientation = agent_profile.get('social_orientation', 0.5)
        
        if source in [InformationSource.OFFICIAL_GOVERNMENT, InformationSource.EMERGENCY_SERVICES]:
            return 0.7 + political_trust * 0.3
        elif source == InformationSource.SOCIAL_MEDIA:
            return 0.5 + tech_savvy * 0.3
        elif source in [InformationSource.FAMILY_FRIENDS, InformationSource.NEIGHBORS]:
            return 0.6 + social_orientation * 0.4
        else:
            return 1.0
    
    def _get_source_weight(self, source: InformationSource) -> float:
        """
        获取信息源权重
        """
        weights = {
            InformationSource.OFFICIAL_GOVERNMENT: 1.0,
            InformationSource.EMERGENCY_SERVICES: 1.0,
            InformationSource.WEATHER_SERVICE: 0.9,
            InformationSource.MEDIA_NEWS: 0.7,
            InformationSource.FAMILY_FRIENDS: 0.6,
            InformationSource.NEIGHBORS: 0.5,
            InformationSource.WORKPLACE: 0.5,
            InformationSource.SOCIAL_MEDIA: 0.3
        }
        return weights.get(source, 0.5)
    
    def _assess_information_consistency(
        self, 
        information_content: Dict[str, Any], 
        sources: List[InformationSource]
    ) -> float:
        """
        评估信息一致性
        """
        # 多源信息一致性
        consistency_score = information_content.get('consistency_score', 0.7)
        
        # 信息源数量影响（更多源头增强一致性可信度）
        source_count_factor = min(len(sources) / 3.0, 1.0)
        
        # 官方源一致性权重更高
        official_sources = [s for s in sources if s in [
            InformationSource.OFFICIAL_GOVERNMENT,
            InformationSource.EMERGENCY_SERVICES,
            InformationSource.WEATHER_SERVICE
        ]]
        
        official_factor = 1.0 + len(official_sources) * 0.1
        
        final_consistency = consistency_score * source_count_factor * official_factor
        
        return min(final_consistency, 1.0)
    
    def _assess_personal_experience_impact(
        self, 
        agent_profile: Dict[str, Any], 
        information_content: Dict[str, Any]
    ) -> float:
        """
        评估个人经验对信任的影响
        """
        # 过往疏散经验
        evacuation_experience = agent_profile.get('evacuation_experience', 'no_experience')
        experience_factor = self.experience_factors.get(evacuation_experience, 1.0)
        
        # 信息与个人认知的匹配度
        cognitive_match = information_content.get('cognitive_match', 0.7)
        
        # 风险感知一致性
        risk_perception_match = information_content.get('risk_perception_match', 0.7)
        
        # 个人知识水平
        knowledge_level = agent_profile.get('disaster_knowledge', 0.5)
        
        experience_impact = (
            experience_factor * 0.4 +
            cognitive_match * 0.3 +
            risk_perception_match * 0.2 +
            knowledge_level * 0.1
        )
        
        return min(experience_impact, 1.0)
    
    def _assess_social_validation(
        self, 
        social_context: Dict[str, Any], 
        agent_profile: Dict[str, Any]
    ) -> float:
        """
        评估社会验证程度
        """
        # 社区共识程度
        community_consensus = social_context.get('community_consensus', 0.6)
        
        # 家庭成员态度
        family_agreement = social_context.get('family_agreement', 0.7)
        
        # 邻居行为
        neighbor_behavior = social_context.get('neighbor_evacuation_rate', 0.5)
        
        # 社会影响敏感度
        social_influence_sensitivity = agent_profile.get('social_influence', 0.6)
        
        # 从众倾向
        conformity_tendency = agent_profile.get('conformity', 0.5)
        
        social_validation = (
            community_consensus * 0.3 +
            family_agreement * 0.4 +
            neighbor_behavior * 0.3
        ) * (social_influence_sensitivity + conformity_tendency) / 2
        
        return min(social_validation, 1.0)
    
    def _assess_information_timeliness(self, information_content: Dict[str, Any]) -> float:
        """
        评估信息时效性
        """
        # 信息发布时间
        time_since_release = information_content.get('time_since_release', 1.0)  # 小时
        
        # 时效性衰减函数
        if time_since_release <= 1:
            timeliness = 1.0
        elif time_since_release <= 6:
            timeliness = 0.9
        elif time_since_release <= 12:
            timeliness = 0.7
        elif time_since_release <= 24:
            timeliness = 0.5
        else:
            timeliness = 0.3
        
        # 更新频率影响
        update_frequency = information_content.get('update_frequency', 'medium')
        frequency_factors = {'high': 1.1, 'medium': 1.0, 'low': 0.8}
        frequency_factor = frequency_factors.get(update_frequency, 1.0)
        
        return min(timeliness * frequency_factor, 1.0)
    
    def _assess_information_clarity(self, information_content: Dict[str, Any]) -> float:
        """
        评估信息清晰度
        """
        # 语言清晰度
        language_clarity = information_content.get('language_clarity', 0.7)
        
        # 指令具体性
        instruction_specificity = information_content.get('instruction_specificity', 0.7)
        
        # 视觉辅助
        has_visual_aids = information_content.get('has_visual_aids', False)
        visual_factor = 1.1 if has_visual_aids else 1.0
        
        # 多语言支持
        multilingual_support = information_content.get('multilingual_support', False)
        language_factor = 1.05 if multilingual_support else 1.0
        
        clarity = (
            language_clarity * 0.5 +
            instruction_specificity * 0.5
        ) * visual_factor * language_factor
        
        return min(clarity, 1.0)
    
    def _apply_demographic_adjustments(
        self, 
        base_trust: float, 
        agent_profile: Dict[str, Any]
    ) -> float:
        """
        应用人口统计学调整
        """
        # 教育水平调整
        education = agent_profile.get('education_level', 'medium')
        education_factor = self.demographic_trust_factors['education'].get(education, 1.0)
        
        # 年龄调整
        age = agent_profile.get('age', 35)
        if age < 30:
            age_category = 'young'
        elif age < 60:
            age_category = 'middle'
        else:
            age_category = 'elderly'
        age_factor = self.demographic_trust_factors['age'].get(age_category, 1.0)
        
        # 收入调整
        income = agent_profile.get('income_level', 'medium')
        income_factor = self.demographic_trust_factors['income'].get(income, 1.0)
        
        adjusted_trust = base_trust * education_factor * age_factor * income_factor
        
        return min(adjusted_trust, 1.0)
    
    def get_trust_explanation(self, trust_components: TrustComponents) -> str:
        """
        生成信任评估的解释说明
        """
        explanations = []
        
        if trust_components.source_credibility > 0.8:
            explanations.append("信息源高度可信")
        elif trust_components.source_credibility < 0.4:
            explanations.append("信息源可信度低")
        
        if trust_components.information_consistency > 0.8:
            explanations.append("多源信息一致")
        elif trust_components.information_consistency < 0.4:
            explanations.append("信息存在矛盾")
        
        if trust_components.social_validation > 0.7:
            explanations.append("社会验证强")
        
        if trust_components.personal_experience > 0.7:
            explanations.append("符合个人经验")
        elif trust_components.personal_experience < 0.4:
            explanations.append("与个人经验冲突")
        
        if not explanations:
            explanations.append("信任程度一般")
        
        return "、".join(explanations)
    
    def calculate_information_seeking_behavior(
        self, 
        current_trust: float, 
        uncertainty_level: float
    ) -> Dict[str, float]:
        """
        计算信息寻求行为倾向
        """
        # 基础信息寻求倾向
        base_seeking = 1.0 - current_trust
        
        # 不确定性增强寻求行为
        uncertainty_boost = uncertainty_level * 0.5
        
        # 各类信息源的寻求概率
        seeking_probabilities = {
            'official_sources': min(base_seeking + uncertainty_boost, 1.0),
            'social_sources': min((base_seeking + uncertainty_boost) * 0.7, 1.0),
            'media_sources': min((base_seeking + uncertainty_boost) * 0.8, 1.0),
            'personal_networks': min((base_seeking + uncertainty_boost) * 0.9, 1.0)
        }
        
        return seeking_probabilities
    
    def apply_trust_updating(
        self, 
        current_trust: float, 
        new_information_quality: float, 
        confirmation_bias: float = 0.3
    ) -> float:
        """
        应用信任更新机制
        基于贝叶斯更新和确认偏差
        """
        # 贝叶斯更新
        prior_trust = current_trust
        likelihood = new_information_quality
        
        # 确认偏差影响（倾向于相信符合现有信念的信息）
        if abs(likelihood - prior_trust) < 0.3:
            # 信息与现有信任一致，增强信任
            bias_factor = 1.0 + confirmation_bias * 0.2
        else:
            # 信息与现有信任冲突，抵制更新
            bias_factor = 1.0 - confirmation_bias * 0.3
        
        # 更新后的信任
        updated_trust = (
            prior_trust * 0.7 + 
            likelihood * 0.3
        ) * bias_factor
        
        return min(max(updated_trust, 0.0), 1.0)