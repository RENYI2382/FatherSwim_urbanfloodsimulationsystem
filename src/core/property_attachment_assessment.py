"""
财产依恋评估模块 (w₄: 保护个人财产的愿望)
基于DDABM论文和恐慌心理模型实现
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AttachmentComponents:
    """财产依恋组件数据类"""
    ownership_attachment: float = 0.0      # 房屋所有权依恋
    residence_attachment: float = 0.0      # 居住年限依恋
    work_attachment: float = 0.0           # 工作职责依恋
    property_value_attachment: float = 0.0 # 财产价值依恋
    emotional_attachment: float = 0.0      # 情感依恋
    total_attachment: float = 0.0          # 总依恋程度

class PropertyAttachmentAssessment:
    """
    财产依恋评估类
    基于DDABM论文的w₄标准：拥有个人财产、有工作要求
    """
    
    def __init__(self):
        # 基于文献的依恋因子权重
        self.attachment_factors = {
            'ownership': 0.30,           # 房屋所有权权重
            'residence_duration': 0.25,  # 居住年限权重
            'work_responsibility': 0.20, # 工作职责权重
            'property_value': 0.15,      # 财产价值权重
            'emotional_factors': 0.10    # 情感因素权重
        }
        
        # 房屋所有权类型影响
        self.ownership_types = {
            'own_outright': 1.0,    # 完全拥有
            'own_mortgage': 0.85,   # 按揭拥有
            'rent': 0.25,           # 租房
            'family_owned': 0.70    # 家族房产
        }
        
        # 工作类型对依恋的影响
        self.work_types = {
            'essential_worker': 0.95,    # 关键岗位工作者
            'business_owner': 0.90,      # 企业主
            'remote_worker': 0.30,       # 远程工作者
            'retired': 0.60,             # 退休人员
            'unemployed': 0.20           # 无业人员
        }
        
        # 恐慌程度对依恋的影响（基于文献3）
        self.panic_reduction_factors = {
            'low': 0.1,      # 低恐慌：依恋影响减少10%
            'medium': 0.25,  # 中等恐慌：依恋影响减少25%
            'high': 0.45     # 高恐慌：依恋影响减少45%
        }
    
    def calculate_attachment_level(
        self, 
        agent_profile: Dict[str, Any],
        panic_level: float = 0.0,
        social_pressure: float = 0.0
    ) -> AttachmentComponents:
        """
        计算财产依恋程度
        
        Args:
            agent_profile: 智能体档案
            panic_level: 恐慌程度 [0,1]
            social_pressure: 社会压力 [0,1]
            
        Returns:
            AttachmentComponents: 依恋组件详情
        """
        try:
            # 1. 房屋所有权依恋
            ownership_attachment = self._calculate_ownership_attachment(agent_profile)
            
            # 2. 居住年限依恋
            residence_attachment = self._calculate_residence_attachment(agent_profile)
            
            # 3. 工作职责依恋
            work_attachment = self._calculate_work_attachment(agent_profile)
            
            # 4. 财产价值依恋
            property_value_attachment = self._calculate_property_value_attachment(agent_profile)
            
            # 5. 情感依恋
            emotional_attachment = self._calculate_emotional_attachment(agent_profile)
            
            # 6. 综合依恋计算
            base_attachment = (
                ownership_attachment * self.attachment_factors['ownership'] +
                residence_attachment * self.attachment_factors['residence_duration'] +
                work_attachment * self.attachment_factors['work_responsibility'] +
                property_value_attachment * self.attachment_factors['property_value'] +
                emotional_attachment * self.attachment_factors['emotional_factors']
            )
            
            # 7. 应用恐慌心理调整
            panic_adjusted_attachment = self._apply_panic_adjustment(
                base_attachment, panic_level
            )
            
            # 8. 应用社会压力调整
            final_attachment = self._apply_social_pressure_adjustment(
                panic_adjusted_attachment, social_pressure
            )
            
            return AttachmentComponents(
                ownership_attachment=ownership_attachment,
                residence_attachment=residence_attachment,
                work_attachment=work_attachment,
                property_value_attachment=property_value_attachment,
                emotional_attachment=emotional_attachment,
                total_attachment=min(final_attachment, 1.0)
            )
            
        except Exception as e:
            logger.error(f"财产依恋计算失败: {e}")
            return AttachmentComponents(total_attachment=0.5)  # 返回默认中等依恋
    
    def _calculate_ownership_attachment(self, agent_profile: Dict[str, Any]) -> float:
        """
        计算房屋所有权依恋
        """
        ownership_type = agent_profile.get('ownership_type', 'rent')
        base_ownership_score = self.ownership_types.get(ownership_type, 0.25)
        
        # 房屋价值影响
        home_value = agent_profile.get('home_value_level', 'medium')
        value_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
        value_multiplier = value_multipliers.get(home_value, 1.0)
        
        # 按揭负担影响（有按揭的人更不愿意离开）
        has_mortgage = agent_profile.get('has_mortgage', False)
        mortgage_factor = 1.15 if has_mortgage else 1.0
        
        ownership_attachment = base_ownership_score * value_multiplier * mortgage_factor
        
        return min(ownership_attachment, 1.0)
    
    def _calculate_residence_attachment(self, agent_profile: Dict[str, Any]) -> float:
        """
        计算居住年限依恋
        """
        residence_years = agent_profile.get('residence_duration', 5)
        
        # 居住年限越长，依恋越强（对数增长）
        if residence_years <= 0:
            residence_score = 0.1
        else:
            residence_score = min(np.log(residence_years + 1) / np.log(21), 1.0)  # 20年为满分
        
        # 年龄影响（老年人对居住地依恋更强）
        age = agent_profile.get('age', 35)
        if age >= 65:
            age_factor = 1.3
        elif age >= 45:
            age_factor = 1.1
        else:
            age_factor = 0.9
        
        # 家庭规模影响（大家庭更难搬迁）
        family_size = agent_profile.get('family_size', 2)
        family_factor = min(1.0 + (family_size - 1) * 0.1, 1.5)
        
        residence_attachment = residence_score * age_factor * family_factor
        
        return min(residence_attachment, 1.0)
    
    def _calculate_work_attachment(self, agent_profile: Dict[str, Any]) -> float:
        """
        计算工作职责依恋
        """
        work_type = agent_profile.get('work_type', 'remote_worker')
        base_work_score = self.work_types.get(work_type, 0.5)
        
        # 工作职责要求
        work_responsibility = agent_profile.get('work_responsibility', False)
        responsibility_factor = 1.4 if work_responsibility else 0.6
        
        # 收入依赖程度
        income_level = agent_profile.get('income_level', 'medium')
        income_factors = {'low': 1.2, 'medium': 1.0, 'high': 0.8}  # 低收入更依赖工作
        income_factor = income_factors.get(income_level, 1.0)
        
        # 企业规模影响（小企业主更难离开）
        is_business_owner = agent_profile.get('is_business_owner', False)
        business_factor = 1.3 if is_business_owner else 1.0
        
        work_attachment = (
            base_work_score * 
            responsibility_factor * 
            income_factor * 
            business_factor
        )
        
        return min(work_attachment, 1.0)
    
    def _calculate_property_value_attachment(self, agent_profile: Dict[str, Any]) -> float:
        """
        计算财产价值依恋
        """
        # 财产总价值水平
        property_value_level = agent_profile.get('property_value_level', 'medium')
        value_scores = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        base_value_score = value_scores.get(property_value_level, 0.6)
        
        # 保险覆盖情况（有保险的人依恋相对较低）
        has_insurance = agent_profile.get('has_property_insurance', True)
        insurance_factor = 0.8 if has_insurance else 1.2
        
        # 收藏品/贵重物品
        has_valuables = agent_profile.get('has_valuable_items', False)
        valuables_factor = 1.2 if has_valuables else 1.0
        
        # 家族传承物品
        has_heirlooms = agent_profile.get('has_family_heirlooms', False)
        heirloom_factor = 1.3 if has_heirlooms else 1.0
        
        property_value_attachment = (
            base_value_score * 
            insurance_factor * 
            valuables_factor * 
            heirloom_factor
        )
        
        return min(property_value_attachment, 1.0)
    
    def _calculate_emotional_attachment(self, agent_profile: Dict[str, Any]) -> float:
        """
        计算情感依恋
        """
        # 宠物因素
        has_pets = agent_profile.get('has_pets', False)
        pet_score = 0.8 if has_pets else 0.2
        
        # 花园/农场
        has_garden = agent_profile.get('has_garden', False)
        garden_score = 0.6 if has_garden else 0.3
        
        # 社区参与度
        community_involvement = agent_profile.get('community_involvement', 0.5)
        
        # 邻里关系
        neighbor_relationships = agent_profile.get('neighbor_relationships', 0.5)
        
        # 情感记忆（重要事件发生地）
        emotional_significance = agent_profile.get('emotional_significance', 0.5)
        
        emotional_attachment = (
            pet_score * 0.3 +
            garden_score * 0.2 +
            community_involvement * 0.2 +
            neighbor_relationships * 0.15 +
            emotional_significance * 0.15
        )
        
        return min(emotional_attachment, 1.0)
    
    def _apply_panic_adjustment(self, base_attachment: float, panic_level: float) -> float:
        """
        应用恐慌心理调整
        基于文献3的恐慌心理模型
        """
        if panic_level <= 0.3:
            panic_category = 'low'
        elif panic_level <= 0.7:
            panic_category = 'medium'
        else:
            panic_category = 'high'
        
        reduction_factor = self.panic_reduction_factors[panic_category]
        adjusted_attachment = base_attachment * (1 - reduction_factor)
        
        return max(adjusted_attachment, 0.0)
    
    def _apply_social_pressure_adjustment(
        self, 
        base_attachment: float, 
        social_pressure: float
    ) -> float:
        """
        应用社会压力调整
        社会压力越大，财产依恋的影响越小
        """
        # 社会压力降低财产依恋的影响
        pressure_reduction = social_pressure * 0.2
        adjusted_attachment = base_attachment * (1 - pressure_reduction)
        
        return max(adjusted_attachment, 0.0)
    
    def get_attachment_explanation(self, attachment_components: AttachmentComponents) -> str:
        """
        生成财产依恋的解释说明
        """
        explanations = []
        
        if attachment_components.ownership_attachment > 0.7:
            explanations.append("房屋所有权依恋强")
        
        if attachment_components.residence_attachment > 0.7:
            explanations.append("居住地情感深厚")
        
        if attachment_components.work_attachment > 0.7:
            explanations.append("工作职责重要")
        
        if attachment_components.property_value_attachment > 0.7:
            explanations.append("财产价值较高")
        
        if attachment_components.emotional_attachment > 0.7:
            explanations.append("情感依恋深厚")
        
        if not explanations:
            explanations.append("财产依恋程度一般")
        
        return "、".join(explanations)
    
    def calculate_protection_behavior_probability(
        self, 
        attachment_level: float, 
        risk_level: float
    ) -> float:
        """
        计算财产保护行为概率
        基于囚徒困境模型
        """
        # 基础保护意愿
        base_protection = attachment_level
        
        # 风险水平调整（风险越高，保护行为越不理性）
        risk_adjustment = 1.0 - risk_level * 0.4
        
        # 保护行为概率
        protection_probability = base_protection * risk_adjustment
        
        return min(protection_probability, 1.0)
    
    def apply_dynamic_learning(
        self, 
        current_attachment: float, 
        experience_factor: float, 
        time_pressure: float
    ) -> float:
        """
        应用动态学习调整
        基于EDFT框架的学习机制
        """
        # 经验因子影响（过往疏散经验降低依恋）
        experience_adjustment = 1.0 - experience_factor * 0.3
        
        # 时间压力影响（时间紧迫时依恋影响降低）
        time_adjustment = 1.0 - time_pressure * 0.2
        
        adjusted_attachment = (
            current_attachment * 
            experience_adjustment * 
            time_adjustment
        )
        
        return max(adjusted_attachment, 0.0)