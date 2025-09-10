"""
Hurricane Mobility Agent 日志管理模块
提供统一的日志配置和管理功能
"""

import logging
import os
from datetime import datetime
from typing import Optional


class HurricaneLogger:
    """飓风移动性智能体日志管理器"""
    
    def __init__(self, name: str = "HurricaneAgent", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志配置"""
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if self.logger.handlers:
            return
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # 文件处理器 - 详细日志
        log_file = os.path.join(self.log_dir, f"{self.name.lower()}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 控制台处理器 - 简化日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # 错误文件处理器
        error_file = os.path.join(self.log_dir, f"{self.name.lower()}_error_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取logger实例"""
        return self.logger
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """记录错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """记录严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def log_agent_action(self, agent_id: str, action: str, details: Optional[dict] = None):
        """记录智能体行为日志"""
        message = f"Agent[{agent_id}] - Action: {action}"
        if details:
            message += f" - Details: {details}"
        self.info(message)
    
    def log_risk_assessment(self, agent_id: str, risk_score: float, factors: dict):
        """记录风险评估日志"""
        message = f"Agent[{agent_id}] - Risk Assessment: Score={risk_score:.3f}, Factors={factors}"
        self.info(message)
    
    def log_evacuation_decision(self, agent_id: str, decision: bool, reason: str):
        """记录疏散决策日志"""
        message = f"Agent[{agent_id}] - Evacuation Decision: {decision}, Reason: {reason}"
        self.info(message)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """记录性能日志"""
        message = f"Performance - {operation}: Duration={duration:.3f}s"
        if metrics:
            message += f", Metrics={metrics}"
        self.debug(message)


# 全局日志实例
_global_logger = None

def get_logger(name: str = "HurricaneAgent", log_dir: str = "logs") -> HurricaneLogger:
    """获取全局日志实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = HurricaneLogger(name, log_dir)
    return _global_logger


def setup_logging(name: str = "HurricaneAgent", log_dir: str = "logs", level: str = "INFO"):
    """快速设置日志配置"""
    logger = get_logger(name, log_dir)
    
    # 设置日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    if level.upper() in level_map:
        logger.get_logger().setLevel(level_map[level.upper()])
    
    return logger


# 便捷函数
def log_info(message: str, **kwargs):
    """快速记录信息日志"""
    get_logger().info(message, **kwargs)

def log_error(message: str, **kwargs):
    """快速记录错误日志"""
    get_logger().error(message, **kwargs)

def log_warning(message: str, **kwargs):
    """快速记录警告日志"""
    get_logger().warning(message, **kwargs)

def log_debug(message: str, **kwargs):
    """快速记录调试日志"""
    get_logger().debug(message, **kwargs)


if __name__ == "__main__":
    # 测试日志功能
    logger = setup_logging("TestLogger", "logs", "DEBUG")
    
    logger.info("测试信息日志")
    logger.debug("测试调试日志")
    logger.warning("测试警告日志")
    logger.error("测试错误日志")
    
    logger.log_agent_action("agent_001", "risk_assessment", {"location": "Miami", "time": "2024-01-01"})
    logger.log_risk_assessment("agent_001", 0.75, {"wind_speed": 120, "distance": 50})
    logger.log_evacuation_decision("agent_001", True, "High risk detected")
    logger.log_performance("simulation", 2.5, agents=100, steps=1000)
    
    print("日志测试完成，请检查logs文件夹中的日志文件")