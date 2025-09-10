"""日志配置模块
配置应用程序的日志记录，包括安全事件日志
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any

class LoggingConfig:
    """日志配置类"""
    
    @staticmethod
    def setup_logging(app_name: str = 'flood_abm_system', log_level: str = 'INFO'):
        """设置应用程序日志"""
        
        # 创建日志目录
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置根日志记录器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 应用程序日志文件处理器
        app_log_file = os.path.join(log_dir, f'{app_name}.log')
        file_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # 错误日志文件处理器
        error_log_file = os.path.join(log_dir, f'{app_name}_error.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        return root_logger
    
    @staticmethod
    def setup_security_logging():
        """设置安全事件日志"""
        
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建安全日志记录器
        security_logger = logging.getLogger('security')
        security_logger.setLevel(logging.WARNING)
        
        # 安全事件格式化器
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 安全日志文件处理器
        security_log_file = os.path.join(log_dir, 'security.log')
        security_handler = logging.handlers.RotatingFileHandler(
            security_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10  # 保留更多安全日志
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(security_formatter)
        security_logger.addHandler(security_handler)
        
        # 安全事件也输出到控制台（仅在开发模式）
        if os.getenv('FLASK_ENV', 'production').lower() in ['development', 'dev']:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            console_handler.setFormatter(security_formatter)
            security_logger.addHandler(console_handler)
        
        return security_logger
    
    @staticmethod
    def setup_access_logging():
        """设置访问日志"""
        
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建访问日志记录器
        access_logger = logging.getLogger('access')
        access_logger.setLevel(logging.INFO)
        
        # 访问日志格式化器
        access_formatter = logging.Formatter(
            '%(asctime)s - ACCESS - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 访问日志文件处理器
        access_log_file = os.path.join(log_dir, 'access.log')
        access_handler = logging.handlers.RotatingFileHandler(
            access_log_file,
            maxBytes=50*1024*1024,  # 50MB，访问日志通常更大
            backupCount=7
        )
        access_handler.setLevel(logging.INFO)
        access_handler.setFormatter(access_formatter)
        access_logger.addHandler(access_handler)
        
        return access_logger

class SecurityLogger:
    """安全事件日志记录器"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
    
    def log_failed_login(self, ip_address: str, user_agent: str = None):
        """记录登录失败事件"""
        self.logger.warning(f"Failed login attempt from {ip_address}, User-Agent: {user_agent}")
    
    def log_rate_limit_violation(self, ip_address: str, endpoint: str, attempts: int):
        """记录速率限制违规"""
        self.logger.warning(f"Rate limit violation: {ip_address} made {attempts} requests to {endpoint}")
    
    def log_input_validation_failure(self, ip_address: str, endpoint: str, error_details: str):
        """记录输入验证失败"""
        self.logger.warning(f"Input validation failed: {ip_address} at {endpoint} - {error_details}")
    
    def log_suspicious_activity(self, ip_address: str, activity_type: str, details: Dict[str, Any]):
        """记录可疑活动"""
        self.logger.warning(f"Suspicious activity: {activity_type} from {ip_address} - {details}")
    
    def log_api_abuse(self, ip_address: str, endpoint: str, abuse_type: str):
        """记录API滥用"""
        self.logger.error(f"API abuse detected: {abuse_type} from {ip_address} at {endpoint}")
    
    def log_security_config_change(self, user: str, change_type: str, details: str):
        """记录安全配置变更"""
        self.logger.info(f"Security config change: {change_type} by {user} - {details}")

class AccessLogger:
    """访问日志记录器"""
    
    def __init__(self):
        self.logger = logging.getLogger('access')
    
    def log_request(self, method: str, path: str, ip_address: str, 
                   user_agent: str = None, response_code: int = None, 
                   response_time: float = None):
        """记录HTTP请求"""
        log_entry = f"{ip_address} - {method} {path}"
        
        if response_code:
            log_entry += f" - {response_code}"
        
        if response_time:
            log_entry += f" - {response_time:.3f}s"
        
        if user_agent:
            log_entry += f" - {user_agent}"
        
        self.logger.info(log_entry)
    
    def log_api_call(self, endpoint: str, ip_address: str, parameters: Dict[str, Any] = None,
                    response_code: int = None, execution_time: float = None):
        """记录API调用"""
        log_entry = f"API: {endpoint} from {ip_address}"
        
        if parameters:
            # 过滤敏感参数
            safe_params = {k: v for k, v in parameters.items() 
                          if k.lower() not in ['password', 'token', 'key', 'secret']}
            log_entry += f" - params: {safe_params}"
        
        if response_code:
            log_entry += f" - {response_code}"
        
        if execution_time:
            log_entry += f" - {execution_time:.3f}s"
        
        self.logger.info(log_entry)

# 全局日志记录器实例
security_logger = SecurityLogger()
access_logger = AccessLogger()

# 便捷函数
def setup_all_logging(app_name: str = 'flood_abm_system', log_level: str = 'INFO'):
    """设置所有日志记录"""
    LoggingConfig.setup_logging(app_name, log_level)
    LoggingConfig.setup_security_logging()
    LoggingConfig.setup_access_logging()
    
    return {
        'main': logging.getLogger(),
        'security': logging.getLogger('security'),
        'access': logging.getLogger('access')
    }

# 向后兼容的别名
setup_logger = LoggingConfig.setup_logging

def get_logger(name: str = None):
    """获取日志记录器"""
    return logging.getLogger(name)