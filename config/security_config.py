"""安全配置模块
定义应用程序的安全策略和配置
"""

import os
from typing import List, Dict, Any

class SecurityConfig:
    """安全配置类"""
    
    # CORS配置
    CORS_ORIGINS = [
        'http://localhost:8080',
        'http://127.0.0.1:8080',
        'http://localhost:3000',  # 开发环境
        'http://127.0.0.1:3000'
    ]
    
    CORS_METHODS = ['GET', 'POST', 'OPTIONS']
    CORS_HEADERS = ['Content-Type', 'Authorization', 'X-Requested-With']
    
    # 速率限制配置
    RATE_LIMIT_CONFIG = {
        'default': {
            'max_requests': 100,
            'window_seconds': 3600  # 1小时
        },
        'api_simulate': {
            'max_requests': 20,
            'window_seconds': 3600  # 仿真API限制更严格
        },
        'api_status': {
            'max_requests': 200,
            'window_seconds': 3600  # 状态检查可以更频繁
        }
    }
    
    # 输入验证配置
    INPUT_VALIDATION = {
        'max_json_size': 1024 * 1024,  # 1MB
        'max_string_length': 1000,
        'max_integer_value': 2147483647,
        'allowed_file_extensions': ['.json', '.yaml', '.yml'],
        'blocked_patterns': [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>.*?</iframe>',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'__import__\s*\('
        ]
    }
    
    # 安全头配置
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'SAMEORIGIN',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' https:; "
            "frame-src 'self'; "
            "frame-ancestors 'self';"
        ),
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
    
    # 日志配置
    SECURITY_LOGGING = {
        'log_failed_requests': True,
        'log_rate_limit_violations': True,
        'log_validation_failures': True,
        'log_suspicious_activity': True,
        'max_log_file_size': 10 * 1024 * 1024,  # 10MB
        'log_retention_days': 30
    }
    
    # 环境变量配置
    REQUIRED_ENV_VARS = [
        'SILICON_FLOW_API_KEY',
        'SECRET_KEY'
    ]
    
    OPTIONAL_ENV_VARS = [
        'ZHIPU_API_KEY',
        'VLLM_API_KEY',
        'AGENT_SOCIETY_API_KEY',
        'FLASK_ENV',
        'CORS_ORIGINS'
    ]
    
    @classmethod
    def get_cors_config(cls) -> Dict[str, Any]:
        """获取CORS配置"""
        # 从环境变量获取额外的允许源
        env_origins = os.getenv('CORS_ORIGINS', '')
        if env_origins:
            additional_origins = [origin.strip() for origin in env_origins.split(',')]
            origins = cls.CORS_ORIGINS + additional_origins
        else:
            origins = cls.CORS_ORIGINS
        
        return {
            'origins': origins,
            'methods': cls.CORS_METHODS,
            'allow_headers': cls.CORS_HEADERS,
            'supports_credentials': False
        }
    
    @classmethod
    def get_rate_limit_config(cls, endpoint: str = 'default') -> Dict[str, int]:
        """获取速率限制配置"""
        return cls.RATE_LIMIT_CONFIG.get(endpoint, cls.RATE_LIMIT_CONFIG['default'])
    
    @classmethod
    def validate_environment(cls) -> List[str]:
        """验证环境变量配置"""
        missing_vars = []
        
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                missing_vars.append(var)
        
        return missing_vars
    
    @classmethod
    def get_security_headers(cls) -> Dict[str, str]:
        """获取安全头配置"""
        return cls.SECURITY_HEADERS.copy()
    
    @classmethod
    def is_development_mode(cls) -> bool:
        """检查是否为开发模式"""
        return os.getenv('FLASK_ENV', 'production').lower() in ['development', 'dev']
    
    @classmethod
    def get_log_level(cls) -> str:
        """获取日志级别"""
        if cls.is_development_mode():
            return 'DEBUG'
        return 'INFO'

# 安全中间件配置
class SecurityMiddleware:
    """安全中间件配置"""
    
    @staticmethod
    def apply_security_headers(response):
        """应用安全头"""
        headers = SecurityConfig.get_security_headers()
        for header, value in headers.items():
            response.headers[header] = value
        return response
    
    @staticmethod
    def log_security_event(event_type: str, details: Dict[str, Any], request_info: Dict[str, Any]):
        """记录安全事件"""
        import logging
        security_logger = logging.getLogger('security')
        
        log_entry = {
            'event_type': event_type,
            'timestamp': str(time.time()),
            'client_ip': request_info.get('remote_addr'),
            'user_agent': request_info.get('user_agent'),
            'endpoint': request_info.get('endpoint'),
            'method': request_info.get('method'),
            'details': details
        }
        
        security_logger.warning(f"Security Event: {log_entry}")

# 导入time模块
import time