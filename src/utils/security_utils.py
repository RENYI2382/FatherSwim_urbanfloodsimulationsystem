"""安全工具模块
提供输入验证、XSS防护和安全相关功能
"""

import re
import html
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from functools import wraps
from flask import request, jsonify

logger = logging.getLogger(__name__)

# 延迟导入安全日志记录器以避免循环导入
def get_security_logger():
    try:
        from config.logging_config import security_logger
        return security_logger
    except ImportError:
        return None

class InputValidator:
    """输入验证器"""
    
    # 常用正则表达式
    PATTERNS = {
        'alphanumeric': r'^[a-zA-Z0-9_-]+$',
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'numeric': r'^\d+$',
        'float': r'^\d*\.?\d+$',
        'scenario_name': r'^[a-zA-Z0-9_-]{1,50}$',
        'simulation_id': r'^sim_\d+$'
    }
    
    # 危险字符和模式
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script标签
        r'javascript:',  # JavaScript协议
        r'on\w+\s*=',  # 事件处理器
        r'<iframe[^>]*>.*?</iframe>',  # iframe标签
        r'<object[^>]*>.*?</object>',  # object标签
        r'<embed[^>]*>.*?</embed>',  # embed标签
        r'<link[^>]*>',  # link标签
        r'<meta[^>]*>',  # meta标签
        r'expression\s*\(',  # CSS表达式
        r'url\s*\(',  # CSS url函数
        r'@import',  # CSS import
    ]
    
    @classmethod
    def validate_string(cls, value: str, pattern: str = None, max_length: int = 1000) -> bool:
        """验证字符串"""
        if not isinstance(value, str):
            return False
        
        if len(value) > max_length:
            return False
        
        # 检查危险模式
        for dangerous_pattern in cls.DANGEROUS_PATTERNS:
            if re.search(dangerous_pattern, value, re.IGNORECASE):
                return False
        
        # 检查指定模式
        if pattern and pattern in cls.PATTERNS:
            return bool(re.match(cls.PATTERNS[pattern], value))
        
        return True
    
    @classmethod
    def validate_integer(cls, value: Union[str, int], min_val: int = None, max_val: int = None) -> bool:
        """验证整数"""
        try:
            int_val = int(value)
            if min_val is not None and int_val < min_val:
                return False
            if max_val is not None and int_val > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def validate_float(cls, value: Union[str, float], min_val: float = None, max_val: float = None) -> bool:
        """验证浮点数"""
        try:
            float_val = float(value)
            if min_val is not None and float_val < min_val:
                return False
            if max_val is not None and float_val > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """清理字符串，移除危险内容"""
        if not isinstance(value, str):
            return str(value)
        
        # HTML编码
        sanitized = html.escape(value)
        
        # 移除危险模式
        for pattern in cls.DANGEROUS_PATTERNS:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()

class SecurityDecorator:
    """安全装饰器"""
    
    @staticmethod
    def validate_json_input(schema: Dict[str, Dict[str, Any]]):
        """验证JSON输入的装饰器
        
        Args:
            schema: 验证模式，格式为 {
                'field_name': {
                    'type': 'string|integer|float',
                    'required': True|False,
                    'pattern': 'pattern_name',
                    'min_val': min_value,
                    'max_val': max_value,
                    'max_length': max_length
                }
            }
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    data = request.get_json()
                    if not data:
                        return jsonify({
                            'error': '无效的JSON数据',
                            'status': 'error',
                            'timestamp': str(time.time())
                        }), 400
                    
                    # 验证每个字段
                    for field_name, field_config in schema.items():
                        value = data.get(field_name)
                        
                        # 检查必填字段
                        if field_config.get('required', False) and value is None:
                            # 记录验证失败事件
                            sec_logger = get_security_logger()
                            if sec_logger:
                                sec_logger.log_input_validation_failure(
                                    request.remote_addr,
                                    request.endpoint,
                                    f'Missing required field: {field_name}'
                                )
                            
                            return jsonify({
                                'error': f'缺少必填字段: {field_name}',
                                'status': 'error',
                                'timestamp': str(time.time())
                            }), 400
                        
                        if value is not None:
                            field_type = field_config.get('type', 'string')
                            
                            # 类型验证
                            if field_type == 'string':
                                if not InputValidator.validate_string(
                                    value, 
                                    field_config.get('pattern'),
                                    field_config.get('max_length', 1000)
                                ):
                                    # 记录验证失败事件
                                    sec_logger = get_security_logger()
                                    if sec_logger:
                                        sec_logger.log_input_validation_failure(
                                            request.remote_addr,
                                            request.endpoint,
                                            f'Invalid string format for field: {field_name}'
                                        )
                                    
                                    return jsonify({
                                        'error': f'字段 {field_name} 格式无效',
                                        'status': 'error',
                                        'timestamp': str(time.time())
                                    }), 400
                                
                                # 清理字符串
                                data[field_name] = InputValidator.sanitize_string(value)
                            
                            elif field_type == 'integer':
                                if not InputValidator.validate_integer(
                                    value,
                                    field_config.get('min_val'),
                                    field_config.get('max_val')
                                ):
                                    return jsonify({
                                        'error': f'字段 {field_name} 数值无效',
                                        'status': 'error',
                                        'timestamp': str(time.time())
                                    }), 400
                            
                            elif field_type == 'float':
                                if not InputValidator.validate_float(
                                    value,
                                    field_config.get('min_val'),
                                    field_config.get('max_val')
                                ):
                                    return jsonify({
                                        'error': f'字段 {field_name} 数值无效',
                                        'status': 'error',
                                        'timestamp': str(time.time())
                                    }), 400
                    
                    # 将验证后的数据传递给原函数
                    request.validated_json = data
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    logger.error(f"输入验证错误: {str(e)}")
                    return jsonify({
                        'error': '输入验证失败',
                        'status': 'error',
                        'timestamp': str(time.time())
                    }), 400
            
            return wrapper
        return decorator
    
    @staticmethod
    def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
        """简单的速率限制装饰器"""
        request_counts = {}
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                import time
                client_ip = request.remote_addr
                current_time = time.time()
                
                # 清理过期记录
                if client_ip in request_counts:
                    request_counts[client_ip] = [
                        timestamp for timestamp in request_counts[client_ip]
                        if current_time - timestamp < window_seconds
                    ]
                else:
                    request_counts[client_ip] = []
                
                # 检查请求频率
                if len(request_counts[client_ip]) >= max_requests:
                    # 记录速率限制违规
                    sec_logger = get_security_logger()
                    if sec_logger:
                        sec_logger.log_rate_limit_violation(
                            client_ip,
                            request.endpoint or 'unknown',
                            len(request_counts[client_ip])
                        )
                    
                    return jsonify({
                        'error': '请求频率过高，请稍后再试',
                        'status': 'error',
                        'timestamp': str(current_time)
                    }), 429
                
                # 记录当前请求
                request_counts[client_ip].append(current_time)
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator

# 导入time模块
import time

# 预定义的验证模式
SIMULATION_SCHEMA = {
    'num_agents': {
        'type': 'integer',
        'required': True,
        'min_val': 1,
        'max_val': 10000
    },
    'scenario': {
        'type': 'string',
        'required': True,
        'pattern': 'scenario_name',
        'max_length': 50
    },
    'time_steps': {
        'type': 'integer',
        'required': False,
        'min_val': 1,
        'max_val': 1000
    },
    'seed': {
        'type': 'integer',
        'required': False,
        'min_val': 0,
        'max_val': 2147483647
    }
}

# 便捷函数
def validate_simulation_input(func):
    """验证仿真输入的便捷装饰器"""
    return SecurityDecorator.validate_json_input(SIMULATION_SCHEMA)(func)

def apply_rate_limit(func):
    """应用速率限制的便捷装饰器"""
    return SecurityDecorator.rate_limit(max_requests=50, window_seconds=3600)(func)