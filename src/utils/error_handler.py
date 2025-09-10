"""
错误处理和重试机制
提供统一的错误处理、重试逻辑和降级策略
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorInfo:
    """错误信息类"""
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: float
    traceback: str
    context: Dict[str, Any]
    retry_count: int = 0

class RetryConfig:
    """重试配置"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.error_callbacks: Dict[str, List[Callable]] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
    
    def register_callback(self, error_type: str, callback: Callable):
        """注册错误回调函数"""
        if error_type not in self.error_callbacks:
            self.error_callbacks[error_type] = []
        self.error_callbacks[error_type].append(callback)
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """注册降级策略"""
        self.fallback_strategies[operation_name] = fallback_func
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> ErrorInfo:
        """处理错误"""
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            timestamp=time.time(),
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        # 记录错误
        self.error_history.append(error_info)
        
        # 记录日志
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"错误处理: {error_info.error_type} - {error_info.message}")
        
        # 执行回调
        callbacks = self.error_callbacks.get(error_info.error_type, [])
        for callback in callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"错误回调执行失败: {e}")
        
        return error_info
    
    def get_fallback(self, operation_name: str) -> Optional[Callable]:
        """获取降级策略"""
        return self.fallback_strategies.get(operation_name)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_counts,
            "severity_distribution": severity_counts,
            "latest_error": self.error_history[-1].timestamp if self.error_history else None
        }

def retry_with_backoff(
    retry_config: RetryConfig = None,
    exceptions: tuple = (Exception,),
    fallback_operation: str = None
):
    """重试装饰器"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_retries:
                        # 最后一次尝试失败，检查是否有降级策略
                        if fallback_operation:
                            fallback = error_handler.get_fallback(fallback_operation)
                            if fallback:
                                logger.warning(f"使用降级策略: {fallback_operation}")
                                return await fallback(*args, **kwargs)
                        
                        # 记录错误
                        error_handler.handle_error(
                            e,
                            context={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "args": str(args)[:100],
                                "kwargs": str(kwargs)[:100]
                            },
                            severity=ErrorSeverity.HIGH
                        )
                        raise
                    
                    # 计算延迟时间
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"重试 {func.__name__} (第{attempt + 1}次), {delay:.2f}秒后重试: {e}")
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == retry_config.max_retries:
                        # 最后一次尝试失败，检查是否有降级策略
                        if fallback_operation:
                            fallback = error_handler.get_fallback(fallback_operation)
                            if fallback:
                                logger.warning(f"使用降级策略: {fallback_operation}")
                                return fallback(*args, **kwargs)
                        
                        # 记录错误
                        error_handler.handle_error(
                            e,
                            context={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "args": str(args)[:100],
                                "kwargs": str(kwargs)[:100]
                            },
                            severity=ErrorSeverity.HIGH
                        )
                        raise
                    
                    # 计算延迟时间
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    
                    if retry_config.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"重试 {func.__name__} (第{attempt + 1}次), {delay:.2f}秒后重试: {e}")
                    time.sleep(delay)
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
):
    """熔断器装饰器"""
    def decorator(func):
        func._failure_count = 0
        func._last_failure_time = None
        func._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            now = time.time()
            
            # 检查熔断器状态
            if func._state == "OPEN":
                if now - func._last_failure_time < recovery_timeout:
                    raise Exception(f"熔断器开启: {func.__name__}")
                else:
                    func._state = "HALF_OPEN"
            
            try:
                result = await func(*args, **kwargs)
                
                # 成功调用，重置计数器
                if func._state == "HALF_OPEN":
                    func._state = "CLOSED"
                func._failure_count = 0
                
                return result
                
            except expected_exception as e:
                func._failure_count += 1
                func._last_failure_time = now
                
                if func._failure_count >= failure_threshold:
                    func._state = "OPEN"
                    logger.error(f"熔断器开启: {func.__name__} (失败次数: {func._failure_count})")
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            now = time.time()
            
            # 检查熔断器状态
            if func._state == "OPEN":
                if now - func._last_failure_time < recovery_timeout:
                    raise Exception(f"熔断器开启: {func.__name__}")
                else:
                    func._state = "HALF_OPEN"
            
            try:
                result = func(*args, **kwargs)
                
                # 成功调用，重置计数器
                if func._state == "HALF_OPEN":
                    func._state = "CLOSED"
                func._failure_count = 0
                
                return result
                
            except expected_exception as e:
                func._failure_count += 1
                func._last_failure_time = now
                
                if func._failure_count >= failure_threshold:
                    func._state = "OPEN"
                    logger.error(f"熔断器开启: {func.__name__} (失败次数: {func._failure_count})")
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 全局错误处理器实例
error_handler = ErrorHandler()

# 注册常见的降级策略
async def llm_fallback(*args, **kwargs):
    """LLM调用降级策略"""
    logger.warning("LLM调用失败，使用默认响应")
    return {
        "content": "LLM服务暂时不可用，使用默认策略",
        "fallback": True
    }

def risk_assessment_fallback(*args, **kwargs):
    """风险评估降级策略"""
    logger.warning("风险评估失败，使用基础算法")
    return 0.5  # 中等风险

def evacuation_decision_fallback(*args, **kwargs):
    """疏散决策降级策略"""
    logger.warning("疏散决策失败，使用保守策略")
    return True  # 保守策略：建议疏散

# 注册降级策略
error_handler.register_fallback("llm_call", llm_fallback)
error_handler.register_fallback("risk_assessment", risk_assessment_fallback)
error_handler.register_fallback("evacuation_decision", evacuation_decision_fallback)