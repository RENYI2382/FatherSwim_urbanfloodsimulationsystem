"""
系统监控和健康检查
提供系统健康状态监控、服务可用性检查和告警功能
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

@dataclass
class HealthCheck:
    """健康检查配置"""
    name: str
    check_func: Callable
    interval: float = 30.0  # 检查间隔（秒）
    timeout: float = 10.0   # 超时时间（秒）
    retries: int = 3        # 重试次数
    critical: bool = False  # 是否为关键服务
    
@dataclass
class HealthResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)

class HealthMonitor:
    """健康监控器"""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthResult] = {}
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks: List[Callable] = []
        self.lock = threading.RLock()
    
    def register_check(self, check: HealthCheck):
        """注册健康检查"""
        with self.lock:
            self.checks[check.name] = check
            logger.info(f"注册健康检查: {check.name}")
    
    def unregister_check(self, name: str):
        """取消注册健康检查"""
        with self.lock:
            if name in self.checks:
                del self.checks[name]
                if name in self.results:
                    del self.results[name]
                logger.info(f"取消注册健康检查: {name}")
    
    def add_callback(self, callback: Callable[[HealthResult], None]):
        """添加状态变化回调"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("健康监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("健康监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self._run_all_checks()
                time.sleep(1)  # 每秒检查一次是否有需要执行的检查
            except Exception as e:
                logger.error(f"健康监控循环错误: {e}")
    
    def _run_all_checks(self):
        """运行所有检查"""
        current_time = time.time()
        
        with self.lock:
            for check in self.checks.values():
                # 检查是否需要执行
                last_result = self.results.get(check.name)
                if (last_result is None or 
                    current_time - last_result.timestamp >= check.interval):
                    
                    # 在新线程中执行检查
                    threading.Thread(
                        target=self._run_single_check,
                        args=(check,),
                        daemon=True
                    ).start()
    
    def _run_single_check(self, check: HealthCheck):
        """运行单个检查"""
        start_time = time.time()
        
        for attempt in range(check.retries + 1):
            try:
                # 执行检查函数
                if asyncio.iscoroutinefunction(check.check_func):
                    # 异步函数
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            asyncio.wait_for(check.check_func(), timeout=check.timeout)
                        )
                    finally:
                        loop.close()
                else:
                    # 同步函数
                    result = check.check_func()
                
                # 处理结果
                if isinstance(result, dict):
                    status = HealthStatus(result.get("status", "healthy"))
                    message = result.get("message", "检查通过")
                    details = result.get("details", {})
                else:
                    status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                    message = "检查通过" if result else "检查失败"
                    details = {}
                
                break
                
            except asyncio.TimeoutError:
                if attempt == check.retries:
                    status = HealthStatus.CRITICAL
                    message = f"检查超时 (>{check.timeout}s)"
                    details = {"timeout": check.timeout}
                else:
                    continue
            except Exception as e:
                if attempt == check.retries:
                    status = HealthStatus.CRITICAL
                    message = f"检查失败: {str(e)}"
                    details = {"error": str(e)}
                else:
                    continue
        
        # 创建结果
        duration = time.time() - start_time
        result = HealthResult(
            name=check.name,
            status=status,
            message=message,
            timestamp=time.time(),
            duration=duration,
            details=details
        )
        
        # 存储结果
        with self.lock:
            old_result = self.results.get(check.name)
            self.results[check.name] = result
            
            # 如果状态发生变化，触发回调
            if old_result is None or old_result.status != result.status:
                for callback in self.callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"健康检查回调错误: {e}")
        
        # 记录日志
        log_level = {
            HealthStatus.HEALTHY: logging.INFO,
            HealthStatus.WARNING: logging.WARNING,
            HealthStatus.CRITICAL: logging.ERROR,
            HealthStatus.DOWN: logging.CRITICAL
        }[status]
        
        logger.log(log_level, f"健康检查 {check.name}: {status.value} - {message}")
    
    def get_status(self, name: str = None) -> Dict[str, Any]:
        """获取健康状态"""
        with self.lock:
            if name:
                result = self.results.get(name)
                return result.__dict__ if result else None
            else:
                return {
                    name: result.__dict__ 
                    for name, result in self.results.items()
                }
    
    def get_overall_status(self) -> HealthStatus:
        """获取整体健康状态"""
        with self.lock:
            if not self.results:
                return HealthStatus.DOWN
            
            statuses = [result.status for result in self.results.values()]
            
            # 如果有任何关键服务DOWN，整体状态为DOWN
            critical_checks = [
                check.name for check in self.checks.values() 
                if check.critical
            ]
            
            for name in critical_checks:
                if name in self.results:
                    if self.results[name].status == HealthStatus.DOWN:
                        return HealthStatus.DOWN
                    elif self.results[name].status == HealthStatus.CRITICAL:
                        return HealthStatus.CRITICAL
            
            # 根据最严重的状态确定整体状态
            if HealthStatus.CRITICAL in statuses:
                return HealthStatus.CRITICAL
            elif HealthStatus.WARNING in statuses:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY

# 预定义的健康检查函数
async def check_llm_service():
    """检查LLM服务"""
    try:
        from ..core.llm_integration import LLMClient
        client = LLMClient()
        
        # 简单的健康检查请求
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=10
        )
        
        if response and "content" in response:
            return {
                "status": "healthy",
                "message": "LLM服务正常",
                "details": {"response_time": response.get("response_time", 0)}
            }
        else:
            return {
                "status": "warning",
                "message": "LLM服务响应异常",
                "details": {"response": response}
            }
    except Exception as e:
        return {
            "status": "critical",
            "message": f"LLM服务不可用: {str(e)}",
            "details": {"error": str(e)}
        }

def check_memory_usage():
    """检查内存使用率"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        if memory.percent > 90:
            status = "critical"
            message = f"内存使用率过高: {memory.percent:.1f}%"
        elif memory.percent > 80:
            status = "warning"
            message = f"内存使用率较高: {memory.percent:.1f}%"
        else:
            status = "healthy"
            message = f"内存使用率正常: {memory.percent:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "percent": memory.percent,
                "available": memory.available,
                "total": memory.total
            }
        }
    except ImportError:
        return {
            "status": "warning",
            "message": "无法检查内存使用率（psutil未安装）"
        }
    except Exception as e:
        return {
            "status": "critical",
            "message": f"内存检查失败: {str(e)}"
        }

def check_disk_space():
    """检查磁盘空间"""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        
        if disk.percent > 95:
            status = "critical"
            message = f"磁盘空间不足: {disk.percent:.1f}%"
        elif disk.percent > 85:
            status = "warning"
            message = f"磁盘空间较少: {disk.percent:.1f}%"
        else:
            status = "healthy"
            message = f"磁盘空间充足: {disk.percent:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "percent": disk.percent,
                "free": disk.free,
                "total": disk.total
            }
        }
    except ImportError:
        return {
            "status": "warning",
            "message": "无法检查磁盘空间（psutil未安装）"
        }
    except Exception as e:
        return {
            "status": "critical",
            "message": f"磁盘检查失败: {str(e)}"
        }

# 全局健康监控器实例
health_monitor = HealthMonitor()

# 注册默认健康检查
health_monitor.register_check(HealthCheck(
    name="llm_service",
    check_func=check_llm_service,
    interval=60.0,
    critical=True
))

health_monitor.register_check(HealthCheck(
    name="memory_usage",
    check_func=check_memory_usage,
    interval=30.0,
    critical=False
))

health_monitor.register_check(HealthCheck(
    name="disk_space",
    check_func=check_disk_space,
    interval=300.0,  # 5分钟检查一次
    critical=False
))

# 状态变化回调
def log_status_change(result: HealthResult):
    """记录状态变化"""
    logger.info(f"健康状态变化: {result.name} -> {result.status.value}: {result.message}")

health_monitor.add_callback(log_status_change)