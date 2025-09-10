"""
性能监控和指标收集
提供系统性能监控、指标收集和分析功能
"""

import time
import asyncio
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    """指标数据类"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "tags": self.tags
        }

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def increment(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """增加计数器"""
        with self.lock:
            self.counters[name] += value
            metric = Metric(
                name=name,
                value=self.counters[name],
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """设置仪表值"""
        with self.lock:
            self.gauges[name] = value
            metric = Metric(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def record_time(self, name: str, duration: float, tags: Dict[str, str] = None):
        """记录时间"""
        with self.lock:
            self.timers[name].append(duration)
            metric = Metric(
                name=name,
                value=duration,
                metric_type=MetricType.TIMER,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics[name].append(metric)
    
    def get_metrics(self, name: str = None) -> List[Metric]:
        """获取指标"""
        with self.lock:
            if name:
                return list(self.metrics.get(name, []))
            else:
                all_metrics = []
                for metric_list in self.metrics.values():
                    all_metrics.extend(metric_list)
                return all_metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.lock:
            summary = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "timers": {}
            }
            
            for name, times in self.timers.items():
                if times:
                    summary["timers"][name] = {
                        "count": len(times),
                        "avg": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "total": sum(times)
                    }
            
            return summary

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics = metrics_collector or MetricsCollector()
        self.active_timers: Dict[str, float] = {}
        self.thresholds: Dict[str, float] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def set_threshold(self, metric_name: str, threshold: float):
        """设置阈值"""
        self.thresholds[metric_name] = threshold
    
    def start_timer(self, name: str) -> str:
        """开始计时"""
        timer_id = f"{name}_{time.time()}"
        self.active_timers[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str, tags: Dict[str, str] = None):
        """停止计时"""
        if timer_id in self.active_timers:
            duration = time.time() - self.active_timers[timer_id]
            name = timer_id.rsplit('_', 1)[0]
            self.metrics.record_time(name, duration, tags)
            del self.active_timers[timer_id]
            
            # 检查阈值
            threshold = self.thresholds.get(name)
            if threshold and duration > threshold:
                self.alerts.append({
                    "type": "performance_alert",
                    "metric": name,
                    "value": duration,
                    "threshold": threshold,
                    "timestamp": time.time(),
                    "tags": tags or {}
                })
                logger.warning(f"性能警告: {name} 耗时 {duration:.3f}s 超过阈值 {threshold}s")
    
    def monitor_function(self, name: str = None, threshold: float = None):
        """函数监控装饰器"""
        def decorator(func):
            metric_name = name or f"{func.__module__}.{func.__name__}"
            
            if threshold:
                self.set_threshold(metric_name, threshold)
            
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    timer_id = self.start_timer(metric_name)
                    try:
                        self.metrics.increment(f"{metric_name}.calls")
                        result = await func(*args, **kwargs)
                        self.metrics.increment(f"{metric_name}.success")
                        return result
                    except Exception as e:
                        self.metrics.increment(f"{metric_name}.errors")
                        raise
                    finally:
                        self.stop_timer(timer_id)
                
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    timer_id = self.start_timer(metric_name)
                    try:
                        self.metrics.increment(f"{metric_name}.calls")
                        result = func(*args, **kwargs)
                        self.metrics.increment(f"{metric_name}.success")
                        return result
                    except Exception as e:
                        self.metrics.increment(f"{metric_name}.errors")
                        raise
                    finally:
                        self.stop_timer(timer_id)
                
                return sync_wrapper
        
        return decorator
    
    def get_alerts(self, clear: bool = False) -> List[Dict[str, Any]]:
        """获取警告"""
        alerts = self.alerts.copy()
        if clear:
            self.alerts.clear()
        return alerts

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics = metrics_collector or MetricsCollector()
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 10  # 秒
    
    def start_monitoring(self):
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"系统监控错误: {e}")
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.set_gauge("system.cpu.usage", cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.metrics.set_gauge("system.memory.usage", memory.percent)
            self.metrics.set_gauge("system.memory.available", memory.available)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            self.metrics.set_gauge("system.disk.usage", disk.percent)
            self.metrics.set_gauge("system.disk.free", disk.free)
            
            # 网络IO
            net_io = psutil.net_io_counters()
            self.metrics.set_gauge("system.network.bytes_sent", net_io.bytes_sent)
            self.metrics.set_gauge("system.network.bytes_recv", net_io.bytes_recv)
            
        except ImportError:
            # psutil未安装，使用基础监控
            self._collect_basic_metrics()
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
    
    def _collect_basic_metrics(self):
        """收集基础指标"""
        import os
        
        # 进程信息
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.metrics.set_gauge("process.cpu.user_time", usage.ru_utime)
            self.metrics.set_gauge("process.cpu.system_time", usage.ru_stime)
            self.metrics.set_gauge("process.memory.max_rss", usage.ru_maxrss)
        except ImportError:
            pass
        
        # 文件描述符
        try:
            import os
            pid = os.getpid()
            fd_count = len(os.listdir(f'/proc/{pid}/fd'))
            self.metrics.set_gauge("process.fd.count", fd_count)
        except (OSError, FileNotFoundError):
            pass

class MetricsExporter:
    """指标导出器"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def export_to_json(self, filename: str = None) -> str:
        """导出为JSON格式"""
        metrics_data = {
            "timestamp": time.time(),
            "summary": self.metrics.get_summary(),
            "metrics": [metric.to_dict() for metric in self.metrics.get_metrics()]
        }
        
        json_data = json.dumps(metrics_data, indent=2, ensure_ascii=False)
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(json_data)
        
        return json_data
    
    def export_prometheus_format(self) -> str:
        """导出为Prometheus格式"""
        lines = []
        summary = self.metrics.get_summary()
        
        # 计数器
        for name, value in summary["counters"].items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # 仪表
        for name, value in summary["gauges"].items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # 计时器
        for name, stats in summary["timers"].items():
            lines.append(f"# TYPE {name}_duration_seconds histogram")
            lines.append(f"{name}_duration_seconds_count {stats['count']}")
            lines.append(f"{name}_duration_seconds_sum {stats['total']}")
        
        return "\n".join(lines)

# 全局监控实例
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor(metrics_collector)
system_monitor = SystemMonitor(metrics_collector)
metrics_exporter = MetricsExporter(metrics_collector)

# 便捷装饰器
def monitor_performance(name: str = None, threshold: float = None):
    """性能监控装饰器"""
    return performance_monitor.monitor_function(name, threshold)

def track_calls(name: str = None):
    """调用跟踪装饰器"""
    def decorator(func):
        metric_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                metrics_collector.increment(f"{metric_name}.calls")
                try:
                    result = await func(*args, **kwargs)
                    metrics_collector.increment(f"{metric_name}.success")
                    return result
                except Exception as e:
                    metrics_collector.increment(f"{metric_name}.errors")
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                metrics_collector.increment(f"{metric_name}.calls")
                try:
                    result = func(*args, **kwargs)
                    metrics_collector.increment(f"{metric_name}.success")
                    return result
                except Exception as e:
                    metrics_collector.increment(f"{metric_name}.errors")
                    raise
            return sync_wrapper
    
    return decorator