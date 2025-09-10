"""
系统管理和监控工具集成
提供统一的系统管理入口和监控功能
"""

import asyncio
import logging
import signal
import sys
from typing import Optional
from pathlib import Path

# 导入所有监控组件
from .error_handler import error_handler, ErrorSeverity
from .performance_monitor import (
    metrics_collector, performance_monitor, system_monitor, 
    metrics_exporter, monitor_performance, track_calls
)
from .health_monitor import health_monitor, HealthCheck, HealthStatus
from .cache_manager import cache_manager, cached, async_cached
from .config_manager import ConfigManager
from .dashboard import dashboard, start_dashboard

logger = logging.getLogger(__name__)

class SystemManager:
    """系统管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_manager = ConfigManager()
        self.running = False
        self.dashboard_task = None
        
        # 加载配置
        if config_path:
            self.config_manager.load_config(config_path)
    
    async def start(self, enable_dashboard: bool = True, dashboard_port: int = 8080):
        """启动系统管理器"""
        logger.info("🚀 启动智能体系统管理器...")
        
        try:
            # 启动各个监控组件
            self._start_monitoring_components()
            
            # 启动仪表板
            if enable_dashboard:
                self.dashboard_task = asyncio.create_task(
                    start_dashboard(port=dashboard_port)
                )
                logger.info(f"📊 监控仪表板已启动: http://localhost:{dashboard_port}")
            
            # 注册信号处理
            self._setup_signal_handlers()
            
            self.running = True
            logger.info("✅ 系统管理器启动完成")
            
            # 保持运行
            if enable_dashboard:
                await self.dashboard_task
            else:
                while self.running:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"❌ 系统管理器启动失败: {e}")
            await self.stop()
            raise
    
    def _start_monitoring_components(self):
        """启动监控组件"""
        # 启动系统监控
        system_monitor.start_monitoring()
        logger.info("📈 系统性能监控已启动")
        
        # 启动健康监控
        health_monitor.start_monitoring()
        logger.info("❤️ 健康状态监控已启动")
        
        # 注册自定义健康检查
        self._register_custom_health_checks()
        
        # 注册错误处理回调
        self._register_error_callbacks()
        
        logger.info("🔧 所有监控组件已启动")
    
    def _register_custom_health_checks(self):
        """注册自定义健康检查"""
        
        def check_config_files():
            """检查配置文件"""
            try:
                config_files = [
                    "llm_config.yaml",
                    "agent_config.yaml"
                ]
                
                missing_files = []
                for file in config_files:
                    if not Path(file).exists():
                        missing_files.append(file)
                
                if missing_files:
                    return {
                        "status": "warning",
                        "message": f"缺少配置文件: {', '.join(missing_files)}",
                        "details": {"missing_files": missing_files}
                    }
                else:
                    return {
                        "status": "healthy",
                        "message": "所有配置文件正常",
                        "details": {"config_files": config_files}
                    }
            except Exception as e:
                return {
                    "status": "critical",
                    "message": f"配置文件检查失败: {str(e)}"
                }
        
        def check_cache_performance():
            """检查缓存性能"""
            try:
                # 获取缓存统计
                l1_size = cache_manager.global_cache.l1_cache.size()
                l2_size = cache_manager.global_cache.l2_cache.size()
                
                if l1_size > 80:  # L1缓存过大
                    return {
                        "status": "warning",
                        "message": f"L1缓存使用率较高: {l1_size}/100",
                        "details": {"l1_size": l1_size, "l2_size": l2_size}
                    }
                else:
                    return {
                        "status": "healthy",
                        "message": "缓存性能正常",
                        "details": {"l1_size": l1_size, "l2_size": l2_size}
                    }
            except Exception as e:
                return {
                    "status": "critical",
                    "message": f"缓存检查失败: {str(e)}"
                }
        
        # 注册健康检查
        health_monitor.register_check(HealthCheck(
            name="config_files",
            check_func=check_config_files,
            interval=300.0,  # 5分钟检查一次
            critical=False
        ))
        
        health_monitor.register_check(HealthCheck(
            name="cache_performance",
            check_func=check_cache_performance,
            interval=60.0,   # 1分钟检查一次
            critical=False
        ))
    
    def _register_error_callbacks(self):
        """注册错误处理回调"""
        
        def critical_error_callback(error_info):
            """关键错误回调"""
            if error_info.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"🚨 关键错误: {error_info.error_type} - {error_info.message}")
                # 这里可以添加告警通知逻辑
        
        def performance_alert_callback(error_info):
            """性能警告回调"""
            if "timeout" in error_info.message.lower():
                logger.warning(f"⏰ 性能警告: {error_info.message}")
                # 记录性能指标
                metrics_collector.increment("performance.timeouts")
        
        # 注册回调
        error_handler.register_callback("Exception", critical_error_callback)
        error_handler.register_callback("TimeoutError", performance_alert_callback)
    
    def _setup_signal_handlers(self):
        """设置信号处理"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，正在关闭系统...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def stop(self):
        """停止系统管理器"""
        logger.info("🛑 正在停止系统管理器...")
        
        self.running = False
        
        # 停止仪表板
        if self.dashboard_task:
            self.dashboard_task.cancel()
            try:
                await self.dashboard_task
            except asyncio.CancelledError:
                pass
        
        # 停止监控组件
        system_monitor.stop_monitoring()
        health_monitor.stop_monitoring()
        
        # 清理缓存
        cache_manager.global_cache.clear()
        
        # 导出最终指标
        try:
            final_metrics = metrics_exporter.export_to_json()
            with open("final_metrics.json", "w", encoding="utf-8") as f:
                f.write(final_metrics)
            logger.info("📊 最终指标已导出到 final_metrics.json")
        except Exception as e:
            logger.error(f"导出最终指标失败: {e}")
        
        logger.info("✅ 系统管理器已停止")
    
    def get_system_info(self) -> dict:
        """获取系统信息"""
        return {
            "status": "running" if self.running else "stopped",
            "health": {
                "overall_status": health_monitor.get_overall_status().value,
                "checks": health_monitor.get_status()
            },
            "performance": {
                "metrics_summary": metrics_collector.get_summary(),
                "alerts": performance_monitor.get_alerts()
            },
            "errors": error_handler.get_error_stats(),
            "cache": {
                "l1_size": cache_manager.global_cache.l1_cache.size(),
                "l2_size": cache_manager.global_cache.l2_cache.size(),
                "namespaces": list(cache_manager.caches.keys())
            }
        }

# 全局系统管理器实例
system_manager = SystemManager()

# 便捷函数
async def start_system(
    config_path: Optional[str] = None,
    enable_dashboard: bool = True,
    dashboard_port: int = 8080
):
    """启动系统"""
    global system_manager
    if config_path:
        system_manager = SystemManager(config_path)
    
    await system_manager.start(enable_dashboard, dashboard_port)

async def stop_system():
    """停止系统"""
    await system_manager.stop()

def get_system_status() -> dict:
    """获取系统状态"""
    return system_manager.get_system_info()

# 导出主要组件
__all__ = [
    # 管理器
    "SystemManager", "system_manager",
    
    # 便捷函数
    "start_system", "stop_system", "get_system_status",
    
    # 监控组件
    "error_handler", "metrics_collector", "performance_monitor",
    "health_monitor", "cache_manager", "dashboard",
    
    # 装饰器
    "monitor_performance", "track_calls", "cached", "async_cached",
    
    # 枚举
    "ErrorSeverity", "HealthStatus"
]