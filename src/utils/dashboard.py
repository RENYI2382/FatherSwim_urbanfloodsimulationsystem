"""
系统管理和监控仪表板
提供系统状态监控、性能指标展示和管理功能的Web界面
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import logging

try:
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

from .performance_monitor import metrics_collector, performance_monitor, system_monitor, metrics_exporter
from .health_monitor import health_monitor, HealthStatus
from .cache_manager import cache_manager
from .error_handler import error_handler

logger = logging.getLogger(__name__)

class SystemDashboard:
    """系统监控仪表板"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.app = None
        self.websocket_connections: List[WebSocket] = []
        
        if FASTAPI_AVAILABLE:
            self._setup_app()
        else:
            logger.warning("FastAPI未安装，无法启动Web仪表板")
    
    def _setup_app(self):
        """设置FastAPI应用"""
        self.app = FastAPI(title="智能体系统监控仪表板", version="1.0.0")
        
        # 路由设置
        self._setup_routes()
        
        # 启动后台任务
        self._setup_background_tasks()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """仪表板首页"""
            return self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_system_status():
            """获取系统状态"""
            return JSONResponse({
                "timestamp": time.time(),
                "overall_status": health_monitor.get_overall_status().value,
                "health_checks": health_monitor.get_status(),
                "metrics_summary": metrics_collector.get_summary(),
                "error_stats": error_handler.get_error_stats(),
                "cache_stats": self._get_cache_stats()
            })
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """获取性能指标"""
            return JSONResponse({
                "timestamp": time.time(),
                "metrics": [metric.to_dict() for metric in metrics_collector.get_metrics()],
                "summary": metrics_collector.get_summary()
            })
        
        @self.app.get("/api/health")
        async def get_health():
            """获取健康状态"""
            return JSONResponse({
                "timestamp": time.time(),
                "overall_status": health_monitor.get_overall_status().value,
                "checks": health_monitor.get_status()
            })
        
        @self.app.get("/api/errors")
        async def get_errors():
            """获取错误信息"""
            return JSONResponse({
                "timestamp": time.time(),
                "stats": error_handler.get_error_stats(),
                "recent_errors": [
                    {
                        "type": error.error_type,
                        "message": error.message,
                        "severity": error.severity.value,
                        "timestamp": error.timestamp,
                        "context": error.context
                    }
                    for error in error_handler.error_history[-10:]  # 最近10个错误
                ]
            })
        
        @self.app.get("/api/cache")
        async def get_cache_info():
            """获取缓存信息"""
            return JSONResponse({
                "timestamp": time.time(),
                "stats": self._get_cache_stats(),
                "namespaces": list(cache_manager.caches.keys())
            })
        
        @self.app.get("/api/export/prometheus")
        async def export_prometheus():
            """导出Prometheus格式指标"""
            return metrics_exporter.export_prometheus_format()
        
        @self.app.get("/api/export/json")
        async def export_json():
            """导出JSON格式指标"""
            return JSONResponse(json.loads(metrics_exporter.export_to_json()))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket连接"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # 保持连接活跃
                    await websocket.receive_text()
            except Exception as e:
                logger.error(f"WebSocket连接错误: {e}")
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    def _setup_background_tasks(self):
        """设置后台任务"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """启动事件"""
            # 启动系统监控
            system_monitor.start_monitoring()
            health_monitor.start_monitoring()
            
            # 启动WebSocket广播任务
            asyncio.create_task(self._websocket_broadcast_task())
            
            logger.info("系统监控仪表板已启动")
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """关闭事件"""
            system_monitor.stop_monitoring()
            health_monitor.stop_monitoring()
            logger.info("系统监控仪表板已关闭")
    
    async def _websocket_broadcast_task(self):
        """WebSocket广播任务"""
        while True:
            try:
                if self.websocket_connections:
                    # 获取最新状态
                    status_data = {
                        "type": "status_update",
                        "timestamp": time.time(),
                        "data": {
                            "overall_status": health_monitor.get_overall_status().value,
                            "metrics_summary": metrics_collector.get_summary(),
                            "error_count": len(error_handler.error_history),
                            "cache_stats": self._get_cache_stats()
                        }
                    }
                    
                    # 广播给所有连接的客户端
                    disconnected = []
                    for websocket in self.websocket_connections:
                        try:
                            await websocket.send_text(json.dumps(status_data))
                        except Exception:
                            disconnected.append(websocket)
                    
                    # 移除断开的连接
                    for websocket in disconnected:
                        self.websocket_connections.remove(websocket)
                
                await asyncio.sleep(5)  # 每5秒广播一次
                
            except Exception as e:
                logger.error(f"WebSocket广播错误: {e}")
                await asyncio.sleep(5)
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        stats = {}
        
        # 全局缓存统计
        stats["global"] = {
            "l1_size": cache_manager.global_cache.l1_cache.size(),
            "l2_size": cache_manager.global_cache.l2_cache.size()
        }
        
        # 命名空间缓存统计
        stats["namespaces"] = {}
        for name, cache in cache_manager.caches.items():
            stats["namespaces"][name] = {
                "size": cache.size()
            }
        
        return stats
    
    def _get_dashboard_html(self) -> str:
        """获取仪表板HTML"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能体系统监控仪表板</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 1rem; text-align: center; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
        .card { background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .card h3 { color: #2c3e50; margin-bottom: 1rem; border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; }
        .status { padding: 0.5rem 1rem; border-radius: 4px; color: white; font-weight: bold; text-align: center; margin: 0.5rem 0; }
        .status.healthy { background: #27ae60; }
        .status.warning { background: #f39c12; }
        .status.critical { background: #e74c3c; }
        .status.down { background: #95a5a6; }
        .metric { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #ecf0f1; }
        .metric:last-child { border-bottom: none; }
        .metric-name { font-weight: 500; color: #2c3e50; }
        .metric-value { color: #3498db; font-weight: bold; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; margin-bottom: 1rem; }
        .refresh-btn:hover { background: #2980b9; }
        .connection-status { position: fixed; top: 10px; right: 10px; padding: 0.5rem 1rem; border-radius: 4px; color: white; font-size: 0.8rem; }
        .connected { background: #27ae60; }
        .disconnected { background: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 智能体系统监控仪表板</h1>
        <p>实时监控系统状态、性能指标和健康状况</p>
    </div>
    
    <div class="connection-status" id="connectionStatus">连接中...</div>
    
    <div class="container">
        <button class="refresh-btn" onclick="refreshData()">🔄 刷新数据</button>
        
        <div class="grid">
            <!-- 系统状态 -->
            <div class="card">
                <h3>📊 系统状态</h3>
                <div id="systemStatus">加载中...</div>
            </div>
            
            <!-- 健康检查 -->
            <div class="card">
                <h3>❤️ 健康检查</h3>
                <div id="healthChecks">加载中...</div>
            </div>
            
            <!-- 性能指标 -->
            <div class="card">
                <h3>⚡ 性能指标</h3>
                <div id="performanceMetrics">加载中...</div>
            </div>
            
            <!-- 错误统计 -->
            <div class="card">
                <h3>🚨 错误统计</h3>
                <div id="errorStats">加载中...</div>
            </div>
            
            <!-- 缓存状态 -->
            <div class="card">
                <h3>💾 缓存状态</h3>
                <div id="cacheStats">加载中...</div>
            </div>
            
            <!-- 最近错误 -->
            <div class="card">
                <h3>📝 最近错误</h3>
                <div id="recentErrors">加载中...</div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let reconnectInterval = null;
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                console.log('WebSocket连接已建立');
                document.getElementById('connectionStatus').textContent = '已连接';
                document.getElementById('connectionStatus').className = 'connection-status connected';
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    updateDashboard(data.data);
                }
            };
            
            ws.onclose = function() {
                console.log('WebSocket连接已关闭');
                document.getElementById('connectionStatus').textContent = '连接断开';
                document.getElementById('connectionStatus').className = 'connection-status disconnected';
                
                // 尝试重连
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 5000);
                }
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket错误:', error);
            };
        }
        
        function updateDashboard(data) {
            // 更新系统状态
            const systemStatus = document.getElementById('systemStatus');
            systemStatus.innerHTML = `
                <div class="status ${data.overall_status}">${data.overall_status.toUpperCase()}</div>
                <div class="metric">
                    <span class="metric-name">更新时间</span>
                    <span class="metric-value">${new Date(data.timestamp * 1000).toLocaleString()}</span>
                </div>
            `;
            
            // 更新性能指标
            const performanceMetrics = document.getElementById('performanceMetrics');
            const metrics = data.metrics_summary;
            let metricsHtml = '';
            
            if (metrics.counters) {
                Object.entries(metrics.counters).forEach(([name, value]) => {
                    metricsHtml += `
                        <div class="metric">
                            <span class="metric-name">${name}</span>
                            <span class="metric-value">${value}</span>
                        </div>
                    `;
                });
            }
            
            performanceMetrics.innerHTML = metricsHtml || '<p>暂无数据</p>';
            
            // 更新错误统计
            const errorStats = document.getElementById('errorStats');
            errorStats.innerHTML = `
                <div class="metric">
                    <span class="metric-name">总错误数</span>
                    <span class="metric-value">${data.error_count}</span>
                </div>
            `;
            
            // 更新缓存状态
            const cacheStats = document.getElementById('cacheStats');
            const cache = data.cache_stats;
            cacheStats.innerHTML = `
                <div class="metric">
                    <span class="metric-name">L1缓存大小</span>
                    <span class="metric-value">${cache.global?.l1_size || 0}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">L2缓存大小</span>
                    <span class="metric-value">${cache.global?.l2_size || 0}</span>
                </div>
            `;
        }
        
        async function refreshData() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                // 更新系统状态
                updateDashboard(data);
                
                // 更新健康检查
                const healthChecks = document.getElementById('healthChecks');
                let healthHtml = '';
                if (data.health_checks) {
                    Object.entries(data.health_checks).forEach(([name, check]) => {
                        healthHtml += `
                            <div class="metric">
                                <span class="metric-name">${name}</span>
                                <span class="status ${check.status}">${check.status}</span>
                            </div>
                        `;
                    });
                }
                healthChecks.innerHTML = healthHtml || '<p>暂无数据</p>';
                
                // 获取最近错误
                const errorsResponse = await fetch('/api/errors');
                const errorsData = await errorsResponse.json();
                const recentErrors = document.getElementById('recentErrors');
                let errorsHtml = '';
                
                if (errorsData.recent_errors && errorsData.recent_errors.length > 0) {
                    errorsData.recent_errors.forEach(error => {
                        errorsHtml += `
                            <div class="metric">
                                <span class="metric-name">${error.type}</span>
                                <span class="metric-value">${error.severity}</span>
                            </div>
                        `;
                    });
                } else {
                    errorsHtml = '<p>暂无错误</p>';
                }
                recentErrors.innerHTML = errorsHtml;
                
            } catch (error) {
                console.error('刷新数据失败:', error);
            }
        }
        
        // 初始化
        connectWebSocket();
        refreshData();
        
        // 定期刷新数据
        setInterval(refreshData, 30000); // 每30秒刷新一次
    </script>
</body>
</html>
        """
    
    async def start(self):
        """启动仪表板"""
        if not FASTAPI_AVAILABLE:
            logger.error("无法启动仪表板：FastAPI未安装")
            return
        
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"启动系统监控仪表板: http://{self.host}:{self.port}")
        await server.serve()

# 全局仪表板实例
dashboard = SystemDashboard()

async def start_dashboard(host: str = "127.0.0.1", port: int = 8080):
    """启动监控仪表板"""
    global dashboard
    dashboard = SystemDashboard(host, port)
    await dashboard.start()