#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mobilkit框架集成模块

本模块实现Mobilkit与洪灾ABM仿真系统的深度集成，包括：
1. 空间分析能力增强
2. 移动轨迹分析集成
3. 分布式计算优化
4. 高级可视化功能

作者: 城市智能体系统
日期: 2025-01-20
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import asyncio
from functools import lru_cache

# Mobilkit核心模块导入
try:
    import dask.dataframe as dd
    import dask.array as da
    from dask.distributed import Client, as_completed
    
    # Mobilkit功能模块
    from mobilkit.spatial import tessellate, haversine_pairwise, rad_of_gyr, total_distance_traveled
    from mobilkit.stats import userStats
    from mobilkit.displacement import process_user_displacement_pings
    from mobilkit.viz import plot_density_map, visualize_boundarymap, compareLinePlot
    from mobilkit.tools import computeClusters
    
    MOBILKIT_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Mobilkit框架加载成功")
    
except ImportError as e:
    MOBILKIT_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Mobilkit框架不可用: {e}")
    logger.warning("将使用备用实现")

# 本地模块导入
from .spatial_movement_modeling import SpatialMovementModeling, SpatialLocation, MovementPath
from .agent_core import AgentProfile, WeatherCondition, HurricanePhase
from ..utils.data_processor import HurricaneDataProcessor
from ..utils.cache_manager import CacheManager

@dataclass
class MobilkitConfig:
    """Mobilkit集成配置"""
    enable_distributed: bool = True
    dask_scheduler_address: str = "localhost:8786"
    cache_size: int = 1000
    spatial_resolution: float = 100.0  # 米
    enable_advanced_viz: bool = True
    max_workers: int = 4
    memory_limit: str = "2GB"

class MobilkitIntegration:
    """Mobilkit集成核心类"""
    
    def __init__(self, config: MobilkitConfig = None):
        self.config = config or MobilkitConfig()
        self.dask_client = None
        self.cache_manager = CacheManager()
        self.spatial_index = None
        
        # 检查Mobilkit可用性
        if not MOBILKIT_AVAILABLE:
            logger.warning("Mobilkit不可用，某些高级功能将被禁用")
            self.config.enable_distributed = False
            self.config.enable_advanced_viz = False
        
        logger.info(f"Mobilkit集成模块初始化完成，配置: {self.config}")
    
    async def initialize_distributed_computing(self) -> bool:
        """初始化分布式计算环境"""
        if not self.config.enable_distributed or not MOBILKIT_AVAILABLE:
            logger.info("分布式计算未启用")
            return False
        
        try:
            self.dask_client = Client(
                self.config.dask_scheduler_address,
                timeout='10s',
                processes=True,
                n_workers=self.config.max_workers,
                memory_limit=self.config.memory_limit
            )
            
            logger.info(f"Dask客户端连接成功: {self.dask_client}")
            return True
            
        except Exception as e:
            logger.error(f"Dask客户端连接失败: {e}")
            self.config.enable_distributed = False
            return False
    
    def setup_spatial_tessellation(self, shapefile_path: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
        """设置高级空间网格化"""
        if not MOBILKIT_AVAILABLE:
            logger.warning("Mobilkit不可用，使用基础空间处理")
            return self._basic_spatial_processing(df)
        
        try:
            logger.info(f"使用Mobilkit进行空间网格化: {shapefile_path}")
            
            # 转换为Dask DataFrame以支持大规模数据
            if self.config.enable_distributed and len(df) > 10000:
                ddf = dd.from_pandas(df, npartitions=self.config.max_workers)
                tessellated_df, tessellation_gdf = tessellate(
                    ddf, shapefile_path, 
                    partitions_number=self.config.max_workers
                )
                tessellated_df = tessellated_df.compute()
            else:
                tessellated_df, tessellation_gdf = tessellate(df, shapefile_path)
            
            logger.info(f"空间网格化完成，处理了{len(tessellated_df)}条记录")
            return tessellated_df, tessellation_gdf
            
        except Exception as e:
            logger.error(f"Mobilkit空间网格化失败: {e}")
            return self._basic_spatial_processing(df)
    
    def analyze_agent_mobility_patterns(self, agent_data: pd.DataFrame) -> Dict[str, Any]:
        """分析智能体移动模式"""
        if not MOBILKIT_AVAILABLE:
            return self._basic_mobility_analysis(agent_data)
        
        try:
            logger.info("使用Mobilkit分析移动模式")
            
            # 计算用户统计信息
            if self.config.enable_distributed and len(agent_data) > 5000:
                ddf = dd.from_pandas(agent_data, npartitions=self.config.max_workers)
                user_stats = userStats(ddf).compute()
            else:
                user_stats = userStats(agent_data)
            
            # 分析位移模式
            displacement_results = []
            for uid in agent_data['uid'].unique():
                user_data = agent_data[agent_data['uid'] == uid]
                if len(user_data) > 1:
                    displacement = self._analyze_user_displacement(user_data)
                    displacement_results.append(displacement)
            
            # 聚类分析
            if len(agent_data) > 100:
                cluster_results = self._perform_mobility_clustering(agent_data)
            else:
                cluster_results = {"clusters": [], "labels": []}
            
            return {
                "user_statistics": user_stats.to_dict('records') if hasattr(user_stats, 'to_dict') else user_stats,
                "displacement_analysis": displacement_results,
                "clustering_results": cluster_results,
                "summary": self._generate_mobility_summary(user_stats, displacement_results)
            }
            
        except Exception as e:
            logger.error(f"Mobilkit移动模式分析失败: {e}")
            return self._basic_mobility_analysis(agent_data)
    
    def _analyze_user_displacement(self, user_data: pd.DataFrame) -> Dict[str, Any]:
        """分析单个用户的位移模式"""
        try:
            # 确保数据包含必要的列
            required_cols = ['lat', 'lon', 'datetime']
            if not all(col in user_data.columns for col in required_cols):
                logger.warning(f"用户数据缺少必要列: {required_cols}")
                return {"error": "数据格式不完整"}
            
            # 添加家庭位置（使用第一个位置作为近似）
            home_lat, home_lon = user_data.iloc[0]['lat'], user_data.iloc[0]['lon']
            user_data = user_data.copy()
            user_data['homelat'] = home_lat
            user_data['homelon'] = home_lon
            user_data['date'] = pd.to_datetime(user_data['datetime']).dt.date
            
            # 使用Mobilkit分析位移
            displacement_stats = process_user_displacement_pings(user_data)
            
            # 计算额外指标
            coordinates = user_data[['lat', 'lon']].values
            rog = rad_of_gyr(coordinates)
            ttd = total_distance_traveled(coordinates)
            
            return {
                "user_id": user_data.iloc[0].get('uid', 'unknown'),
                "displacement_stats": displacement_stats.to_dict('records') if hasattr(displacement_stats, 'to_dict') else displacement_stats,
                "radius_of_gyration": float(rog),
                "total_travel_distance": float(ttd),
                "home_location": {"lat": home_lat, "lon": home_lon}
            }
            
        except Exception as e:
            logger.error(f"用户位移分析失败: {e}")
            return {"error": str(e)}
    
    def _perform_mobility_clustering(self, agent_data: pd.DataFrame) -> Dict[str, Any]:
        """执行移动模式聚类分析"""
        try:
            # 准备聚类特征
            features = []
            user_ids = []
            
            for uid in agent_data['uid'].unique():
                user_data = agent_data[agent_data['uid'] == uid]
                if len(user_data) > 5:  # 确保有足够的数据点
                    # 计算特征向量
                    coords = user_data[['lat', 'lon']].values
                    rog = rad_of_gyr(coords)
                    ttd = total_distance_traveled(coords)
                    
                    # 计算活动范围
                    lat_range = user_data['lat'].max() - user_data['lat'].min()
                    lon_range = user_data['lon'].max() - user_data['lon'].min()
                    
                    features.append([rog, ttd, lat_range, lon_range, len(user_data)])
                    user_ids.append(uid)
            
            if len(features) < 3:
                return {"clusters": [], "labels": [], "message": "数据不足以进行聚类"}
            
            # 使用Mobilkit的聚类功能
            features_array = np.array(features)
            cluster_results = computeClusters(
                features_array, 
                distance_metric='euclidean',
                n_clusters=min(5, len(features)//2)
            )
            
            return {
                "clusters": cluster_results.get('clusters', []),
                "labels": cluster_results.get('labels', []),
                "user_ids": user_ids,
                "features": features,
                "n_clusters": len(set(cluster_results.get('labels', [])))
            }
            
        except Exception as e:
            logger.error(f"聚类分析失败: {e}")
            return {"clusters": [], "labels": [], "error": str(e)}
    
    def create_advanced_visualizations(self, agent_data: pd.DataFrame, boundary_data: gpd.GeoDataFrame = None) -> Dict[str, Any]:
        """创建高级可视化"""
        if not self.config.enable_advanced_viz or not MOBILKIT_AVAILABLE:
            return self._basic_visualization(agent_data)
        
        try:
            import matplotlib.pyplot as plt
            
            visualizations = {}
            
            # 1. 密度热力图
            if 'lat' in agent_data.columns and 'lon' in agent_data.columns:
                lats = agent_data['lat'].values
                lons = agent_data['lon'].values
                center = (np.mean(lats), np.mean(lons))
                
                fig, ax = plt.subplots(figsize=(12, 8))
                plot_density_map(lats, lons, center, bins=50, radius=1000, ax=ax)
                
                if boundary_data is not None:
                    visualize_boundarymap(boundary_data)
                
                plt.title('智能体移动密度热力图')
                visualizations['density_heatmap'] = fig
            
            # 2. 移动轨迹对比图
            if 'datetime' in agent_data.columns:
                # 按时间聚合数据
                hourly_data = agent_data.copy()
                hourly_data['hour'] = pd.to_datetime(hourly_data['datetime']).dt.hour
                hourly_counts = hourly_data.groupby('hour').size().reset_index(name='count')
                
                fig, ax = plt.subplots(figsize=(10, 6))
                compareLinePlot(
                    x_scatter='hour', x_line='hour', y='count',
                    data=hourly_counts, ax=ax
                )
                plt.title('每小时移动活动分布')
                plt.xlabel('小时')
                plt.ylabel('活动次数')
                visualizations['hourly_activity'] = fig
            
            logger.info(f"创建了{len(visualizations)}个高级可视化")
            return visualizations
            
        except Exception as e:
            logger.error(f"高级可视化创建失败: {e}")
            return self._basic_visualization(agent_data)
    
    @lru_cache(maxsize=100)
    def get_spatial_neighbors(self, location_id: int, radius: float = 1000.0) -> List[int]:
        """获取空间邻居（带缓存）"""
        # 这里应该实现基于空间索引的邻居查找
        # 暂时返回空列表作为占位符
        return []
    
    def optimize_data_processing(self, data: pd.DataFrame, operation: str) -> Any:
        """优化数据处理性能"""
        if not self.config.enable_distributed or len(data) < 1000:
            # 小数据集使用单线程处理
            return self._single_thread_processing(data, operation)
        
        try:
            # 大数据集使用分布式处理
            ddf = dd.from_pandas(data, npartitions=self.config.max_workers)
            
            if operation == 'mobility_stats':
                result = userStats(ddf).compute()
            elif operation == 'spatial_analysis':
                result = self._distributed_spatial_analysis(ddf)
            else:
                result = ddf.compute()  # 默认操作
            
            logger.info(f"分布式处理完成: {operation}")
            return result
            
        except Exception as e:
            logger.error(f"分布式处理失败: {e}，回退到单线程处理")
            return self._single_thread_processing(data, operation)
    
    def _basic_spatial_processing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, None]:
        """基础空间处理（Mobilkit不可用时的备用方案）"""
        logger.info("使用基础空间处理")
        # 添加简单的网格ID
        df = df.copy()
        df['tile_ID'] = (df['lat'] * 100).astype(int) * 1000 + (df['lon'] * 100).astype(int)
        return df, None
    
    def _basic_mobility_analysis(self, agent_data: pd.DataFrame) -> Dict[str, Any]:
        """基础移动分析（Mobilkit不可用时的备用方案）"""
        logger.info("使用基础移动分析")
        
        # 计算基础统计
        stats = {
            "total_agents": len(agent_data['uid'].unique()) if 'uid' in agent_data.columns else len(agent_data),
            "total_records": len(agent_data),
            "date_range": {
                "start": agent_data['datetime'].min() if 'datetime' in agent_data.columns else None,
                "end": agent_data['datetime'].max() if 'datetime' in agent_data.columns else None
            }
        }
        
        return {
            "user_statistics": stats,
            "displacement_analysis": [],
            "clustering_results": {"clusters": [], "labels": []},
            "summary": {"message": "使用基础分析模式"}
        }
    
    def _basic_visualization(self, agent_data: pd.DataFrame) -> Dict[str, Any]:
        """基础可视化（Mobilkit不可用时的备用方案）"""
        import matplotlib.pyplot as plt
        
        visualizations = {}
        
        if 'lat' in agent_data.columns and 'lon' in agent_data.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(agent_data['lon'], agent_data['lat'], alpha=0.6, s=1)
            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_title('智能体位置分布')
            visualizations['basic_scatter'] = fig
        
        return visualizations
    
    def _single_thread_processing(self, data: pd.DataFrame, operation: str) -> Any:
        """单线程数据处理"""
        logger.info(f"单线程处理: {operation}")
        
        if operation == 'mobility_stats':
            return self._basic_mobility_analysis(data)
        elif operation == 'spatial_analysis':
            return self._basic_spatial_processing(data)
        else:
            return data
    
    def _distributed_spatial_analysis(self, ddf) -> Any:
        """分布式空间分析"""
        # 实现分布式空间分析逻辑
        return ddf.compute()
    
    def _generate_mobility_summary(self, user_stats: Any, displacement_results: List[Dict]) -> Dict[str, Any]:
        """生成移动模式摘要"""
        summary = {
            "total_users": len(displacement_results),
            "avg_radius_of_gyration": np.mean([r.get('radius_of_gyration', 0) for r in displacement_results if 'radius_of_gyration' in r]),
            "avg_travel_distance": np.mean([r.get('total_travel_distance', 0) for r in displacement_results if 'total_travel_distance' in r]),
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
        
        return summary
    
    async def cleanup(self):
        """清理资源"""
        if self.dask_client:
            await self.dask_client.close()
            logger.info("Dask客户端已关闭")

# 全局集成实例
_mobilkit_integration = None

def get_mobilkit_integration(config: MobilkitConfig = None) -> MobilkitIntegration:
    """获取Mobilkit集成实例（单例模式）"""
    global _mobilkit_integration
    if _mobilkit_integration is None:
        _mobilkit_integration = MobilkitIntegration(config)
    return _mobilkit_integration

async def initialize_mobilkit_integration(config: MobilkitConfig = None) -> MobilkitIntegration:
    """异步初始化Mobilkit集成"""
    integration = get_mobilkit_integration(config)
    await integration.initialize_distributed_computing()
    return integration

async def cleanup_mobilkit_integration():
    """清理Mobilkit集成资源"""
    global _mobilkit_integration
    if _mobilkit_integration:
        await _mobilkit_integration.cleanup()
        _mobilkit_integration = None