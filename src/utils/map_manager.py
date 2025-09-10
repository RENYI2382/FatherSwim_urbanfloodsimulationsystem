#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图管理模块

功能：
1. 地图格式转换和验证
2. 场景切换和管理
3. 地图预览和可视化
4. 多种地图格式支持
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import logging

logger = logging.getLogger(__name__)

class MapManager:
    """地图管理器类"""
    
    def __init__(self, maps_dir: str = "maps", scenarios_dir: str = "scenarios"):
        """初始化地图管理器"""
        self.maps_dir = Path(maps_dir)
        self.scenarios_dir = Path(scenarios_dir)
        
        # 创建目录
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        
        # 支持的地图格式
        self.supported_formats = {
            'numpy': ['.npy', '.npz'],
            'image': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
            'geojson': ['.geojson', '.json'],
            'csv': ['.csv'],
            'text': ['.txt', '.asc']
        }
        
        # 地图类型定义
        self.map_types = {
            'terrain': {'name': '地形图', 'color_map': 'terrain'},
            'elevation': {'name': '高程图', 'color_map': 'viridis'},
            'flood_risk': {'name': '洪水风险图', 'color_map': 'Blues'},
            'accessibility': {'name': '可达性图', 'color_map': 'RdYlGn'},
            'safety': {'name': '安全性图', 'color_map': 'RdYlBu'},
            'population': {'name': '人口密度图', 'color_map': 'Reds'},
            'custom': {'name': '自定义地图', 'color_map': 'viridis'}
        }
        
        # 当前活动场景
        self.current_scenario = None
        self.scenario_config_file = self.scenarios_dir / "current_scenario.json"
        
        # 加载当前场景
        self._load_current_scenario()
    
    def _load_current_scenario(self):
        """加载当前活动场景"""
        if self.scenario_config_file.exists():
            try:
                with open(self.scenario_config_file, 'r', encoding='utf-8') as f:
                    self.current_scenario = json.load(f)
                logger.info(f"加载当前场景: {self.current_scenario.get('name', 'Unknown')}")
            except Exception as e:
                logger.error(f"加载当前场景失败: {e}")
                self.current_scenario = None
    
    def _save_current_scenario(self):
        """保存当前活动场景"""
        try:
            with open(self.scenario_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_scenario, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存当前场景失败: {e}")
    
    def convert_map_format(self, input_file: Path, output_format: str, 
                          map_type: str = 'custom') -> Tuple[bool, str, Optional[Path]]:
        """转换地图格式"""
        try:
            input_ext = input_file.suffix.lower()
            output_file = input_file.parent / f"{input_file.stem}_converted"
            
            # 读取输入文件
            if input_ext in ['.npy']:
                data = np.load(input_file)
            elif input_ext in ['.csv']:
                df = pd.read_csv(input_file, header=None)
                data = df.values
            elif input_ext in ['.png', '.jpg', '.jpeg']:
                img = Image.open(input_file)
                data = np.array(img.convert('L'))  # 转换为灰度
            elif input_ext in ['.txt', '.asc']:
                data = np.loadtxt(input_file)
            else:
                return False, f"不支持的输入格式: {input_ext}", None
            
            # 确保数据是二维数组
            if len(data.shape) != 2:
                return False, "地图数据必须是二维数组", None
            
            # 转换输出格式
            if output_format == 'numpy':
                output_file = output_file.with_suffix('.npy')
                np.save(output_file, data)
            elif output_format == 'csv':
                output_file = output_file.with_suffix('.csv')
                pd.DataFrame(data).to_csv(output_file, index=False, header=False)
            elif output_format == 'image':
                output_file = output_file.with_suffix('.png')
                self._save_map_as_image(data, output_file, map_type)
            elif output_format == 'geojson':
                output_file = output_file.with_suffix('.geojson')
                self._save_map_as_geojson(data, output_file)
            else:
                return False, f"不支持的输出格式: {output_format}", None
            
            return True, "转换成功", output_file
            
        except Exception as e:
            logger.error(f"地图格式转换失败: {e}")
            return False, f"转换失败: {str(e)}", None
    
    def _save_map_as_image(self, data: np.ndarray, output_file: Path, map_type: str):
        """将地图数据保存为图像"""
        plt.figure(figsize=(10, 10))
        
        color_map = self.map_types.get(map_type, {}).get('color_map', 'viridis')
        plt.imshow(data, cmap=color_map, origin='lower')
        plt.colorbar(label=self.map_types.get(map_type, {}).get('name', '数值'))
        plt.title(f"{self.map_types.get(map_type, {}).get('name', '地图')} ({data.shape[0]}x{data.shape[1]})")
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_map_as_geojson(self, data: np.ndarray, output_file: Path):
        """将地图数据保存为GeoJSON格式"""
        features = []
        rows, cols = data.shape
        
        for i in range(rows):
            for j in range(cols):
                if data[i, j] != 0:  # 只保存非零值
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [j, i], [j+1, i], [j+1, i+1], [j, i+1], [j, i]
                            ]]
                        },
                        "properties": {
                            "value": float(data[i, j]),
                            "row": i,
                            "col": j
                        }
                    }
                    features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
    
    def create_map_preview(self, map_file: Path, map_type: str = 'custom', 
                          output_dir: Optional[Path] = None) -> Tuple[bool, str, Optional[Path]]:
        """创建地图预览图"""
        try:
            if output_dir is None:
                output_dir = map_file.parent
            
            preview_file = output_dir / f"{map_file.stem}_preview.png"
            
            # 读取地图数据
            file_ext = map_file.suffix.lower()
            if file_ext == '.npy':
                data = np.load(map_file)
            elif file_ext == '.csv':
                data = pd.read_csv(map_file, header=None).values
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                img = Image.open(map_file)
                data = np.array(img.convert('L'))
            else:
                return False, f"不支持的文件格式: {file_ext}", None
            
            # 创建预览图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"地图预览: {map_file.name}", fontsize=16)
            
            # 原始地图
            color_map = self.map_types.get(map_type, {}).get('color_map', 'viridis')
            im1 = axes[0, 0].imshow(data, cmap=color_map, origin='lower')
            axes[0, 0].set_title('原始地图')
            plt.colorbar(im1, ax=axes[0, 0])
            
            # 数据分布直方图
            axes[0, 1].hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('数值分布')
            axes[0, 1].set_xlabel('数值')
            axes[0, 1].set_ylabel('频次')
            
            # 统计信息
            stats_text = f"""形状: {data.shape}
最小值: {data.min():.3f}
最大值: {data.max():.3f}
平均值: {data.mean():.3f}
标准差: {data.std():.3f}
非零值: {np.count_nonzero(data)}"""
            
            axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes,
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 0].set_title('统计信息')
            axes[1, 0].axis('off')
            
            # 热力图（缩放版本）
            if data.shape[0] > 50 or data.shape[1] > 50:
                # 对大地图进行下采样
                step_x = max(1, data.shape[1] // 50)
                step_y = max(1, data.shape[0] // 50)
                sampled_data = data[::step_y, ::step_x]
            else:
                sampled_data = data
            
            im2 = axes[1, 1].imshow(sampled_data, cmap='hot', origin='lower')
            axes[1, 1].set_title('热力图视图')
            plt.colorbar(im2, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig(preview_file, dpi=200, bbox_inches='tight')
            plt.close()
            
            return True, "预览图创建成功", preview_file
            
        except Exception as e:
            logger.error(f"创建地图预览失败: {e}")
            return False, f"预览创建失败: {str(e)}", None
    
    def create_scenario(self, scenario_name: str, maps: Dict[str, Path], 
                      config: Optional[Dict] = None) -> Tuple[bool, str, Optional[Dict]]:
        """创建新场景"""
        try:
            scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            scenario_dir = self.scenarios_dir / scenario_id
            scenario_dir.mkdir(exist_ok=True)
            
            # 验证地图文件
            validated_maps = {}
            for map_type, map_file in maps.items():
                if not map_file.exists():
                    return False, f"地图文件不存在: {map_file}", None
                
                # 复制地图文件到场景目录
                target_file = scenario_dir / f"{map_type}{map_file.suffix}"
                import shutil
                shutil.copy2(map_file, target_file)
                validated_maps[map_type] = str(target_file.relative_to(self.scenarios_dir))
            
            # 创建场景配置
            scenario_config = {
                'id': scenario_id,
                'name': scenario_name,
                'created_time': datetime.now().isoformat(),
                'maps': validated_maps,
                'config': config or {},
                'metadata': {
                    'version': '1.0',
                    'creator': 'MapManager',
                    'description': f"场景: {scenario_name}"
                }
            }
            
            # 保存场景配置
            config_file = scenario_dir / "scenario_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(scenario_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"场景创建成功: {scenario_name} ({scenario_id})")
            return True, "场景创建成功", scenario_config
            
        except Exception as e:
            logger.error(f"创建场景失败: {e}")
            return False, f"场景创建失败: {str(e)}", None
    
    def switch_scenario(self, scenario_id: str) -> Tuple[bool, str, Optional[Dict]]:
        """切换到指定场景"""
        try:
            scenario_dir = self.scenarios_dir / scenario_id
            config_file = scenario_dir / "scenario_config.json"
            
            if not config_file.exists():
                return False, f"场景配置文件不存在: {scenario_id}", None
            
            # 加载场景配置
            with open(config_file, 'r', encoding='utf-8') as f:
                scenario_config = json.load(f)
            
            # 验证场景文件
            for map_type, map_path in scenario_config['maps'].items():
                full_path = self.scenarios_dir / map_path
                if not full_path.exists():
                    return False, f"场景地图文件缺失: {map_path}", None
            
            # 设置为当前场景
            self.current_scenario = scenario_config
            self._save_current_scenario()
            
            logger.info(f"切换到场景: {scenario_config['name']} ({scenario_id})")
            return True, "场景切换成功", scenario_config
            
        except Exception as e:
            logger.error(f"切换场景失败: {e}")
            return False, f"场景切换失败: {str(e)}", None
    
    def get_available_scenarios(self) -> List[Dict]:
        """获取可用场景列表"""
        scenarios = []
        
        try:
            for scenario_dir in self.scenarios_dir.iterdir():
                if scenario_dir.is_dir():
                    config_file = scenario_dir / "scenario_config.json"
                    if config_file.exists():
                        try:
                            with open(config_file, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            # 添加状态信息
                            config['is_current'] = (self.current_scenario and 
                                                   config['id'] == self.current_scenario.get('id'))
                            config['path'] = str(scenario_dir)
                            scenarios.append(config)
                            
                        except Exception as e:
                            logger.warning(f"读取场景配置失败 {scenario_dir}: {e}")
            
            # 按创建时间排序
            scenarios.sort(key=lambda x: x.get('created_time', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"获取场景列表失败: {e}")
        
        return scenarios
    
    def get_current_scenario(self) -> Optional[Dict]:
        """获取当前活动场景"""
        return self.current_scenario
    
    def delete_scenario(self, scenario_id: str) -> Tuple[bool, str]:
        """删除场景"""
        try:
            scenario_dir = self.scenarios_dir / scenario_id
            
            if not scenario_dir.exists():
                return False, f"场景不存在: {scenario_id}"
            
            # 检查是否为当前场景
            if (self.current_scenario and 
                self.current_scenario.get('id') == scenario_id):
                self.current_scenario = None
                self._save_current_scenario()
            
            # 删除场景目录
            import shutil
            shutil.rmtree(scenario_dir)
            
            logger.info(f"场景删除成功: {scenario_id}")
            return True, "场景删除成功"
            
        except Exception as e:
            logger.error(f"删除场景失败: {e}")
            return False, f"删除场景失败: {str(e)}"
    
    def get_map_statistics(self) -> Dict[str, Any]:
        """获取地图管理统计信息"""
        try:
            scenarios = self.get_available_scenarios()
            
            # 统计地图文件
            total_maps = 0
            map_types_count = {}
            total_size = 0
            
            for scenario in scenarios:
                for map_type in scenario.get('maps', {}).keys():
                    total_maps += 1
                    map_types_count[map_type] = map_types_count.get(map_type, 0) + 1
            
            # 计算目录大小
            for path in [self.maps_dir, self.scenarios_dir]:
                if path.exists():
                    for file_path in path.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            
            return {
                'total_scenarios': len(scenarios),
                'current_scenario': self.current_scenario.get('name') if self.current_scenario else None,
                'total_maps': total_maps,
                'map_types_distribution': map_types_count,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'supported_formats': self.supported_formats,
                'available_map_types': list(self.map_types.keys())
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def validate_map_compatibility(self, map_files: List[Path]) -> Tuple[bool, str, Dict]:
        """验证地图兼容性"""
        try:
            if not map_files:
                return False, "没有提供地图文件", {}
            
            # 检查所有地图的维度是否一致
            shapes = []
            file_info = {}
            
            for map_file in map_files:
                if not map_file.exists():
                    return False, f"文件不存在: {map_file}", {}
                
                # 读取地图数据
                file_ext = map_file.suffix.lower()
                try:
                    if file_ext == '.npy':
                        data = np.load(map_file)
                    elif file_ext == '.csv':
                        data = pd.read_csv(map_file, header=None).values
                    else:
                        continue  # 跳过不支持的格式
                    
                    shapes.append(data.shape)
                    file_info[str(map_file)] = {
                        'shape': data.shape,
                        'dtype': str(data.dtype),
                        'size_mb': round(map_file.stat().st_size / (1024 * 1024), 2)
                    }
                    
                except Exception as e:
                    return False, f"读取文件失败 {map_file}: {e}", {}
            
            # 检查维度一致性
            if len(set(shapes)) > 1:
                return False, f"地图维度不一致: {shapes}", file_info
            
            return True, "地图兼容性验证通过", file_info
            
        except Exception as e:
            logger.error(f"地图兼容性验证失败: {e}")
            return False, f"验证失败: {str(e)}", {}