#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件上传管理器
支持用户自主上传数据和地图文件
"""

import os
import json
import shutil
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import numpy as np
import pandas as pd
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class FileUploadManager:
    """文件上传管理器"""
    
    def __init__(self, upload_base_dir: str = "uploads"):
        self.upload_base_dir = Path(upload_base_dir)
        self.upload_base_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.maps_dir = self.upload_base_dir / "maps"
        self.data_dir = self.upload_base_dir / "data"
        self.scenarios_dir = self.upload_base_dir / "scenarios"
        self.temp_dir = self.upload_base_dir / "temp"
        
        for dir_path in [self.maps_dir, self.data_dir, self.scenarios_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 支持的文件类型
        self.supported_formats = {
            'maps': {
                'image': ['.png', '.jpg', '.jpeg', '.tiff', '.tif'],
                'data': ['.npy', '.npz', '.json', '.geojson'],
                'shapefile': ['.shp', '.shx', '.dbf', '.prj']
            },
            'data': {
                'csv': ['.csv'],
                'excel': ['.xlsx', '.xls'],
                'json': ['.json'],
                'numpy': ['.npy', '.npz']
            },
            'scenarios': {
                'config': ['.json', '.yaml', '.yml'],
                'script': ['.py']
            }
        }
        
        # 文件大小限制 (MB)
        self.size_limits = {
            'maps': 50,  # 50MB
            'data': 100,  # 100MB
            'scenarios': 10  # 10MB
        }
        
        # 上传记录
        self.upload_log_file = self.upload_base_dir / "upload_log.json"
        self.upload_log = self._load_upload_log()
    
    def _load_upload_log(self) -> Dict:
        """加载上传日志"""
        if self.upload_log_file.exists():
            try:
                with open(self.upload_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载上传日志失败: {e}")
        
        return {
            'uploads': [],
            'total_files': 0,
            'total_size': 0,
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_upload_log(self):
        """保存上传日志"""
        try:
            self.upload_log['last_updated'] = datetime.now().isoformat()
            with open(self.upload_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.upload_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存上传日志失败: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _validate_file(self, file: FileStorage, file_type: str) -> Tuple[bool, str]:
        """验证上传文件"""
        if not file or not file.filename:
            return False, "未选择文件"
        
        # 检查文件名安全性
        filename = secure_filename(file.filename)
        if not filename:
            return False, "文件名无效"
        
        # 检查文件扩展名
        file_ext = Path(filename).suffix.lower()
        supported_exts = []
        for format_type, exts in self.supported_formats[file_type].items():
            supported_exts.extend(exts)
        
        if file_ext not in supported_exts:
            return False, f"不支持的文件格式。支持的格式: {', '.join(supported_exts)}"
        
        # 检查文件大小
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置到文件开头
        
        max_size = self.size_limits[file_type] * 1024 * 1024  # 转换为字节
        if file_size > max_size:
            return False, f"文件过大。最大允许大小: {self.size_limits[file_type]}MB"
        
        return True, "验证通过"
    
    def _validate_map_data(self, file_path: Path) -> Tuple[bool, str, Dict]:
        """验证地图数据文件"""
        try:
            file_ext = file_path.suffix.lower()
            metadata = {}
            
            if file_ext == '.npy':
                # 验证numpy数组
                data = np.load(file_path)
                if len(data.shape) != 2:
                    return False, "地图数据必须是二维数组", {}
                
                metadata = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'min_value': float(data.min()),
                    'max_value': float(data.max()),
                    'unique_values': len(np.unique(data))
                }
                
            elif file_ext == '.json':
                # 验证JSON格式
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查必要字段
                required_fields = ['width', 'height', 'data']
                for field in required_fields:
                    if field not in data:
                        return False, f"缺少必要字段: {field}", {}
                
                metadata = {
                    'width': data['width'],
                    'height': data['height'],
                    'data_type': type(data['data']).__name__
                }
                
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff']:
                # 验证图像文件
                with Image.open(file_path) as img:
                    metadata = {
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode,
                        'format': img.format
                    }
            
            return True, "地图数据验证通过", metadata
            
        except Exception as e:
            return False, f"地图数据验证失败: {str(e)}", {}
    
    def _validate_csv_data(self, file_path: Path) -> Tuple[bool, str, Dict]:
        """验证CSV数据文件"""
        try:
            df = pd.read_csv(file_path)
            
            metadata = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            
            # 检查是否包含智能体数据的必要字段
            agent_fields = ['id', 'x', 'y']
            has_agent_fields = all(field in df.columns for field in agent_fields)
            
            if has_agent_fields:
                metadata['data_type'] = 'agent_data'
                metadata['agent_count'] = len(df)
            else:
                metadata['data_type'] = 'general_data'
            
            return True, "CSV数据验证通过", metadata
            
        except Exception as e:
            return False, f"CSV数据验证失败: {str(e)}", {}
    
    def upload_file(self, file: FileStorage, file_type: str, 
                   description: str = "") -> Dict[str, Any]:
        """上传文件"""
        try:
            # 验证文件
            is_valid, message = self._validate_file(file, file_type)
            if not is_valid:
                return {
                    'success': False,
                    'message': message
                }
            
            # 生成安全文件名
            original_filename = file.filename
            secure_name = secure_filename(original_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{secure_name}"
            
            # 确定保存路径
            if file_type == 'maps':
                save_dir = self.maps_dir
            elif file_type == 'data':
                save_dir = self.data_dir
            else:
                save_dir = self.scenarios_dir
            
            file_path = save_dir / filename
            
            # 保存文件
            file.save(str(file_path))
            
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path)
            
            # 验证文件内容
            content_valid = True
            content_message = "文件上传成功"
            metadata = {}
            
            if file_type == 'maps':
                content_valid, content_message, metadata = self._validate_map_data(file_path)
            elif file_type == 'data' and file_path.suffix.lower() == '.csv':
                content_valid, content_message, metadata = self._validate_csv_data(file_path)
            
            if not content_valid:
                # 删除无效文件
                file_path.unlink()
                return {
                    'success': False,
                    'message': content_message
                }
            
            # 记录上传信息
            upload_record = {
                'id': len(self.upload_log['uploads']) + 1,
                'original_filename': original_filename,
                'saved_filename': filename,
                'file_path': str(file_path.relative_to(self.upload_base_dir)),
                'file_type': file_type,
                'file_size': file_path.stat().st_size,
                'file_hash': file_hash,
                'description': description,
                'metadata': metadata,
                'upload_time': datetime.now().isoformat(),
                'status': 'active'
            }
            
            self.upload_log['uploads'].append(upload_record)
            self.upload_log['total_files'] += 1
            self.upload_log['total_size'] += upload_record['file_size']
            self._save_upload_log()
            
            logger.info(f"文件上传成功: {original_filename} -> {filename}")
            
            return {
                'success': True,
                'message': content_message,
                'file_id': upload_record['id'],
                'filename': filename,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"文件上传失败: {str(e)}")
            return {
                'success': False,
                'message': f"文件上传失败: {str(e)}"
            }
    
    def get_uploaded_files(self, file_type: Optional[str] = None) -> List[Dict]:
        """获取已上传文件列表"""
        files = [f for f in self.upload_log['uploads'] if f['status'] == 'active']
        
        if file_type:
            files = [f for f in files if f['file_type'] == file_type]
        
        return files
    
    def delete_file(self, file_id: int) -> Dict[str, Any]:
        """删除上传的文件"""
        try:
            # 查找文件记录
            file_record = None
            for record in self.upload_log['uploads']:
                if record['id'] == file_id and record['status'] == 'active':
                    file_record = record
                    break
            
            if not file_record:
                return {
                    'success': False,
                    'message': '文件不存在'
                }
            
            # 删除物理文件
            file_path = self.upload_base_dir / file_record['file_path']
            if file_path.exists():
                file_path.unlink()
            
            # 更新记录状态
            file_record['status'] = 'deleted'
            file_record['delete_time'] = datetime.now().isoformat()
            
            self.upload_log['total_files'] -= 1
            self.upload_log['total_size'] -= file_record['file_size']
            self._save_upload_log()
            
            logger.info(f"文件删除成功: {file_record['saved_filename']}")
            
            return {
                'success': True,
                'message': '文件删除成功'
            }
            
        except Exception as e:
            logger.error(f"文件删除失败: {str(e)}")
            return {
                'success': False,
                'message': f'文件删除失败: {str(e)}'
            }
    
    def get_file_info(self, file_id: int) -> Optional[Dict]:
        """获取文件详细信息"""
        for record in self.upload_log['uploads']:
            if record['id'] == file_id and record['status'] == 'active':
                return record
        return None
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """获取上传统计信息"""
        active_files = [f for f in self.upload_log['uploads'] if f['status'] == 'active']
        
        stats = {
            'total_files': len(active_files),
            'total_size_mb': sum(f['file_size'] for f in active_files) / (1024 * 1024),
            'by_type': {},
            'recent_uploads': sorted(active_files, key=lambda x: x['upload_time'], reverse=True)[:5]
        }
        
        # 按类型统计
        for file_type in ['maps', 'data', 'scenarios']:
            type_files = [f for f in active_files if f['file_type'] == file_type]
            stats['by_type'][file_type] = {
                'count': len(type_files),
                'size_mb': sum(f['file_size'] for f in type_files) / (1024 * 1024)
            }
        
        return stats
    
    def create_scenario_from_uploads(self, scenario_name: str, 
                                   map_file_id: int, 
                                   data_file_id: Optional[int] = None,
                                   config: Optional[Dict] = None) -> Dict[str, Any]:
        """从上传的文件创建场景"""
        try:
            # 获取地图文件
            map_file = self.get_file_info(map_file_id)
            if not map_file:
                return {
                    'success': False,
                    'message': '地图文件不存在'
                }
            
            # 获取数据文件（可选）
            data_file = None
            if data_file_id:
                data_file = self.get_file_info(data_file_id)
                if not data_file:
                    return {
                        'success': False,
                        'message': '数据文件不存在'
                    }
            
            # 创建场景配置
            scenario_config = {
                'name': scenario_name,
                'created_time': datetime.now().isoformat(),
                'map_file': {
                    'id': map_file['id'],
                    'filename': map_file['saved_filename'],
                    'path': map_file['file_path'],
                    'metadata': map_file['metadata']
                },
                'data_file': None,
                'config': config or {},
                'status': 'active'
            }
            
            if data_file:
                scenario_config['data_file'] = {
                    'id': data_file['id'],
                    'filename': data_file['saved_filename'],
                    'path': data_file['file_path'],
                    'metadata': data_file['metadata']
                }
            
            # 保存场景配置
            scenario_filename = f"scenario_{secure_filename(scenario_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            scenario_path = self.scenarios_dir / scenario_filename
            
            with open(scenario_path, 'w', encoding='utf-8') as f:
                json.dump(scenario_config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"场景创建成功: {scenario_name}")
            
            return {
                'success': True,
                'message': '场景创建成功',
                'scenario_file': scenario_filename,
                'config': scenario_config
            }
            
        except Exception as e:
            logger.error(f"场景创建失败: {str(e)}")
            return {
                'success': False,
                'message': f'场景创建失败: {str(e)}'
            }
    
    def get_scenarios(self) -> List[Dict]:
        """获取所有场景"""
        scenarios = []
        
        for scenario_file in self.scenarios_dir.glob("scenario_*.json"):
            try:
                with open(scenario_file, 'r', encoding='utf-8') as f:
                    scenario_config = json.load(f)
                    scenario_config['scenario_file'] = scenario_file.name
                    scenarios.append(scenario_config)
            except Exception as e:
                logger.error(f"读取场景文件失败 {scenario_file}: {str(e)}")
        
        return sorted(scenarios, key=lambda x: x['created_time'], reverse=True)