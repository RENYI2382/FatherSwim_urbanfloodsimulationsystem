#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图管理API路由

提供地图格式转换、场景切换、地图预览等功能的RESTful接口
"""

import os
import json
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import logging

from ..utils.map_manager import MapManager

logger = logging.getLogger(__name__)

# 创建蓝图
map_bp = Blueprint('map_management', __name__, url_prefix='/api/maps')

# 全局地图管理器实例
map_manager = None

def init_map_manager(app):
    """初始化地图管理器"""
    global map_manager
    upload_dir = app.config.get('UPLOAD_FOLDER', 'uploads')
    maps_dir = os.path.join(upload_dir, 'maps')
    scenarios_dir = os.path.join(upload_dir, 'scenarios')
    map_manager = MapManager(maps_dir, scenarios_dir)
    logger.info(f"地图管理器初始化完成，地图目录: {maps_dir}, 场景目录: {scenarios_dir}")

@map_bp.route('/formats', methods=['GET'])
def get_supported_formats():
    """获取支持的地图格式"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        return jsonify({
            'success': True,
            'data': {
                'supported_formats': map_manager.supported_formats,
                'map_types': map_manager.map_types
            }
        })
        
    except Exception as e:
        logger.error(f"获取支持格式失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取支持格式失败: {str(e)}'
        }), 500

@map_bp.route('/convert', methods=['POST'])
def convert_map_format():
    """转换地图格式"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        # 获取参数
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '缺少请求数据'
            }), 400
        
        input_file_path = data.get('input_file')
        output_format = data.get('output_format')
        map_type = data.get('map_type', 'custom')
        
        if not input_file_path or not output_format:
            return jsonify({
                'success': False,
                'message': '缺少必要参数: input_file, output_format'
            }), 400
        
        input_file = Path(input_file_path)
        if not input_file.exists():
            return jsonify({
                'success': False,
                'message': f'输入文件不存在: {input_file_path}'
            }), 404
        
        # 执行格式转换
        success, message, output_file = map_manager.convert_map_format(
            input_file, output_format, map_type
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'data': {
                    'output_file': str(output_file),
                    'input_file': str(input_file),
                    'output_format': output_format,
                    'map_type': map_type
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        logger.error(f"地图格式转换失败: {e}")
        return jsonify({
            'success': False,
            'message': f'格式转换失败: {str(e)}'
        }), 500

@map_bp.route('/preview', methods=['POST'])
def create_map_preview():
    """创建地图预览"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '缺少请求数据'
            }), 400
        
        map_file_path = data.get('map_file')
        map_type = data.get('map_type', 'custom')
        
        if not map_file_path:
            return jsonify({
                'success': False,
                'message': '缺少地图文件路径'
            }), 400
        
        map_file = Path(map_file_path)
        if not map_file.exists():
            return jsonify({
                'success': False,
                'message': f'地图文件不存在: {map_file_path}'
            }), 404
        
        # 创建预览图
        success, message, preview_file = map_manager.create_map_preview(
            map_file, map_type
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'data': {
                    'preview_file': str(preview_file),
                    'map_file': str(map_file),
                    'map_type': map_type
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        logger.error(f"创建地图预览失败: {e}")
        return jsonify({
            'success': False,
            'message': f'预览创建失败: {str(e)}'
        }), 500

@map_bp.route('/scenarios', methods=['GET'])
def get_scenarios():
    """获取所有场景"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        scenarios = map_manager.get_available_scenarios()
        current_scenario = map_manager.get_current_scenario()
        
        return jsonify({
            'success': True,
            'data': {
                'scenarios': scenarios,
                'current_scenario': current_scenario,
                'total_count': len(scenarios)
            }
        })
        
    except Exception as e:
        logger.error(f"获取场景列表失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取场景列表失败: {str(e)}'
        }), 500

@map_bp.route('/scenarios', methods=['POST'])
def create_scenario():
    """创建新场景"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '缺少请求数据'
            }), 400
        
        scenario_name = data.get('name')
        maps_data = data.get('maps', {})
        config = data.get('config', {})
        
        if not scenario_name:
            return jsonify({
                'success': False,
                'message': '缺少场景名称'
            }), 400
        
        if not maps_data:
            return jsonify({
                'success': False,
                'message': '缺少地图文件信息'
            }), 400
        
        # 转换地图路径
        maps = {}
        for map_type, map_path in maps_data.items():
            map_file = Path(map_path)
            if not map_file.exists():
                return jsonify({
                    'success': False,
                    'message': f'地图文件不存在: {map_path}'
                }), 404
            maps[map_type] = map_file
        
        # 创建场景
        success, message, scenario_config = map_manager.create_scenario(
            scenario_name, maps, config
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'data': scenario_config
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        logger.error(f"创建场景失败: {e}")
        return jsonify({
            'success': False,
            'message': f'创建场景失败: {str(e)}'
        }), 500

@map_bp.route('/scenarios/<scenario_id>/switch', methods=['POST'])
def switch_scenario(scenario_id):
    """切换到指定场景"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        # 切换场景
        success, message, scenario_config = map_manager.switch_scenario(scenario_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message,
                'data': scenario_config
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        logger.error(f"切换场景失败: {e}")
        return jsonify({
            'success': False,
            'message': f'切换场景失败: {str(e)}'
        }), 500

@map_bp.route('/scenarios/<scenario_id>', methods=['DELETE'])
def delete_scenario(scenario_id):
    """删除场景"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        # 删除场景
        success, message = map_manager.delete_scenario(scenario_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        logger.error(f"删除场景失败: {e}")
        return jsonify({
            'success': False,
            'message': f'删除场景失败: {str(e)}'
        }), 500

@map_bp.route('/scenarios/current', methods=['GET'])
def get_current_scenario():
    """获取当前活动场景"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        current_scenario = map_manager.get_current_scenario()
        
        return jsonify({
            'success': True,
            'data': {
                'current_scenario': current_scenario
            }
        })
        
    except Exception as e:
        logger.error(f"获取当前场景失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取当前场景失败: {str(e)}'
        }), 500

@map_bp.route('/validate', methods=['POST'])
def validate_map_compatibility():
    """验证地图兼容性"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'message': '缺少请求数据'
            }), 400
        
        map_files_paths = data.get('map_files', [])
        
        if not map_files_paths:
            return jsonify({
                'success': False,
                'message': '缺少地图文件列表'
            }), 400
        
        # 转换为Path对象
        map_files = []
        for file_path in map_files_paths:
            map_file = Path(file_path)
            if not map_file.exists():
                return jsonify({
                    'success': False,
                    'message': f'地图文件不存在: {file_path}'
                }), 404
            map_files.append(map_file)
        
        # 验证兼容性
        success, message, file_info = map_manager.validate_map_compatibility(map_files)
        
        return jsonify({
            'success': success,
            'message': message,
            'data': {
                'file_info': file_info,
                'compatible': success
            }
        })
        
    except Exception as e:
        logger.error(f"地图兼容性验证失败: {e}")
        return jsonify({
            'success': False,
            'message': f'兼容性验证失败: {str(e)}'
        }), 500

@map_bp.route('/statistics', methods=['GET'])
def get_map_statistics():
    """获取地图管理统计信息"""
    try:
        if not map_manager:
            return jsonify({
                'success': False,
                'message': '地图管理器未初始化'
            }), 500
        
        statistics = map_manager.get_map_statistics()
        
        return jsonify({
            'success': True,
            'data': statistics
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        return jsonify({
            'success': False,
            'message': f'获取统计信息失败: {str(e)}'
        }), 500

@map_bp.route('/download/<path:file_path>', methods=['GET'])
def download_file(file_path):
    """下载地图文件"""
    try:
        # 安全检查：确保文件路径在允许的目录内
        safe_path = Path(file_path).resolve()
        upload_dir = Path(current_app.config.get('UPLOAD_FOLDER', 'uploads')).resolve()
        
        if not str(safe_path).startswith(str(upload_dir)):
            return jsonify({
                'success': False,
                'message': '文件路径不安全'
            }), 403
        
        if not safe_path.exists():
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
        
        return send_file(
            safe_path,
            as_attachment=True,
            download_name=safe_path.name
        )
        
    except Exception as e:
        logger.error(f"文件下载失败: {e}")
        return jsonify({
            'success': False,
            'message': f'文件下载失败: {str(e)}'
        }), 500

# 错误处理
@map_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': '请求的资源不存在'
    }), 404

@map_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'message': '服务器内部错误'
    }), 500