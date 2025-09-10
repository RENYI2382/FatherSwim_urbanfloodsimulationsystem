#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件上传API路由
提供文件上传、管理和场景创建的RESTful接口
"""

import os
import json
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from pathlib import Path
import logging

# 导入文件上传管理器
try:
    from src.utils.file_upload_manager import FileUploadManager
except ImportError:
    from utils.file_upload_manager import FileUploadManager

logger = logging.getLogger(__name__)

# 创建蓝图
upload_bp = Blueprint('upload', __name__, url_prefix='/api/upload')

# 初始化文件上传管理器
upload_manager = None

def init_upload_manager(app):
    """初始化文件上传管理器"""
    global upload_manager
    upload_base_dir = app.config.get('UPLOAD_FOLDER', 'uploads')
    upload_manager = FileUploadManager(upload_base_dir)
    logger.info(f"文件上传管理器初始化完成，上传目录: {upload_base_dir}")

@upload_bp.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """处理文件过大错误"""
    return jsonify({
        'success': False,
        'message': '文件过大，请选择较小的文件'
    }), 413

@upload_bp.route('/config', methods=['GET'])
def get_upload_config():
    """获取上传配置信息"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        config = {
            'supported_formats': upload_manager.supported_formats,
            'size_limits': upload_manager.size_limits,
            'upload_statistics': upload_manager.get_upload_statistics()
        }
        
        return jsonify({
            'success': True,
            'config': config
        })
        
    except Exception as e:
        logger.error(f"获取上传配置失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取配置失败: {str(e)}'
        }), 500

@upload_bp.route('/file', methods=['POST'])
def upload_file():
    """上传文件"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        file = request.files['file']
        file_type = request.form.get('file_type', 'data')
        description = request.form.get('description', '')
        
        # 验证文件类型参数
        if file_type not in ['maps', 'data', 'scenarios']:
            return jsonify({
                'success': False,
                'message': '无效的文件类型，支持: maps, data, scenarios'
            }), 400
        
        # 上传文件
        result = upload_manager.upload_file(file, file_type, description)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'文件上传失败: {str(e)}'
        }), 500

@upload_bp.route('/files', methods=['GET'])
def get_files():
    """获取已上传文件列表"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        file_type = request.args.get('file_type')
        files = upload_manager.get_uploaded_files(file_type)
        
        return jsonify({
            'success': True,
            'files': files,
            'total': len(files)
        })
        
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取文件列表失败: {str(e)}'
        }), 500

@upload_bp.route('/file/<int:file_id>', methods=['GET'])
def get_file_info(file_id):
    """获取文件详细信息"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        file_info = upload_manager.get_file_info(file_id)
        
        if file_info:
            return jsonify({
                'success': True,
                'file': file_info
            })
        else:
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
            
    except Exception as e:
        logger.error(f"获取文件信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取文件信息失败: {str(e)}'
        }), 500

@upload_bp.route('/file/<int:file_id>', methods=['DELETE'])
def delete_file(file_id):
    """删除文件"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        result = upload_manager.delete_file(file_id)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"删除文件失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'删除文件失败: {str(e)}'
        }), 500

@upload_bp.route('/file/<int:file_id>/download', methods=['GET'])
def download_file(file_id):
    """下载文件"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        file_info = upload_manager.get_file_info(file_id)
        
        if not file_info:
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
        
        file_path = upload_manager.upload_base_dir / file_info['file_path']
        
        if not file_path.exists():
            return jsonify({
                'success': False,
                'message': '文件不存在于磁盘'
            }), 404
        
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name=file_info['original_filename']
        )
        
    except Exception as e:
        logger.error(f"下载文件失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'下载文件失败: {str(e)}'
        }), 500

@upload_bp.route('/scenario', methods=['POST'])
def create_scenario():
    """创建场景"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '请求数据为空'
            }), 400
        
        scenario_name = data.get('scenario_name')
        map_file_id = data.get('map_file_id')
        data_file_id = data.get('data_file_id')
        config = data.get('config', {})
        
        if not scenario_name:
            return jsonify({
                'success': False,
                'message': '场景名称不能为空'
            }), 400
        
        if not map_file_id:
            return jsonify({
                'success': False,
                'message': '必须选择地图文件'
            }), 400
        
        result = upload_manager.create_scenario_from_uploads(
            scenario_name, map_file_id, data_file_id, config
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"创建场景失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'创建场景失败: {str(e)}'
        }), 500

@upload_bp.route('/scenarios', methods=['GET'])
def get_scenarios():
    """获取所有场景"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        scenarios = upload_manager.get_scenarios()
        
        return jsonify({
            'success': True,
            'scenarios': scenarios,
            'total': len(scenarios)
        })
        
    except Exception as e:
        logger.error(f"获取场景列表失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取场景列表失败: {str(e)}'
        }), 500

@upload_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """获取上传统计信息"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        stats = upload_manager.get_upload_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取统计信息失败: {str(e)}'
        }), 500

@upload_bp.route('/validate', methods=['POST'])
def validate_file():
    """验证文件（不保存）"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': '未选择文件'
            }), 400
        
        file = request.files['file']
        file_type = request.form.get('file_type', 'data')
        
        # 验证文件类型参数
        if file_type not in ['maps', 'data', 'scenarios']:
            return jsonify({
                'success': False,
                'message': '无效的文件类型，支持: maps, data, scenarios'
            }), 400
        
        # 只进行验证，不保存
        is_valid, message = upload_manager._validate_file(file, file_type)
        
        return jsonify({
            'success': is_valid,
            'message': message,
            'file_info': {
                'filename': file.filename,
                'size': len(file.read()),
                'type': file_type
            }
        })
        
    except Exception as e:
        logger.error(f"文件验证失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'文件验证失败: {str(e)}'
        }), 500

# 批量操作接口
@upload_bp.route('/batch/delete', methods=['POST'])
def batch_delete_files():
    """批量删除文件"""
    try:
        if not upload_manager:
            return jsonify({
                'success': False,
                'message': '上传管理器未初始化'
            }), 500
        
        data = request.get_json()
        file_ids = data.get('file_ids', [])
        
        if not file_ids:
            return jsonify({
                'success': False,
                'message': '未指定要删除的文件'
            }), 400
        
        results = []
        success_count = 0
        
        for file_id in file_ids:
            result = upload_manager.delete_file(file_id)
            results.append({
                'file_id': file_id,
                'success': result['success'],
                'message': result['message']
            })
            if result['success']:
                success_count += 1
        
        return jsonify({
            'success': True,
            'message': f'批量删除完成，成功删除 {success_count}/{len(file_ids)} 个文件',
            'results': results,
            'success_count': success_count,
            'total_count': len(file_ids)
        })
        
    except Exception as e:
        logger.error(f"批量删除文件失败: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'批量删除失败: {str(e)}'
        }), 500