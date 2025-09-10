// 地图管理系统JavaScript
class MapManager {
    constructor() {
        this.currentScenario = null;
        this.scenarios = [];
        this.selectedFormat = null;
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadScenarios();
        this.loadStatistics();
    }
    
    bindEvents() {
        // 场景管理事件
        document.getElementById('refreshScenariosBtn').addEventListener('click', () => {
            this.loadScenarios();
        });
        
        // 格式转换事件
        document.getElementById('convertFileInput').addEventListener('change', (e) => {
            const convertBtn = document.getElementById('convertMapBtn');
            convertBtn.disabled = !e.target.files.length || !this.selectedFormat;
        });
        
        document.getElementById('convertMapBtn').addEventListener('click', () => {
            this.convertMapFormat();
        });
        
        // 地图预览事件
        document.getElementById('previewFileInput').addEventListener('change', (e) => {
            const previewBtn = document.getElementById('createPreviewBtn');
            previewBtn.disabled = !e.target.files.length;
        });
        
        document.getElementById('createPreviewBtn').addEventListener('click', () => {
            this.createMapPreview();
        });
        
        // 格式选择事件
        document.querySelectorAll('.format-option').forEach(option => {
            option.addEventListener('click', (e) => {
                this.selectFormat(e.currentTarget);
            });
        });
        
        // 创建场景事件
        document.getElementById('saveScenarioBtn').addEventListener('click', () => {
            this.createScenario();
        });
    }
    
    // 加载场景列表
    async loadScenarios() {
        try {
            const response = await fetch('/api/map/scenarios');
            const data = await response.json();
            
            if (data.success) {
                this.scenarios = data.scenarios;
                this.currentScenario = data.current_scenario;
                this.renderScenarios();
                this.updateCurrentScenario();
            } else {
                this.showError('加载场景失败: ' + data.message);
            }
        } catch (error) {
            console.error('加载场景错误:', error);
            this.showError('网络错误，无法加载场景列表');
        }
    }
    
    // 渲染场景列表
    renderScenarios() {
        const container = document.getElementById('scenariosList');
        
        if (this.scenarios.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-folder-open fa-2x mb-2"></i>
                    <p>暂无可用场景</p>
                    <small>点击"创建场景"按钮开始</small>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.scenarios.map(scenario => `
            <div class="scenario-card ${scenario.name === this.currentScenario ? 'active' : ''}" 
                 data-scenario="${scenario.name}">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <h6 class="mb-1">
                            <i class="fas fa-map-marked-alt"></i>
                            ${scenario.name}
                        </h6>
                        <p class="mb-2 text-muted small">${scenario.description || '无描述'}</p>
                        <div class="d-flex gap-2 align-items-center">
                            <span class="status-badge ${
                                scenario.name === this.currentScenario ? 'status-active' : 'status-inactive'
                            }">
                                ${scenario.name === this.currentScenario ? '当前活动' : '未激活'}
                            </span>
                            <small class="text-muted">
                                <i class="fas fa-layer-group"></i>
                                ${scenario.map_count || 0} 个地图
                            </small>
                        </div>
                    </div>
                    <div class="dropdown">
                        <button class="btn btn-sm btn-outline-secondary dropdown-toggle" 
                                data-bs-toggle="dropdown">
                            <i class="fas fa-ellipsis-v"></i>
                        </button>
                        <ul class="dropdown-menu">
                            ${scenario.name !== this.currentScenario ? `
                                <li><a class="dropdown-item" href="#" 
                                       onclick="mapManager.switchScenario('${scenario.name}')">
                                    <i class="fas fa-play text-success"></i> 激活场景
                                </a></li>
                            ` : ''}
                            <li><a class="dropdown-item" href="#" 
                                   onclick="mapManager.previewScenario('${scenario.name}')">
                                <i class="fas fa-eye text-info"></i> 预览场景
                            </a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item text-danger" href="#" 
                                   onclick="mapManager.deleteScenario('${scenario.name}')">
                                <i class="fas fa-trash"></i> 删除场景
                            </a></li>
                        </ul>
                    </div>
                </div>
            </div>
        `).join('');
    }
    
    // 更新当前场景显示
    updateCurrentScenario() {
        const display = document.getElementById('currentScenarioDisplay');
        
        if (this.currentScenario) {
            const scenario = this.scenarios.find(s => s.name === this.currentScenario);
            display.innerHTML = `
                <div class="d-flex align-items-center">
                    <i class="fas fa-check-circle text-success me-2"></i>
                    <div>
                        <strong>${this.currentScenario}</strong>
                        <br><small class="text-muted">${scenario?.description || '无描述'}</small>
                    </div>
                </div>
            `;
            display.className = 'alert alert-success';
        } else {
            display.innerHTML = `
                <div class="d-flex align-items-center">
                    <i class="fas fa-exclamation-triangle text-warning me-2"></i>
                    <span>未选择场景</span>
                </div>
            `;
            display.className = 'alert alert-warning';
        }
    }
    
    // 切换场景
    async switchScenario(scenarioName) {
        if (!confirm(`确定要切换到场景 "${scenarioName}" 吗？`)) {
            return;
        }
        
        this.showLoading('正在切换场景...');
        
        try {
            const response = await fetch('/api/map/switch_scenario', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ scenario_name: scenarioName })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.currentScenario = scenarioName;
                this.renderScenarios();
                this.updateCurrentScenario();
                this.showSuccess('场景切换成功！');
            } else {
                this.showError('场景切换失败: ' + data.message);
            }
        } catch (error) {
            console.error('切换场景错误:', error);
            this.showError('网络错误，场景切换失败');
        } finally {
            this.hideLoading();
        }
    }
    
    // 删除场景
    async deleteScenario(scenarioName) {
        if (!confirm(`确定要删除场景 "${scenarioName}" 吗？此操作不可撤销！`)) {
            return;
        }
        
        this.showLoading('正在删除场景...');
        
        try {
            const response = await fetch(`/api/map/scenarios/${scenarioName}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.loadScenarios();
                this.showSuccess('场景删除成功！');
            } else {
                this.showError('场景删除失败: ' + data.message);
            }
        } catch (error) {
            console.error('删除场景错误:', error);
            this.showError('网络错误，场景删除失败');
        } finally {
            this.hideLoading();
        }
    }
    
    // 创建场景
    async createScenario() {
        const name = document.getElementById('scenarioName').value.trim();
        const description = document.getElementById('scenarioDescription').value.trim();
        
        if (!name) {
            this.showError('请输入场景名称');
            return;
        }
        
        this.showLoading('正在创建场景...');
        
        try {
            const response = await fetch('/api/map/scenarios', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    name: name,
                    description: description,
                    maps: {} // 暂时为空，后续可以添加地图选择功能
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 关闭模态框
                const modal = bootstrap.Modal.getInstance(document.getElementById('createScenarioModal'));
                modal.hide();
                
                // 清空表单
                document.getElementById('createScenarioForm').reset();
                
                // 重新加载场景列表
                this.loadScenarios();
                this.showSuccess('场景创建成功！');
            } else {
                this.showError('场景创建失败: ' + data.message);
            }
        } catch (error) {
            console.error('创建场景错误:', error);
            this.showError('网络错误，场景创建失败');
        } finally {
            this.hideLoading();
        }
    }
    
    // 选择格式
    selectFormat(element) {
        // 移除其他选中状态
        document.querySelectorAll('.format-option').forEach(opt => {
            opt.classList.remove('selected');
        });
        
        // 添加选中状态
        element.classList.add('selected');
        this.selectedFormat = element.dataset.format;
        
        // 更新转换按钮状态
        const fileInput = document.getElementById('convertFileInput');
        const convertBtn = document.getElementById('convertMapBtn');
        convertBtn.disabled = !fileInput.files.length || !this.selectedFormat;
    }
    
    // 转换地图格式
    async convertMapFormat() {
        const fileInput = document.getElementById('convertFileInput');
        const file = fileInput.files[0];
        
        if (!file || !this.selectedFormat) {
            this.showError('请选择文件和目标格式');
            return;
        }
        
        this.showLoading('正在转换格式...');
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('target_format', this.selectedFormat);
        
        try {
            const response = await fetch('/api/map/convert', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                // 下载转换后的文件
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `converted_${file.name.split('.')[0]}.${this.getFileExtension(this.selectedFormat)}`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showSuccess('格式转换成功！文件已下载');
            } else {
                const data = await response.json();
                this.showError('格式转换失败: ' + data.message);
            }
        } catch (error) {
            console.error('转换格式错误:', error);
            this.showError('网络错误，格式转换失败');
        } finally {
            this.hideLoading();
        }
    }
    
    // 创建地图预览
    async createMapPreview() {
        const fileInput = document.getElementById('previewFileInput');
        const file = fileInput.files[0];
        
        if (!file) {
            this.showError('请选择地图文件');
            return;
        }
        
        this.showLoading('正在生成预览...');
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/map/preview', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.displayPreview(data.preview_url, data.stats);
                this.showSuccess('预览生成成功！');
            } else {
                this.showError('预览生成失败: ' + data.message);
            }
        } catch (error) {
            console.error('生成预览错误:', error);
            this.showError('网络错误，预览生成失败');
        } finally {
            this.hideLoading();
        }
    }
    
    // 显示预览
    displayPreview(previewUrl, stats) {
        const container = document.getElementById('previewContainer');
        
        container.innerHTML = `
            <div class="row">
                <div class="col-md-8">
                    <img src="${previewUrl}" class="preview-image" alt="地图预览">
                </div>
                <div class="col-md-4">
                    <h6>统计信息</h6>
                    <ul class="list-unstyled">
                        <li><strong>尺寸:</strong> ${stats.shape?.join(' × ') || '未知'}</li>
                        <li><strong>数据类型:</strong> ${stats.dtype || '未知'}</li>
                        <li><strong>最小值:</strong> ${stats.min?.toFixed(2) || '未知'}</li>
                        <li><strong>最大值:</strong> ${stats.max?.toFixed(2) || '未知'}</li>
                        <li><strong>平均值:</strong> ${stats.mean?.toFixed(2) || '未知'}</li>
                    </ul>
                </div>
            </div>
        `;
        
        container.style.display = 'block';
    }
    
    // 加载统计信息
    async loadStatistics() {
        try {
            const response = await fetch('/api/map/statistics');
            const data = await response.json();
            
            if (data.success) {
                this.updateStatistics(data.statistics);
            }
        } catch (error) {
            console.error('加载统计信息错误:', error);
        }
    }
    
    // 更新统计信息显示
    updateStatistics(stats) {
        const container = document.getElementById('statisticsContainer');
        const items = container.querySelectorAll('.stat-item');
        
        if (items.length >= 4) {
            items[0].querySelector('.stat-number').textContent = stats.total_scenarios || 0;
            items[1].querySelector('.stat-number').textContent = stats.total_maps || 0;
            items[2].querySelector('.stat-number').textContent = (stats.storage_size_mb || 0).toFixed(1);
            items[3].querySelector('.stat-number').textContent = stats.supported_formats || 0;
        }
    }
    
    // 获取文件扩展名
    getFileExtension(format) {
        const extensions = {
            'numpy': 'npy',
            'csv': 'csv',
            'image': 'png',
            'geojson': 'geojson'
        };
        return extensions[format] || 'txt';
    }
    
    // 显示加载提示
    showLoading(message = '处理中...') {
        document.getElementById('loadingMessage').textContent = message;
        const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
        modal.show();
    }
    
    // 隐藏加载提示
    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
        if (modal) {
            modal.hide();
        }
    }
    
    // 显示成功消息
    showSuccess(message) {
        this.showToast(message, 'success');
    }
    
    // 显示错误消息
    showError(message) {
        this.showToast(message, 'error');
    }
    
    // 显示提示消息
    showToast(message, type = 'info') {
        // 创建toast元素
        const toastId = 'toast_' + Date.now();
        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-white bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'primary'} border-0" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        // 添加到页面
        let toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
            toastContainer.style.zIndex = '9999';
            document.body.appendChild(toastContainer);
        }
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        
        // 显示toast
        const toastElement = document.getElementById(toastId);
        const toast = new bootstrap.Toast(toastElement, {
            autohide: true,
            delay: type === 'error' ? 5000 : 3000
        });
        toast.show();
        
        // 自动移除
        toastElement.addEventListener('hidden.bs.toast', () => {
            toastElement.remove();
        });
    }
}

// 初始化地图管理器
let mapManager;
document.addEventListener('DOMContentLoaded', () => {
    mapManager = new MapManager();
});