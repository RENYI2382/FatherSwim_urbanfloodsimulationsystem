/**
 * 文件上传管理系统前端脚本
 * 支持拖拽上传、文件管理和场景创建
 */

class FileUploadManager {
    constructor() {
        this.selectedFiles = [];
        this.uploadConfig = null;
        this.currentFileId = null;
        
        this.init();
    }
    
    async init() {
        this.setupEventListeners();
        await this.loadUploadConfig();
        await this.loadStatistics();
        await this.loadFiles();
        await this.loadScenarios();
        this.updateFileSelects();
    }
    
    setupEventListeners() {
        // 拖拽上传
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadZone.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadZone.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // 按钮事件
        document.getElementById('uploadBtn').addEventListener('click', this.uploadFiles.bind(this));
        document.getElementById('clearBtn').addEventListener('click', this.clearFiles.bind(this));
        
        // 文件类型筛选
        document.getElementById('fileTypeFilter').addEventListener('change', this.filterFiles.bind(this));
        
        // 场景创建表单
        document.getElementById('createScenarioForm').addEventListener('submit', this.createScenario.bind(this));
        
        // 标签切换事件
        document.querySelectorAll('[data-bs-toggle="pill"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', this.handleTabChange.bind(this));
        });
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadZone').classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadZone').classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        document.getElementById('uploadZone').classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        this.addFiles(files);
    }
    
    handleFileSelect(e) {
        const files = Array.from(e.target.files);
        this.addFiles(files);
    }
    
    addFiles(files) {
        this.selectedFiles = [...this.selectedFiles, ...files];
        this.updateUploadButton();
        this.showMessage(`已选择 ${files.length} 个文件`, 'info');
    }
    
    clearFiles() {
        this.selectedFiles = [];
        document.getElementById('fileInput').value = '';
        this.updateUploadButton();
        this.hideProgress();
    }
    
    updateUploadButton() {
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.disabled = this.selectedFiles.length === 0;
        
        if (this.selectedFiles.length > 0) {
            uploadBtn.innerHTML = `<i class="bi bi-upload"></i> 上传 ${this.selectedFiles.length} 个文件`;
        } else {
            uploadBtn.innerHTML = `<i class="bi bi-upload"></i> 开始上传`;
        }
    }
    
    async uploadFiles() {
        if (this.selectedFiles.length === 0) return;
        
        const uploadBtn = document.getElementById('uploadBtn');
        uploadBtn.classList.add('loading');
        uploadBtn.disabled = true;
        
        this.showProgress();
        
        const fileType = document.getElementById('fileType').value;
        const description = document.getElementById('fileDescription').value;
        
        let successCount = 0;
        let totalFiles = this.selectedFiles.length;
        
        for (let i = 0; i < this.selectedFiles.length; i++) {
            const file = this.selectedFiles[i];
            
            try {
                this.updateProgress((i / totalFiles) * 100, `正在上传: ${file.name}`);
                
                const result = await this.uploadSingleFile(file, fileType, description);
                
                if (result.success) {
                    successCount++;
                } else {
                    this.showMessage(`上传失败: ${file.name} - ${result.message}`, 'error');
                }
                
            } catch (error) {
                this.showMessage(`上传错误: ${file.name} - ${error.message}`, 'error');
            }
        }
        
        this.updateProgress(100, `上传完成: ${successCount}/${totalFiles} 个文件成功`);
        
        setTimeout(() => {
            this.hideProgress();
            uploadBtn.classList.remove('loading');
            this.clearFiles();
            
            if (successCount > 0) {
                this.showMessage(`成功上传 ${successCount} 个文件`, 'success');
                this.loadStatistics();
                this.loadFiles();
                this.updateFileSelects();
            }
        }, 2000);
    }
    
    async uploadSingleFile(file, fileType, description) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('file_type', fileType);
        formData.append('description', description);
        
        const response = await fetch('/api/upload/file', {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    showProgress() {
        document.getElementById('uploadProgress').style.display = 'block';
    }
    
    hideProgress() {
        document.getElementById('uploadProgress').style.display = 'none';
    }
    
    updateProgress(percent, text) {
        const progressBar = document.querySelector('.progress-bar');
        const progressText = document.getElementById('progressText');
        
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        progressText.textContent = text;
    }
    
    async loadUploadConfig() {
        try {
            const response = await fetch('/api/upload/config');
            const data = await response.json();
            
            if (data.success) {
                this.uploadConfig = data.config;
                this.displayUploadConfig();
            }
        } catch (error) {
            console.error('加载上传配置失败:', error);
        }
    }
    
    displayUploadConfig() {
        if (!this.uploadConfig) return;
        
        // 显示支持的格式
        const formatsContainer = document.getElementById('supportedFormats');
        let formatsHtml = '';
        
        Object.entries(this.uploadConfig.supported_formats).forEach(([type, formats]) => {
            formatsHtml += `<div class="mb-2">`;
            formatsHtml += `<strong>${this.getTypeDisplayName(type)}:</strong><br>`;
            
            Object.entries(formats).forEach(([category, exts]) => {
                formatsHtml += `<small class="text-muted">${category}: ${exts.join(', ')}</small><br>`;
            });
            
            formatsHtml += `</div>`;
        });
        
        formatsContainer.innerHTML = formatsHtml;
        
        // 显示大小限制
        const limitsContainer = document.getElementById('sizeLimits');
        let limitsHtml = '';
        
        Object.entries(this.uploadConfig.size_limits).forEach(([type, limit]) => {
            limitsHtml += `<div class="mb-1">`;
            limitsHtml += `<strong>${this.getTypeDisplayName(type)}:</strong> ${limit}MB`;
            limitsHtml += `</div>`;
        });
        
        limitsContainer.innerHTML = limitsHtml;
    }
    
    getTypeDisplayName(type) {
        const names = {
            'maps': '地图文件',
            'data': '数据文件',
            'scenarios': '场景配置'
        };
        return names[type] || type;
    }
    
    async loadStatistics() {
        try {
            const response = await fetch('/api/upload/statistics');
            const data = await response.json();
            
            if (data.success) {
                this.displayStatistics(data.statistics);
            }
        } catch (error) {
            console.error('加载统计信息失败:', error);
        }
    }
    
    displayStatistics(stats) {
        document.getElementById('totalFiles').textContent = stats.total_files;
        document.getElementById('totalSize').textContent = `${stats.total_size_mb.toFixed(1)} MB`;
        document.getElementById('mapFiles').textContent = stats.by_type.maps.count;
        document.getElementById('dataFiles').textContent = stats.by_type.data.count;
    }
    
    async loadFiles(fileType = null) {
        try {
            let url = '/api/upload/files';
            if (fileType) {
                url += `?file_type=${fileType}`;
            }
            
            const response = await fetch(url);
            const data = await response.json();
            
            if (data.success) {
                this.displayFiles(data.files);
            }
        } catch (error) {
            console.error('加载文件列表失败:', error);
        }
    }
    
    displayFiles(files) {
        const container = document.getElementById('filesList');
        
        if (files.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-inbox" style="font-size: 3rem; color: #6c757d;"></i>
                    <p class="text-muted mt-2">暂无文件</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        
        files.forEach(file => {
            const uploadDate = new Date(file.upload_time).toLocaleString('zh-CN');
            const fileSize = (file.file_size / 1024 / 1024).toFixed(2);
            
            html += `
                <div class="file-item position-relative">
                    <div class="d-flex align-items-center">
                        <div class="file-icon">
                            ${this.getFileIcon(file.file_type, file.saved_filename)}
                        </div>
                        <div class="flex-grow-1">
                            <h6 class="mb-1">${file.original_filename}</h6>
                            <div class="d-flex flex-wrap align-items-center">
                                <span class="badge bg-primary me-2">${this.getTypeDisplayName(file.file_type)}</span>
                                <small class="text-muted me-3">${fileSize} MB</small>
                                <small class="text-muted me-3">${uploadDate}</small>
                            </div>
                            ${file.description ? `<p class="text-muted small mb-1">${file.description}</p>` : ''}
                            ${this.renderMetadata(file.metadata)}
                        </div>
                        <div class="btn-group">
                            <button class="btn btn-sm btn-outline-primary" onclick="fileManager.showFileDetail(${file.id})">
                                <i class="bi bi-eye"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-success" onclick="fileManager.downloadFile(${file.id})">
                                <i class="bi bi-download"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger" onclick="fileManager.deleteFile(${file.id})">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    getFileIcon(fileType, filename) {
        const ext = filename.split('.').pop().toLowerCase();
        
        if (fileType === 'maps') {
            if (['png', 'jpg', 'jpeg', 'tiff'].includes(ext)) {
                return '<i class="bi bi-image text-success"></i>';
            } else {
                return '<i class="bi bi-map text-primary"></i>';
            }
        } else if (fileType === 'data') {
            if (ext === 'csv') {
                return '<i class="bi bi-table text-info"></i>';
            } else {
                return '<i class="bi bi-file-earmark-spreadsheet text-warning"></i>';
            }
        } else {
            return '<i class="bi bi-gear text-secondary"></i>';
        }
    }
    
    renderMetadata(metadata) {
        if (!metadata || Object.keys(metadata).length === 0) {
            return '';
        }
        
        let html = '<div class="mt-2">';
        
        Object.entries(metadata).forEach(([key, value]) => {
            if (typeof value === 'object') {
                html += `<span class="badge bg-secondary metadata-badge">${key}: ${JSON.stringify(value)}</span>`;
            } else {
                html += `<span class="badge bg-secondary metadata-badge">${key}: ${value}</span>`;
            }
        });
        
        html += '</div>';
        return html;
    }
    
    async showFileDetail(fileId) {
        try {
            const response = await fetch(`/api/upload/file/${fileId}`);
            const data = await response.json();
            
            if (data.success) {
                this.displayFileDetail(data.file);
                this.currentFileId = fileId;
                
                const modal = new bootstrap.Modal(document.getElementById('fileDetailModal'));
                modal.show();
            }
        } catch (error) {
            console.error('获取文件详情失败:', error);
        }
    }
    
    displayFileDetail(file) {
        const container = document.getElementById('fileDetailContent');
        const uploadDate = new Date(file.upload_time).toLocaleString('zh-CN');
        const fileSize = (file.file_size / 1024 / 1024).toFixed(2);
        
        let html = `
            <div class="row">
                <div class="col-md-6">
                    <h6>基本信息</h6>
                    <table class="table table-sm">
                        <tr><td>原始文件名</td><td>${file.original_filename}</td></tr>
                        <tr><td>文件类型</td><td>${this.getTypeDisplayName(file.file_type)}</td></tr>
                        <tr><td>文件大小</td><td>${fileSize} MB</td></tr>
                        <tr><td>上传时间</td><td>${uploadDate}</td></tr>
                        <tr><td>文件哈希</td><td><small>${file.file_hash}</small></td></tr>
                    </table>
                </div>
                <div class="col-md-6">
                    <h6>元数据</h6>
        `;
        
        if (file.metadata && Object.keys(file.metadata).length > 0) {
            html += '<table class="table table-sm">';
            Object.entries(file.metadata).forEach(([key, value]) => {
                html += `<tr><td>${key}</td><td>${typeof value === 'object' ? JSON.stringify(value) : value}</td></tr>`;
            });
            html += '</table>';
        } else {
            html += '<p class="text-muted">无元数据</p>';
        }
        
        html += `
                </div>
            </div>
        `;
        
        if (file.description) {
            html += `
                <div class="mt-3">
                    <h6>描述</h6>
                    <p>${file.description}</p>
                </div>
            `;
        }
        
        container.innerHTML = html;
        
        // 设置下载按钮
        document.getElementById('downloadFileBtn').onclick = () => this.downloadFile(file.id);
    }
    
    async downloadFile(fileId) {
        try {
            const response = await fetch(`/api/upload/file/${fileId}/download`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = response.headers.get('Content-Disposition')?.split('filename=')[1] || 'download';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            } else {
                this.showMessage('下载失败', 'error');
            }
        } catch (error) {
            console.error('下载文件失败:', error);
            this.showMessage('下载失败', 'error');
        }
    }
    
    async deleteFile(fileId) {
        if (!confirm('确定要删除这个文件吗？')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/upload/file/${fileId}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showMessage('文件删除成功', 'success');
                this.loadFiles();
                this.loadStatistics();
                this.updateFileSelects();
            } else {
                this.showMessage(`删除失败: ${data.message}`, 'error');
            }
        } catch (error) {
            console.error('删除文件失败:', error);
            this.showMessage('删除失败', 'error');
        }
    }
    
    filterFiles() {
        const fileType = document.getElementById('fileTypeFilter').value;
        this.loadFiles(fileType);
    }
    
    async loadScenarios() {
        try {
            const response = await fetch('/api/upload/scenarios');
            const data = await response.json();
            
            if (data.success) {
                this.displayScenarios(data.scenarios);
            }
        } catch (error) {
            console.error('加载场景列表失败:', error);
        }
    }
    
    displayScenarios(scenarios) {
        const container = document.getElementById('scenariosList');
        
        if (scenarios.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i class="bi bi-collection" style="font-size: 3rem; color: #6c757d;"></i>
                    <p class="text-muted mt-2">暂无场景</p>
                </div>
            `;
            return;
        }
        
        let html = '';
        
        scenarios.forEach(scenario => {
            const createDate = new Date(scenario.created_time).toLocaleString('zh-CN');
            
            html += `
                <div class="scenario-card">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <h5 class="mb-2">${scenario.name}</h5>
                            <div class="mb-2">
                                <span class="badge bg-primary me-2">地图: ${scenario.map_file.filename}</span>
                                ${scenario.data_file ? `<span class="badge bg-info">数据: ${scenario.data_file.filename}</span>` : ''}
                            </div>
                            <small class="text-muted">创建时间: ${createDate}</small>
                        </div>
                        <div class="btn-group">
                            <button class="btn btn-sm btn-outline-primary" onclick="fileManager.useScenario('${scenario.scenario_file}')">
                                <i class="bi bi-play"></i> 使用
                            </button>
                            <button class="btn btn-sm btn-outline-info" onclick="fileManager.showScenarioDetail('${scenario.scenario_file}')">
                                <i class="bi bi-eye"></i> 详情
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    async updateFileSelects() {
        try {
            // 加载地图文件
            const mapsResponse = await fetch('/api/upload/files?file_type=maps');
            const mapsData = await mapsResponse.json();
            
            // 加载数据文件
            const dataResponse = await fetch('/api/upload/files?file_type=data');
            const dataData = await dataResponse.json();
            
            // 更新地图文件选择器
            const mapSelect = document.getElementById('mapFileSelect');
            mapSelect.innerHTML = '<option value="">请选择地图文件</option>';
            
            if (mapsData.success) {
                mapsData.files.forEach(file => {
                    mapSelect.innerHTML += `<option value="${file.id}">${file.original_filename}</option>`;
                });
            }
            
            // 更新数据文件选择器
            const dataSelect = document.getElementById('dataFileSelect');
            dataSelect.innerHTML = '<option value="">无</option>';
            
            if (dataData.success) {
                dataData.files.forEach(file => {
                    dataSelect.innerHTML += `<option value="${file.id}">${file.original_filename}</option>`;
                });
            }
            
        } catch (error) {
            console.error('更新文件选择器失败:', error);
        }
    }
    
    async createScenario(e) {
        e.preventDefault();
        
        const scenarioName = document.getElementById('scenarioName').value;
        const mapFileId = document.getElementById('mapFileSelect').value;
        const dataFileId = document.getElementById('dataFileSelect').value || null;
        
        if (!scenarioName || !mapFileId) {
            this.showMessage('请填写场景名称并选择地图文件', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/upload/scenario', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    scenario_name: scenarioName,
                    map_file_id: parseInt(mapFileId),
                    data_file_id: dataFileId ? parseInt(dataFileId) : null,
                    config: {}
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showMessage('场景创建成功', 'success');
                document.getElementById('createScenarioForm').reset();
                this.loadScenarios();
            } else {
                this.showMessage(`创建失败: ${data.message}`, 'error');
            }
        } catch (error) {
            console.error('创建场景失败:', error);
            this.showMessage('创建场景失败', 'error');
        }
    }
    
    useScenario(scenarioFile) {
        // 这里可以实现场景使用逻辑，比如跳转到仿真页面
        this.showMessage(`正在加载场景: ${scenarioFile}`, 'info');
        // 可以跳转到仿真页面并传递场景参数
        // window.location.href = `/simulation?scenario=${scenarioFile}`;
    }
    
    showScenarioDetail(scenarioFile) {
        // 显示场景详情
        this.showMessage(`场景详情: ${scenarioFile}`, 'info');
    }
    
    handleTabChange(e) {
        const targetId = e.target.getAttribute('data-bs-target');
        
        if (targetId === '#files') {
            this.loadFiles();
        } else if (targetId === '#scenarios') {
            this.loadScenarios();
            this.updateFileSelects();
        }
    }
    
    showMessage(message, type = 'info') {
        const container = document.getElementById('messageContainer');
        const alertClass = {
            'success': 'alert-success',
            'error': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';
        
        const alertId = 'alert-' + Date.now();
        
        const alertHtml = `
            <div class="alert ${alertClass} alert-dismissible fade show" id="${alertId}" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        container.insertAdjacentHTML('beforeend', alertHtml);
        
        // 自动消失
        setTimeout(() => {
            const alert = document.getElementById(alertId);
            if (alert) {
                const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                bsAlert.close();
            }
        }, 5000);
    }
}

// 全局函数
function refreshData() {
    if (window.fileManager) {
        window.fileManager.loadStatistics();
        window.fileManager.loadFiles();
        window.fileManager.loadScenarios();
        window.fileManager.updateFileSelects();
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    window.fileManager = new FileUploadManager();
});