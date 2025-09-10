// 全局变量
let simulationStatus = "未开始";
let currentStep = 0;
let totalSteps = 100;
let startTime = null;
let elapsedTime = 0;
let simulationInterval = null;
let mainMap = null;
let animationMap = null;
let heatLayer = null;
let agentMarkers = [];
let timelineInterval = null;
let animationInterval = null;
let charts = {};
let simulationData = {
    steps: [],
    agentStats: [],
    floodData: [],
    networkData: []
};

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 隐藏加载动画
    document.getElementById('loadingOverlay').style.display = 'none';
    
    // 初始化导航菜单
    initNavigation();
    
    // 初始化地图
    initMaps();
    
    // 初始化图表
    initCharts();
    
    // 初始化社交网络图
    initSocialNetworkGraph();
    
    // 初始化动画帧
    initAnimationFrames();
    
    // 初始化事件监听
    initEventListeners();
    
    // 获取初始状态
    fetchSimulationStatus();
});

// 初始化导航菜单
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link[data-section]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // 移除所有活动状态
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // 添加当前活动状态
            this.parentElement.classList.add('active');
            
            // 隐藏所有部分
            document.querySelectorAll('.dashboard-section').forEach(section => {
                section.style.display = 'none';
            });
            
            // 显示选中部分
            const sectionId = this.getAttribute('data-section');
            document.getElementById(sectionId).style.display = 'block';
            
            // 如果是地图视图，需要刷新地图大小
            if (sectionId === 'map-view' && mainMap) {
                mainMap.invalidateSize();
            }
            
            // 如果是动态模拟，需要刷新动画地图大小
            if (sectionId === 'animation' && animationMap) {
                animationMap.invalidateSize();
            }
        });
    });
}

// 初始化地图
function initMaps() {
    // 初始化主地图
    mainMap = L.map('mainMap').setView([23.13, 113.26], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(mainMap);
    
    // 初始化动画地图
    animationMap = L.map('animationMap').setView([23.13, 113.26], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(animationMap);
    
    // 添加初始热力图层
    updateHeatmapLayer('water');
}

// 更新热力图层
function updateHeatmapLayer(layerType) {
    // 移除现有热力图层
    if (heatLayer) {
        mainMap.removeLayer(heatLayer);
    }
    
    // 清除现有标记
    agentMarkers.forEach(marker => mainMap.removeLayer(marker));
    agentMarkers = [];
    
    // 根据选择的图层类型添加相应的可视化
    if (layerType === 'water') {
        // 模拟水位数据点
        const heatPoints = [];
        for (let i = 0; i < 100; i++) {
            const lat = 23.08 + Math.random() * 0.1;
            const lng = 113.2 + Math.random() * 0.12;
            const intensity = Math.random() * 1.0;
            heatPoints.push([lat, lng, intensity]);
        }
        
        heatLayer = L.heatLayer(heatPoints, {
            radius: 25,
            blur: 15,
            maxZoom: 17,
            gradient: {0.4: '#1a73e8', 0.65: '#fbbc05', 0.9: '#ea4335'}
        }).addTo(mainMap);
        
        // 更新图例
        updateMapLegend('water');
    } else if (layerType === 'agents') {
        // 模拟智能体分布
        const agentStatuses = ['normal', 'evacuating', 'trapped', 'rescued'];
        const agentColors = {
            'normal': '#34a853',
            'evacuating': '#fbbc05',
            'trapped': '#ea4335',
            'rescued': '#1a73e8'
        };
        
        for (let i = 0; i < 100; i++) {
            const lat = 23.08 + Math.random() * 0.1;
            const lng = 113.2 + Math.random() * 0.12;
            const status = agentStatuses[Math.floor(Math.random() * agentStatuses.length)];
            
            const marker = L.circleMarker([lat, lng], {
                radius: 5,
                fillColor: agentColors[status],
                color: '#fff',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(mainMap);
            
            marker.bindTooltip(`智能体 #${i}<br>状态: ${status}`);
            agentMarkers.push(marker);
        }
        
        // 更新图例
        updateMapLegend('agents');
    } else if (layerType === 'risk') {
        // 模拟风险评估热图
        const heatPoints = [];
        for (let i = 0; i < 100; i++) {
            const lat = 23.08 + Math.random() * 0.1;
            const lng = 113.2 + Math.random() * 0.12;
            const intensity = Math.random() * 1.0;
            heatPoints.push([lat, lng, intensity]);
        }
        
        heatLayer = L.heatLayer(heatPoints, {
            radius: 25,
            blur: 15,
            maxZoom: 17,
            gradient: {0.4: '#34a853', 0.65: '#fbbc05', 0.9: '#ea4335'}
        }).addTo(mainMap);
        
        // 更新图例
        updateMapLegend('risk');
    } else if (layerType === 'evacuation') {
        // 模拟疏散路线
        const evacuationRoutes = [
            [[23.13, 113.26], [23.135, 113.27], [23.14, 113.28], [23.145, 113.29]],
            [[23.12, 113.25], [23.125, 113.26], [23.13, 113.27], [23.135, 113.28]],
            [[23.11, 113.24], [23.115, 113.25], [23.12, 113.26], [23.125, 113.27]]
        ];
        
        evacuationRoutes.forEach((route, index) => {
            const polyline = L.polyline(route, {
                color: '#1a73e8',
                weight: 5,
                opacity: 0.7
            }).addTo(mainMap);
            
            polyline.bindTooltip(`疏散路线 #${index + 1}`);
            agentMarkers.push(polyline);
            
            // 添加起点和终点标记
            const startMarker = L.circleMarker(route[0], {
                radius: 8,
                fillColor: '#34a853',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(mainMap);
            
            const endMarker = L.circleMarker(route[route.length - 1], {
                radius: 8,
                fillColor: '#1a73e8',
                color: '#fff',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(mainMap);
            
            startMarker.bindTooltip('起点');
            endMarker.bindTooltip('安全区域');
            
            agentMarkers.push(startMarker);
            agentMarkers.push(endMarker);
        });
        
        // 更新图例
        updateMapLegend('evacuation');
    }
}

// 更新地图图例
function updateMapLegend(layerType) {
    const legendEl = document.getElementById('mapLegend');
    let legendHTML = '';
    
    if (layerType === 'water') {
        legendHTML = `
            <div class="legend-item">
                <div class="legend-color" style="background-color: #1a73e8"></div>
                <span>低水位 (0-1m)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #fbbc05"></div>
                <span>中水位 (1-2m)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ea4335"></div>
                <span>高水位 (>2m)</span>
            </div>
        `;
    } else if (layerType === 'agents') {
        legendHTML = `
            <div class="legend-item">
                <div class="legend-color" style="background-color: #34a853"></div>
                <span>正常</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #fbbc05"></div>
                <span>疏散中</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ea4335"></div>
                <span>受困</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #1a73e8"></div>
                <span>已救援</span>
            </div>
        `;
    } else if (layerType === 'risk') {
        legendHTML = `
            <div class="legend-item">
                <div class="legend-color" style="background-color: #34a853"></div>
                <span>低风险</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #fbbc05"></div>
                <span>中风险</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ea4335"></div>
                <span>高风险</span>
            </div>
        `;
    } else if (layerType === 'evacuation') {
        legendHTML = `
            <div class="legend-item">
                <div class="legend-color" style="background-color: #34a853"></div>
                <span>起点</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #1a73e8"></div>
                <span>安全区域</span>
            </div>
            <div class="legend-item">
                <div style="height: 3px; width: 20px; background-color: #1a73e8; margin-right: 8px;"></div>
                <span>疏散路线</span>
            </div>
        `;
    }
    
    legendEl.innerHTML = legendHTML;
}

// 初始化图表
function initCharts() {
    // 疏散图表
    const evacuationCtx = document.getElementById('evacuationChart').getContext('2d');
    charts.evacuation = new Chart(evacuationCtx, {
        type: 'line',
        data: {
            labels: ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
            datasets: [{
                label: '疏散率',
                data: [0, 5, 15, 30, 45, 60, 70, 80, 85, 90, 95],
                borderColor: '#1a73e8',
                backgroundColor: 'rgba(26, 115, 232, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '疏散进度'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '疏散率 (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '模拟进度'
                    }
                }
            }
        }
    });
    
    // 水位图表
    const waterLevelCtx = document.getElementById('waterLevelChart').getContext('2d');
    charts.waterLevel = new Chart(waterLevelCtx, {
        type: 'line',
        data: {
            labels: ['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
            datasets: [{
                label: '最大水位',
                data: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.0, 2.8, 2.5, 2.0],
                borderColor: '#ea4335',
                backgroundColor: 'rgba(234, 67, 53, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: '平均水位',
                data: [0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0],
                borderColor: '#fbbc05',
                backgroundColor: 'rgba(251, 188, 5, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '水位变化'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '水位 (m)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '模拟进度'
                    }
                }
            }
        }
    });
    
    // 智能体状态图表
    const agentStatusCtx = document.getElementById('agentStatusChart').getContext('2d');
    charts.agentStatus = new Chart(agentStatusCtx, {
        type: 'pie',
        data: {
            labels: ['正常', '疏散中', '受困', '已救援'],
            datasets: [{
                data: [40, 30, 20, 10],
                backgroundColor: ['#34a853', '#fbbc05', '#ea4335', '#1a73e8'],
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '智能体状态分布'
                },
                legend: {
                    position: 'right'
                }
            }
        }
    });
    
    // 疏散率变化图表
    const evacuationRateCtx = document.getElementById('evacuationRateChart').getContext('2d');
    charts.evacuationRate = new Chart(evacuationRateCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 11}, (_, i) => `${i * 10}%`),
            datasets: [{
                label: '总体疏散率',
                data: [0, 5, 15, 30, 45, 60, 70, 80, 85, 90, 95],
                borderColor: '#1a73e8',
                backgroundColor: 'rgba(26, 115, 232, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: '家庭疏散率',
                data: [0, 8, 20, 35, 50, 65, 75, 85, 90, 95, 98],
                borderColor: '#34a853',
                backgroundColor: 'rgba(52, 168, 83, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: '独居疏散率',
                data: [0, 3, 10, 25, 40, 55, 65, 75, 80, 85, 90],
                borderColor: '#fbbc05',
                backgroundColor: 'rgba(251, 188, 5, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '疏散率变化'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: '疏散率 (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '模拟进度'
                    }
                }
            }
        }
    });
    
    // 风险分布图表
    const riskDistributionCtx = document.getElementById('riskDistributionChart').getContext('2d');
    charts.riskDistribution = new Chart(riskDistributionCtx, {
        type: 'bar',
        data: {
            labels: ['0级', '1级', '2级', '3级', '4级', '5级'],
            datasets: [{
                label: '智能体数量',
                data: [10, 20, 30, 25, 10, 5],
                backgroundColor: [
                    'rgba(52, 168, 83, 0.7)',
                    'rgba(52, 168, 83, 0.5)',
                    'rgba(251, 188, 5, 0.7)',
                    'rgba(251, 188, 5, 0.5)',
                    'rgba(234, 67, 53, 0.7)',
                    'rgba(234, 67, 53, 0.5)'
                ],
                borderColor: [
                    'rgba(52, 168, 83, 1)',
                    'rgba(52, 168, 83, 1)',
                    'rgba(251, 188, 5, 1)',
                    'rgba(251, 188, 5, 1)',
                    'rgba(234, 67, 53, 1)',
                    'rgba(234, 67, 53, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '智能体风险分布'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '智能体数量'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '风险等级'
                    }
                }
            }
        }
    });
    
    // 水位变化趋势图表
    const waterLevelTrendCtx = document.getElementById('waterLevelTrendChart').getContext('2d');
    charts.waterLevelTrend = new Chart(waterLevelTrendCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 11}, (_, i) => `${i * 10}%`),
            datasets: [{
                label: '最大水位',
                data: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.2, 3.0, 2.8, 2.5, 2.0],
                borderColor: '#ea4335',
                backgroundColor: 'rgba(234, 67, 53, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: '平均水位',
                data: [0.2, 0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0],
                borderColor: '#fbbc05',
                backgroundColor: 'rgba(251, 188, 5, 0.1)',
                fill: true,
                tension: 0.4
            }, {
                label: '最小水位',
                data: [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.0, 0.8, 0.5, 0.3],
                borderColor: '#1a73e8',
                backgroundColor: 'rgba(26, 115, 232, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '水位变化趋势'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '水位 (m)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '模拟进度'
                    }
                }
            }
        }
    });
    
    // 影响区域变化图表
    const affectedAreaCtx = document.getElementById('affectedAreaChart').getContext('2d');
    charts.affectedArea = new Chart(affectedAreaCtx, {
        type: 'line',
        data: {
            labels: Array.from({length: 11}, (_, i) => `${i * 10}%`),
            datasets: [{
                label: '影响面积',
                data: [5, 10, 15, 20, 25, 30, 35, 40, 38, 35, 30],
                borderColor: '#1a73e8',
                backgroundColor: 'rgba(26, 115, 232, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '影响区域变化'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: '面积 (km²)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '模拟进度'
                    }
                }
            }
        }
    });
    
    // 互助行为统计图表
    const aidActionsCtx = document.getElementById('aidActionsChart').getContext('2d');
    charts.aidActions = new Chart(aidActionsCtx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 11}, (_, i) => `${i * 10}%`),
            datasets: [{
                label: '血缘互助',
                data: [0, 5, 15, 25, 40, 55, 70, 85, 95, 100, 105],
                backgroundColor: 'rgba(234, 67, 53, 0.7)',
                borderColor: 'rgba(234, 67, 53, 1)',
                borderWidth: 1
            }, {
                label: '地缘互助',
                data: [0, 3, 10, 20, 35, 45, 60, 70, 75, 80, 85],
                backgroundColor: 'rgba(251, 188, 5, 0.7)',
                borderColor: 'rgba(251, 188, 5, 1)',
                borderWidth: 1
            }, {
                label: '业缘互助',
                data: [0, 2, 8, 15, 25, 35, 45, 55, 60, 65, 70],
                backgroundColor: 'rgba(52, 168, 83, 0.7)',
                borderColor: 'rgba(52, 168, 83, 1)',
                borderWidth: 1
            }, {
                label: '学缘互助',
                data: [0, 1, 5, 10, 20, 30, 40, 45, 50, 55, 60],
                backgroundColor: 'rgba(26, 115, 232, 0.7)',
                borderColor: 'rgba(26, 115, 232, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '互助行为统计'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    stacked: true,
                    title: {
                        display: true,
                        text: '互助行为数量'
                    }
                },
                x: {
                    stacked: true,
                    title: {
                        display: true,
                        text: '模拟进度'
                    }
                }
            }
        }
    });
}

// 初始化社交网络图
function initSocialNetworkGraph() {
    const container = document.getElementById('socialNetworkGraph');
    
    // 使用D3.js创建社交网络图
    const width = container.clientWidth;
    const height = container.clientHeight;
    
    // 创建SVG元素
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height);
    
    // 创建模拟数据
    const nodes = [];
    const links = [];
    
    // 创建节点
    for (let i = 0; i < 50; i++) {
        nodes.push({
            id: i,
            group: Math.floor(Math.random() * 4) // 0: 血缘, 1: 地缘, 2: 业缘, 3: 学缘
        });
    }
    
    // 创建连接
    for (let i = 0; i < 100; i++) {
        const source = Math.floor(Math.random() * 50);
        const target = Math.floor(Math.random() * 50);
        if (source !== target) {
            links.push({
                source,
                target,
                value: Math.floor(Math.random() * 10) + 1,
                type: Math.floor(Math.random() * 4) // 0: 血缘, 1: 地缘, 2: 业缘, 3: 学缘
            });
        }
    }
    
    // 创建力导向图
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id))
        .force('charge', d3.forceManyBody().strength(-50))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    // 定义连接线颜色
    const linkColors = ['#ea4335', '#fbbc05', '#34a853', '#1a73e8'];
    
    // 绘制连接线
    const link = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter()
        .append('line')
        .attr('stroke', d => linkColors[d.type])
        .attr('stroke-width', d => Math.sqrt(d.value));
    
    // 定义节点颜色
    const nodeColors = ['#ea4335', '#fbbc05', '#34a853', '#1a73e8'];
    
    // 绘制节点
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', 5)
        .attr('fill', d => nodeColors[d.group])
        .call(d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended));
    
    // 添加节点交互
    node.append('title')
        .text(d => `智能体 #${d.id}`);
    
    // 更新力导向图
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    });
    
    // 拖拽函数
    function dragstarted(event, d) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    function dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
    
    // 添加图例
    const legend = svg.append('g')
        .attr('transform', 'translate(20, 20)');
    
    const legendLabels = ['血缘关系', '地缘关系', '业缘关系', '学缘关系'];
    
    legendLabels.forEach((label, i) => {
        const legendRow = legend.append('g')
            .attr('transform', `translate(0, ${i * 20})`);
        
        legendRow.append('rect')
            .attr('width', 10)
            .attr('height', 10)
            .attr('fill', nodeColors[i]);
        
        legendRow.append('text')
            .attr('x', 15)
            .attr('y', 10)
            .text(label)
            .style('font-size', '12px')
            .attr('alignment-baseline', 'middle');
    });
    
    // 关系类型选择器事件监听
    document.getElementById('relationshipTypeSelect').addEventListener('change', function() {
        const selectedType = this.value;
        
        // 更新连接线可见性
        link.style('opacity', d => {
            if (selectedType === 'all') return 1;
            return d.type === ['family', 'neighbor', 'colleague', 'classmate'].indexOf(selectedType) ? 1 : 0.1;
        });
        
        // 更新节点可见性
        node.style('opacity', d => {
            if (selectedType === 'all') return 1;
            
            // 检查节点是否有选定类型的连接
            const hasConnection = links.some(link => 
                (link.source.id === d.id || link.target.id === d.id) && 
                link.type === ['family', 'neighbor', 'colleague', 'classmate'].indexOf(selectedType)
            );
            
            return hasConnection ? 1 : 0.3;
        });
    });
}

// 初始化动画帧
function initAnimationFrames() {
    const container = document.getElementById('animationFrames');
    
    // 创建10个动画帧缩略图
    for (let i = 0; i < 10; i++) {
        const thumbnail = document.createElement('div');
        thumbnail.className = 'frame-thumbnail';
        thumbnail.style.backgroundColor = `hsl(${210 + i * 15}, 70%, ${70 - i * 5}%)`;
        thumbnail.dataset.frame = i;
        
        thumbnail.addEventListener('click', function() {
            // 移除所有活动状态
            document.querySelectorAll('.frame-thumbnail').forEach(item => {
                item.classList.remove('active');
            });
            
            // 添加当前活动状态
            this.classList.add('active');
            
            // 更新动画滑块
            document.getElementById('animationSlider').value = i * 10;
            
            // 更新动画时间标签
            document.getElementById('animationTimeLabel').textContent = `时间: ${i * 2}小时`;
            
            // 更新动画地图
            updateAnimationMap(i);
        });
        
        container.appendChild(thumbnail);
    }
    
    // 默认选中第一帧
    container.querySelector('.frame-thumbnail').classList.add('active');
}

// 更新动画地图
function updateAnimationMap(frame) {
    // 清除现有标记
    if (animationMap) {
        animationMap.eachLayer(layer => {
            if (layer instanceof L.Marker || layer instanceof L.CircleMarker || layer instanceof L.Polyline || layer instanceof L.Polygon) {
                animationMap.removeLayer(layer);
            }
        });
    }
    
    // 根据帧索引更新地图
    // 这里使用模拟数据，实际应用中应该使用真实的模拟数据
    
    // 模拟水位数据
    const waterLevel = frame * 0.3; // 水位随时间增加
    
    // 模拟受影响区域
    const affectedArea = L.circle([23.13, 113.26], {
        radius: 1000 + frame * 500,
        color: '#1a73e8',
        fillColor: '#1a73e8',
        fillOpacity: 0.2,
        weight: 1
    }).addTo(animationMap);
    
    // 模拟智能体位置
    for (let i = 0; i < 50; i++) {
        // 智能体位置随时间变化
        const lat = 23.13 + (Math.random() - 0.5) * 0.1 + frame * 0.01;
        const lng = 113.26 + (Math.random() - 0.5) * 0.1 + frame * 0.01;
        
        // 根据帧索引确定智能体状态
        let status;
        if (frame < 3) {
            status = 'normal';
        } else if (frame < 6) {
            status = Math.random() > 0.5 ? 'normal' : 'evacuating';
        } else {
            status = Math.random() > 0.7 ? 'evacuating' : 'rescued';
        }
        
        // 根据状态设置颜色
        const colors = {
            'normal': '#34a853',
            'evacuating': '#fbbc05',
            'trapped': '#ea4335',
            'rescued': '#1a73e8'
        };
        
        L.circleMarker([lat, lng], {
            radius: 4,
            fillColor: colors[status],
            color: '#fff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(animationMap);
    }
}

// 初始化事件监听
function initEventListeners() {
    // 开始仿真按钮
    document.getElementById('startSimulation').addEventListener('click', function() {
        startSimulation();
    });
    
    // 暂停仿真按钮
    document.getElementById('pauseSimulation').addEventListener('click', function() {
        pauseSimulation();
    });
    
    // 停止仿真按钮
    document.getElementById('stopSimulation').addEventListener('click', function() {
        stopSimulation();
    });
    
    // 刷新统计按钮
    document.getElementById('refreshStats').addEventListener('click', function() {
        fetchSimulationStatus();
    });
    
    // 导出数据按钮
    document.getElementById('exportData').addEventListener('click', function() {
        showExportModal();
    });
    
    // 关闭导出弹窗按钮
    document.getElementById('closeExportModal').addEventListener('click', function() {
        hideExportModal();
    });
    
    // 下载数据按钮
    document.getElementById('downloadData').addEventListener('click', function() {
        downloadData();
    });
    
    // 地图图层选择器
    document.getElementById('mapLayerSelect').addEventListener('change', function() {
        updateHeatmapLayer(this.value);
    });
    
    // 时间轴滑块
    document.getElementById('timelineSlider').addEventListener('input', function() {
        updateTimeline(this.value);
    });
    
    // 播放时间轴按钮
    document.getElementById('playTimeline').addEventListener('click', function() {
        playTimeline();
    });
    
    // 暂停时间轴按钮
    document.getElementById('pauseTimeline').addEventListener('click', function() {
        pauseTimeline();
    });
    
    // 重置时间轴按钮
    document.getElementById('resetTimeline').addEventListener('click', function() {
        resetTimeline();
    });
    
    // 动画控制按钮
    document.getElementById('playAnimation').addEventListener('click', function() {
        playAnimation();
    });
    
    document.getElementById('pauseAnimation').addEventListener('click', function() {
        pauseAnimation();
    });
    
    document.getElementById('resetAnimation').addEventListener('click', function() {
        resetAnimation();
    });
    
    // 动画滑块
    document.getElementById('animationSlider').addEventListener('input', function() {
        const frame = Math.floor(this.value / 10);
        document.getElementById('animationTimeLabel').textContent = `时间: ${frame * 2}小时`;
        updateAnimationMap(frame);
        
        // 更新缩略图选中状态
        document.querySelectorAll('.frame-thumbnail').forEach(item => {
            item.classList.remove('active');
        });
        
        const thumbnails = document.querySelectorAll('.frame-thumbnail');
        if (thumbnails[frame]) {
            thumbnails[frame].classList.add('active');
        }
    });
}

// 开始仿真
function startSimulation() {
    // 发送开始仿真请求
    fetch('/api/simulation/start', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            simulationStatus = "运行中";
            document.getElementById('simulationStatus').textContent = simulationStatus;
            document.getElementById('startSimulation').disabled = true;
            document.getElementById('pauseSimulation').disabled = false;
            document.getElementById('stopSimulation').disabled = false;
            
            // 记录开始时间
            startTime = new Date();
            
            // 启动定时器更新状态
            simulationInterval = setInterval(updateSimulationStatus, 1000);
        } else {
            alert('启动仿真失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('启动仿真请求失败:', error);
        alert('启动仿真请求失败，请检查网络连接');
    });
}

// 暂停仿真
function pauseSimulation() {
    // 发送暂停仿真请求
    fetch('/api/simulation/pause', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            simulationStatus = "已暂停";
            document.getElementById('simulationStatus').textContent = simulationStatus;
            document.getElementById('startSimulation').disabled = false;
            document.getElementById('pauseSimulation').disabled = true;
            
            // 停止定时器
            clearInterval(simulationInterval);
        } else {
            alert('暂停仿真失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('暂停仿真请求失败:', error);
        alert('暂停仿真请求失败，请检查网络连接');
    });
}

// 停止仿真
function stopSimulation() {
    // 发送停止仿真请求
    fetch('/api/simulation/stop', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            simulationStatus = "已停止";
            document.getElementById('simulationStatus').textContent = simulationStatus;
            document.getElementById('startSimulation').disabled = false;
            document.getElementById('pauseSimulation').disabled = true;
            document.getElementById('stopSimulation').disabled = true;
            
            // 停止定时器
            clearInterval(simulationInterval);
            
            // 重置开始时间
            startTime = null;
            elapsedTime = 0;
            document.getElementById('elapsedTime').textContent = '00:00';
            
            // 重置进度
            currentStep = 0;
            document.getElementById('currentStep').textContent = currentStep;
            document.getElementById('progressBar').style.width = '0%';
        } else {
            alert('停止仿真失败: ' + data.message);
        }
    })
    .catch(error => {
        console.error('停止仿真请求失败:', error);
        alert('停止仿真请求失败，请检查网络连接');
    });
}

// 更新仿真状态
function updateSimulationStatus() {
    // 更新运行时间
    if (startTime) {
        const now = new Date();
        elapsedTime = Math.floor((now - startTime) / 1000);
        const minutes = Math.floor(elapsedTime / 60).toString().padStart(2, '0');
        const seconds = (elapsedTime % 60).toString().padStart(2, '0');
        document.getElementById('elapsedTime').textContent = `${minutes}:${seconds}`;
    }
    
    // 模拟进度更新
    if (currentStep < totalSteps) {
        currentStep++;
        document.getElementById('currentStep').textContent = currentStep;
        const progress = (currentStep / totalSteps) * 100;
        document.getElementById('progressBar').style.width = `${progress}%`;
        
        // 如果完成，停止仿真
        if (currentStep >= totalSteps) {
            stopSimulation();
        }
    }
}

// 获取仿真状态
function fetchSimulationStatus() {
    // 显示加载动画
    document.getElementById('loadingOverlay').style.display = 'flex';
    
    // 发送获取状态请求
    fetch('/api/simulation/status')
    .then(response => response.json())
    .then(data => {
        // 隐藏加载动画
        document.getElementById('loadingOverlay').style.display = 'none';
        
        if (data.success) {
            // 更新状态
            simulationStatus = data.status;
            document.getElementById('simulationStatus').textContent = simulationStatus;
            
            // 更新步骤
            currentStep = data.current_step;
            totalSteps = data.total_steps;
            document.getElementById('currentStep').textContent = currentStep;
            const progress = (currentStep / totalSteps) * 100;
            document.getElementById('progressBar').style.width = `${progress}%`;
            
            // 更新智能体数量
            document.getElementById('agentCount').textContent = data.agent_count;
            
            // 更新水位数据
            document.getElementById('maxWaterLevel').textContent = data.max_water_level.toFixed(1);
            document.getElementById('avgWaterLevel').textContent = data.avg_water_level.toFixed(1);
            document.getElementById('affectedArea').textContent = data.affected_area.toFixed(1);
            document.getElementById('waterLevelChangeRate').textContent = data.water_level_change_rate.toFixed(2);
            
            // 更新社交网络数据
            document.getElementById('totalAidActions').textContent = data.total_aid_actions;
            document.getElementById('networkDensity').textContent = data.network_density.toFixed(2);
            document.getElementById('avgRelationshipStrength').textContent = data.avg_relationship_strength.toFixed(2);
            document.getElementById('familyAidRatio').textContent = `${(data.family_aid_ratio * 100).toFixed(0)}%`;
            
            // 更新图表数据
            updateCharts(data);
        } else {
            alert('获取仿真状态失败: ' + data.message);
        }
    })
    .catch(error => {
        // 隐藏加载动画
        document.getElementById('loadingOverlay').style.display = 'none';
        
        console.error('获取仿真状态请求失败:', error);
        alert('获取仿真状态请求失败，请检查网络连接');
    });
}

// 更新图表
function updateCharts(data) {
    // 更新疏散图表
    if (charts.evacuation && data.evacuation_data) {
        charts.evacuation.data.datasets[0].data = data.evacuation_data;
        charts.evacuation.update();
    }
    
    // 更新水位图表
    if (charts.waterLevel && data.water_level_data) {
        charts.waterLevel.data.datasets[0].data = data.water_level_data.max;
        charts.waterLevel.data.datasets[1].data = data.water_level_data.avg;
        charts.waterLevel.update();
    }
    
    // 更新智能体状态图表
    if (charts.agentStatus && data.agent_status_data) {
        charts.agentStatus.data.datasets[0].data = [
            data.agent_status_data.normal,
            data.agent_status_data.evacuating,
            data.agent_status_data.trapped,
            data.agent_status_data.rescued
        ];
        charts.agentStatus.update();
    }
    
    // 更新其他图表...
}

// 显示导出数据弹窗
function showExportModal() {
    document.getElementById('exportModal').style.display = 'flex';
}

// 隐藏导出数据弹窗
function hideExportModal() {
    document.getElementById('exportModal').style.display = 'none';
}

// 下载数据
function downloadData() {
    const dataType = document.getElementById('dataTypeSelect').value;
    const formatType = document.getElementById('formatTypeSelect').value;
    
    // 发送下载请求
    fetch(`/api/data/export?type=${dataType}&format=${formatType}`)
    .then(response => response.blob())
    .then(blob => {
        // 创建下载链接
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `${dataType}_data.${formatType}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
        // 隐藏弹窗
        hideExportModal();
    })
    .catch(error => {
        console.error('下载数据请求失败:', error);
        alert('下载数据请求失败，请检查网络连接');
    });
}

// 更新时间轴
function updateTimeline(value) {
    const percent = value;
    document.getElementById('timelineLabel').textContent = `时间: ${Math.floor(percent / 10)}小时`;
    
    // 更新地图数据
    // 这里使用模拟数据，实际应用中应该使用真实的模拟数据
    updateHeatmapLayer(document.getElementById('mapLayerSelect').value);
}

// 播放时间轴
function playTimeline() {
    // 停止现有定时器
    pauseTimeline();
    
    // 获取当前值
    let value = parseInt(document.getElementById('timelineSlider').value);
    
    // 启动定时器
    timelineInterval = setInterval(() => {
        value += 1;
        if (value > 100) {
            value = 0;
        }
        
        document.getElementById('timelineSlider').value = value;
        updateTimeline(value);
    }, 500);
}

// 暂停时间轴
function pauseTimeline() {
    if (timelineInterval) {
        clearInterval(timelineInterval);
        timelineInterval = null;
    }
}

// 重置时间轴
function resetTimeline() {
    pauseTimeline();
    document.getElementById('timelineSlider').value = 0;
    updateTimeline(0);
}

// 播放动画
function playAnimation() {
    // 停止现有定时器
    pauseAnimation();
    
    // 获取当前值
    let value = parseInt(document.getElementById('animationSlider').value);
    
    // 启动定时器
    animationInterval = setInterval(() => {
        value += 1;
        if (value > 100) {
            value = 0;
        }
        
        document.getElementById('animationSlider').value = value;
        const frame = Math.floor(value / 10);
        document.getElementById('animationTimeLabel').textContent = `时间: ${frame * 2}小时`;
        updateAnimationMap(frame);
        
        // 更新缩略图选中状态
        document.querySelectorAll('.frame-thumbnail').forEach(item => {
            item.classList.remove('active');
        });
        
        const thumbnails = document.querySelectorAll('.frame-thumbnail');
        if (thumbnails[frame]) {
            thumbnails[frame].classList.add('active');
        }
    }, 500);
}

// 暂停动画
function pauseAnimation() {
    if (animationInterval) {
        clearInterval(animationInterval);
        animationInterval = null;
    }
}

// 重置动画
function resetAnimation() {
    pauseAnimation();
    document.getElementById('animationSlider').value = 0;
    const frame = 0;
    document.getElementById('animationTimeLabel').textContent = `时间: ${frame * 2}小时`;
    updateAnimationMap(frame);
    
    // 更新缩略图选中状态
    document.querySelectorAll('.frame-thumbnail').forEach(item => {
        item.classList.remove('active');
    });
    
    const thumbnails = document.querySelectorAll('.frame-thumbnail');
    if (thumbnails[frame]) {
        thumbnails[frame].classList.add('active');
    }
}
