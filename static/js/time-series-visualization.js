/**
 * 时间序列可视化模块
 * 用于实现洪灾ABM仿真系统的时间序列图表和动态可视化功能
 */

// 时间序列可视化模块
const TimeSeriesVisualization = (function() {
    // 私有变量
    let timeSeriesCharts = {};
    let animationFrames = [];
    let currentFrameIndex = 0;
    let animationInterval = null;
    let animationSpeed = 1000; // 默认动画速度（毫秒）
    let isPlaying = false;
    
    // 模拟数据（实际应用中应该从API获取）
    const simulatedData = {
        waterLevel: {
            labels: Array.from({length: 24}, (_, i) => `${i}小时`),
            max: Array.from({length: 24}, (_, i) => {
                if (i < 8) return 0.5 + i * 0.3;
                else if (i < 16) return 2.9 - (i - 8) * 0.1;
                else return 2.1 - (i - 16) * 0.15;
            }),
            avg: Array.from({length: 24}, (_, i) => {
                if (i < 8) return 0.2 + i * 0.2;
                else if (i < 16) return 1.8 - (i - 8) * 0.05;
                else return 1.4 - (i - 16) * 0.1;
            }),
            min: Array.from({length: 24}, (_, i) => {
                if (i < 8) return 0.0 + i * 0.15;
                else if (i < 16) return 1.2 - (i - 8) * 0.03;
                else return 0.96 - (i - 16) * 0.06;
            })
        },
        evacuationRate: {
            labels: Array.from({length: 24}, (_, i) => `${i}小时`),
            total: Array.from({length: 24}, (_, i) => {
                if (i < 4) return i * 5;
                else if (i < 12) return 20 + (i - 4) * 8;
                else if (i < 20) return 84 + (i - 12) * 2;
                else return 100 - Math.exp(-(i - 20));
            }),
            family: Array.from({length: 24}, (_, i) => {
                if (i < 4) return i * 8;
                else if (i < 12) return 32 + (i - 4) * 7;
                else if (i < 20) return 88 + (i - 12) * 1.5;
                else return 100 - Math.exp(-(i - 18));
            }),
            single: Array.from({length: 24}, (_, i) => {
                if (i < 4) return i * 3;
                else if (i < 12) return 12 + (i - 4) * 7;
                else if (i < 20) return 68 + (i - 12) * 3;
                else return 92 + (i - 20) * 1;
            })
        },
        agentStatus: {
            labels: Array.from({length: 24}, (_, i) => `${i}小时`),
            normal: Array.from({length: 24}, (_, i) => {
                if (i < 8) return 100 - i * 10;
                else if (i < 16) return 20 - (i - 8) * 2;
                else return 4;
            }),
            evacuating: Array.from({length: 24}, (_, i) => {
                if (i < 4) return i * 5;
                else if (i < 12) return 20 + (i - 4) * 5;
                else if (i < 20) return 60 - (i - 12) * 7;
                else return 4;
            }),
            trapped: Array.from({length: 24}, (_, i) => {
                if (i < 8) return i * 3;
                else if (i < 16) return 24 - (i - 8) * 2;
                else return 8;
            }),
            rescued: Array.from({length: 24}, (_, i) => {
                if (i < 4) return 0;
                else if (i < 12) return (i - 4) * 5;
                else if (i < 20) return 40 + (i - 12) * 6;
                else return 88 - (i - 20) * 2;
            })
        },
        affectedArea: {
            labels: Array.from({length: 24}, (_, i) => `${i}小时`),
            area: Array.from({length: 24}, (_, i) => {
                if (i < 8) return 5 + i * 4;
                else if (i < 16) return 37 + (i - 8) * 0.5;
                else return 41 - (i - 16) * 2;
            })
        },
        events: [
            { time: 0, description: "模拟开始，初始水位设置" },
            { time: 2, description: "洪水开始上涨，首批预警发出" },
            { time: 5, description: "部分区域开始疏散" },
            { time: 8, description: "洪水达到峰值，大规模疏散行动" },
            { time: 12, description: "互助网络形成，社区自救行动开始" },
            { time: 16, description: "洪水开始退去，救援行动加强" },
            { time: 20, description: "大部分地区水位下降，重建工作开始" }
        ]
    };
    
    // 地理位置数据（模拟广州市部分区域）
    const geoData = {
        center: [23.13, 113.26], // 广州市中心坐标
        floodPoints: Array.from({length: 24}, (_, timeStep) => {
            // 随时间变化的洪水点
            return Array.from({length: 100}, (_, i) => {
                // 模拟洪水扩散
                const radius = Math.min(0.05 + timeStep * 0.003, 0.12);
                const angle = Math.random() * Math.PI * 2;
                const distance = Math.random() * radius;
                const lat = 23.13 + distance * Math.cos(angle);
                const lng = 113.26 + distance * Math.sin(angle);
                
                // 水位强度随时间变化
                let intensity;
                if (timeStep < 8) {
                    intensity = 0.3 + timeStep * 0.1;
                } else if (timeStep < 16) {
                    intensity = 1.1 - (timeStep - 8) * 0.05;
                } else {
                    intensity = 0.7 - (timeStep - 16) * 0.05;
                }
                intensity = Math.max(0.1, Math.min(1.0, intensity));
                
                return [lat, lng, intensity];
            });
        }),
        agentPositions: Array.from({length: 24}, (_, timeStep) => {
            // 随时间变化的智能体位置
            return Array.from({length: 50}, (_, i) => {
                // 初始位置
                let baseLat = 23.13 + (Math.random() - 0.5) * 0.1;
                let baseLng = 113.26 + (Math.random() - 0.5) * 0.1;
                
                // 模拟智能体移动（向安全区域移动）
                if (timeStep > 4) {
                    // 安全区域方向
                    const safeDirection = Math.atan2(23.18 - baseLat, 113.31 - baseLng);
                    const moveDistance = Math.min(0.001 * (timeStep - 4), 0.05);
                    
                    baseLat += moveDistance * Math.cos(safeDirection);
                    baseLng += moveDistance * Math.sin(safeDirection);
                }
                
                // 随机扰动
                const jitter = 0.002;
                const lat = baseLat + (Math.random() - 0.5) * jitter;
                const lng = baseLng + (Math.random() - 0.5) * jitter;
                
                // 智能体状态
                let status;
                if (timeStep < 4) {
                    status = 'normal';
                } else if (timeStep < 12) {
                    status = Math.random() > 0.7 ? 'normal' : 'evacuating';
                } else if (timeStep < 18) {
                    status = Math.random() > 0.8 ? 'evacuating' : 'rescued';
                } else {
                    status = Math.random() > 0.9 ? 'evacuating' : 'rescued';
                }
                
                // 有小概率被困
                if (timeStep > 6 && timeStep < 16 && Math.random() < 0.1) {
                    status = 'trapped';
                }
                
                return { lat, lng, status };
            });
        })
    };
    
    // 初始化函数
    function init() {
        console.log("初始化时间序列可视化模块");
        
        // 初始化时间序列图表
        initTimeSeriesCharts();
        
        // 初始化动画帧
        generateAnimationFrames();
        
        // 初始化事件监听
        initEventListeners();
    }
    
    // 初始化时间序列图表
    function initTimeSeriesCharts() {
        // 水位时间序列图表
        const waterLevelTimeSeriesCtx = document.getElementById('waterLevelTimeSeriesChart');
        if (waterLevelTimeSeriesCtx) {
            timeSeriesCharts.waterLevel = new Chart(waterLevelTimeSeriesCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: simulatedData.waterLevel.labels,
                    datasets: [{
                        label: '最大水位',
                        data: simulatedData.waterLevel.max,
                        borderColor: '#ea4335',
                        backgroundColor: 'rgba(234, 67, 53, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: '平均水位',
                        data: simulatedData.waterLevel.avg,
                        borderColor: '#fbbc05',
                        backgroundColor: 'rgba(251, 188, 5, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: '最小水位',
                        data: simulatedData.waterLevel.min,
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
                            text: '水位时间序列'
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
                                text: '时间'
                            }
                        }
                    }
                }
            });
        }
        
        // 疏散率时间序列图表
        const evacuationRateTimeSeriesCtx = document.getElementById('evacuationRateTimeSeriesChart');
        if (evacuationRateTimeSeriesCtx) {
            timeSeriesCharts.evacuationRate = new Chart(evacuationRateTimeSeriesCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: simulatedData.evacuationRate.labels,
                    datasets: [{
                        label: '总体疏散率',
                        data: simulatedData.evacuationRate.total,
                        borderColor: '#1a73e8',
                        backgroundColor: 'rgba(26, 115, 232, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: '家庭疏散率',
                        data: simulatedData.evacuationRate.family,
                        borderColor: '#34a853',
                        backgroundColor: 'rgba(52, 168, 83, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: '独居疏散率',
                        data: simulatedData.evacuationRate.single,
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
                            text: '疏散率时间序列'
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
                                text: '时间'
                            }
                        }
                    }
                }
            });
        }
        
        // 智能体状态时间序列图表
        const agentStatusTimeSeriesCtx = document.getElementById('agentStatusTimeSeriesChart');
        if (agentStatusTimeSeriesCtx) {
            timeSeriesCharts.agentStatus = new Chart(agentStatusTimeSeriesCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: simulatedData.agentStatus.labels,
                    datasets: [{
                        label: '正常',
                        data: simulatedData.agentStatus.normal,
                        borderColor: '#34a853',
                        backgroundColor: 'rgba(52, 168, 83, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: '疏散中',
                        data: simulatedData.agentStatus.evacuating,
                        borderColor: '#fbbc05',
                        backgroundColor: 'rgba(251, 188, 5, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: '受困',
                        data: simulatedData.agentStatus.trapped,
                        borderColor: '#ea4335',
                        backgroundColor: 'rgba(234, 67, 53, 0.1)',
                        fill: true,
                        tension: 0.4
                    }, {
                        label: '已救援',
                        data: simulatedData.agentStatus.rescued,
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
                            text: '智能体状态时间序列'
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
                                text: '智能体数量 (%)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: '时间'
                            }
                        }
                    }
                }
            });
        }
        
        // 影响区域时间序列图表
        const affectedAreaTimeSeriesCtx = document.getElementById('affectedAreaTimeSeriesChart');
        if (affectedAreaTimeSeriesCtx) {
            timeSeriesCharts.affectedArea = new Chart(affectedAreaTimeSeriesCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: simulatedData.affectedArea.labels,
                    datasets: [{
                        label: '影响面积',
                        data: simulatedData.affectedArea.area,
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
                            text: '影响区域时间序列'
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
                                text: '时间'
                            }
                        }
                    }
                }
            });
        }
    }
    
    // 生成动画帧
    function generateAnimationFrames() {
        // 生成24个时间步的动画帧
        for (let i = 0; i < 24; i++) {
            animationFrames.push({
                timeStep: i,
                waterLevel: {
                    max: simulatedData.waterLevel.max[i],
                    avg: simulatedData.waterLevel.avg[i],
                    min: simulatedData.waterLevel.min[i]
                },
                evacuationRate: {
                    total: simulatedData.evacuationRate.total[i],
                    family: simulatedData.evacuationRate.family[i],
                    single: simulatedData.evacuationRate.single[i]
                },
                agentStatus: {
                    normal: simulatedData.agentStatus.normal[i],
                    evacuating: simulatedData.agentStatus.evacuating[i],
                    trapped: simulatedData.agentStatus.trapped[i],
                    rescued: simulatedData.agentStatus.rescued[i]
                },
                affectedArea: simulatedData.affectedArea.area[i],
                floodPoints: geoData.floodPoints[i],
                agentPositions: geoData.agentPositions[i]
            });
        }
    }
    
    // 初始化事件监听
    function initEventListeners() {
        // 动态时间轴滑块
        const dynamicTimelineSlider = document.getElementById('dynamicTimelineSlider');
        if (dynamicTimelineSlider) {
            dynamicTimelineSlider.addEventListener('input', function() {
                const frameIndex = Math.floor(this.value / (100 / 23));
                updateVisualizationToFrame(frameIndex);
            });
        }
        
        // 播放按钮
        const playDynamicTimeline = document.getElementById('playDynamicTimeline');
        if (playDynamicTimeline) {
            playDynamicTimeline.addEventListener('click', function() {
                playAnimation();
            });
        }
        
        // 暂停按钮
        const pauseDynamicTimeline = document.getElementById('pauseDynamicTimeline');
        if (pauseDynamicTimeline) {
            pauseDynamicTimeline.addEventListener('click', function() {
                pauseAnimation();
            });
        }
        
        // 停止按钮
        const stopDynamicTimeline = document.getElementById('stopDynamicTimeline');
        if (stopDynamicTimeline) {
            stopDynamicTimeline.addEventListener('click', function() {
                stopAnimation();
            });
        }
        
        // 动画速度选择器
        const animationSpeedSelect = document.getElementById('animationSpeedSelect');
        if (animationSpeedSelect) {
            animationSpeedSelect.addEventListener('change', function() {
                animationSpeed = parseInt(this.value);
                if (isPlaying) {
                    pauseAnimation();
                    playAnimation();
                }
            });
        }
    }
    
    // 更新可视化到指定帧
    function updateVisualizationToFrame(frameIndex) {
        if (frameIndex < 0 || frameIndex >= animationFrames.length) return;
        
        currentFrameIndex = frameIndex;
        const frame = animationFrames[frameIndex];
        
        // 更新时间标签
        const dynamicTimelineLabel = document.getElementById('dynamicTimelineLabel');
        if (dynamicTimelineLabel) {
            dynamicTimelineLabel.textContent = `时间: ${frame.timeStep}小时`;
        }
        
        // 更新滑块位置
        const dynamicTimelineSlider = document.getElementById('dynamicTimelineSlider');
        if (dynamicTimelineSlider) {
            dynamicTimelineSlider.value = (frameIndex / 23) * 100;
        }
        
        // 更新图表
        updateCharts(frame);
        
        // 更新地图
        updateMap(frame);
        
        // 更新数据点详情
        updateDataPointDetails(frame);
    }
    
    // 更新图表
    function updateCharts(frame) {
        // 添加垂直线标记当前时间点
        Object.keys(timeSeriesCharts).forEach(chartKey => {
            const chart = timeSeriesCharts[chartKey];
            if (chart) {
                // 移除现有的垂直线
                if (chart.options.plugins.annotation && chart.options.plugins.annotation.annotations) {
                    chart.options.plugins.annotation.annotations = {};
                } else {
                    chart.options.plugins.annotation = {
                        annotations: {}
                    };
                }
                
                // 添加新的垂直线
                chart.options.plugins.annotation.annotations.currentTime = {
                    type: 'line',
                    xMin: frame.timeStep,
                    xMax: frame.timeStep,
                    borderColor: 'rgba(255, 0, 0, 0.7)',
                    borderWidth: 2
                };
                
                chart.update();
            }
        });
    }
    
    // 更新地图
    function updateMap(frame) {
        const dynamicMap = document.getElementById('dynamicMap');
        if (!dynamicMap || !window.L) return;
        
        // 获取地图实例
        let map = dynamicMap._leaflet_map;
        if (!map) {
            // 初始化地图
            map = L.map(dynamicMap).setView(geoData.center, 12);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);
            
            // 保存地图实例
            dynamicMap._leaflet_map = map;
        }
        
        // 清除现有图层
        map.eachLayer(layer => {
            if (layer instanceof L.Marker || layer instanceof L.CircleMarker || layer instanceof L.Polyline || layer instanceof L.Polygon || layer instanceof L.LayerGroup) {
                map.removeLayer(layer);
            }
        });
        
        // 添加底图
        if (!map._hasBaseLayer) {
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);
            map._hasBaseLayer = true;
        }
        
        // 添加热力图层
        if (frame.floodPoints && frame.floodPoints.length > 0) {
            const heatLayer = L.heatLayer(frame.floodPoints, {
                radius: 25,
                blur: 15,
                maxZoom: 17,
                gradient: {0.4: '#1a73e8', 0.65: '#fbbc05', 0.9: '#ea4335'}
            }).addTo(map);
        }
        
        // 添加智能体标记
        if (frame.agentPositions && frame.agentPositions.length > 0) {
            const agentColors = {
                'normal': '#34a853',
                'evacuating': '#fbbc05',
                'trapped': '#ea4335',
                'rescued': '#1a73e8'
            };
            
            frame.agentPositions.forEach((agent, index) => {
                const marker = L.circleMarker([agent.lat, agent.lng], {
                    radius: 5,
                    fillColor: agentColors[agent.status],
                    color: '#fff',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }).addTo(map);
                
                marker.bindTooltip(`智能体 #${index}<br>状态: ${agent.status}`);
            });
        }
        
        // 添加安全区域标记
        const safeZone = L.circle([23.18, 113.31], {
            radius: 1000,
            color: '#34a853',
            fillColor: '#34a853',
            fillOpacity: 0.2,
            weight: 2
        }).addTo(map);
        safeZone.bindTooltip('安全区域');
    }
    
    // 更新数据点详情
    function updateDataPointDetails(frame) {
        const dataPointDetails = document.getElementById('dataPointDetails');
        if (!dataPointDetails) return;
        
        // 查找当前时间点的事件
        const events = simulatedData.events.filter(event => event.time === frame.timeStep);
        
        if (events.length > 0) {
            let detailsHTML = '<div class="event-alert">';
            events.forEach(event => {
                detailsHTML += `<div class="event-title">时间 ${event.time}小时: ${event.description}</div>`;
            });
            detailsHTML += '</div>';
            
            dataPointDetails.innerHTML = detailsHTML;
            dataPointDetails.style.display = 'block';
            
            // 添加淡出效果
            setTimeout(() => {
                dataPointDetails.classList.add('fade-out');
                setTimeout(() => {
                    dataPointDetails.style.display = 'none';
                    dataPointDetails.classList.remove('fade-out');
                }, 3000);
            }, 5000);
        } else {
            dataPointDetails.style.display = 'none';
        }
    }
    
    // 播放动画
    function playAnimation() {
        // 停止现有动画
        pauseAnimation();
        
        isPlaying = true;
        
        // 启动动画定时器
        animationInterval = setInterval(() => {
            currentFrameIndex = (currentFrameIndex + 1) % animationFrames.length;
            updateVisualizationToFrame(currentFrameIndex);
        }, animationSpeed);
    }
    
    // 暂停动画
    function pauseAnimation() {
        if (animationInterval) {
            clearInterval(animationInterval);
            animationInterval = null;
        }
        
        isPlaying = false;
    }
    
    // 停止动画
    function stopAnimation() {
        pauseAnimation();
        currentFrameIndex = 0;
        updateVisualizationToFrame(currentFrameIndex);
    }
    
    // 公开API
    return {
        init: init,
        playAnimation: playAnimation,
        pauseAnimation: pauseAnimation,
        stopAnimation: stopAnimation,
        updateVisualizationToFrame: updateVisualizationToFrame
    };
})();

// 导出模块
window.TimeSeriesVisualization = TimeSeriesVisualization;