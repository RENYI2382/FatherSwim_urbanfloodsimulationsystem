# GitHub Pages 设置指南

## 🌐 启用 GitHub Pages

为了让您的项目在 GitHub Pages 上正确显示，请按照以下步骤操作：

### 1. 进入仓库设置
1. 打开您的 GitHub 仓库：https://github.com/RENYI2382/FatherSwim_urbanfloodsimulationsystem
2. 点击仓库顶部的 **Settings** 标签
3. 在左侧菜单中找到 **Pages** 选项

### 2. 配置 Pages 设置
1. 在 **Source** 部分，选择 **GitHub Actions**
2. 系统会自动检测到我们创建的 `.github/workflows/deploy-pages.yml` 工作流
3. 点击 **Save** 保存设置

### 3. 等待部署完成
1. 返回仓库主页，点击 **Actions** 标签
2. 您会看到 "Deploy to GitHub Pages" 工作流正在运行
3. 等待工作流完成（通常需要 1-3 分钟）

### 4. 访问您的网站
部署完成后，您的网站将可以通过以下地址访问：
```
https://renyi2382.github.io/FatherSwim_urbanfloodsimulationsystem/
```

## 🎯 网站功能

### 主页导航
- **系统主页**: 进入原有的系统主界面
- **交互式仿真**: 实时洪灾仿真和智能体分析
- **数据仪表盘**: 仿真数据统计和可视化
- **增强仪表盘**: 高级数据分析功能
- **地图管理**: 地图数据上传和编辑
- **项目展示**: 功能演示和技术特性

### 技术特点
- ✅ 响应式设计，支持移动端和桌面端
- ✅ 现代化 UI 设计，毛玻璃效果
- ✅ 自动部署，每次推送代码都会更新网站
- ✅ 静态资源优化，快速加载

## 🔧 故障排除

### 如果网站显示 404 错误
1. 检查 GitHub Pages 是否已启用
2. 确认工作流是否成功运行
3. 等待几分钟让 DNS 传播完成

### 如果样式显示异常
1. 清除浏览器缓存
2. 检查控制台是否有 JavaScript 错误
3. 确认所有静态资源路径正确

## 📝 注意事项

1. **静态网站限制**: GitHub Pages 只能托管静态网站，不支持服务器端功能（如 Flask 应用）
2. **交互功能**: 某些需要后端支持的功能在 GitHub Pages 上无法正常工作
3. **演示目的**: 这个部署主要用于展示项目结构和前端界面

## 🚀 下一步

如果您需要完整的功能体验（包括仿真功能），建议：
1. 在本地运行 Flask 应用：`python app.py --port 8080`
2. 或者部署到支持 Python 的云平台（如 Heroku、Railway 等）

---

**项目地址**: https://github.com/RENYI2382/FatherSwim_urbanfloodsimulationsystem  
**GitHub Pages**: https://renyi2382.github.io/FatherSwim_urbanfloodsimulationsystem/