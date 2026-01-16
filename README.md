# DefenderV - 音频深度伪造检测平台

基于知识蒸馏的轻量化音频真伪检测系统，可检测音频是真人声音还是AI合成声音。

## 📁 项目结构

```
DefenderV/
├── backend/                  # 后端服务 (需要GPU服务器)
│   ├── api_server.py        # REST API 服务器
│   ├── detector.py           # 检测器核心代码
│   ├── test_setup.py         # 环境测试脚本
│   ├── requirements.txt      # Python依赖
│   ├── models/              # 模型定义
│   │   ├── decouple.py      # SpeechTokenizer模型
│   │   ├── safeear.py       # DefenderV检测模型
│   │   └── modules/         # 模型组件
│   └── weights/             # 模型权重 (需要下载)
│       ├── SpeechTokenizer.pt
│       └── student_model_weights.pth
│
├── frontend/                 # 前端服务 (普通服务器/开发板)
│   ├── app.py               # Flask Web应用
│   ├── requirements.txt     # Python依赖
│   ├── templates/           # HTML模板
│   │   └── index.html       # 主页面
│   └── uploads/             # 临时上传目录
│
└── README.md                # 本文件
```

---

## 🚀 快速开始

### 方案一：前后端分离部署（推荐）

**架构：**
```
用户浏览器 → 前端服务器 (端口5000) → GPU服务器 (端口8000)
```

#### Step 1: 部署后端 (GPU服务器)

**1.1 上传文件到GPU服务器**

```bash
# 在本地打包
cd F:\pythonprojects\pythonproject\DefenderV
tar -czvf DefenderV_backend.tar.gz backend/

# 上传到GPU服务器 (替换 YOUR_GPU_SERVER_IP)
scp DefenderV_backend.tar.gz root@YOUR_GPU_SERVER_IP:/root/

# 在服务器上解压
ssh root@YOUR_GPU_SERVER_IP
cd /root
tar -xzvf DefenderV_backend.tar.gz
cd backend
```

**1.2 安装依赖**

```bash
# 创建conda环境
conda create -n defenderv python=3.9 -y
conda activate defenderv

# 安装PyTorch (根据CUDA版本选择)
# CUDA 11.7
pip install torch==1.13.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# CUDA 11.8
pip install torch==2.0.0+cu118 torchaudio==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

**1.3 上传模型权重**

确保 `weights/` 目录下有：
- `SpeechTokenizer.pt` (约460MB)
- `student_model_weights.pth` (约40MB)

如果还没有，从训练好的模型复制过来。

**1.4 测试环境**

```bash
python test_setup.py
```

应该看到所有测试通过。

**1.5 启动API服务**

```bash
# 前台运行 (测试)
python api_server.py

# 后台运行 (生产)
nohup python api_server.py > api.log 2>&1 &

# 查看日志
tail -f api.log
```

**1.6 验证API**

```bash
# 测试健康检查
curl http://localhost:8000/health

# 应该返回:
# {"status":"healthy","cuda_available":true,"device":"cuda:0"}
```

**重要：** 确保防火墙开放8000端口！

```bash
# Ubuntu/Debian
sudo ufw allow 8000

# CentOS
sudo firewall-cmd --add-port=8000/tcp --permanent
sudo firewall-cmd --reload
```

---

#### Step 2: 部署前端

**2.1 修改API地址**

编辑 `frontend/app.py`，找到第22行：

```python
# 修改为你的GPU服务器IP
DEFENDERV_API_URL = 'http://YOUR_GPU_SERVER_IP:8000'
```

例如：
```python
DEFENDERV_API_URL = 'http://123.45.67.89:8000'
```

**2.2 上传到前端服务器**

```bash
# 上传frontend文件夹到服务器
scp -r frontend/ root@YOUR_FRONTEND_SERVER_IP:/root/DefenderV/

# 或上传到开发板
scp -r frontend/ user@YOUR_DEVICE_IP:/home/user/DefenderV/
```

**2.3 安装依赖**

```bash
ssh root@YOUR_FRONTEND_SERVER_IP
cd /root/DefenderV/frontend

# 创建虚拟环境 (可选)
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**2.4 启动前端服务**

```bash
# 前台运行
python app.py

# 后台运行
nohup python app.py > frontend.log 2>&1 &

# 使用gunicorn (更稳定)
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**2.5 访问测试**

浏览器打开：`http://YOUR_FRONTEND_SERVER_IP:5000`

上传音频文件测试！

---

### 方案二：单服务器部署

如果只有一台GPU服务器，可以前后端都部署在同一台：

```bash
# 1. 上传整个DefenderV文件夹
scp -r DefenderV/ root@GPU_SERVER:/root/

# 2. 安装后端依赖
cd /root/DefenderV/backend
conda activate defenderv
pip install -r requirements.txt

# 3. 安装前端依赖
cd /root/DefenderV/frontend
pip install -r requirements.txt

# 4. 修改前端API地址为本地
# 在 frontend/app.py 中:
DEFENDERV_API_URL = 'http://localhost:8000'

# 5. 启动后端 (端口8000)
cd /root/DefenderV/backend
nohup python api_server.py > api.log 2>&1 &

# 6. 启动前端 (端口5000)
cd /root/DefenderV/frontend
nohup python app.py > frontend.log 2>&1 &
```

访问：`http://GPU_SERVER_IP:5000`

---

## 🔧 配置说明

### 后端配置 (`backend/api_server.py`)

默认配置：
- 端口：8000
- 上传限制：16MB
- 支持格式：wav, mp3, m4a, ogg, flac, webm

### 前端配置 (`frontend/app.py`)

```python
# 检测模式: 'api' 或 'local'
DETECTION_MODE = 'api'  # 使用远程API

# DefenderV API 地址
DEFENDERV_API_URL = 'http://YOUR_GPU_SERVER_IP:8000'
```

---

## 📊 API 接口文档

### 健康检查

```bash
GET /health
```

响应：
```json
{
  "status": "healthy",
  "cuda_available": true,
  "device": "cuda:0"
}
```

### 音频检测

```bash
POST /detect
Content-Type: multipart/form-data
```

请求：
- 字段名：`audio`
- 文件：音频文件

响应：
```json
{
  "status": "success",
  "is_fake": false,
  "confidence": 0.95,
  "fake_probability": 0.05,
  "real_probability": 0.95,
  "label": "真人声音 (Real)",
  "result": "real",
  "detection_result": "真人声音",
  "confidence_percent": "95.0%"
}
```

---

## 🐛 常见问题

### Q1: API连接失败？

**检查清单：**
1. GPU服务器API是否运行：`curl http://GPU_SERVER_IP:8000/health`
2. 防火墙是否开放8000端口
3. 前端配置的API地址是否正确

### Q2: 模型加载失败？

**检查：**
1. `weights/` 目录下是否有模型文件
2. 文件路径是否正确
3. 运行 `python test_setup.py` 查看详细错误

### Q3: GPU内存不足？

**解决方案：**
- 减小batch_size（如果支持）
- 使用CPU模式（较慢）
- 升级GPU

### Q4: 检测很慢？

**优化建议：**
1. 确认使用GPU：检查 `device: cuda`
2. 检查网络延迟（前后端分离时）
3. 音频文件不要太大（建议<10MB）

### Q5: 前端无法访问？

**检查：**
1. `app.run(host='0.0.0.0')` 允许外部访问
2. 防火墙开放5000端口
3. 服务器IP是否正确

---

## 📝 使用 systemd 管理服务 (Linux)

### 后端服务

创建 `/etc/systemd/system/defenderv-api.service`:

```ini
[Unit]
Description=DefenderV API Server
After=network.target

[Service]
User=root
WorkingDirectory=/root/DefenderV/backend
Environment="PATH=/root/miniconda3/envs/defenderv/bin"
ExecStart=/root/miniconda3/envs/defenderv/bin/python api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### 前端服务

创建 `/etc/systemd/system/defenderv-frontend.service`:

```ini
[Unit]
Description=DefenderV Frontend
After=network.target

[Service]
User=root
WorkingDirectory=/root/DefenderV/frontend
ExecStart=/usr/bin/python3 app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### 管理命令

```bash
# 启动
sudo systemctl start defenderv-api
sudo systemctl start defenderv-frontend

# 停止
sudo systemctl stop defenderv-api
sudo systemctl stop defenderv-frontend

# 开机自启
sudo systemctl enable defenderv-api
sudo systemctl enable defenderv-frontend

# 查看状态
sudo systemctl status defenderv-api
sudo systemctl status defenderv-frontend
```

---

## 🔒 安全建议

1. **HTTPS**: 生产环境使用Nginx反向代理 + SSL证书
2. **防火墙**: 只开放必要端口
3. **API认证**: 可添加API Key验证（可选）

### Nginx 配置示例

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📈 性能指标

| 指标 | 数值 |
|------|------|
| 模型大小 | ~40 MB (学生模型) |
| EER | ~2.63% |
| 推理速度 | ~50ms/样本 (GPU) |
| 支持格式 | WAV, MP3, FLAC, OGG等 |
| 最大文件 | 16 MB |

---

## 📞 快速检查清单

部署前确认：

- [x] GPU服务器租用完成
- [x] 模型权重文件已上传到 `backend/weights/`
- [ ] 后端依赖安装完成
- [ ] 后端API测试通过 (`/health`)
- [ ] 前端API地址已配置
- [ ] 前端依赖安装完成
- [ ] 防火墙端口已开放
- [ ] 浏览器可以访问前端页面

---

## 📜 许可证

仅供研究和学习使用。

---

## 🆘 获取帮助

如遇问题，请检查：
1. 日志文件：`api.log` 和 `frontend.log`
2. 运行测试：`python test_setup.py`
3. 查看本文档的"常见问题"部分





### 快速部署步骤

#### 1. 部署后端 (GPU服务器)

```
# 上传backend文件夹
scp -r backend/ root@GPU_SERVER:/root/DefenderV/

# 在服务器上
cd /root/DefenderV/backend
conda create -n defenderv python=3.9 -y
conda activate defenderv
pip install -r requirements.txt
python test_setup.py  # 测试
python api_server.py  # 启动
```

#### 2. 部署前端

```
# 修改 frontend/app.py 第22行
DEFENDERV_API_URL = 'http://YOUR_GPU_SERVER_IP:8000'

# 上传frontend文件夹
scp -r frontend/ root@FRONTEND_SERVER:/root/DefenderV/

# 在服务器上
cd /root/DefenderV/frontend
pip install -r requirements.txt
python app.py  # 启动
```

#### 3. 访问

浏览器打开：http://FRONTEND_SERVER_IP:5000

### 详细说明

查看 README.md，包含：

- 完整部署步骤

- 配置说明

- API 文档

- 常见问题

- systemd 服务配置

- 安全建议

所有文件已就绪，可直接部署使用。
