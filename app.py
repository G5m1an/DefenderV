"""
DefenderV - 音频深度伪造检测平台 (本地单机版)
"""
import os
import sys
import uuid
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ============ 核心配置 ============
# 1. 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))    # frontend/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                 # DefenderV/
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')      # backend/
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')        # frontend/uploads/

# 2. 将后端加入 Python 路径 (关键)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# 3. Flask 配置
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm'}
# ==================================

# 全局检测器实例
local_detector = None

def get_detector():
    """懒加载检测器，避免启动时卡顿"""
    global local_detector
    if local_detector is None:
        try:
            print("正在初始化 DefenderV 本地检测核心...")
            from detector import DefenderVDetector
            # 这里不需要传路径，detector.py 内部已经自适应了
            local_detector = DefenderVDetector()
            print("DefenderV 检测核心加载完毕！")
        except Exception as e:
            print(f"严重错误：无法加载检测模块。请检查 backend 目录依赖。\n错误信息: {e}")
            raise e
    return local_detector

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "没有文件"})
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"status": "error", "message": "未选择文件"})
    
    if file and allowed_file(file.filename):
        # 保存文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # 调用本地检测器
            det = get_detector()
            result = det.detect(file_path)
            
            response = {
                "status": "success",
                "detection_result": "AI合成声音" if result["is_fake"] else "真人声音",
                "confidence_percent": f"{result['confidence']*100:.1f}%",
                "is_fake": result["is_fake"],
                "fake_probability": f"{result['fake_probability']*100:.1f}%",
                "real_probability": f"{result['real_probability']*100:.1f}%",
                "label": result["label"]
            }
            return jsonify(response)
            
        except Exception as e:
            return jsonify({"status": "error", "message": f"检测出错: {str(e)}"})
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({"status": "error", "message": "不支持的文件格式"})

@app.route('/api/status', methods=['GET'])
def api_status():
    """让前端页面知道我们运行在本地模式"""
    return jsonify({
        "status": "online",
        "mode": "local",
        "message": "DefenderV 本地运行中"
    })

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    print("\n" + "="*50)
    print("DefenderV 本地版启动中...")
    print("请稍候，浏览器将自动打开...")
    print("="*50 + "\n")
    
    # 预加载模型（可选，如果你想启动时就加载好，取消下面注释）
    # get_detector()
    
    Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)