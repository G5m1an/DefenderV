"""
DefenderV API 服务器
提供 REST API 接口供前端调用
"""
import os
import uuid
import torch
import torchaudio
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# 导入检测器
from detector import DefenderVDetector

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局检测器实例（启动时加载模型）
detector = None

def get_detector():
    """懒加载检测器"""
    global detector
    if detector is None:
        print("Loading DefenderV detector...")
        detector = DefenderVDetector(
            speech_tokenizer_path="weights/SpeechTokenizer.pt",
            student_model_path="weights/student_model_weights.pth",
            device=None  # 自动选择
        )
        print("Detector loaded!")
    return detector


# 允许的音频格式
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """健康检查"""
    return jsonify({
        "status": "online",
        "service": "DefenderV Audio Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect": "POST - 检测音频文件",
            "/health": "GET - 健康检查"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_detector().device)
    })


@app.route('/detect', methods=['POST'])
def detect_audio():
    """
    检测上传的音频文件
    
    请求：
        - multipart/form-data
        - 字段名: audio
        
    响应：
        {
            "status": "success",
            "is_fake": true/false,
            "confidence": 0.95,
            "fake_probability": 0.95,
            "real_probability": 0.05,
            "label": "AI合成 (Fake)" / "真人声音 (Real)",
            "result": "fake" / "real"
        }
    """
    # 检查文件
    if 'audio' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No audio file provided"
        }), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({
            "status": "error", 
            "message": "No file selected"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # 保存临时文件
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # 检测音频
            det = get_detector()
            result = det.detect(file_path)
            
            # 返回结果
            response = {
                "status": "success",
                "is_fake": result["is_fake"],
                "confidence": round(result["confidence"], 4),
                "fake_probability": round(result["fake_probability"], 4),
                "real_probability": round(result["real_probability"], 4),
                "label": result["label"],
                "result": "fake" if result["is_fake"] else "real",
                # 前端友好的中文结果
                "detection_result": "AI合成声音" if result["is_fake"] else "真人声音",
                "confidence_percent": f"{result['confidence']*100:.1f}%"
            }
            
            return jsonify(response)
            
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/detect/url', methods=['POST'])
def detect_from_url():
    """
    从URL下载并检测音频（可选功能）
    
    请求JSON：
        {"url": "https://example.com/audio.wav"}
    """
    try:
        import requests
        from io import BytesIO
        
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "status": "error",
                "message": "URL is required"
            }), 400
        
        url = data['url']
        
        # 下载音频
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 保存临时文件
        unique_filename = f"{uuid.uuid4().hex}.wav"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        try:
            # 检测
            det = get_detector()
            result = det.detect(file_path)
            
            return jsonify({
                "status": "success",
                "is_fake": result["is_fake"],
                "confidence": round(result["confidence"], 4),
                "label": result["label"],
                "result": "fake" if result["is_fake"] else "real"
            })
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    # 预加载模型
    print("Pre-loading model...")
    get_detector()
    
    # 启动服务器
    print("\n" + "="*50)
    print("DefenderV API Server")
    print("="*50)
    print("Endpoints:")
    print("  POST /detect  - Detect audio file")
    print("  GET  /health  - Health check")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

