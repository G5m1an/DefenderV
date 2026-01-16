"""
DefenderV 音频深度伪造检测器
用于检测音频是真人声音还是AI合成声音
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
# 新增：使用 librosa 加载音频（支持 m4a、mp3 等）
import librosa
from typing import Union, Tuple, Dict

# 获取当前脚本所在目录 (backend/)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# 确保 backend 目录在 path 中，以便能导入 models
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# 导入模型
from models.decouple import SpeechTokenizer
from models.safeear import SafeEarLite, TransformerClassifier

class DefenderVDetector:
    """
    DefenderV 音频检测器
    """
    
    def __init__(
        self,
        # 修改：默认路径改为绝对路径，防止在不同目录下运行报错
        speech_tokenizer_path: str = None,
        student_model_path: str = None,
        device: str = None
    ):
        # 设置默认路径
        if speech_tokenizer_path is None:
            speech_tokenizer_path = os.path.join(BACKEND_DIR, "weights", "SpeechTokenizer.pt")
        if student_model_path is None:
            student_model_path = os.path.join(BACKEND_DIR, "weights", "student_model_weights.pth")

        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading models on: {self.device}")
        print(f"Tokenizer path: {speech_tokenizer_path}")
        print(f"Detector path:  {student_model_path}")
        
        # 加载模型
        self._load_models(speech_tokenizer_path, student_model_path)
        
        # 音频参数
        self.sample_rate = 16000
        self.max_len = 64600  # 约4秒
        
    def _load_models(self, speech_tokenizer_path: str, student_model_path: str):
        if not os.path.exists(speech_tokenizer_path):
            raise FileNotFoundError(f"找不到 SpeechTokenizer 权重文件: {speech_tokenizer_path}")
        if not os.path.exists(student_model_path):
            raise FileNotFoundError(f"找不到检测模型权重文件: {student_model_path}")

        print("正在加载 SpeechTokenizer...")
        self.speech_tokenizer = SpeechTokenizer(
            n_filters=64,
            dimension=1024,
            strides=[8, 5, 4, 2],
            lstm_layers=2,
            bidirectional=True,
            dilation_base=2,
            residual_kernel_size=3,
            n_residual_layers=1,
            activation="ELU",
            sample_rate=16000,
            n_q=8,
            semantic_dimension=768,
            codebook_size=1024
        )
        
        st_weights = torch.load(speech_tokenizer_path, map_location=self.device)
        self.speech_tokenizer.load_state_dict(st_weights)
        self.speech_tokenizer.to(self.device)
        self.speech_tokenizer.eval()
        
        print("正在加载学生检测模型...")
        self.detector = SafeEarLite(
            front=None,
            embedding_dim=1024,
            student_embedding_dim=512,
            dropout_rate=0.1,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            num_layers=2,
            num_heads=8,
            student_num_layers=1,
            student_num_heads=4,
            num_classes=2,
            positional_embedding='sine',
            mlp_ratio=1.0,
            student_mlp_ratio=1.0
        )
        
        detector_weights = torch.load(student_model_path, map_location=self.device)
        self.detector.load_state_dict(detector_weights)
        self.detector.to(self.device)
        self.detector.eval()
        
        print("模型加载完成!")
        
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        预处理音频文件
        """
        try:
            # 使用 librosa 加载音频，支持 m4a、mp3、wav 等
            # sr=16000 自动重采样
            waveform, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,    
                mono=False
            )
            
            # 转为 torch tensor
            waveform = torch.from_numpy(waveform).float()
            
            # 转为单声道
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # 确保是 [1, T] 形状
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            # 处理长度
            if waveform.shape[1] > self.max_len:
                waveform = waveform[:, :self.max_len]
            elif waveform.shape[1] < self.max_len:
                pad_len = self.max_len - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
                
        except Exception as e:
            raise RuntimeError(f"音频加载失败 ({audio_path}): {str(e)}")
        
        return waveform
    
    @torch.no_grad()
    def detect(self, audio_path: str) -> Dict:
        # 预处理音频
        waveform = self._preprocess_audio(audio_path)
        waveform = waveform.unsqueeze(0).to(self.device)  # [1, 1, T]
        
        # 使用SpeechTokenizer提取acoustic tokens
        quantized_list = self.speech_tokenizer.forward_feature(waveform, layers=list(range(8)))
        
        # 只使用acoustic tokens (层1-7)
        acoustic_tokens = quantized_list[1:]
        
        # 通过检测模型
        logits, _ = self.detector(acoustic_tokens)
        
        # 计算概率
        probs = torch.softmax(logits, dim=-1)
        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()
        
        is_fake = fake_prob > real_prob
        confidence = max(real_prob, fake_prob)
        
        return {
            "is_fake": is_fake,
            "confidence": confidence,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "label": "AI合成 (Fake)" if is_fake else "真人声音 (Real)"
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_files", nargs="+", help="Path to audio files")
    args = parser.parse_args()
    
    # 简单测试用
    det = DefenderVDetector()
    for f in args.audio_files:
        print(det.detect(f))