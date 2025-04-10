import torch
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Transformer.model_transformer import TransformerClassifier
from Transformer.predict_transformer import predict_sentiment

# ✅ 한글 글꼴 설정 (Windows 기준)
matplotlib.rc('font', family='Malgun Gothic')  # '맑은 고딕' 사용
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# ✅ 기본 설정
embedding_dim = 100
hidden_dim = 128
max_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드
checkpoint = torch.load(
    "../checkpoint/saved_transformer_model.pth",
    map_location=device
)
vocab = checkpoint['vocab']
label_names = checkpoint['label_names']

input_dim = len(vocab)
output_dim = len(label_names)
pad_idx = vocab["<pad>"]

model = TransformerClassifier(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# ✅ 대사 파일 경로
faust_path = "../../preprocess/TWO_extract_lines/faust_dialogues.txt"
#mephi_path = "../../preprocess/TWO_extract_lines/mephisto_dialogues.txt"

# ✅ 텍스트 로드
with open(faust_path, "r", encoding="utf-8") as f:
    faust_lines = [line.strip() for line in f if line.strip()]

# ✅ 감정 예측
results = []
for line in faust_lines:
    label, confidence, _ = predict_sentiment(
        line, model, vocab, label_names, max_len=max_len, device=device
    )
    results.append((line, label, confidence))

# ✅ 대표 감정 흐름 시각화
dominant_labels = [label for _, label, _ in results]
label_to_index = {label: i for i, label in enumerate(label_names)}
label_indices = [label_to_index[label] for label in dominant_labels]

plt.figure(figsize=(14, 5))
plt.plot(label_indices, marker='o', linestyle='-', alpha=0.7)
plt.xticks(range(0, len(label_indices), 50))
plt.yticks(range(len(label_names)), label_names)
plt.title("[Faust] 각 문장의 대표 감정 흐름")
plt.xlabel("대사 순서")
plt.ylabel("대표 감정")
plt.grid(True)
plt.tight_layout()
plt.show()

# 감정 → 숫자 매핑
label_to_index = {label: i for i, label in enumerate(label_names)}
index_to_label = {i: label for label, i in label_to_index.items()}

# 문장별 대표 감정 인덱스
label_indices = [label_to_index[label] for _, label, _ in results]

# 이동 평균 계산 (윈도우 크기 조정 가능)
window_size = 15
smoothed = np.convolve(label_indices, np.ones(window_size)/window_size, mode='valid')

# 시각화
plt.figure(figsize=(14, 5))
plt.plot(smoothed, color='darkorange', label='감정 이동 평균')
plt.yticks(range(len(label_names)), label_names)
plt.title(f"[Faust] 감정 흐름 (이동 평균 - window={window_size})")
plt.xlabel("문장 순서 (평균 시작 위치)")
plt.ylabel("대표 감정 (스무딩)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
