import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from collections import defaultdict
from itertools import pairwise  # Python 3.10 이상

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from Transformer.model_transformer import TransformerClassifier
from Transformer.predict_transformer import predict_sentiment

# 한글 글꼴 설정
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 기본 설정
embedding_dim = 100
hidden_dim = 128
max_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
checkpoint = torch.load("../../checkpoint/saved_transformer_model.pth", map_location=device)
vocab = checkpoint['vocab']
label_names = checkpoint['label_names']  # ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
input_dim = len(vocab)
output_dim = len(label_names)
pad_idx = vocab["<pad>"]

model = TransformerClassifier(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 대사 파일 경로
faust_path = "../../../preprocess/TWO_extract_lines/faust_dialogues.txt"
mephi_path = "../../../preprocess/TWO_extract_lines/mephisto_dialogues.txt"

# 감정 라벨 정의 (surprise 제외)
label_names_filtered = [label for label in label_names if label != "surprise"]


# 텍스트 로드 함수
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# 감정 분석 함수
def analyze_emotions(lines):
    results = []
    for line in lines:
        label, confidence, _ = predict_sentiment(
            line, model, vocab, label_names, max_len=max_len, device=device
        )
        results.append((line, label, confidence))
    return results

# 감정 전이 행렬 계산 함수 수정
def compute_transition_matrix(results, filtered_labels):
    transitions = defaultdict(lambda: defaultdict(int))
    labels = [label for _, label, _ in results if label in filtered_labels]
    for a, b in pairwise(labels):
        transitions[a][b] += 1

    matrix = np.zeros((len(filtered_labels), len(filtered_labels)))
    for i, src in enumerate(filtered_labels):
        total = sum(transitions[src].values())
        for j, dst in enumerate(filtered_labels):
            matrix[i][j] = transitions[src][dst] / total if total > 0 else 0.0
    return matrix


# 히트맵 시각화
def plot_transition_heatmap(matrix, label_names, title, save_path, cmap="Blues"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=label_names, yticklabels=label_names,
                cbar_kws={"label": "전이 확률"})
    plt.title(title)
    plt.xlabel("다음 감정")
    plt.ylabel("이전 감정")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# 실행
faust_lines = load_lines(faust_path)
mephi_lines = load_lines(mephi_path)

results_faust = analyze_emotions(faust_lines)
results_mephi = analyze_emotions(mephi_lines)

matrix_faust = compute_transition_matrix(results_faust, label_names_filtered)
matrix_mephi = compute_transition_matrix(results_mephi, label_names_filtered)

# 실행
plot_transition_heatmap(matrix_faust, label_names_filtered, 
                        "Faust 감정 전이 히트맵 (Transformer)", 
                        "faust_transition_heatmap_Transformer.png",
                        cmap="Blues")

plot_transition_heatmap(matrix_mephi, label_names_filtered, 
                        "Mephistopheles 감정 전이 히트맵 (Transformer)", 
                        "mephi_transition_heatmap_Transformer.png",
                        cmap="Reds")
