import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import pearsonr

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

# 텍스트 불러오기
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# 감정 분석
def analyze_emotions(lines):
    results = []
    for line in lines:
        label, confidence, _ = predict_sentiment(line, model, vocab, label_names, max_len=max_len, device=device)
        results.append((line, label, confidence))
    return results

# 감정 점수 이진화: joy/love → 1, sadness/anger/fear → -1
def compute_binary_emotion_scores(results):
    positive = {"joy", "love"}
    negative = {"sadness", "anger", "fear"}
    return [1 if label in positive else -1 for _, label, _ in results if label in positive.union(negative)]

# 이동 평균 smoothing
def smooth_sequence(seq, window_size=5):
    if len(seq) < window_size:
        return seq
    kernel = np.ones(window_size) / window_size
    return np.convolve(seq, kernel, mode='same')

# 시각화 함수
def plot_emotion_scene_continuous(f_scores, m_scores, title, save_path=None, window_size=5):
    f_smooth = smooth_sequence(f_scores, window_size)
    m_smooth = smooth_sequence(m_scores, window_size)

    min_len = min(len(f_smooth), len(m_smooth))
    f_trim = f_smooth[:min_len]
    m_trim = m_smooth[:min_len]

    x = np.linspace(0, 100, min_len)
    r, p = pearsonr(f_trim, m_trim)

    plt.figure(figsize=(10, 4))
    plt.plot(x, f_trim, label="Faust", color="blue")
    plt.plot(x, m_trim, label="Mephistopheles", color="red")
    plt.title(f"{title}\n(Pearson r={r:.3f}, p={p:.3f})")
    plt.xlabel("서사 진행 비율")
    plt.ylabel("감정 극성 점수")
    plt.ylim(-1, 1)  # ✅ 감정 점수 범위 조정
    plt.xticks(np.linspace(0, 100, 11), [f"{int(i)}%" for i in np.linspace(0, 100, 11)])
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"📊 Pearson 상관계수: r = {r:.3f}, p = {p:.3f}")

# 실행 함수
def process_scene(faust_path, mephi_path, title, save_name):
    faust_lines = load_lines(faust_path)
    mephi_lines = load_lines(mephi_path)
    results_faust = analyze_emotions(faust_lines)
    results_mephi = analyze_emotions(mephi_lines)
    faust_scores = compute_binary_emotion_scores(results_faust)
    mephi_scores = compute_binary_emotion_scores(results_mephi)

    plot_emotion_scene_continuous(
        faust_scores,
        mephi_scores,
        title=title,
        save_path=save_name,
        window_size=5
    )

# 장면별 실행
process_scene(
    faust_path="../faust_IV.txt",
    mephi_path="../mephi_IV.txt",
    title="계약 장면 감정 흐름 비교 (Transformer)",
    save_name="scene_contract_emotion_continuous_flow_transformer.png"
)

process_scene(
    faust_path="../faust_XXIII.txt",
    mephi_path="../mephi_XXIII.txt",
    title="Dreary Day 장면 감정 흐름 비교 (Transformer)",
    save_name="scene_drearyday_emotion_continuous_flow_transformer.png"
)

process_scene(
    faust_path="../faust_XXV.txt",
    mephi_path="../mephi_XXV.txt",
    title="감옥 장면 감정 흐름 비교 (Transformer)",
    save_name="scene_prison_emotion_continuous_flow_transformer.png"
)
