import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from RNN.model import RNN
from RNN.predict import predict_sentiment

# 한글 글꼴 설정
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 기본 설정
embedding_dim = 100
hidden_dim = 128
max_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
checkpoint = torch.load("../../checkpoint/saved_rnn_model.pth", map_location=device)
vocab = checkpoint['vocab']
label_names = checkpoint['label_names']  # ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
input_dim = len(vocab)
output_dim = len(label_names)
pad_idx = vocab["<pad>"]

model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# 대사 파일 경로
faust_path = "../../../preprocess/TWO_extract_lines/faust_dialogues.txt"
mephi_path = "../../../preprocess/TWO_extract_lines/mephisto_dialogues.txt"

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

# 시각화 함수
def plot_positive_negative_trend(results, title, save_path, window_size=10):
    """
    대사 순서에 따른 긍정 감정 비율의 흐름을 라인 차트로 시각화
    - results: [(line, label, confidence)] 형식
    - window_size: 이동 평균 계산에 사용할 윈도우 크기
    """
    # 긍정/부정 감정 정의
    positive_emotions = {"joy", "love"}
    negative_emotions = {"anger", "sadness", "fear"}
    
    # 1: positive, 0: negative, -1: surprise or 기타
    polarity_sequence = []
    for _, label, _ in results:
        if label in positive_emotions:
            polarity_sequence.append(1)
        elif label in negative_emotions:
            polarity_sequence.append(0)
        else:
            polarity_sequence.append(None)  # surprise or other
    
    # None 제거
    polarity_sequence = [p for p in polarity_sequence if p is not None]
    
    # 이동 평균 (긍정 감정 비율)
    moving_avg = []
    for i in range(len(polarity_sequence)):
        window = polarity_sequence[max(0, i - window_size + 1): i + 1]
        avg = sum(window) / len(window)
        moving_avg.append(avg)

    # 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(moving_avg, label="긍정 감정 비율", color="tab:blue")
    plt.ylim(0, 1)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel("대사 순서", fontsize=12)
    plt.ylabel("긍정 감정 비율 (이동 평균)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# 실행
faust_lines = load_lines(faust_path)
mephi_lines = load_lines(mephi_path)

results_faust = analyze_emotions(faust_lines)
results_mephi = analyze_emotions(mephi_lines)

plot_positive_negative_trend(
    results_faust,
    "Faust 감정 흐름 (긍정 감정 비율)",
    "faust_positive_trend.png",
    window_size=50
)

plot_positive_negative_trend(
    results_mephi,
    "Mephistopheles 감정 흐름 (긍정 감정 비율)",
    "mephi_positive_trend.png",
    window_size=50
)