import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib


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

# 실행
faust_lines = load_lines(faust_path)
mephi_lines = load_lines(mephi_path)

results_faust = analyze_emotions(faust_lines)
results_mephi = analyze_emotions(mephi_lines)

def plot_dual_positive_trend(results_faust, results_mephi, save_path, window_size=50):
    positive_emotions = {"joy", "love"}
    negative_emotions = {"anger", "sadness", "fear"}

    def get_polarity_seq(results):
        return [
            1 if label in positive_emotions else
            0 if label in negative_emotions else
            None
            for _, label, _ in results
        ]

    def moving_avg(seq, w):
        return [sum(seq[max(0, i-w+1):i+1])/len(seq[max(0, i-w+1):i+1]) for i in range(len(seq))]

    # 시퀀스 추출
    faust_seq = [x for x in get_polarity_seq(results_faust) if x is not None]
    mephi_seq = [x for x in get_polarity_seq(results_mephi) if x is not None]

    faust_avg = moving_avg(faust_seq, window_size)
    mephi_avg = moving_avg(mephi_seq, window_size)

 
    # 시각화
    plt.figure(figsize=(12, 5))
    plt.plot(faust_avg, label="Faust", color='blue')
    plt.plot(mephi_avg, label="Mephistopheles", color='red')
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    plt.title("Faust vs Mephistopheles 긍정 감정 비율 (Transformer))")
    plt.xlabel("대사 순서")
    plt.ylabel("긍정 감정 비율")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# 사용
plot_dual_positive_trend(
    results_faust,
    results_mephi,
    "faust_mephi_positive_trend_transformer.png",
    window_size=75
)
