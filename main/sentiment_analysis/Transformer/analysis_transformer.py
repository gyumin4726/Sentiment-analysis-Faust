import torch
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter

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

# 시각화 함수
def plot_emotion_distribution(results, title, save_path):
    labels = [label for _, label, _ in results]
    counter = Counter(labels)
    emotion_order = label_names
    counts = [counter.get(emotion, 0) for emotion in emotion_order]

    sns.set(style="whitegrid", font="Malgun Gothic", palette="pastel")
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(x=emotion_order, y=counts, palette="coolwarm")

    for idx, val in enumerate(counts):
        plt.text(idx, val + 1, f"{val}", ha='center', va='bottom', fontsize=10)

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("감정", fontsize=13)
    plt.ylabel("등장 빈도", fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# 실행
faust_lines = load_lines(faust_path)
mephi_lines = load_lines(mephi_path)

results_faust = analyze_emotions(faust_lines)
results_mephi = analyze_emotions(mephi_lines)

plot_emotion_distribution(results_faust, "Faust 감정 분포 (Transformer)", "faust_Transformer.png")
plot_emotion_distribution(results_mephi, "Mephistopheles 감정 분포 (Transformer)", "mephi_Transformer.png")
