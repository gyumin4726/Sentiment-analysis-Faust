import torch
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RNN.model import RNN
from RNN.predict import predict_sentiment

# ✅ 기본 설정
embedding_dim = 100
hidden_dim = 128
max_len = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드
checkpoint = torch.load(
    "../checkpoint/saved_rnn_model.pth",
    map_location=device
)
vocab = checkpoint['vocab']
label_names = checkpoint['label_names']

input_dim = len(vocab)
output_dim = len(label_names)
pad_idx = vocab["<pad>"]

model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# ✅ 대사 파일 경로
faust_path = "../../preprocess/THREE_NLTK_stop_words/faust_dialogues_final.txt"
#mephi_path = "../../preprocess/THREE_NLTK_stop_words/mephisto_dialogues_final.txt"

# ✅ 텍스트 로드
with open(faust_path, "r", encoding="utf-8") as f:
    faust_lines = [line.strip() for line in f if line.strip()]

# ✅ 감정 예측 실행
results = []
all_probs = []  # 각 대사의 감정 확률 저장

for line in faust_lines:
    label, confidence, probs = predict_sentiment(
        line, model, vocab, label_names, max_len=max_len, device=device
    )
    results.append((line, label, confidence))
    all_probs.append(probs.tolist())  # 감정별 확률 저장

# ✅ 감정 확률 DataFrame 생성
df = pd.DataFrame(all_probs, columns=label_names)

# ✅ 시각화
plt.figure(figsize=(14, 6))
for label in label_names:
    plt.plot(df[label], label=label)
plt.title("[Faust] 감정 흐름 시각화")
plt.xlabel("대사 순서")
plt.ylabel("감정 확률")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("\n✅ 전체 파우스트 감정 분석 및 시각화 완료! → faust_emotion_probs.csv")
