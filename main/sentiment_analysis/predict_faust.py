import torch
from RNN.model import RNN
from RNN.predict import predict_sentiment

# ✅ 기본 설정
embedding_dim = 100
hidden_dim = 128
max_len = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 저장된 모델 로드
checkpoint = torch.load("saved_rnn_model.pth", map_location=device)
#checkpoint = torch.load("saved_lstm_model.pth", map_location=device)
#checkpoint = torch.load("saved_transformer_model.pth", map_location=device)
vocab = checkpoint['vocab']
label_names = checkpoint['label_names']

input_dim = len(vocab)
output_dim = len(label_names)
pad_idx = vocab["<pad>"]

model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# ✅ 파우스트 대사 파일 경로
faust_path = "../../preprocess/THREE_NLTK_stop_words/faust_dialogues_final.txt"

# ✅ 텍스트 불러오기
with open(faust_path, "r", encoding="utf-8") as f:
    faust_lines = [line.strip() for line in f if line.strip()]

# ✅ 감정 예측 실행
results = []
for line in faust_lines:
    label, confidence, _ = predict_sentiment(
        line, model, vocab, label_names, max_len=max_len, device=device
    )
    results.append((line, label, confidence))

# ✅ 상위 10개 예시 출력
print("\n📜 [Faust 감정 예측 결과 Top 10]")
for i, (line, label, conf) in enumerate(results[:10]):
    print(f"{i+1:>2}. \"{line[:40]}...\" → {label} ({conf:.4f})")

# ✅ CSV로 저장
with open("faust_emotion_results.csv", "w", encoding="utf-8") as f:
    f.write("text,predicted_emotion,confidence\n")
    for line, label, conf in results:
        f.write(f'"{line}","{label}",{conf:.4f}\n')

print("\n✅ 전체 파우스트 대사 감정 분석 및 저장 완료! → faust_emotion_results.csv")
