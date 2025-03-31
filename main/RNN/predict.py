import torch
import re

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def encode(tokens, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

def predict_sentiment(sentence, model, vocab, label_names, max_len=50, device='cpu'):
    model.eval()
    tokens = tokenize(sentence)
    indexed = encode(tokens, vocab)[:max_len]
    indexed += [vocab["<pad>"]] * (max_len - len(indexed))  # padding

    tensor = torch.LongTensor(indexed).unsqueeze(0).to(device)  # [1, seq_len]

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze()  # [output_dim]

    predicted_idx = torch.argmax(probs).item()
    return label_names[predicted_idx], probs[predicted_idx].item(), probs


# ✅ 직접 실행 테스트 (예: 파우스트 문장)
if __name__ == "__main__":
    from model import RNN
    import torch.nn as nn
    import torch.optim as optim
    import os

    # 모델 불러오기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("../checkpoint/saved_rnn_model.pth", map_location=device)
    vocab = checkpoint['vocab']
    label_names = checkpoint['label_names']

    embedding_dim = 100
    hidden_dim = 128
    max_len = 50
    input_dim = len(vocab)
    output_dim = len(label_names)
    pad_idx = vocab["<pad>"]

    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 예측할 문장
    sentence = "confess forth wander steps slight obstacle controlled— wizardsfoot threshold made"

    # 예측
    label, confidence, probs = predict_sentiment(sentence, model, vocab, label_names, max_len=max_len, device=device)

    # 출력
    print(f"\n💬 예측 문장: {sentence}")
    print(f"🔍 예측 감정: {label} (확률: {confidence:.4f})\n")

    print("📊 감정별 확률:")
    for i, p in enumerate(probs):
        print(f"{label_names[i]:<15}: {p.item():.4f}")
