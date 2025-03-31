import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from dataset.load_dataset import get_dataloaders
from LSTM.model import LSTM
from LSTM.train import train, evaluate
from LSTM.predict import predict_sentiment

# ✅ 설정값
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 100
hidden_dim = 128
num_epochs = 5
batch_size = 64
max_len = 50

# ✅ 데이터 로딩
train_loader, test_loader, vocab, label_names = get_dataloaders(batch_size=batch_size, max_len=max_len)

# ✅ 모델 초기화
input_dim = len(vocab)
output_dim = len(label_names)
pad_idx = vocab["<pad>"]

model = LSTM(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# ✅ 학습 루프
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    print(f"\n🟢 Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")

# ✅ 최종 평가
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\n✅ Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ✅ 모델 저장
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'label_names': label_names
}, "../checkpoint/saved_lstm_model.pth")

print("💾 모델 저장 완료: ../checkpoint/saved_lstm_model.pth")
