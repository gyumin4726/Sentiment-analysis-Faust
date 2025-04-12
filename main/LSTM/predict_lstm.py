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
