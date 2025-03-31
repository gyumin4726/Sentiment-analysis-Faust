from datasets import load_dataset
from collections import Counter
import re
import torch
from torch.utils.data import Dataset, DataLoader

# ------------------- Tokenization & Encoding -------------------
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_vocab(tokenized_texts, min_freq=2):
    word_counter = Counter(word for sent in tokenized_texts for word in sent)
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in word_counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def encode(tokens, vocab):
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

# ------------------- PyTorch Dataset -------------------
class EmotionDataset(Dataset):
    def __init__(self, X, y, vocab, max_len=50):
        self.X = [encode(x, vocab)[:max_len] for x in X]
        self.X = [x + [0] * (max_len - len(x)) for x in self.X]  # padding
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# ------------------- Data Loader Function -------------------
def get_dataloaders(batch_size=64, max_len=50):
    # ✅ 전체 감정 포함
    dataset = load_dataset("dair-ai/emotion", split="train")
    label_names = dataset.features['label'].names

    texts = [item['text'] for item in dataset]
    labels = [item['label'] for item in dataset]

    tokenized = [tokenize(t) for t in texts]
    vocab = build_vocab(tokenized)

    train_dataset = EmotionDataset(tokenized, labels, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ✅ 테스트셋도 동일하게 처리
    test_set = load_dataset("dair-ai/emotion", split="test")
    test_texts = [item['text'] for item in test_set]
    test_labels = [item['label'] for item in test_set]
    tokenized_test = [tokenize(t) for t in test_texts]

    test_dataset = EmotionDataset(tokenized_test, test_labels, vocab, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader, vocab, label_names
