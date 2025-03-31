import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False  # 유지: permute 방식
        )

        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, text):
        # text: [batch_size, sent_len] → RNN: [sent_len, batch_size]
        text = text.permute(1, 0)
        embedded = self.embedding(text)  # [sent_len, batch_size, emb_dim]
        output, hidden = self.rnn(embedded)

        # 양방향 → 마지막 layer의 forward & backward hidden 결합
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_dim * 2]
        # output: [seq_len, batch, hidden*2]
        context = torch.mean(output, dim=0)  # 평균 pooling
        normed = self.norm(context)
        return self.fc_layers(normed)

 
