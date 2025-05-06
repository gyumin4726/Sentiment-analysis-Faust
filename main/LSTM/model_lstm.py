import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=False  
        )
        
        self.norm = nn.LayerNorm(hidden_dim * 2 + embedding_dim * 2)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2 + embedding_dim * 2, output_dim)
        )

    def forward(self, text):
        # text: [batch_size, sent_len] → LSTM: [sent_len, batch_size]
        text = text.permute(1, 0)
        embedded = self.embedding(text)  # [seq_len, batch, emb_dim]

        output, (hidden, cell) = self.lstm(embedded)  # output: [seq_len, batch, hidden*2]
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden_dim*2]

        # 💡 평균/최대 임베딩: seq_len 차원 평균/최대
        avg_emb = embedded.mean(dim=0)  # [batch, emb_dim]
        max_emb, _ = embedded.max(dim=0)  # [batch, emb_dim]

        # 🔗 결합: LSTM hidden + 평균/최대 임베딩
        combined = torch.cat((hidden_cat, avg_emb, max_emb), dim=1)  # [batch, hidden_dim*2 + emb_dim*2]
        normed = self.norm(combined)
        return self.fc_layers(normed)
