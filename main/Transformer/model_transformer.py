import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, emb_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, emb_dim]
        x = x + self.pe[:x.size(0)]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, pad_idx=0, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.2,
            batch_first=False  # 유지: permute 방식
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 💡 Transformer에서는 hidden 대신 첫 토큰 + 평균/최대 pooling 사용
        self.norm = nn.LayerNorm(embedding_dim * 3)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(embedding_dim * 3, output_dim)
        )

    def forward(self, text):
        # text: [batch_size, seq_len] → [seq_len, batch_size]
        text = text.permute(1, 0)
        embedded = self.embedding(text)  # [seq_len, batch_size, emb_dim]
        embedded = self.pos_encoder(embedded)

        transformer_output = self.transformer_encoder(embedded)  # [seq_len, batch_size, emb_dim]
        first_token = transformer_output[0]  # [batch_size, emb_dim]
        avg_emb = embedded.mean(dim=0)  # [batch_size, emb_dim]
        max_emb, _ = embedded.max(dim=0)  # [batch_size, emb_dim]

        combined = torch.cat((first_token, avg_emb, max_emb), dim=1)  # [batch_size, emb_dim*3]
        normed = self.norm(combined)
        return self.fc_layers(normed)
