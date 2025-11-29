import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerGenerator(nn.Module):
    def __init__(self, feat_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, feat_dim)

    def forward(self, src):
        src = self.pos_encoder(self.embedding(src))
        encoded = self.transformer_encoder(src)
        return self.decoder(encoded), encoded

class Discriminator(nn.Module):
    def __init__(self, feat_dim, window_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feat_dim * window_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

def contrastive_loss(feat1, feat2):
    feat1 = torch.nn.functional.normalize(feat1, dim=2)
    feat2 = torch.nn.functional.normalize(feat2, dim=2)
    return torch.mean((feat1 - feat2) ** 2)