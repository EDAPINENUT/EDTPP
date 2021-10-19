import torch.nn as nn
import torch

class TrigonoTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wt = nn.Linear(1, embed_dim // 2, bias=False)

    def forward(self, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        pe_sin = torch.sin(phi)
        pe_cos = torch.cos(phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe

class LinearTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wt = nn.Linear(1, embed_dim, bias=False)
    
    def forward(self, interval):
        emb = self.Wt(interval.unsqueeze(-1))
        return emb