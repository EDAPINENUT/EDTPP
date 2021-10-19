import torch
from torch import nn
import torch.nn.functional as F

class RelationDiscover(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer('relation_weight', nn.Parameter(torch.rand((2, self.embed_dim, self.embed_dim))))
    
    def forward(self, embed_matrix):

        out = torch.matmul(embed_matrix, self.relation_weight)
        out = torch.matmul(out, embed_matrix.transpose(0,1))
        out[0] += torch.eye(out.shape[1]).to(out) * 1e8

        return F.softmax(out.permute(1,2,0), dim=-1)