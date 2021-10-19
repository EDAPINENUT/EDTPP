import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        scores = torch.clip(scores, min=1e-9, max=1e9)

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask==False, -1e9)
            else:
                scores = scores * mask + (mask.abs() <= 1e-9).float() * -1e9 #.masked_fill(mask.abs() <= 1e-9, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn