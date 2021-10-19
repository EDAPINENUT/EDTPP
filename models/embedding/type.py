import torch.nn as nn

class TypeEmbedding(nn.Embedding):
    def __init__(self, num_event_type, embed_dim, padding_idx):
        super().__init__(num_event_type, embed_dim, padding_idx=padding_idx)