import torch.nn as nn
import torch
from torch.distributions import Categorical

class Intensity(nn.Module):
    def __init__(
            self,
            embed_dim,
            layer_num,
            event_type_num,
            intra_encoding,
            **kwargs
        ):
        super().__init__()
        self.layer_num = layer_num
        self.embed_dim = embed_dim
        self.event_type_num = event_type_num
        self.intra_encoding = intra_encoding
        
        if self.intra_encoding == False:
            self.mark_linear = nn.Linear(self.embed_dim, self.event_type_num + 1)
        else:
            self.mark_linear = nn.Linear(self.embed_dim, 1)

    def mark_logit(self, history_embedding, seq_types):
        history_embedding = history_embedding[:,:,0,...] if self.intra_encoding == False else history_embedding
        self.mark_logits = torch.log_softmax(self.mark_linear(history_embedding).squeeze(), dim=-1)  # (batch_size, seq_len, num_marks)
        mark_dist = Categorical(logits=self.mark_logits)
        mask = ~(seq_types == self.event_type_num)
        mark_dist = -(mark_dist.log_prob(seq_types) * mask).sum()
        return mark_dist
    
    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=100):
        raise NotImplementedError()