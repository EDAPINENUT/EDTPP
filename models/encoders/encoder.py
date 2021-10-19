from models.embedding.sequence import SequenceEmbedding
from torch import nn
import torch
from models.graph.granger import *
from models.embedding import *
from models.lib.gumbel import *

class Encoder(nn.Module):
    def __init__(
        self,
        event_type_num: int,
        input_size: int = 32, 
        hidden_size: int = 32, 
        layer_num: int = 1, 
        lag_step: int = 20,
        gumble_tau: float = 0.2,
        dropout: float = 0.1,
        activation: str='tanh',
        intra_encoding: bool=True,
        device: str='cuda',
    ):  
        """
        Graph Discovery History Encoder

        Args:
            input_size: The input vector's dimension 
            hidden_size: The output vector's dimension 
            layer_num: The layer number in the Encoder
            lag_step: How many history hidden vector are used to compute the history embedding
            gumble_tau: The temperature for gumble sampling smoothness
            dropout: The dropout ratio
            activation: the activation function used
            intra_encoding: if use intra_type encoding
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.lag_step = lag_step
        self.event_type_num = event_type_num
        self.gumble_tau = gumble_tau
        self.dropout = dropout
        self.device = device
        self.is_intra_encoding = intra_encoding

        self.activation = getattr(torch, activation)
        self.bn = nn.LayerNorm((self.hidden_size, self.event_type_num + 1))
        self.drop = nn.Dropout(p=self.dropout)
        self.recurrent_nn = None
        self.register_buffer('mask_weight', nn.Parameter(torch.rand((self.lag_step, self.event_type_num + 1))))
        self.register_buffer('mask_bias', nn.Parameter(torch.zeros((1, self.event_type_num + 1))))


    def _get_pad_tuples(self, shape, pad_dim, value):
        _one_pos = 2 * (len(shape) - pad_dim) - 1
        _pad_pos = [0 for t in range(_one_pos + _one_pos % 2)]
        _pad_pos[_one_pos - 1] = value
        return tuple(_pad_pos)

    def _pad_transform(self, variable, lag_step, pad_value, pad_dim=1):
        new_variable = [variable]
        for i in range(1, lag_step):
            variable = self._shift_pad(variable, pad_value)
            new_variable.append(variable)
        return torch.stack(new_variable, dim=-1)    

    def _shift_pad(self, variable, pad_value):
        shape = variable.shape
        padding = torch.ones_like(variable) * pad_value 
        padding = padding[...,[0]]
        shift = variable[...,:-1]
        return torch.cat([padding, shift], dim = -1)
        
    def _generate_graph_mask(self, seq_types, edges, lag_step, pad_value):
        seq_types_pad = self._pad_transform(seq_types, lag_step, pad_value) 
        mask = mapping_granger_graph(seq_types_pad, edges)
        return mask
    
    def _multi_to_uni(self, history_embedding_multivariate, seq_positions_multivariate, seq_length):
        batch_size, event_type_num, seq_len, emb_dim = history_embedding_multivariate.shape
        seq_positions_multivariate[seq_positions_multivariate == -1] = seq_length

        seq_positions_multivariate = seq_positions_multivariate[:,:,:seq_len]
        history_embedding_multivariate = history_embedding_multivariate.reshape(batch_size, event_type_num * seq_len, emb_dim)
        seq_positions_multivariate = seq_positions_multivariate.reshape(batch_size, event_type_num * seq_len, -1)
        seq_positions_multivariate = seq_positions_multivariate.expand(batch_size, event_type_num * seq_len, emb_dim)

        container = torch.zeros(batch_size, event_type_num * seq_len + 1, emb_dim).to(history_embedding_multivariate.device)
        recover_embedding = container.scatter_(dim=1, index=seq_positions_multivariate, src=history_embedding_multivariate)
        recover_embedding[:,:-1,:][seq_positions_multivariate == seq_length] = 0.0
        return recover_embedding[:, :seq_length, ...]

    def _mask_nn(self, mask, hidden):
        history_embedding = torch.matmul(hidden, mask * self.mask_weight[None, None, ...]) + self.mask_bias[None, None, ...]
        history_embedding = self.drop(self.bn(history_embedding))
        return history_embedding

    def _recurrent_hist_embedding(self, embdedding):
        '''
         The intra-type history encoding using recurrent neural network 
         input: [batch_size, event_type_num, seq_len, emb_dim]
         output: [batch_size, event_type_num, seq_len, emb_dim]

         '''
        batch_size, event_type_num, seq_len, emb_dim = embdedding.shape

        embdedding = embdedding.reshape(batch_size * event_type_num, seq_len, emb_dim)

        if self.recurrent_nn == None:
            raise NotImplementedError()

        history_embdedding, _ = self.recurrent_nn(embdedding)
        history_embdedding = history_embdedding.reshape(batch_size, event_type_num, seq_len, emb_dim)

        return history_embdedding
    
    def _inter_encoding(self, seq_types, embedding_univariate, seq_positions_multivariate):
        raise NotImplementedError()

    def _intra_encoding(self, seq_types, embedding_univariate, seq_positions_multivariate):
        raise NotImplementedError()

    def _pad_dim(self, x, pad_value=0.0, dim=1):
        shape = list(x.shape)
        shape[dim] = 1
        return torch.cat([torch.ones(shape).to(x) * pad_value, x], dim=dim)