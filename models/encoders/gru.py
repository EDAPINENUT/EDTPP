import torch
from torch import nn
from .encoder import Encoder


class GDGRU(Encoder):
    def __init__(
        self,
        event_type_num: int,
        input_size: int = 32, 
        hidden_size: int = 32, 
        layer_num: int = 1, 
        lag_step: int = 20,
        gumble_tau: float = 0.2,
        dropout: float = 0.1, 
        prior_graph=None,
        activation='tanh',
        intra_encoding: bool=True,
        device: str='cuda',
        **kwargs
    ):

        super().__init__(event_type_num, input_size, hidden_size, layer_num, lag_step, gumble_tau, dropout, activation, intra_encoding, device)

        self.recurrent_nn = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = layer_num, batch_first=True)
        
    def forward(self, seq_types, embedding, seq_positions_multivariate, edges=None):
        
        if self.is_intra_encoding == True:
            assert edges is not None, ('No relation given.')
            return self._intra_encoding(seq_types, embedding, seq_positions_multivariate, edges)
        else:
            return self._inter_encoding(seq_types, embedding, seq_positions_multivariate)

    def _inter_encoding(self, seq_types, embedding_univariate, seq_positions_multivariate):
        
        history_embedding = self._recurrent_hist_embedding(embedding_univariate[:,None,...])
        
        return history_embedding.permute(0,2,3,1).repeat(1,1,1,self.event_type_num + 1)

    def _intra_encoding(self, seq_types, embedding_multivariate, seq_positions_multivariate, edges):
        
        mask = self._generate_graph_mask(seq_types, edges, self.lag_step, self.event_type_num)[...,0]
        
        seq_length = seq_types.shape[1]
        
        history_embedding_multivariate = self._recurrent_hist_embedding(embedding_multivariate)
        
        history_embedding_univariate = self._multi_to_uni(history_embedding_multivariate, seq_positions_multivariate, seq_length)
        
        history_embedding_univariate_pad = self._pad_transform(history_embedding_univariate, self.lag_step, 0.0)  
                
        history_embeding = self._mask_nn(mask, history_embedding_univariate_pad)
        
        return history_embeding

                
            
        
