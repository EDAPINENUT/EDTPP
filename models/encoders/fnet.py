from models.layers.fnetblock import FNetBlock
import torch
from torch import nn
from .encoder import Encoder
from models.layers import *
from models.graph.order import MaskBatch
from models.graph.granger import *

class GDFNet(Encoder):
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

        self.top_frequency = self.lag_step
        self.input_sublayer = SublayerConnection(size=self.hidden_size, dropout=self.dropout)
        self.output_sublayer = SublayerConnection(size=self.hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=self.hidden_size, d_ff=self.hidden_size * 4, dropout=dropout)
        fnetblock = [FNetBlock(self.top_frequency) for i in range(self.layer_num)]
        self.fnetblock = nn.Sequential(*fnetblock)
        self.batch_masker = MaskBatch(self.event_type_num, self.device)

    def forward(self, seq_types, embedding, seq_positions_multivariate, edges=None):

        if self.is_intra_encoding == True:
            assert edges is not None, ('No relation given.')
            return self._intra_encoding(seq_types, embedding, seq_positions_multivariate, edges)
        else:
            return self._inter_encoding(seq_types, embedding, seq_positions_multivariate)


    def _inter_encoding(self, seq_types, embedding_univariate, seq_positions_multivariate):
        
        history_embedding = self._inter_fnet_hist_embedding(embedding_univariate, seq_types)
        
        return history_embedding.unsqueeze(dim=-1).repeat(1,1,1,self.event_type_num + 1)


    def _intra_encoding(self, seq_types, embedding_multivariate, seq_positions_multivariate, edges):
        
        mask = self._generate_graph_mask(seq_types, edges, self.lag_step, self.event_type_num)[...,0]
        
        seq_length = seq_types.shape[1] 
        history_embedding_multivariate = self._intra_fnet_hist_embedding(embedding_multivariate, seq_positions_multivariate)

        history_embedding_univariate = self._multi_to_uni(history_embedding_multivariate, seq_positions_multivariate, seq_length)
        
        history_embedding_univariate = self._pad_transform(history_embedding_univariate, self.lag_step, 0.0)
        
        history_embedding = self._mask_nn(mask, history_embedding_univariate)
        
        return history_embedding

    
    def _inter_fnet_hist_embedding(self, embedding, seq_types):
        batch_size, seq_length, embed_dim = embedding.shape
        src_mask = self.batch_masker.make_std_mask(seq_types).reshape(-1, seq_length, seq_length)
        src_mask[:,:,0] = True
        x = embedding.reshape(-1, seq_length, embed_dim)

        for i in range(self.layer_num):
            x = self.input_sublayer(x, lambda _x: self.fnetblock[i].forward(_x, mask=src_mask))
            x = self.drop(self.output_sublayer(x, self.feed_forward))
        
        history_embedding = x.reshape(batch_size, seq_length, embed_dim)
        return history_embedding

    def _intra_fnet_hist_embedding(self, embedding_multivariate, seq_positions_multivariate):
        '''
         The intra-type history encoding using Fnet encoder
         input: [batch_size, event_type_num, seq_len, emb_dim]
         output: [batch_size, event_type_num, seq_len, emb_dim]

         '''
        max_uni_len = ((~(seq_positions_multivariate == -1)).sum(dim=-1) + 1).max()
        seq_positions_multivariate = seq_positions_multivariate[..., :max_uni_len]
        embedding_multivariate = embedding_multivariate[..., :max_uni_len, :]
        batch_size, event_num, seq_length, embed_dim = embedding_multivariate.shape
        src_mask = self.batch_masker.make_std_mask(seq_positions_multivariate).reshape(-1, seq_length, seq_length)
        x = embedding_multivariate.reshape(-1, seq_length, embed_dim)

        for i in range(self.layer_num):
            x = self.input_sublayer(x, lambda _x: self.fnetblock[i].forward(_x, mask=src_mask))
            x = self.drop(self.output_sublayer(x, self.feed_forward))
        
        history_embedding_multivariate = x.reshape(batch_size, event_num, seq_length, embed_dim)
        return history_embedding_multivariate
    
