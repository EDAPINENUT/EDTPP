import torch
import torch.nn as nn
from models.embedding import *
import models
import math
from models.lib.gumbel import *
from models.graph.relation import *

class EDTPP(nn.Module):
    """
    Deep Neural Network based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        event_type_num: Number of event types 
        embed_dim: Dimension of the history embedding vector, including event type embedding and time embedding, whose dims are both embed_dim//2
        time_embed_type: Which type of time embedding is used, chosen in {"Linear", "Trigono"}
        encoder_type: Which Encoder to use, possible choices {"RNN", "GRU", "LSTM", "Attention", "FNet"}
        layer_num: The layer number in the Encoder
        lag_step: How many history hidden vector are used to compute the history embedding
        intensity_type: The intensity type for modelling the point process and maximize the log likelihood, chosen in {"Exp","LogNorm","FNN", "NormFlow"},
        prior_graph: The given prior graph of shape (event_type_num, event_type_num).  
                         If the hidden granger graph discovery module is not used, 
                         set prior graph as torch.ones(event_type_num, event_type_num)
    """
    def __init__(
        self,
        event_type_num: int,
        time_embed_type: str = 'Linear',
        embed_dim: int = 32,
        encoder_type: str = 'RNN',
        layer_num: int = 1,
        lag_step: int = 16, 
        intensity_type: str = 'FNN',
        prior_graph = None,
        gumbel_tau: float = 0.2,
        dropout: float = 0.1,
        device: str = 'cuda',
        intra_encoding: bool = True,
        l1_lambda: float = None,
        variational_discovery: bool = True,
        **kwargs
    ):
        super().__init__()
        self.event_type_num = event_type_num
        self.embed_dim = embed_dim
        assert self.embed_dim//2 != 0, ('embed_dim must be an event number.')
        self.time_embed_type = time_embed_type + 'TimeEmbedding'
        self.encoder_type = 'GD' + encoder_type
        self.intensity_type = intensity_type
        self.layer_num = layer_num
        self.lag_step = lag_step
        self.type_emb = models.embedding.TypeEmbedding(self.event_type_num + 1, self.embed_dim//2, self.event_type_num)
        self.time_emb = getattr(models.embedding, self.time_embed_type)(self.embed_dim//2)
        self.device = device
        self.gumbel_tau = gumbel_tau
        self.l1_lambda = l1_lambda
        self.is_intra_encoding = intra_encoding
        self.variational_discovery = variational_discovery

        self.encoder = getattr(models.encoders, self.encoder_type)(
            event_type_num=self.event_type_num,
            input_size=self.embed_dim, 
            hidden_size=self.embed_dim,
            layer_num=self.layer_num, 
            lag_step=self.lag_step,
            gumbel_tau=gumbel_tau,
            dropout=dropout,
            intra_encoding=intra_encoding,
            device=self.device,
            **kwargs
        )

        self.log_loss = getattr(models.intensity, self.intensity_type)(
            embed_dim=self.embed_dim,
            layer_num=self.layer_num,
            event_type_num=self.event_type_num,
            intra_encoding=intra_encoding,
            **kwargs
        )

        if prior_graph is not None:
            self.register_buffer('granger_graph', torch.FloatTensor(prior_graph))
            self.edges = self.granger_graph.bool().float().to(self.device)
        else:
            self.granger_graph = None

        if self.variational_discovery == False:
            self.relation_discover = RelationDiscover(self.embed_dim//2)
        else: 
            self.variational_relation_discover =  SequenceEmbedding(self.embed_dim, event_type_num)

    def _update_tau(self, decay_ratio, epoch):
        self.gumbel_tau = max(self.gumbel_tau * (decay_ratio ** epoch), 1e-6)
        return self.gumbel_tau

    def _event_embedding(self, seq_times, seq_types):
        """
        Calculate the embedding from the sequence of events.

        Args:
            seq_times: Time interval of events (batch_size, ... ,seq_len)
            seq_types: Sequence of event types (batch_size, ... ,seq_len)
        Returns:
            embedding: The embedding of time and event types (batch_size, ..., seq_len, embed_dim)

        """
        type_embedding = self.type_emb(seq_types) * math.sqrt(self.embed_dim//2)  #
        time_embedding = self.time_emb(seq_times)
        embedding = torch.cat([time_embedding, type_embedding], dim=-1)
        return embedding

    def get_embedding(self, seq_times, seq_types):
        """
        Get the embedding of given sequence of events.

        Args:
            seq_times: Time interval of events (batch_size, ... ,seq_len)
            seq_types: Sequence of event types (batch_size, ... ,seq_len)
        Returns:
            embedding: The embedding of time and event types. The first is time, second is event (batch_size, ..., seq_len, embed_dim//2)
        """
        embedding = self._event_embedding(seq_times, seq_types)
        return (embedding[...,:self.embed_dim//2], embedding[self.embed_dim//2:])


    def kl_categorical_uniform(self, preds, num_edge_types=1, add_const=False,
                           eps=1e-16):
        kl_div = preds * torch.log(preds + eps)
        if add_const:
            const = math.log(num_edge_types)
            kl_div += const
        return kl_div.sum()

    @property
    def hidden_adjacency(self):
        return self.prob[...,0].mean(dim=0)

    def l1_hidden_loss(self):
        try:
            hidden_batch_adj = self.prob[...,0].sum(dim=0)
            l1_loss = hidden_batch_adj.abs().mean()
            return l1_loss
        except:
            return torch.tensor(0)

    def _convert_prior_graph(self, generate_num):
        assert self.granger_graph is not None, ('You must give the prior hidden granger graph!')
        edges = self.edges
        edges = nn.ZeroPad2d((0,1,0,1))(edges)
        edges = torch.stack([edges, torch.zeros_like(edges)], dim=-1)
        edges = edges[None, :, :, :].repeat(generate_num, 1, 1, 1) #batch, event_num, event_num, 2
        self.prob = edges[:,:-1,:-1,:]
        return edges

    def _graph_discovery(self, generate_num):
        type_embedding = self.type_emb.weight[:-1]
        self.hidden_relation = self.relation_discover(type_embedding)[None, ...].repeat(generate_num, 1 , 1, 1)
        self.edges = gumbel_softmax(
            self.hidden_relation, 
            tau=self.gumbel_tau
            ).reshape(-1, self.event_type_num, self.event_type_num, 2)
        self.prob = my_softmax(self.hidden_relation, -1)
        edges = self.edges.permute(0,3,1,2).contiguous()
        edges = nn.ZeroPad2d((0,1,0,1))(edges).permute(0,2,3,1).contiguous() #pad the graph with a padded event
        return edges

    def _variational_graph_discovery(self, embedding_multivariate):
        self.hidden_relation = self.variational_relation_discover(embedding_multivariate)
        self.edges = gumbel_softmax(
            self.hidden_relation, 
            tau=self.gumbel_tau
            ).reshape(-1, self.event_type_num, self.event_type_num, 2)
        self.prob = my_softmax(self.hidden_relation, -1)
        edges = self.edges.permute(0,3,1,2).contiguous()
        edges = nn.ZeroPad2d((0,1,0,1))(edges).permute(0,2,3,1).contiguous() #pad the graph with a padded event
        return edges

    def forward(self, batch, *args):
        """
        Calculate the history embedding of the sequence.

        Args:
            batch
        Returns:
            history_embedding: The history embedding of each events, in agreement of order of seq_types (batch_size, ... ,seq_len)

        """
        seq_times, seq_types, seq_times_multivariate, seq_types_multivariate, seq_positions_multivariate = \
            batch.in_times, batch.in_types, batch.in_multi_times, batch.in_multi_types, batch.in_multi_positions

        if self.is_intra_encoding == True:
            return self._intra_encoding(seq_times, seq_types, seq_times_multivariate, seq_types_multivariate, seq_positions_multivariate)
        else:
            return self._inter_encoding(seq_times, seq_types, seq_times_multivariate, seq_types_multivariate, seq_positions_multivariate)

    def _intra_encoding(self, seq_times, seq_types, seq_times_multivariate, seq_types_multivariate, seq_positions_multivariate):
        assert seq_times_multivariate.shape == seq_types_multivariate.shape
        assert len(seq_times_multivariate.shape) == len(seq_types_multivariate.shape) == 3

        embedding_multivariate = self._event_embedding(seq_times_multivariate, seq_types_multivariate)

        generate_num = seq_times.shape[0]
        if self.granger_graph is not None:
            edges = self._convert_prior_graph(generate_num)
        elif self.variational_discovery == True:
            edges = self._variational_graph_discovery(embedding_multivariate)
        else:
            edges = self._graph_discovery(generate_num)

        history_embedding = self.encoder(seq_types, embedding_multivariate, seq_positions_multivariate, edges)
        self.history_embedding = history_embedding.permute(0,1,3,2)
        return self.history_embedding
    
    def _inter_encoding(self, seq_times, seq_types, seq_times_multivariate, seq_types_multivariate, seq_positions_multivariate):
        embedding_unitivariate = self._event_embedding(seq_times, seq_types)
        history_embedding = self.encoder(seq_types, embedding_unitivariate, seq_positions_multivariate)
        self.history_embedding = history_embedding.permute(0,1,3,2)
        return self.history_embedding

    def compute_loss(self, batch, *args, train=False):
        """
        Calculate the history embedding of the sequence.

        Args:
            seq_dts: Time intervals of events (batch_size, ... ,seq_len)
            seq_types: Types of events in the sequence (batch_size, ... ,seq_len)
            seq_onehots: Padded sequences of event time interval (batch_size, event_type_num, ... ,seq_len)

        Returns:
            loss

        """
        history_embedding = self.history_embedding #remove the final history embedding which is used for prediction
        seq_dts, seq_types, seq_onehots = batch.out_dts, batch.out_types, batch.out_onehots
        log_loss, mark_loss = self.log_loss(seq_dts, seq_types, seq_onehots, history_embedding)
        if train == False:
            return log_loss, mark_loss
        
        try:
            loss_kl = self.kl_categorical_uniform(self.encoder.prob)
        except:
            loss_kl = torch.tensor(0).to(log_loss)

        if self.l1_lambda is None:
            return log_loss + loss_kl + mark_loss
        else:
            l1_loss = self.l1_hidden_loss()
            return log_loss + loss_kl + mark_loss + l1_loss * self.l1_lambda

    def learn(self, batch, *args, epoch):
        self._update_tau(0.97, epoch)
        return self.forward(batch, *args)

    def evaluate(self, batch, *args):
        return self.forward(batch, *args)
    
    def predict_event_time(self, max_t, resolution=100):
        max_t = max_t.item() if torch.is_tensor(max_t) else max_t
        history_embedding = self.history_embedding
        return self.log_loss.inter_time_dist_pred(history_embedding, max_t, resolution)
    
    def predict_event_type(self):
        if hasattr(self.log_loss, 'mark_logits'):
            return self.log_loss.mark_logits
        else:
            return None