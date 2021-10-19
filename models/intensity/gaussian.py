import torch
import torch
import torch.nn as nn
from torch.nn.functional import embedding, one_hot
from .intensity import Intensity
from models.layers.normlayer import LayerNorm
import torch.distributions as D
from models.distributions import Normal, MixtureSameFamily, TransformedDistribution
from models.lib.utils import clamp_preserve_gradients

class GaussianMixtureDistribution(TransformedDistribution):
    """
    Mixture of normal distributions.

    x ~ GaussianMixtureModel(locs, log_scales, log_weights)
    y = std_log_inter_time * x + mean_log_inter_time


    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_scales: Logarithms of scale parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        log_weights: Logarithms of mixing probabilities for the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time
        std_log_inter_time: Std of log-inter-event-times
    """
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        transforms = []
        
        super().__init__(GMM, transforms)

    def mean(self, *args) -> torch.Tensor:

        loc = self.base_dist._component_distribution.loc
        variance = self.base_dist._component_distribution.variance
        log_weights = self.base_dist._mixture_distribution.logits
        return (log_weights +  loc).logsumexp(-1).clamp(max=50).exp()


class GaussianMix(Intensity):
    """

    The distribution of the inter-event times given the history is modeled with a Gaussian mixture distribution.

    Args:
        embed_dim: Dimension of the embedding vectors
        layer_num: Number of layers for non-linear transformations
        event_type_num: Number of event type to consturct the number of intensity functions
        mean_log_inter_time: Average log-inter-event-time
        std_log_inter_time: Std of log-inter-event-times
        num_mix_components: Number of mixture components in the inter-event time distribution.
    """

    def __init__(
        self,
        embed_dim: int,
        layer_num: int,
        event_type_num: int,
        num_mix_components: int = 16,
        **kwargs
        ):
        super().__init__(embed_dim, layer_num, event_type_num, **kwargs)
        self.num_mix_components = num_mix_components
        linear = nn.ModuleList([nn.Linear(self.embed_dim, 3*self.num_mix_components, bias=True)])
        for i in range(layer_num - 1):
            linear.append(nn.Linear(3*self.num_mix_components, 3*self.num_mix_components, bias=True))
        self.linear = nn.Sequential(*linear)
        self.norm = LayerNorm(3*self.num_mix_components)

    def get_inter_time_dist(self, history_embedding: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the history_embedding.

        Args:
            history_embedding: history_embedding vector used to condition the distribution of each event,
                shape (batch_size, seq_len, event_type_num, embed_dim)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, event_type_num, seq_len)

        """
        raw_params = self.norm(self.linear(history_embedding))  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        return GaussianMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights
        )

    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        batch_size, seq_length, event_num, embed_dim = history_embedding.shape

        inter_time_dist = self.get_inter_time_dist(history_embedding)
        seq_dts = seq_dts.clamp(1e-10)
        seq_dts_expand = seq_dts[:,:,None].expand(batch_size, seq_length, event_num)
        # remove the final intensity which is the useless padded event type
        log_intensity = inter_time_dist.log_intensity(seq_dts_expand)[:,:,:-1]
        one_hot_mask = seq_onehots
        log_intensity = log_intensity * one_hot_mask
        int_intensity = inter_time_dist.int_intensity(seq_dts_expand)[:,:,:-1]
        log_loss = -log_intensity + int_intensity
        mask = (seq_types != self.event_type_num)
        log_loss = (log_loss * mask[:,:,None]).sum()
        mark_logits = self.mark_logit(history_embedding, seq_types)
        return log_loss, mark_logits

    def inter_time_dist_pred(self, history_embedding, *args):
        inter_time_dist = self.get_inter_time_dist(history_embedding)
        return inter_time_dist.mean()


class GaussianMixSingle(Intensity):
    """

    The distribution of the inter-event times given the history is modeled with a Gaussian mixture distribution.

    Args:
        embed_dim: Dimension of the embedding vectors
        layer_num: Number of layers for non-linear transformations
        event_type_num: Number of event type to consturct the number of intensity functions
        mean_log_inter_time: Average log-inter-event-time
        std_log_inter_time: Std of log-inter-event-times
        num_mix_components: Number of mixture components in the inter-event time distribution.
    """

    def __init__(
        self,
        embed_dim: int,
        layer_num: int,
        event_type_num: int,
        num_mix_components: int = 16,
        **kwargs
        ):
        super().__init__(embed_dim, layer_num, event_type_num, **kwargs)
        self.num_mix_components = num_mix_components
        linear = nn.ModuleList([nn.Linear(self.embed_dim, 3*self.num_mix_components, bias=True)])
        for i in range(layer_num - 1):
            linear.append(nn.Linear(3*self.num_mix_components, 3*self.num_mix_components, bias=True))
        self.linear = nn.Sequential(*linear)
        self.norm = LayerNorm(3*self.num_mix_components)

    def get_inter_time_dist(self, history_embedding: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the history_embedding.

        Args:
            history_embedding: history_embedding vector used to condition the distribution of each event,
                shape (batch_size, seq_len, event_type_num, embed_dim)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, event_type_num, seq_len)

        """
        raw_params = self.norm(self.linear(history_embedding))  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        locs = raw_params[..., :self.num_mix_components]
        log_scales = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        log_weights = torch.log_softmax(log_weights, dim=-1)
        return GaussianMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights
        )

    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        history_embedding_single = history_embedding[:,:,0,:]
        batch_size, seq_length, embed_dim = history_embedding_single.shape

        inter_time_dist = self.get_inter_time_dist(history_embedding_single)
        seq_dts = seq_dts.clamp(1e-10)
        # remove the final intensity which is the useless padded event type
        log_intensity = inter_time_dist.log_intensity(seq_dts)
        mask = (seq_types != self.event_type_num)
        int_intensity = inter_time_dist.int_intensity(seq_dts)
        log_loss = -log_intensity + int_intensity
        log_loss = (log_loss * mask).sum()
        mark_logits = self.mark_logit(history_embedding, seq_types)
        return log_loss, mark_logits

    def inter_time_dist_pred(self, history_embedding, *args):
        history_embedding_single = history_embedding[:,:,0,:]
        inter_time_dist = self.get_inter_time_dist(history_embedding_single)
        return inter_time_dist.mean()