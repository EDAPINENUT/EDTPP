import torch
import torch
from torch.distributions import beta
import torch.nn as nn
from .intensity import Intensity
from models.layers.normlayer import LayerNorm
import torch.distributions as D
from models.distributions import LogCauchy, MixtureSameFamily, TransformedDistribution

class LogCauchyMixtureDistribution(TransformedDistribution):
    """
    Mixture of LogCauchy distributions.

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        mu: mu of the component LogCauchy distributions, which is not all positive,
            shape (batch_size, seq_len, num_mix_components)
        r_sigma: sigma of the component LogCauchy distributions, which is not all positive,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time
        std_log_inter_time: Std of log-inter-event-times
    """
    def __init__(
        self,
        mu: torch.Tensor,
        r_sigma: torch.Tensor,
        log_weights: torch.Tensor,
        **kwargs
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = LogCauchy(mu=mu, sigma=r_sigma.exp())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        transforms = []
        super().__init__(GMM, transforms)

    def mean(self, max_t=40, resolution=100) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        Using Trapeze Numerical Integration.

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        log_weights = self.base_dist._mixture_distribution.logits
        from models.lib.utils import numerical_expectation
        mean = numerical_expectation(self.base_dist._component_distribution.interval_prob, log_weights, max_t, resolution)
        return mean


class LogCauMix(Intensity):
    """

    The distribution of the inter-event times given the history is modeled with a LogCauchy mixture distribution.

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
        linear = nn.ModuleList([nn.Linear(self.embed_dim, 3*num_mix_components, bias=True)])
        for i in range(layer_num - 1):
            linear.append(nn.Linear(3*num_mix_components, 3*num_mix_components, bias=True))
        self.linear = nn.Sequential(*linear)
        self.norm = LayerNorm(3*num_mix_components)

    def get_inter_time_dist(self, history_embedding: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the history_embedding.

        Args:
            history_embedding: history_embedding vector used to condition the distribution of each event,
                shape (baWtch_size, seq_len, embed_dim)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.norm(self.linear(history_embedding))  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        mu = raw_params[..., :self.num_mix_components]
        r_sigma = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_weights = torch.log_softmax(log_weights, dim=-1)

        return LogCauchyMixtureDistribution(
            mu=mu,
            r_sigma=r_sigma,
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

    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=100):
        inter_time_dist = self.get_inter_time_dist(history_embedding)
        return inter_time_dist.mean(resolution=resolution, max_t=max_t)


class LogCauMixSingle(Intensity):
    """

    The distribution of the inter-event times given the history is modeled with a LogCauchy mixture distribution.

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
        linear = nn.ModuleList([nn.Linear(self.embed_dim, 3*num_mix_components, bias=True)])
        for i in range(layer_num - 1):
            linear.append(nn.Linear(3*num_mix_components, 3*num_mix_components, bias=True))
        self.linear = nn.Sequential(*linear)
        self.norm = LayerNorm(3*num_mix_components)

    def get_inter_time_dist(self, history_embedding: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the history_embedding.

        Args:
            history_embedding: history_embedding vector used to condition the distribution of each event,
                shape (baWtch_size, seq_len, embed_dim)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.norm(self.linear(history_embedding))  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        mu = raw_params[..., :self.num_mix_components]
        r_sigma = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_weights = torch.log_softmax(log_weights, dim=-1)

        return LogCauchyMixtureDistribution(
            mu=mu,
            r_sigma=r_sigma,
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

    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=100):
        history_embedding_single = history_embedding[:,:,0,:]
        inter_time_dist = self.get_inter_time_dist(history_embedding_single)
        return inter_time_dist.mean()