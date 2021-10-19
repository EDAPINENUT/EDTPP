import torch
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily


class MixtureSameFamily(TorchMixtureSameFamily):
    def log_cdf(self, value):
        value = self._pad(value)
        log_cdf_x = self.component_distribution.log_cdf(value)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, value):
        value = self._pad(value)
        log_sf_x = self.component_distribution.log_survival_function(value)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)
    
    def log_prob(self, value):
        value = self._pad(value)
        log_prob_x = self.component_distribution.log_prob(value)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_prob_x + mix_logits, dim=-1)

    def log_intensity(self, value):
        return self.log_prob(value) - self.log_survival_function(value)