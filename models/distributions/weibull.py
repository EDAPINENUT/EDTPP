from math import e
import torch
from numbers import Number
import math
from torch.distributions import Distribution as TorchDistribution
from models.lib.utils import clamp_preserve_gradients
from torch.distributions.utils import broadcast_all

class Weibull(TorchDistribution):
    '''
    Weibull distribution for TorchDistribution class
    whose pdf: f(t) = eta * beta * (eta * t) ** (beta - 1) * exp(-(eta * t) ** beta)
          cdf: F(t) = 1 - exp(1 (eta * t) ** beta)
          intensity: lamda(t) = eta * beta *(eta * t) ** (beta - 1)
          sampling: u ~ U[0,1]  
                    ln(u) ~ exp(1) 
                    1/beta * ln(1 - beta / eta * ln(u) ) ~ Weib(eta, beta)
     '''
    def __init__(self, eta, beta, validate_args=False):
        assert (beta>=0).float().prod() > 0 and (eta>=0).float().prod() > 0, \
            ('Wrong parameter!')
        self.eta, self.beta = broadcast_all(eta, beta)
        if isinstance(eta, Number) and isinstance(beta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.eta.size()
        super(Weibull, self).__init__(batch_shape, validate_args=validate_args)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        eta, beta = self._clamp_params()
        return 1 - torch.exp(-(eta * value) ** beta)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            pass

    def log_cdf(self, value):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(value), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, value):
        eta, beta = self._clamp_params()
        return clamp_preserve_gradients(-(eta * value) ** beta, 1e-7, 1e7)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        eta, beta = self._clamp_params()
        return eta.log() + beta.log() + (beta - 1) * (eta * value).log() - (eta * value) ** beta


    def _clamp_params(self):
        eta = clamp_preserve_gradients(self.eta, 1e-7, 1e7)
        beta = clamp_preserve_gradients(self.beta, 1e-7, 1e7)
        return eta, beta