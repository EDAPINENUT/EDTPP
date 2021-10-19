import torch.nn as nn
import torch
import numpy as np
from .intensity import Intensity
import torch.nn.functional as F

class NonnegativeDiagonalLinear(nn.Linear):

    def __init__(self, event_type_num: int, embed_dim: int, in_size: int, out_size: int, bias: bool=True):
        super().__init__(in_size, out_size, bias=bias)
        self.n = event_type_num
        self.e = embed_dim

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # Make weight non-negative at initialization
        self.weight.data.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(0.0)
        mask = torch.zeros_like(self.weight.data) 
        for i in range(self.n):
            s, e = i * self.e, (i + 1) * self.e
            mask[s:e, s:e] = 1
        return F.linear(input, self.weight * mask, self.bias)

class DiagonalLinear(nn.Linear):

    def __init__(self, event_type_num: int, embed_dim: int, in_size: int, out_size: int, bias: bool=True):
        super().__init__(in_size, out_size, bias=bias)
        self.n = event_type_num
        self.e = embed_dim

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # Make weight non-negative at initialization
        self.weight.data.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        mask = torch.zeros_like(self.weight.data) 
        for i in range(self.n):
            s, e = i * self.e, (i + 1) * self.e
            mask[s:e, s:e] = 1
        return F.linear(input, self.weight * mask, self.bias)

class FNNIntegral(Intensity):
    def __init__(
            self,
            embed_dim,
            layer_num,
            event_type_num,
            **kwargs
        ):
        super().__init__(embed_dim, layer_num, event_type_num, **kwargs)

        self.hidden_embed_dim = embed_dim
        self.linear_layers = nn.ModuleList([
            NonnegativeDiagonalLinear((event_type_num + 1), 
            embed_dim, embed_dim * (event_type_num + 1), 
            embed_dim * (event_type_num + 1)) for _ in range(self.layer_num - 1)
        ])
        self.linear_time = NonnegativeDiagonalLinear((event_type_num + 1), 
                                                    embed_dim, (event_type_num + 1), 
                                                    embed_dim * (event_type_num + 1))

        self.final_layer = NonnegativeDiagonalLinear((event_type_num + 1), embed_dim, embed_dim * (event_type_num + 1), (event_type_num + 1))
        self.linear_rnn = DiagonalLinear((event_type_num + 1), 
                                        embed_dim, embed_dim * (event_type_num + 1), 
                                        embed_dim * (event_type_num + 1), bias=False)

        self.register_buffer('base_int', nn.Parameter(torch.rand(event_type_num + 1)))
  
    def mlp(self, tau, history_embedding=None):
        hidden = self.linear_time(tau)
        if history_embedding is not None:
            hidden += self.linear_rnn(history_embedding)
        hidden = torch.tanh(hidden)

        for linear in self.linear_layers:
            hidden = torch.tanh(linear(hidden))
        hidden = self.final_layer(hidden) + self.base_int.abs()[None, None, :] * tau
        return hidden

    def cdf(self, tau, h=None):
        output = self.mlp(tau, h)
        integral = F.softplus(output)
        return -torch.expm1(-integral)

    def log_cdf(self, tau, h=None):
        return torch.log(self.cdf(tau, h) + 1e-8)

    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding):
        batch_size, seq_len, event_type_num, embed_dim = history_embedding.shape
        tau = seq_dts[...,None].expand(batch_size, seq_len, event_type_num)
        history_embedding = history_embedding.reshape(batch_size, seq_len, event_type_num*embed_dim)
        tau.requires_grad_()
        output = self.mlp(tau, history_embedding)
        integral = F.softplus(output)
        intensity = torch.autograd.grad(integral, tau, torch.ones_like(output), create_graph=True)[0]
        one_hot_mask = seq_onehots
        log_loss = - torch.log(intensity[:,:,:-1] * one_hot_mask + 1e-8) + integral[:,:,:-1]
        mark_logits = self.mark_logit(history_embedding.reshape(batch_size, seq_len, event_type_num, embed_dim), seq_types)
        mask = (seq_types != self.event_type_num)
        log_loss = (log_loss * mask[:,:,None]).sum()
        return log_loss, mark_logits
    

    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=100):
        batch_size, seq_len, event_type_num, embed_dim = history_embedding.shape
        time_step = max_t / resolution
        x_axis = torch.linspace(0, max_t, resolution).to(history_embedding)
        taus = x_axis[None,None,:,None].expand(batch_size, seq_len, -1, event_type_num).detach()
        taus.requires_grad_()
        history_embedding = history_embedding.reshape(batch_size, seq_len, event_type_num*embed_dim)
        output = self.mlp(taus, history_embedding[:,:,None,:])
        integral = F.softplus(output)
        intensity = torch.autograd.grad(integral, taus, torch.ones_like(output), create_graph=True)[0]
        heights = intensity * torch.exp(-integral)
        expectation = (taus.squeeze() * heights * time_step).sum(dim=-2) #xf(x)dx
        return expectation