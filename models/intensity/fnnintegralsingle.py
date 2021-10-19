import torch.nn as nn
import torch
import numpy as np
from .intensity import Intensity
import torch.nn.functional as F

class NonnegativeLinear(nn.Linear):
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
        return F.linear(input, self.weight, self.bias)

class FNNIntegralSingle(Intensity):
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
            NonnegativeLinear(embed_dim, embed_dim) for _ in range(self.layer_num - 1)
        ])
        self.linear_time = NonnegativeLinear(1, embed_dim)
        self.final_layer = NonnegativeLinear(embed_dim, 1)
        self.linear_rnn = nn.Linear(embed_dim, embed_dim, bias=False)
        self.register_buffer('base_int', nn.Parameter(torch.rand(1)[0]))
  
    def mlp(self, tau, history_embedding=None):
        tau = tau.unsqueeze(-1)
        hidden = self.linear_time(tau)
        if history_embedding is not None:
            hidden += self.linear_rnn(history_embedding)
        hidden = torch.tanh(hidden)

        for linear in self.linear_layers:
            hidden = torch.tanh(linear(hidden))
        hidden = self.final_layer(hidden) + self.base_int.abs() * tau
        return hidden.squeeze(-1)

    def cdf(self, tau, h=None):
        output = self.mlp(tau, h)
        integral = F.softplus(output)
        return -torch.expm1(-integral)

    def log_cdf(self, tau, h=None):
        return torch.log(self.cdf(tau, h) + 1e-8)

    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding):
        history_embedding_single = history_embedding[:,:,0,:]
        batch_size, seq_len, embed_dim = history_embedding_single.shape
        tau = seq_dts.expand(batch_size, seq_len)

        tau.requires_grad_()
        output = self.mlp(tau, history_embedding_single)
        integral = F.softplus(output)
        intensity = torch.autograd.grad(integral, tau, torch.ones_like(output), create_graph=True)[0]
        log_loss = -torch.log(intensity + 1e-8) + integral
        mark_logits = self.mark_logit(history_embedding, seq_types)
        mask = (seq_types != self.event_type_num)
        log_loss = (log_loss * mask).sum()
        return log_loss, mark_logits
    

    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=100):
        history_embedding_single = history_embedding[:,:,0,:]
        batch_size, seq_len, embed_dim = history_embedding_single.shape
        time_step = max_t / resolution
        x_axis = torch.linspace(0, max_t, resolution).to(history_embedding_single)
        taus = x_axis[None,None,:].expand(batch_size, seq_len, -1).detach()
        taus.requires_grad_() 
        output = self.mlp(taus, history_embedding_single[:,:,None,:])
        integral = F.softplus(output)
        intensity = torch.autograd.grad(integral, taus, torch.ones_like(output), create_graph=True)[0]
        heights = intensity * torch.exp(-integral)
        expectation = (taus.squeeze() * heights * time_step).sum(dim=-1) #xf(x)dx

        return expectation