import torch.fft as fft
from torch import nn
import torch

class FNetBlock(nn.Module):
  def __init__(self, lag_step):
    super().__init__()
    self.lag_step = lag_step
    self.seq_to_emb = nn.Linear(self.lag_step * 2, 1)

  def forward(self, x, mask):
    x_ = (x[:,:,None,:] * mask[:,:,:,None])
    x_ = fft.rfft(fft.fft(x_, dim=-1).real, dim=-2).real
    x_, idx = torch.sort(x_, dim=-2)
    x_pos = x_[:,:,-self.lag_step:,:]
    x_neg = x_[:,:,:self.lag_step,:]
    seq = torch.cat([x_neg, x_pos], dim=-2)
    if seq.shape[-2] < self.lag_step * 2:
      seq = torch.cat([seq, torch.zeros_like(seq)[...,[0],:].repeat(1,1,self.lag_step * 2 - seq.shape[-2],1)],dim=-2)

    emb = self.seq_to_emb(seq.permute(0,1,3,2)).squeeze()
    return emb