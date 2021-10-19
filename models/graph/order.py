import torch
from torch.autograd import Variable

class MaskBatch():
    "object for holding a batch of data with mask during training"
    def __init__(self, pad_index, device):
        self.pad = pad_index
        self.device = device

    def make_std_mask(self, tgt):
        "create a mask to hide padding and future input"
        tgt_mask = (tgt != self.pad).unsqueeze(-2).to(self.device)
        tgt_mask = tgt_mask & Variable(self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)).to(self.device)
        return tgt_mask
    
    def subsequent_mask(self, size):
        "mask out subsequent positions"
        atten_shape = (1,size,size)
        mask = torch.triu(torch.ones(atten_shape), diagonal=1)
        m = mask == 0
        return m
