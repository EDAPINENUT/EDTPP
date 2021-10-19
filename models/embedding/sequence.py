import torch.nn as nn
import torch
from models.lib.utils import encode_onehot 
import numpy as np
from torch.nn import functional as F

class SequenceEmbedding(nn.Module):
    def __init__(self, d_model, num_nodes, embedding_dim=8, conv_size=4):
        super(SequenceEmbedding, self).__init__()
        self.d_model = d_model
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.conv_size = conv_size
        self.conv1 = nn.Conv2d(d_model, d_model * 2, (1, conv_size), stride=1)
        self.conv2 = nn.Conv2d(d_model * 2, d_model * 4, (1, conv_size), stride=1)

        self.bn1 = nn.BatchNorm2d(d_model * 2)
        self.bn2 = nn.BatchNorm2d(d_model * 4)
        self.bn3 = nn.BatchNorm2d(d_model * 4)

        self.read_out = nn.AdaptiveMaxPool2d(embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim ** 2 * 2, 2)
        # Generate off-diagonal interaction graph
        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.register_buffer('rel_rec', torch.FloatTensor(rel_rec)) 
        self.register_buffer('rel_send', torch.FloatTensor(rel_send)) 
        self.register_buffer('self_loop', torch.stack([
            torch.eye(self.num_nodes, self.num_nodes),\
            torch.zeros(self.num_nodes, self.num_nodes)
            ], dim=-1))

    def forward(self, x):
        # batch_size, num_node, seq_len, channels
        x = x.permute(0,3,1,2).contiguous()
        # batch_size, channels, num_node, seq_len
        x = F.pad(self.conv1(x),(2,2,0,0))
        x = F.relu(x)
        x = self.bn1(x)
        x = F.pad(self.conv2(x),(2,2,0,0))
        x = F.relu(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 1, 3)
        batch_size, num_node, channels, seq_len = x.shape
        x = self.read_out(x).reshape(batch_size, num_node, -1)
        receivers = torch.matmul(self.rel_rec, x)
        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=-1)
        x = F.relu(self.fc_out(x)).reshape(-1, self.num_nodes, self.num_nodes, 2)
        x = self.self_loop * 1e7 + x
        return x