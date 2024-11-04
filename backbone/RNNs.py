import torch
from torch import layout, nn, tensor
from torch.nn import functional as F
import numpy as np

class LSTMBlock(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=1024, layer=1, out_dim=2):
        super(LSTMBlock, self).__init__()
        self.LSTM = nn.LSTM(in_dim, hidden_size=hidden_dim, num_layers=layer)
        self.ops = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim,out_dim)
        )
    def forward(self, x):
        x,_ = self.LSTM(x)
        return self.ops(x)

class GRUBlock(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=1024, layer=1, out_dim=2):
        super(GRUBlock, self).__init__()
        self.GRU = nn.GRU(in_dim, hidden_size=hidden_dim, num_layers=layer)
        self.ops = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim,out_dim)
        )
    def forward(self, x):
        x,_ = self.GRU(x)
        return self.ops(x)

class BLSTMBlock(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=1024, layer=1, out_dim=2):
        super(BLSTMBlock, self).__init__()
        self.LSTM = nn.LSTM(in_dim, hidden_size=hidden_dim, num_layers=layer, bidirectional=True)
        self.ops = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim*2,out_dim)
        )
    def forward(self, x):
        x,_ = self.LSTM(x)
        return self.ops(x)

class BGRUBlock(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=1024, layer=1, out_dim=2):
        super(BGRUBlock, self).__init__()
        self.GRU = nn.GRU(in_dim, hidden_size=hidden_dim, num_layers=layer, bidirectional=True)
        self.ops = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim*2,out_dim)
        )
    def forward(self, x):
        x,_ = self.GRU(x)
        return self.ops(x)

if __name__ == '__main__':
    lstm = BLSTMBlock()
    # Seq, Batch(?), Feature
    tens = torch.randn(64,256,1024)
    res = lstm(tens)
    import pdb; pdb.set_trace()