import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.configs import opts

def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)


class RNN(nn.Module):
    def __init__(self,args=None):
        super(RNN,self).__init__()
        self.args = args
        self.rnn = nn.GRU(512, opts.rnn_hid_size, opts.rnn_layer, batch_first=True,
                          dropout=opts.rnn_dropout,bidirectional=opts.bidirectional)
        if not opts.bidirectional:
              self.fc = nn.Linear(opts.rnn_hid_size,1)
              
        else:
          self.fc = nn.Linear(opts.rnn_hid_size*2,1)
        
       
    def forward(self, input):
        feat, hx = self.rnn(input)
        if self.args.fusion_type == 'early':
            return feat
        else:
            x = self.fc(feat)
            x = x.squeeze()
            if self.args.fusion_type == 'late' or self.args.modality =='c3d':
                return x
            elif self.args.fusion_type == 'mixture':
                return feat, x


