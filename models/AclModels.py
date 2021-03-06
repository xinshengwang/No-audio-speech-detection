import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.configs import opts

def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)


class CRNN(nn.Module):
    def __init__(self,args=None):
        super(CRNN,self).__init__()
        self.args=args
        self.Conv1 = nn.Conv1d(in_channels=opts.in_channel,out_channels=opts.hid_channel[0],
                              kernel_size=opts.kernel[0],stride=opts.stride[0],padding=opts.padding[0])
        self.Conv2 = nn.Conv1d(in_channels=opts.hid_channel[0],out_channels=opts.hid_channel[1],
                              kernel_size=opts.kernel[1],stride=opts.stride[1],padding=opts.padding[1])
        self.Conv3 = nn.Conv1d(in_channels=opts.hid_channel[1],out_channels=opts.out_channel,
                              kernel_size=opts.kernel[2],stride=opts.stride[2],padding=opts.padding[1])
        self.bnorm1 = nn.BatchNorm1d(opts.hid_channel[0])
        self.bnorm2 = nn.BatchNorm1d(opts.hid_channel[1])
        self.rnn = nn.GRU(opts.out_channel, opts.rnn_hid_size, opts.rnn_layer, batch_first=True,
                          dropout=opts.rnn_dropout,bidirectional=opts.bidirectional)
        if not opts.bidirectional:
              self.fc = nn.Linear(opts.rnn_hid_size,1)
              
        else:
          self.fc = nn.Linear(opts.rnn_hid_size*2,1)
        
       
    def forward(self, input):
        input = input.transpose(2,1)
        x = self.Conv1(input)
        x = self.bnorm1(x)
        x = self.Conv2(x)
        x = self.bnorm2(x)
        x = self.Conv3(x)
        x = x.transpose(2,1)
        feat, hx = self.rnn(x)
        if self.args.fusion_type == 'early' and self.args.modality !='acl':
            return feat
        else:
            x = self.fc(feat)
            x = x.squeeze()
            if self.args.fusion_type == 'late' or self.args.modality =='acl':
                return x
            elif self.args.fusion_type == 'mixture':
                return feat, x


