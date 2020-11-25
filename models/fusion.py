import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.configs import opts
from models import c3dModels,AclModels
import pdb

def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)


class Early_fusion(nn.Module):
    def __init__(self,args=None):
        super(Early_fusion,self).__init__()
        self.args = args
        if args.modality == 'acl_c3d':
            self.model1 = AclModels.CRNN(args)
            self.model2 = c3dModels.RNN(args) 
        if self.args.fusion_type == 'late' or self.args.fusion_type == 'mixture':
            self.fc_l = nn.Linear(2,1)
            self.fc_m = nn.Linear(3,1)
        if not opts.bidirectional:
            self.fc1 = nn.Linear(opts.rnn_hid_size*2,opts.rnn_hid_size)
            self.fc2 = nn.Linear(opts.rnn_hid_size,1)
        else:
            self.fc1 = nn.Linear(opts.rnn_hid_size*4,opts.rnn_hid_size)
            self.fc2 = nn.Linear(opts.rnn_hid_size,1)
       
    def forward(self, input1,input2):
        if self.args.fusion_type == 'early':
            x1 = self.model1(input1)
            x2 = self.model2(input2)
            x = torch.cat((x1,x2),dim=2)
            x = self.fc1(x)
            # x = torch.relu(x)
            x = self.fc2(x)
            return x.squeeze()
        elif self.args.fusion_type == 'late':
            x1 = self.model1(input1)
            x2 = self.model2(input2)
            x = torch.cat((x1,x2),dim=2)
            x = self.fc_l(x)
            return x.squeeze()
        elif self.args.fusion_type == 'mixture':
            feat1,x1 = self.model1(input1)
            feat2,x2 = self.model2(input2)
            feat = torch.cat((feat1,feat2),dim=2)
            x = self.fc1(feat)
            x = torch.Relu(x)
            x = self.fc2(x)
            x = torch.cat((x,x1,x2),dim=2)
            x = self.fc_m(x)
            return x.squeeze()

class mix_fusion(nn.Module):
    def __init__(self,args=None):
        super(mix_fusion,self).__init__()
        self.args = args
        if not opts.bidirectional:
            self.fc1 = nn.Linear(opts.rnn_hid_size*2,opts.rnn_hid_size)
            self.fc2 = nn.Linear(opts.rnn_hid_size,1)
        else:
            self.fc1 = nn.Linear(opts.rnn_hid_size*4,opts.rnn_hid_size*2)
            self.fc2 = nn.Linear(opts.rnn_hid_size*2,1)
       
    def forward(self,feat1,feat2,score1,score2):
        input = torch.cat((feat1,feat2),dim=2)
        """
        a1 = self.fc1(feat1)
        a2 = self.fc2(feat2)
        # pdb.set_trace()
        x = torch.cat((a1,a2),dim=2) # ,a1,a2  (score1.unsqueeze(-1),score2.unsqueeze(-1)
        x = self.fc_m(x)
        """
        x = self.fc1(input)
        x = self.fc2(x)

        return x.squeeze()