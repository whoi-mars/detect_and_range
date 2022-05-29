# Code adapted from https://github.com/locuslab/TCN

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

##################################################################
#                       TCN Building Blocks                      #
##################################################################

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class BranchedTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, branch=2):
        super(BranchedTemporalConvNet, self).__init__()
        self.branch = branch
        if self.branch < 1:
            raise ValueError("To branch the end of the network, you must specify a positive number of branches.")
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            if i == num_levels - 1:
                self.ends = nn.ModuleList([TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                    padding=(kernel_size-1) * dilation_size, dropout=dropout) for j in range(self.branch)])
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        y1 = self.network(x)
        outs = [self.ends[i](y1) for i in range(self.branch)]
        return outs

##################################################################
#                           TCN Models                           #
##################################################################

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        y1 = self.tcn(inputs)
        o = self.linear(y1[:,:,-1])
        o[:,1] = self.sigmoid(o[:,1])
        return o

class BranchedTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(BranchedTCN, self).__init__()
        if output_size % 2 != 0:
            raise ValueError("output_size must be divisible by 2.")
        self.btcn = BranchedTemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, branch=2)
        self.linear1 = nn.Linear(num_channels[-1], output_size // 2)
        self.linear2 = nn.Linear(num_channels[-1], output_size // 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        y1 = self.btcn(inputs)
        o1 = self.linear1(y1[0][:,:,-1])
        o2 = self.linear2(y1[1][:,:,-1])
        o2 = self.sigmoid(o2)
        return torch.cat((o1, o2), 1)