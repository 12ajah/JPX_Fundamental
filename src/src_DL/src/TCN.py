# coding=utf-8
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

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
        self.relu1 = nn.ReLU() #nn.SiLU()
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

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.fc1 = nn.Linear(106, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.LeakyReLU1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #self.SiLU1 = nn.SiLU()
        
        self.fc2 = nn.Linear(128,128)
        self.dropout2 = nn.Dropout(dropout)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.LeakyReLU2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #self.SiLU2 = nn.SiLU()
        
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, inputs_ts, inputs_cs):
        y1 = self.tcn(inputs_ts)  # input should have dimension (N, C, L)
        y1 = torch.flatten(y1, start_dim=1)
        
        # concate:y1(N, 60) + inputs_cs(N, 46) -> (N, 106)
        y1 = torch.cat([y1, inputs_cs], dim=1)
        
        y1 = self.fc1(y1)
        y1 = self.batch_norm1(y1)
        y1 = self.LeakyReLU1(y1)
        #y1 = self.SiLU1(y1)
        
        y1 = self.dropout1(y1)
        
        y1 = self.fc2(y1)
        y1 = self.batch_norm2(y1)
        y1 = self.LeakyReLU2(y1)
        #y1 = self.SiLU2(y1)
        
        y1 = self.dropout2(y1)
        
        o = self.fc3(y1)
        return o