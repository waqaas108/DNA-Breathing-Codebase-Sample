from unicodedata import bidirectional
from xml.dom.pulldom import IGNORABLE_WHITESPACE
import torch
import torch.nn as nn
import numpy as np

class CNN_LSTM(torch.nn.Module):
    def __init__(self, num_target, input_channels = 4):
        super(CNN_LSTM, self).__init__()
        self.conv_kernel = 5
        self.pool_kernel = 3

        self.H1 = 64
        self.H2 = 128
        self.H3 = 256
        self.input_channels = input_channels

        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, self.H1, kernel_size = self.conv_kernel),
            nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
            nn.GELU(),

            nn.Conv1d(self.H1, self.H2, kernel_size = self.conv_kernel),
            nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
            nn.GELU(),


            nn.Conv1d(self.H2, self.H3, kernel_size = self.conv_kernel),
            nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
            nn.GELU(),

            nn.Conv1d(self.H3, 128, kernel_size = 1),
            nn.ReLU(inplace = True),
            nn.GELU(),
            )
        self.biLSTM = nn.LSTM(input_size = 128, hidden_size = 512, num_layers = 3, batch_first = True, bidirectional = True, proj_size = 128)
        self.n_channels = 16

        self.classifier = nn.Sequential(
                nn.Linear(256 * self.n_channels, num_target),
                nn.Sigmoid()
                )

    def forward(self,x):
        print("Output before conv_nets:", x.size())
        out = self.conv_net(x)
        print("Output after conv_nets:", out.size())
        out = out.permute(0,2,1)
        print("Output after permute:", out.size())
        out,_ = self.biLSTM(out)
        print("Output after biLSTM:", out.size())
        out = out.reshape(out.size(0), 256 * self.n_channels)
        print("Output after reshape:", out.size())
        out = self.classifier(out)
        
        return out

if __name__ == '__main__':
    import sys
    sys.path.append('../data_processing')
    import TF_data_loader
    from torch.utils.data import DataLoader
    model = CNN_LSTM(num_target = 130)
    dset = TF_data_loader.TF_data(data_path = '/home/blai/Breathing/data/Chipseq_data/seq_breathing_feat.pkl', partition = 'train')
    loader = DataLoader(dset, batch_size = 128, shuffle=True)
    for seq, bio_feat, label in loader:
        out = model(seq)
        print(out.size())
        print(label.squeeze().size())

    