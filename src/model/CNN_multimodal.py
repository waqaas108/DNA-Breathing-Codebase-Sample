import torch
import torch.nn as nn
import numpy as np

class CNN_multimodal(torch.nn.Module):
    def __init__(self, num_target, input_channels = 4):
        super(CNN_multimodal, self).__init__()
        self.conv_kernel = 5
        self.pool_kernel = 3

        self.H1 = 64
        self.H2 = 128
        self.H3 = 256
        self.input_channels = input_channels

        self.seq_conv = nn.Sequential(
                nn.Conv1d(4, self.H1, kernel_size = self.conv_kernel),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
                nn.GELU(),

                nn.Conv1d(self.H1, self.H2, kernel_size = self.conv_kernel),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
                nn.GELU(),


                nn.Conv1d(self.H2, self.H3, kernel_size = self.conv_kernel),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
                nn.GELU(),

                nn.Conv1d(self.H3, 128, kernel_size = 1),
                nn.GELU(),
                )
        
        self.breath_conv = nn.Sequential(
                nn.Conv1d(3, self.H1, kernel_size = self.conv_kernel),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
                nn.GELU(),

                nn.Conv1d(self.H1, self.H2, kernel_size = self.conv_kernel),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
                nn.GELU(),


                nn.Conv1d(self.H2, self.H3, kernel_size = self.conv_kernel),
                nn.MaxPool1d(kernel_size = self.pool_kernel, stride=self.pool_kernel),
                nn.GELU(),

                nn.Conv1d(self.H3, 128, kernel_size = 1),
                nn.GELU(),
                )
                
        self.n_channels = 16

        self.classifier = nn.Sequential(
                nn.Linear(128 * self.n_channels * 2, num_target),
                nn.Sigmoid()
                )

    def forward(self,seq, bio_feat):
        print("Output before conv_nets: seq:", seq.size(), "bio_feat:", bio_feat.size())
        seq_out = self.seq_conv(seq)
        print("seq conv out:", seq_out.size())
        bio_out = self.breath_conv(bio_feat)
        print("bio_feat conv out:", bio_out.size())
        seq_out = seq_out.view(seq_out.size(0), 128 * self.n_channels)
        print("seq conv out:", seq_out.size())
        bio_out = bio_out.view(bio_out.size(0), 128 * self.n_channels)
        print("bio_feat conv out:", bio_out.size())
        out = torch.cat([seq_out, bio_out], dim = -1)
        print("multimodlal conv out:", out.size())
        out = self.classifier(out)
        print("final out:", out.size())
        
        return out

if __name__ == '__main__':
    import sys
    sys.path.append('../data_processing')
    import TF_data_loader
    from torch.utils.data import DataLoader
    model = CNN_multimodal(num_target = 130)
    dset = TF_data_loader.TF_data(data_path = '/home/blai/Breathing/data/Chipseq_data/seq_breathing_feat.pkl', partition = 'train')
    loader = DataLoader(dset, batch_size = 128, shuffle=True)
    for seq, bio_feat, label in loader:
        out = model(seq, bio_feat)
        print(out.size())
        #print(seq.size())
        #print(bio_feat.size())
        #print(label.squeeze().size())

    