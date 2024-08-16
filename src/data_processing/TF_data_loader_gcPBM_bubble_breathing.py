import numpy as np
import torch
from torch.utils import data as D
import pickle

def seq2onehot(seq):
    window_size = 36
    matrix = np.zeros(shape = (4, window_size), dtype = np.uint8)
    for i, nt in enumerate(seq):
        if nt == "A":
            matrix[0][i] = 1
        elif nt == "G":
            matrix[1][i] = 1
        elif nt == "C":
            matrix[2][i] = 1
        elif nt == "T":
            matrix[3][i] = 1
        else:
            continue
    return matrix

class TF_data(D.Dataset):
    def __init__(self, data_path, partition):
        assert(partition in ['train', 'test', 'valid'])
        self.data_dict = pickle.load(open(data_path, 'rb'))[partition]
        self.id_list = list(self.data_dict)
        self.len = len(self.id_list)

    def __getitem__(self, index):
        seq_id = self.id_list[index]
        sample = self.data_dict[seq_id]
        seq = torch.from_numpy(seq2onehot(sample['seq'])).float()
        # load label as array and convert to tensor
        label = torch.from_numpy(np.array(sample['label'])).float()
        coord_feat = torch.from_numpy(sample['coords']).float()
        coordsq_feat = torch.from_numpy(sample['coordssquared']).float()
        flip_feat = torch.from_numpy(sample['flipping']).float()
        bubble_feat = torch.from_numpy(sample['bubble']).float()
        bio_feat = torch.stack([coord_feat, coordsq_feat, flip_feat], dim = 0).float()
        #bio_feat = torch.cat([seq, bio_feat], dim = 0).float()
        return seq, bio_feat, bubble_feat, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    Dset = TF_data(data_path='/home/blai/Breathing/data/Chipseq_data/seq_breathing_feat.pkl', partition = 'train')
    for feat in Dset:
        print(feat[0].size())
        print(feat[1].size())
        print(feat[2].size())      

         
            
