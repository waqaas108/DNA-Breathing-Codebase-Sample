import os
import sys
import glob
import importlib
import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import argparse
from torch.utils import data as D
sys.path.append('model')
import CNN
from data_processing import TF_data_loader_gcPBM_bubble_linear
import matplotlib.pyplot as plt

def train_model(train_params, epoch, model_path):
    model.train()
    torch.cuda.set_device(train_params['device'])
    running_loss = []
    pbar = tqdm(total=len(train_params['train_loader']))

    loss_fn = train_params['loss_fn']
    train_loader = train_params['train_loader']
    val_loader = train_params['val_loader']
    optimizer = train_params['optim']

    for seq, bubble_feat, label in train_loader:
        seq_batch, bubble_batch, label_batch = seq.cuda(), bubble_feat.cuda(), label.cuda()
        out_batch = model(torch.cat([seq_batch, bubble_batch.unsqueeze(0)], dim=1)).squeeze()
        loss = loss_fn(out_batch, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

        pbar.update()
        pbar.set_description(str(epoch) + '/' + str(train_params['training_epoch']) + '  ' +
                             'Loss: ' + str(np.mean(running_loss))[:6])

    tmp_loss = []
    for seq, bubble_feat, label in val_loader:
        val_batch, bubble_batch, label_batch = seq.cuda(), bubble_feat.cuda(), label.cuda()
        out_batch = model(torch.cat([val_batch, bubble_batch.unsqueeze(0)], dim=1)).squeeze()
        loss = loss_fn(out_batch, label_batch)
        tmp_loss.append(loss.item())
    pbar.set_description(str(epoch) + '/' + str(np.mean(tmp_loss))[:6])

    model_states = {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "loss": running_loss}
    torch.save(model_states, model_path)

    return np.mean(tmp_loss), model_states


if __name__ == "__main__":
    print('------------Starting TFChrom Training------------' + '\n')

    parser = argparse.ArgumentParser(description='Training script for TFChrom')
    parser.add_argument('--Device', type=int, help='CUDA device for training', default=0)
    parser.add_argument('--lr', type=float,  help='Learning rate for the optimizer',default=5e-4)
    parser.add_argument('--BatchSize', help='Size of the minibatch' ,type=int, default=128)
    parser.add_argument('--DataDir', help='Directory contains the training sequence and label', type=str)
    parser.add_argument('--ModelOut', help='Destination for saving the trained model' ,type=str)
    parser.add_argument('--NumTarget', help='Number of epigenomic features', type = int)
    parser.add_argument('--Epoch', help='Number of training epochs', type=int, default=100)
    parser.add_argument('--TF', help='Name of the TF', type=str, default='mad')
    args = parser.parse_args()

    if args.ModelOut is None:
        print('Error: Please provide model output directory')
        sys.exit(1)
    elif not os.path.isdir(args.ModelOut):
        os.makedirs(args.ModelOut)
    else:
        pass

    TFs = glob.glob(f'../data/gcPBM_data/{args.TF}.pkl', recursive=True)

    for TF in TFs:
        directory, filename = os.path.split(TF)
        TF_name = os.path.splitext(filename)[0]
        #preparing the data loader
        train_dset = TF_data_loader_gcPBM_bubble_linear.TF_data(TF, partition='train')
        valid_dset = TF_data_loader_gcPBM_bubble_linear.TF_data(TF, partition='test')

        train_loader = D.DataLoader(train_dset, batch_size=args.BatchSize, num_workers=20)
        val_loader = D.DataLoader(valid_dset, batch_size=args.BatchSize, num_workers=20)
        for kernel in [6,8,10]:
            for filters in [64]:
                for layers in [3]:
                    for pool in [1]:
                        # create the model
                        torch.cuda.set_device(args.Device)
                        model = CNN.CNN(kernel=kernel, pool=pool, filters=filters, layers=layers, input_channels=5).cuda()
                        # prepare the optimizer
                        optimizer = optim.SGD(model.parameters(), lr=args.lr)
                        loss_fn = torch.nn.MSELoss()

                        train_param_dict = {
                            'model': model,
                            'optim': optimizer,
                            'loss_fn': loss_fn,
                            'train_loader': train_loader,
                            'val_loader': val_loader,
                            'device': args.Device,
                            'training_epoch': args.Epoch
                        }

                        output_path = os.path.join("../trained_model/gcPBM/bubble_linear",
                                                str(layers), str(filters), str(kernel), str(pool))
                        os.makedirs(output_path, exist_ok=True)
                        output = os.path.join(output_path, TF_name)

                        print('DataDir: ' + args.DataDir)
                        print('TF: ' + TF)
                        print('filters: ' + str(filters))
                        print('layers: ' + str(layers))
                        print('kernel size: ' + str(kernel))
                        print('pooling size: ' + str(pool))
                        print('Number of Training Sequence: ' + str(len(train_dset)))
                        print('Batch Size: ' + str(args.BatchSize))
                        print('Learning Rate: ' + str(args.lr))
                        print('Number of Epochs: ' + str(args.Epoch))
                        print('Saving trained model at: ' + output)

                        patience = 5
                        best_val_loss = np.inf
                        best_epoch = 0
                        best_model = None
                        counter = 0

                        for epoch in range(1,args.Epoch + 1):
                            val_loss, current_model = train_model(train_params = train_param_dict, epoch = epoch ,model_path = output)
                            # val_loss = round(val_loss, 4)
                            if val_loss<best_val_loss:
                                best_val_loss = val_loss
                                best_epoch = epoch
                                counter = 0
                                best_model=current_model
                            counter += 1
                            if counter > patience:
                                break
                        print('Training stopped after' + str(best_epoch) + ' Epochs')
                        print('Saving models at' + output)

                        torch.save(best_model, output)
