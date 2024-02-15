from alive_progress import alive_bar
import pickle
from filesplit.split import Split
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 
import torch
import torch.optim as optim
import torch.nn as nn


from datasets.dataset import Dataset
from models.cnn_resnet import ResNet512
from engines.train import cnn_train
from engines.validation import binary_validate

def main():
    with open('data/partition', 'rb') as file1, open('data/labels', 'rb') as file2:
        partition = pickle.load(file1)
        labels = pickle.load(file2)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    train_params = {'batch_size':32,
                    'n_samples':len(partition['train']),
                    'max_epochs': 100,
                    'device': device}

    res_layers = [18, 18, 18, 18]

    optim_params = {'lr': 0.01,
                    'momentum': 0.9}


    training_set = Dataset(partition['train'], labels)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size = train_params['batch_size'], shuffle = True)

    test_set = Dataset(partition['test'], labels)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = train_params['batch_size'], shuffle = True)

    net = ResNet512(res_layers)
    net = net.to(device)

    criterion = nn.BCELoss()
    optimiser = optim.SGD(net.parameters(), **optim_params)

    results = cnn_train(net, training_loader, optimiser, criterion, **train_params)
    plot = results['conv_plot']

    torch.save(results['net'], 'trained_models/res_net_18/res_net_18.pt')
    split = Split(inputfile= 'trained_models/res_net_18/res_net_18.pt', outputdir =  'trained_models/res_net_18')
    os.remove('trained_models/res_net_18/res_net_18.pt')
    plot.savefig('trained_models/res_net_18/res_net_18_convergence_plot.png')

if __name__ == "__main__":
    main()