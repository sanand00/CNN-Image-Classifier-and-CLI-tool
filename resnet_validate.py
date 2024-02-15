from alive_progress import alive_bar
import pickle
from filesplit.merge import Merge
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd

from datasets.dataset import Dataset
from engines.validation import binary_validate

with open('data/partition', 'rb') as file1, open('data/labels', 'rb') as file2:
        partition = pickle.load(file1)
        labels = pickle.load(file2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

merge = Merge(inputdir = 'trained_models/res_net_18',
              outputdir = 'trained_models/res_net_18',
              outputfilename = 'res_net_18.pt')
merge.merge()
net = torch.load('trained_models/res_net_18/res_net_18.pt')
os.remove('trained_models/res_net_18/res_net_18.pt')
net.eval()

batch_size = 32

test_set = Dataset(partition['test'], labels)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle = True)

validation_results = binary_validate(net, test_loader, device)

disp_cm = metrics.ConfusionMatrixDisplay(validation_results['confusion_matrix'])
disp_cm.plot()
disp_cm.figure_.savefig('trained_models/res_net_18/res_net_18_confusion_matrix.png')

disp_roc = validation_results['roc_curve']
disp_roc.plot()
disp_roc.figure_.savefig('trained_models/res_net_18/res_net_18_roc_curve.png')

pd.DataFrame(validation_results['report']).transpose().to_csv('trained_models/res_net_18/res_net_18_classification_report.csv')