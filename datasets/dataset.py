import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, transform = True):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        ID = self.list_IDs[index]

        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y