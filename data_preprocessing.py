import numpy as np
import torch
import pickle
from torchvision.transforms import v2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
train_batch = [unpickle("cifar10/data_batch_" + str(i)) for i in range(1,6)]
test_batch = unpickle("cifar10/test_batch")
for key in ["batch", "labels", "data", "filenames"]:
    for i in range(5):
        train_batch[i][key] = train_batch[i].pop(list(train_batch[i].keys())[0])
    test_batch[key] = test_batch.pop(list(test_batch.keys())[0])

partition = {'train': [],
             'test': []}
labels = {}

normalise = v2.Normalize([127.5, 127.5,127.5], [127.5, 127.5,127.5])

for i in range(5):
    batch = train_batch[i]
    rows = np.where((np.asarray(batch['labels'])==3) | (np.asarray(batch['labels'])==5))[0]
    for j in rows:
        img_tensor = normalise(torch.reshape(torch.tensor(batch['data'][j, ]).float(), (3, 32, 32)))
        name = batch['filenames'][j].decode("utf-8")[:-4]
        torch.save(img_tensor, 'data/' + name + '.pt')
        partition['train'].append(name)
        labels[str(name)] = int(batch['labels'][j]==3)


rows = np.where((np.asarray(test_batch['labels'])==3) | (np.asarray(test_batch['labels'])==5))[0]
for j in rows:
    img_tensor = normalise(torch.reshape(torch.tensor(test_batch['data'][j, ]).float(), (3, 32, 32)))
    name = test_batch['filenames'][j].decode("utf-8")[:-4]
    torch.save(img_tensor, 'data/' + name + '.pt')
    partition['test'].append(name)
    labels[str(name)] = int(test_batch['labels'][j]==3)
    
# Data Augmentation
transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomCrop(32, 4)
])

num = len(partition['train'])

for i in range(num):
    name = partition['train'][i]
    img = transform(torch.load('data/' + name + '.pt'))
    new_name = name + '_t'
    partition['train'].append(new_name)
    labels[str(new_name)] = labels[str(name)]
    torch.save(img, 'data/' + new_name + '.pt')
    
partition_file = open('data/partition', 'wb')
pickle.dump(partition, partition_file)
partition_file.close()

labels_file = open('data/labels', 'wb')
pickle.dump(labels, labels_file)
labels_file.close()