import numpy as np
import torch
from sklearn import metrics

def binary_validate(net, test_loader, device):
    test_pred = []
    test_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, test_lab = data[0].to(device), data[1].to(device)
            outputs = net(images)
            test_pred += outputs.flatten().to(torch.device('cpu')).tolist()
            test_labels += test_lab.flatten().to(torch.device('cpu')).tolist()
    
    fpr, tpr, thresholds = metrics.roc_curve(test_labels,test_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    roc_curve = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    test_pred = np.array(test_pred).round(0).astype(int)
    test_labels = np.array(test_labels)
    cm = metrics.confusion_matrix(test_labels, test_pred, normalize='true')
    report = metrics.classification_report(test_labels, test_pred, output_dict=True)
    return({
        'auc': auc,
        'confusion_matrix':cm,
        'roc_curve': roc_curve,
        'report': report,
        'predictions': test_pred,
        'true': test_labels
    })