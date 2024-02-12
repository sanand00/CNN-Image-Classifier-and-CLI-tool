import numpy as np
import torch
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import math

def cnn_train(net, training_loader, optimiser, loss_function, n_samples, batch_size, max_epochs, device):
    """_summary_

    _extended_summary_

    Args:
        net (_type_): _description_
        training_loader (_type_): _description_
        optimiser (_type_): _description_
        loss_function (_type_): _description_
        n_samples (_type_): _description_
        batch_size (_type_): _description_
        max_epochs (_type_): _description_
        device (_type_): _description_
    """
    loss_array = []
    with alive_bar(int(math.ceil(n_samples / batch_size) * max_epochs), title='Processing', force_tty = True, length = 20) as bar:
        for _ in range(max_epochs):
            for data in training_loader:

                inputs, train_labels = data[0].to(device), data[1].to(device)

                optimiser.zero_grad()

                outputs = net(inputs)
                loss = loss_function(outputs,train_labels.reshape((-1,1)).float())
                loss.backward()
                optimiser.step()

                loss_array.append(loss.item())
                bar()
    
    loss_array = np.mean(np.array(loss_array).reshape(max_epochs,-1), axis=1)
    
    fig, ax = plt.subplots()
    ax.plot(range(max_epochs), np.mean(np.array(loss_array).reshape(max_epochs,-1), axis=1))
    
    return({
        'net': net,
        'conv_plot': fig,
        'loss_array': loss_array
    })