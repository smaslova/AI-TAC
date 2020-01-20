import numpy as np
from numpy import random
import pandas as pd

from sklearn.model_selection import KFold
import torch.utils.data
import matplotlib
import os
import sys
matplotlib.use('Agg')

import aitac
import plot_utils

#create output directory
output_file_path = "../outputs/valid10x10/"
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    print("Creating directory %s" % output_file_path)
    os.makedirs(directory)
else:
     print("Directory %s exists" % output_file_path)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 81
batch_size = 100
learning_rate = 0.001
num_filters = 300
run_num = sys.argv[1]

# Load all data
x = np.load(sys.argv[2])
x = x.astype(np.float32)
y = np.load(sys.argv[3])
y = y.astype(np.float32)
peak_names = np.load(sys.argv[4])


def cross_validate(x, y, peak_names, output_file_path):
    kf = KFold(n_splits=10, shuffle=True)

    pred_all = []
    corr_all = []
    peak_order = []
    for train_index, test_index in kf.split(x):
        train_data, eval_data = x[train_index, :, :], x[test_index, :, :]
        train_labels, eval_labels = y[train_index, :], y[test_index, :]
        train_names, eval_name = peak_names[train_index], peak_names[test_index]

        # Data loader
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

        eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(eval_data), torch.from_numpy(eval_labels))
        eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)


        # create model 
        model = aitac.ConvNet(num_classes, num_filters).to(device)

        # Loss and optimizer
        criterion = aitac.pearson_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # train model
        model, best_loss = aitac.train_model(train_loader, eval_loader, model, device, criterion,  optimizer, num_epochs, output_file_path)

        # Predict on test set
        predictions, max_activations, max_act_index = aitac.test_model(eval_loader, model, device)

        # plot the correlations histogram
        correlations = plot_utils.plot_cors(eval_labels, predictions, output_file_path)

        pred_all.append(predictions)
        corr_all.append(correlations)
        peak_order.append(eval_name)
    
    pred_all = np.vstack(pred_all)
    corr_all = np.hstack(corr_all)
    peak_order = np.hstack(peak_order)

    return pred_all, corr_all, peak_order


predictions, correlations, peak_order = cross_validate(x, y, peak_names, output_file_path)

np.save(output_file_path + "predictions_trial" + run_num + ".npy", predictions)
np.save(output_file_path + "correlations_trial" + run_num + ".npy", correlations)
np.save(output_file_path + "peak_order" + run_num + ".npy", peak_order)
