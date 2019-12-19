import numpy as np

import torch
import torch.utils.data
import matplotlib
import os
import sys
matplotlib.use('Agg')

import aitac
import plot_utils

import time
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_classes = 81
batch_size = 100
num_filters = 300

#create output figure directory
output_file_path = "../outputs/" + sys.argv[1] + "/"
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Load all data
x= np.load(sys.argv[2])
x = x.astype(np.float32)
y= np.load(sys.argv[3])
y= y[:,[ 0,  1,  2,  9, 14, 16, 11, 17]]
peak_names= np.load(sys.argv[4])

model_name = sys.argv[5]

# Data loader
dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

# load trained model
model = aitac.ConvNet(num_classes, num_filters).to(device)
checkpoint = torch.load('../models/' + model_name + '.ckpt')
model.load_state_dict(checkpoint)


# run predictions with full model on all data
mouse_predictions, max_activations, act_index = aitac.test_model(data_loader, model, device)

# convert predictions from mouse cell types to human cell types
map = np.genfromtxt("../human_data/mouse_human_celltypes.txt",dtype='str')
mouse_cell_types = np.genfromtxt("../data/cell_type_names.txt", dtype='str')
predictions, cell_names = plot_utils.mouse2human(mouse_predictions, mouse_cell_types, map)
print(cell_names)

#-------------------------------------------#
#               Create Plots                #
#-------------------------------------------#

# plot the correlations histogram
# returns correlation measurement for every prediction-label pair
print("Creating plots...")

correlations = plot_utils.plot_cors(y, predictions, output_file_path)

plot_utils.plot_corr_variance(y, correlations, output_file_path)

quantile_indx = plot_utils.plot_piechart(correlations, y, output_file_path)


#-------------------------------------------#
#                 Save Files                #
#-------------------------------------------#

#save predictions
np.save(output_file_path + "predictions.npy", predictions)

#save correlations
np.save(output_file_path + "correlations.npy", correlations)
