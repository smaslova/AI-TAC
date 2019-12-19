import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
import torch.utils.data
import matplotlib
import os
import sys
matplotlib.use('Agg')

import aitac_v2
import plot_utils


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 81
batch_size = 100
learning_rate = 0.001

#pre-trained model to use
model_name = sys.argv[1]

#create output figure directory
output_file_path = "../../outputs/" + model_name + "/layer2_motifs/"
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    print("Creating directory %s" % output_file_path)
    os.makedirs(directory)
else:
     print("Directory %s exists" % output_file_path)


# Load all data
x = np.load(sys.argv[2])
x = x.astype(np.float32)
y = np.load(sys.argv[3])
y = y.astype(np.float32)
peak_names = np.load(sys.argv[4])

#load names of test set from original model
test_peaks = np.load("../../outputs/" + model_name + "/training/test_OCR_names.npy")
idx = np.in1d(peak_names, test_peaks)

# split the data into training and test sets
eval_data, eval_labels, eval_names = x[idx, :, :], y[idx, :], peak_names[idx]
train_data, train_labels, train_names = x[~idx, :, :], y[~idx, :], peak_names[~idx]

# Data loader
eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(eval_data), torch.from_numpy(eval_labels))
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

#load trained model weights
checkpoint = torch.load("../../models/" + model_name + ".ckpt")

# initialize model 
model = aitac_v2.ConvNet(num_classes).to(device)
checkpoint2 = model.state_dict()

#copy original model weights into new model
for i, (layer_name, layer_weights) in enumerate(checkpoint.items()):
        new_name = list(checkpoint2.keys())[i]
        checkpoint2[new_name] = layer_weights

#load weights into new model
model.load_state_dict(checkpoint2)

#get layer 2 motifs
predictions, max_act_layer2, activations_layer2, act_index_layer2 = aitac_v2.test_model(eval_loader, model, device)

correlations = plot_utils.plot_cors(eval_labels, predictions, output_file_path)

#get PWMs of second layer motifs
plot_utils.get_memes2(activations_layer2, eval_data, eval_labels, output_file_path)

#save files
np.save(output_file_path + "second_layer_maximum_activations.npy", max_act_layer2)
np.save(output_file_path + "second_layer_maxact_index.npy", act_index_layer2)

