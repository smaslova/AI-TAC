import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
import torch.utils.data
import matplotlib
import os
import sys
matplotlib.use('Agg')

import aitac
import plot_utils


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 81
batch_size = 100
learning_rate = 0.0001
num_filters = 99

#create output figure directory
model_name = sys.argv[1]
output_file_path = "../outputs/" + model_name + "/training/"
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

#file containing original model weights
original_model = sys.argv[5]

#load names of test set from original model
test_peaks = np.load("../outputs/" + original_model + "/training/test_OCR_names.npy")
idx = np.in1d(peak_names, test_peaks)

# split the data into training and test sets
eval_data, eval_labels, eval_names = x[idx, :, :], y[idx, :], peak_names[idx]
train_data, train_labels, train_names = x[~idx, :, :], y[~idx, :], peak_names[~idx]

# split the data into training and validation sets
train_data, valid_data, train_labels, valid_labels, train_names, valid_names = train_test_split(train_data, train_labels, train_names, test_size=0.1, random_state=40)

# Data loader
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(valid_data), torch.from_numpy(valid_labels))
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(eval_data), torch.from_numpy(eval_labels))
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)


# create model 
model = aitac.ConvNet(num_classes, num_filters).to(device)

# weights from model with 300 filters
checkpoint = torch.load("../models/" + original_model + ".ckpt")

#indices of filters in original model
filters = np.loadtxt('../data/filter_set99_index.txt')

#load new weights into model
checkpoint2 = model.state_dict()

#copy original model weights into new model
for i, (layer_name, layer_weights) in enumerate(checkpoint.items()):

    # for all first layer weights take subset
    if i<6:
        subset_weights = layer_weights[filters, ...]
        checkpoint2[layer_name] = subset_weights
    elif i==6: 
        subset_weights = layer_weights[:,filters, ...]
        checkpoint2[layer_name] = subset_weights

    #for remainder of layers take all weights
    else:
        checkpoint2[layer_name] = layer_weights

model.load_state_dict(checkpoint2)

#freeze first layer
def freeze_layer(layer):
 for param in layer.parameters():
  param.requires_grad = False

freeze_layer(model.layer1_conv)


# Loss and optimizer
criterion = aitac.pearson_loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# train model
model, best_loss_valid = aitac.train_model(train_loader, valid_loader, model, device, criterion,  optimizer, num_epochs, output_file_path)

# save the model checkpoint
torch.save(model.state_dict(), '../models/model' + model_name + '.ckpt')

#save the whole model
torch.save(model, '../models/model' + model_name + '.pth')

# Predict on test set
predictions, max_activations, max_act_index = aitac.test_model(eval_loader, model, device)


#-------------------------------------------#
#               Create Plots                #
#-------------------------------------------#

# plot the correlations histogram
# returns correlation measurement for every prediction-label pair
print("Creating plots...")

#plot_utils.plot_training_loss(training_loss, output_file_path)

correlations = plot_utils.plot_cors(eval_labels, predictions, output_file_path)

plot_utils.plot_corr_variance(eval_labels, correlations, output_file_path)

quantile_indx = plot_utils.plot_piechart(correlations, eval_labels, output_file_path)


#-------------------------------------------#
#                 Save Files                #
#-------------------------------------------#

#save predictions
np.save(output_file_path + "predictions.npy", predictions)

#save correlations
np.save(output_file_path + "correlations.npy", correlations)

#save max first layer activations
np.save(output_file_path + "max_activations.npy", max_activations)
np.save(output_file_path + "max_activation_index.npy", max_act_index)

#save test data set
np.save(output_file_path + "test_OCR_names.npy", eval_names)
