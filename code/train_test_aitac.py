import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
import torch.utils.data
import matplotlib
import os
import sys
matplotlib.use('Agg')

import cnn_model
import plot_utils


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 81
batch_size = 100
learning_rate = 0.001

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


# split the data into training and test sets
train_data, eval_data, train_labels, eval_labels, train_names, eval_names = train_test_split(x, y, peak_names, test_size=0.1, random_state=40)

# Data loader
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(eval_data), torch.from_numpy(eval_labels))
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)


# create model 
model = cnn_model.ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = cnn_model.pearson_loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train model
model = cnn_model.train_model(train_loader, eval_loader, model, device, criterion,  optimizer, num_epochs, output_file_path)

# save the model checkpoint
torch.save(model.state_dict(), '../models/model' + model_name + '.ckpt')

#save the whole model
torch.save(model, '../models/model' + model_name + '.pth')

# Predict on test set
predictions, max_activations, max_act_index = cnn_model.test_model(eval_loader, model, device)


#-------------------------------------------#
#               Create Plots                #
#-------------------------------------------#

# plot the correlations histogram
# returns correlation measurement for every prediction-label pair
print("Creating plots...")

plot_utils.plot_training_loss(training_loss, output_file_path)

correlations = plot_utils.plot_cors(eval_labels, predictions, output_file_path)

plot_utils.plot_corr_variance(eval_labels, correlations, output_file_path)

quantile_indx = plot_utils.plot_piechart(correlations, eval_labels, output_file_path)

plot_utils.plot_random_predictions(eval_labels, predictions, correlations, quantile_indx, eval_names, output_file_path)


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
