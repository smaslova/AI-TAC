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
model_name = sys.argv[1]
output_file_path = "../outputs/" + model_name + "/motifs/"
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    os.makedirs(directory)


# Load all data
x = np.load(sys.argv[2])
x = x.astype(np.float32)
y = np.load(sys.argv[3])
peak_names = np.load(sys.argv[4])

# Data loader
dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

# load trained model
model = aitac.ConvNet(num_classes, num_filters).to(device)
checkpoint = torch.load('../models/' + model_name + '.ckpt')
model.load_state_dict(checkpoint)

#copy trained model weights to motif extraction model
motif_model = aitac.motifCNN(model).to(device)
motif_model.load_state_dict(model.state_dict())

# run predictions with full model on all data
pred_full_model, max_activations, activation_idx = aitac.test_model(data_loader, model, device)
correlations = plot_utils.plot_cors(y, pred_full_model, output_file_path)


# find well predicted OCRs
idx = np.argwhere(np.asarray(correlations)>0.75).squeeze()

#get data subset for well predicted OCRs to run further test
x2 = x[idx, :, :]
y2 = y[idx, :]

dataset = torch.utils.data.TensorDataset(torch.from_numpy(x2), torch.from_numpy(y2))
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

# non-modified results for well-predicted OCRs only
pred_full_model2 = pred_full_model[idx,:]
correlations2 = plot_utils.plot_cors(y2, pred_full_model2, output_file_path)


# get first layer activations and predictions with leave-one-filter-out
start = time.time()
activations, predictions = aitac.get_motifs(data_loader, motif_model, device)
print(time.time()- start)

filt_corr, filt_infl, ave_filt_infl = plot_utils.plot_filt_corr(predictions, y2, correlations2, output_file_path)

infl, infl_by_OCR = plot_utils.plot_filt_infl(pred_full_model2, predictions, output_file_path)

pwm, act_ind, nseqs, activated_OCRs, n_activated_OCRs, OCR_matrix = plot_utils.get_memes(activations, x2, y2, output_file_path)


#save predictions
np.save(output_file_path + "filter_predictions.npy", predictions)
np.save(output_file_path + "predictions.npy", pred_full_model)

#save correlations
np.save(output_file_path + "correlations.npy", correlations)
np.save(output_file_path + "correaltions_per_filter.npy", filt_corr)

#overall influence:
np.save(output_file_path + "influence.npy", ave_filt_infl)
np.save(output_file_path + "influence_by_OCR.npy", filt_infl)

#influence by cell type:
np.save(output_file_path + "filter_cellwise_influence.npy", infl)
np.save(output_file_path + "cellwise_influence_by_OCR.npy", infl_by_OCR)

#other metrics
np.savetxt(output_file_path + "nseqs_per_filters.txt", nseqs)
np.save(output_file_path + "mean_OCR_activation.npy", activated_OCRs)
np.save(output_file_path + "n_activated_OCRs.npy",  n_activated_OCRs)
np.save(output_file_path + "OCR_matrix.npy", OCR_matrix)
