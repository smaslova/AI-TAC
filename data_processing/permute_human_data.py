import numpy as np

directory = "../human_data/"

x = np.load(directory + 'one_hot_seqs.npy')
y = np.load(directory + 'cell_type_array.npy')
peak_names = np.load(directory + 'peak_names.npy')

#shuffle order of ocrs in peak intensity array
idx = np.random.permutation(y.shape[0])
y_shuffled_ocrs = y[idx, :]
peak_names_shuffled = peak_names[idx]
np.save(directory + 'shuffled_ocrs/peak_intensities.npy', y_shuffled_ocrs)
np.save(directory + 'shuffled_ocrs/one_hot_seqs.npy', x)
np.save(directory + 'shuffled_ocrs/peak_names.npy', peak_names_shuffled)

#shuffle sequences for each sample
x_shuffled = np.zeros(x.shape)
for sample in range(x.shape[0]):
    idx = np.random.permutation(np.arange(x.shape[2]))
    x_shuffled[sample, :, :] = x[np.ix_([sample], np.arange(4), idx)]
np.save(directory + "shuffled_seqs/peak_intensities.npy", y)
np.save(directory + "shuffled_seqs/one_hot_seqs.npy", x_shuffled)
np.save(directory + "shuffled_seqs/peak_names.npy", peak_names)


#shuffle order of cells in peak intensity array
y_shuffled_cells = np.zeros(y.shape)
for peak in range(y.shape[0]):
    idx = np.random.permutation(np.arange(y.shape[1]))
    y_shuffled_cells[peak, :] = y[peak, idx]

np.save(directory + "shuffled_cells/peak_intensities.npy", y_shuffled_cells)
np.save(directory + "shuffled_cells/one_hot_seqs.npy", x)
np.save(directory + "shuffled_cells/peak_names.npy", peak_names)
