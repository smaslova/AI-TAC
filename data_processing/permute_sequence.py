import numpy as np

x = np.load('../data/processed/one_hot_seqs.npy')
y = np.load('../data/processed/cell_type_array.npy')
peak_names = np.load('../data/processed/peak_names.npy')

#shuffle order of ocrs in peak intensity array
idx = np.random.permutation(y.shape[0])
y_shuffled_ocrs = y[idx, :]
peak_names_shuffled = peak_names[idx]
np.save("../data/shuffled_ocrs/peak_intensities.npy", y_shuffled_ocrs)
np.save("../data/shuffled_ocrs/one_hot_seqs.npy", x)
np.save("../data/shuffled_ocrs/peak_names.npy", peak_names_shuffled)

#shuffle sequences for each sample
x_shuffled = np.zeros(x.shape)
for sample in range(x.shape[0]):
    idx = np.random.permutation(np.arange(x.shape[2]))
    x_shuffled[sample, :, :] = x[np.ix_([sample], np.arange(4), idx)]
np.save("../data/shuffled_seqs/peak_intensities.npy", y)
np.save("../data/shuffled_seqs/one_hot_seqs.npy", x_shuffled)
np.save("../data/shuffled_seqs/peak_names.npy", peak_names)


#shuffle order of cells in peak intensity array
y_shuffled_cells = np.zeros(y.shape)
for peak in range(y.shape[0]):
    idx = np.random.permutation(np.arange(y.shape[1]))
    y_shuffled_cells[peak, :] = y[peak, idx]

np.save("../data/shuffled_cells/peak_intensities.npy", y_shuffled_cells)
np.save("../data/shuffled_cells/one_hot_seqs.npy", x)
np.save("../data/shuffled_cells/peak_names.npy", peak_names)
