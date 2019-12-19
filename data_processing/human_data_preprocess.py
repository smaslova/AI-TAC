import numpy as np
import json
import os
import _pickle as pickle

import preprocess_utils

#parameters
num_chr = 23

#input files:
data_file = sys.argv[1] #path to bed file with human data
intensity_file = sys.argv[2] #path to file with normalized ATAC-Seq data
reference_genome = sys.argv[3] #path to reference genome fasta files "../human_data/hg19/"
output_directory = "../human_data/"


# read bed file with peak positions, and keep only entries with valid activity vectors
positions = preprocess_utils.read_bed(data_file)

# read reference genome fasta file into dictionary
if not os.path.exists('../human_data/hg19_chr_chr_dict.pickle'):
    chr_dict = preprocess_utils.read_fasta(reference_genome, num_chr)
    pickle.dump(chr_dict, open("../human_data/hg19_chr_dict.pickle", "wb"))

chr_dict = pickle.load(open("../human_data/hg19_chr_dict.pickle", "rb"))


one_hot_seqs, peak_seqs, invalid_ids, peak_names = preprocess_utils.get_sequences(positions, chr_dict, num_chr)

# remove invalid ids from intensities file so sequence/intensity files match
cell_type_array, peak_names2 = preprocess_utils.format_intensities(intensity_file, invalid_ids)

cell_type_array = cell_type_array.astype(np.float32)


#take one_hot_sequences of only peaks that have associated intensity values in cell_type_array
peak_ids = np.intersect1d(peak_names, peak_names2)

idx = np.isin(peak_names, peak_ids)
peak_names = peak_names[idx]
one_hot_seqs = one_hot_seqs[idx, :, :]
peak_seqs = peak_seqs[idx]


idx2 = np.isin(peak_names2, peak_ids)
peak_names2 = peak_names2[idx2]
cell_type_array = cell_type_array[idx2, :]

if np.sum(peak_names != peak_names2) > 0:
    print("Order of peaks not matching for sequences/intensities!")


# write to file
np.save(output_directory + 'one_hot_seqs.npy', one_hot_seqs)
np.save(output_directory + 'peak_names.npy', peak_names)
np.save(output_directory + 'peak_seqs.npy', peak_seqs)

with open(output_directory + 'invalid_ids.txt', 'w') as f:
    f.write(json.dumps(invalid_ids))

np.save(output_directory + 'cell_type_array.npy', cell_type_array)