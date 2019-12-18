from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import numpy as np

# takes DNA sequence, outputs one-hot-encoded matrix with rows A, T, G, C
def one_hot_encoder(sequence):
    l = len(sequence)
    x = np.zeros((4,l),dtype = 'int8')
    for j, i in enumerate(sequence):
        if i == "A" or i == "a":
            x[0][j] = 1
        elif i == "T" or i == "t":
            x[1][j] = 1
        elif i == "G" or i == "g":
            x[2][j] = 1
        elif i == "C" or i == "c":
            x[3][j] = 1
        else:
            return "contains_N"
    return x

#read names and postions from bed file
def read_bed(filename):
    positions = defaultdict(list)
    with open(filename) as f:
        for line in f:
            name, chr, start, stop = line.split()
            positions[name].append((chr, int(start), int(stop)))

    return positions


# parse fasta file and turn into dictionary
def read_fasta(genome_dir, num_chr):
    chr_dict = dict()
    for chr in range(1, num_chr):
        chr_file_path = genome_dir + "chr{}.fa".format(chr)
        chr_dict.update(SeqIO.to_dict(SeqIO.parse(open(chr_file_path), 'fasta')))

    return chr_dict


#get sequences for peaks from reference genome
def get_sequences(positions, chr_dict, num_chr):
    one_hot_seqs = []
    peak_seqs = []
    invalid_ids = []
    peak_names = []

    target_chr = ['chr{}'.format(i) for i in range(1, num_chr)]

    for name in positions:
        for (chr, start, stop) in positions[name]:
            if chr in target_chr:
                chr_seq = chr_dict[chr].seq
                peak_seq = str(chr_seq)[start - 1:stop].lower()
                one_hot_seq = one_hot_encoder(peak_seq)

                if isinstance(one_hot_seq, np.ndarray):  # it is valid sequence
                    peak_names.append(name)
                    peak_seqs.append(peak_seq)
                    one_hot_seqs.append(one_hot_seq)
                else:
                    invalid_ids.append(name[20:])
            else:
                invalid_ids.append(name[20:])

    one_hot_seqs = np.stack(one_hot_seqs)
    peak_seqs = np.stack(peak_seqs)
    peak_names = np.stack(peak_names)

    return one_hot_seqs, peak_seqs, invalid_ids, peak_names


def format_intensities(intensity_file, invalid_ids):
    cell_type_array = []
    peak_names = []
    with open(intensity_file) as f:
        for i, line in enumerate(f):
            if i == 0: continue
            columns = line.split()
            peak_name = columns[0]
            if '\x1a' not in columns:
                cell_act = columns[1:]
                cell_type_array.append(cell_act)
                peak_names.append(peak_name)

    cell_type_array = np.stack(cell_type_array)
    peak_names = np.stack(peak_names)

    return cell_type_array, peak_names



