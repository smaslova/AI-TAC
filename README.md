# ![Overview of AI-TAC](figures/logo.png)
# Convolutional neural network to predict immune cell chromatin state
AI-TAC is a deep convoluional network for predicting mouse immune cell ATAC-seq signal from peak sequences using data from the Immunological Genome Project:  http://www.immgen.org/.

![Overview of AI-TAC](https://github.com/smaslova/AI-TAC/blob/master/figures/AI-TAC.png)


## Requirements
The code was written in python v3.6.3 and pytorch v0.4.0, and run on NVIDIA P100 Pascal GPUs.

## Tutorial
The required input is a bed file with ATAC-seq peak locations, the reference genome and a file with normalized peak heights.  The code for processing raw data is in data_processing/; for example, to convert the ImmGen mouse data set to one-hot encoded sequences and save in the data directory, run:

```python
python process_data.py "../data/ImmGenATAC1219.peak.filteredSM0.05.bed" "../data/ATAC_Data_Intensity_FilteredPeaksLogQuantile.txt" "../mm10/" "../data/"
```

The model can then be trained by running:
```python
python train_test_aitac.py model_name '../data/one_hot_seqs.npy' '../data/cell_type_array.npy' '../data/peak_names.npy'
```

To extract first layer motifs run:
```python
python -u extract_motifs.py model_name '../data/one_hot_seqs.npy' '../data/cell_type_array.npy' '../data/peak_names.npy'
```
