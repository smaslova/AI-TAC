import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#create output figure directory
model_name = sys.argv[1]
file_path = "../outputs/" + model_name + "/motifs/"
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    print("Creating directory %s" % output_file_path)
    os.makedirs(directory)
else:
     print("Directory %s exists" % output_file_path)

#files
meme_file = file_path + "filter_motifs_pwm.meme"
infl_file = file_path + "influence.npy"
cell_infl_file = file_path + "filter_cellwise_influence.npy"
nseqs_file = file_path + "nseqs_per_filters.txt"

cell_type_file = "../data/cell_type_names.npy"
tomtom_file = file_path + "tomtom/tomtom.tsv"
  
#load cell type names 
col_names = np.load(cell_type_file)
#filter names
rows = ['filter' + str(i) for i in range(300)]

#tomtom results
#load tomtom results
tomtom = pd.read_csv(tomtom_file, sep='\t')
tomtom = tomtom[:-3]

#read in motif TF information
#files downloaded from the CisBP database 
TF_info = pd.read_csv('../data/Mus_musculus_2018_09_27_3_44_pm/TF_Information_all_motifs.txt', sep='\t')

#leave only rows with q-value <0.05
tomtom = tomtom.loc[tomtom['q-value']<0.05]

#merge results with TF information 
tomtom = tomtom.merge(TF_info[['Motif_ID', 'TF_Name', 'Family_Name', 'TF_Status', 'Motif_Type']], how='left', left_on='Target_ID', right_on='Motif_ID')
tomtom = tomtom.drop(columns=['Motif_ID']) 

# get top 5 transcription factor matches for each motif
temp = tomtom[['Target_ID', 'TF_Name', 'Family_Name', 'TF_Status', 'Motif_Type']].drop_duplicates()
temp = temp.groupby(['Target_ID']).agg(lambda x : ', '.join(x.astype(str)))

# combine matrices
tomtom = tomtom.drop(['TF_Name', 'Family_Name', 'TF_Status', 'Motif_Type'], axis=1)
tomtom = tomtom.merge(temp, how='left', left_on=['Target_ID'], right_on=['Target_ID']).drop_duplicates()

tomtom['log_qval'] = -np.log(tomtom['q-value'])
tomtom['log_qval'].fillna(0, inplace=True)

#load influence score
infl = np.load(infl_file)

rows = ['filter' + str(i) for i in range(infl.shape[0])]
infl = pd.DataFrame(infl, columns=["Influence"])
infl.index = rows 
tomtom = tomtom.merge(infl, how='outer', left_on='Query_ID',right_index=True)

#load cell-wise influence for individual filters
cell_infl = np.load(cell_infl_file)
cell_infl = pd.DataFrame(cell_infl, index=rows, columns = col_names[:,1])
tomtom = tomtom.merge(cell_infl, how='outer', left_on='Query_ID', right_index=True)

# number of sequences per filter:
nseqs = pd.read_csv(nseqs_file, header = None, names = ['num_seqs'])
nseqs.index = rows

#add to data frame
tomtom = tomtom.merge(nseqs, how='outer', left_on='Query_ID', right_index=True)


#read in meme file
with open(meme_file) as fp:
    line = fp.readline()
    motifs=[]
    motif_names=[]
    while line:
        #determine length of next motif
        if line.split(" ")[0]=='MOTIF':
            #add motif number to separate array
            motif_names.append(line.split(" ")[1])
            
            #get length of motif
            line2=fp.readline().split(" ")
            motif_length = int(float(line2[5]))
            
            #read in motif 
            current_motif=np.zeros((19, 4))
            for i in range(motif_length):
                current_motif[i,:] = fp.readline().split("\t")
            
            motifs.append(current_motif)

        line = fp.readline()
        
    motifs = np.stack(motifs)
    motif_names = np.stack(motif_names)

fp.close()

#set background frequencies of nucleotides
bckgrnd = [0.25, 0.25, 0.25, 0.25]

#compute information content of each motif
info_content = []
position_ic = []
for i in range(motifs.shape[0]):
    length = motifs[i,:,:].shape[0]
    position_wise_ic = np.subtract(np.sum(np.multiply(motifs[i,:,:],np.log2(motifs[i,:,:] + 0.00000000001)), axis=1),np.sum(np.multiply(bckgrnd,np.log2(bckgrnd))))
                                        
    position_ic.append(position_wise_ic)
    ic = np.sum(position_wise_ic, axis=0)
    info_content.append(ic)
    
info_content = np.stack(info_content)
position_ic = np.stack(position_ic)

#length of motif with high info content
n_info = np.sum(position_ic>0.2, axis=1)

#"length of motif", i.e. difference between first and last informative base
ic_idx = pd.DataFrame(np.argwhere(position_ic>0.2), columns=['row', 'idx']).groupby('row')['idx'].apply(list)
motif_length = []
for row in ic_idx:
    motif_length.append(np.max(row)-np.min(row) + 1)

motif_length = np.stack(motif_length)

#create pandas data frame:
info_content_df = pd.DataFrame(data=[motif_names, info_content, n_info, pd.to_numeric(motif_length)]).T
info_content_df.columns=['Filter', 'IC', 'Num_Informative_Bases', 'Motif_Length']

tomtom = tomtom.merge(info_content_df, how='left', left_on='Query_ID', right_on='Filter')
tomtom = tomtom.drop(columns=['Filter'])
tomtom['IC'] = pd.to_numeric(tomtom['IC'])
tomtom['Num_Informative_Bases'] = pd.to_numeric(tomtom['Num_Informative_Bases'])

#compute consensus sequence
for filter in tomtom[pd.isnull(tomtom['Query_consensus'])]['Query_ID']:
    sequence = ''
    indx = np.argwhere(motif_names == filter)
    pwm = motifs[indx,:,:].squeeze()
    for j in range(pwm.shape[0]):
        letter = np.argmax(pwm[j, :])
        if(letter==0):
            sequence = sequence + 'A'
        if(letter==1):
            sequence = sequence + 'C'
        if(letter==2):
            sequence = sequence + 'G'
        if(letter==3):
            sequence = sequence + 'T'
            
    tomtom.loc[(tomtom['Query_ID']==filter), 'Query_consensus'] = sequence


tomtom.to_csv("../outputs/motif_summary.csv")
