import numpy as np
import math
import random
import matplotlib
from matplotlib.colors import LogNorm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_loss(training_loss, output_file_path):
    #plot line plot of loss function per epoch
    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(len(training_loss)), training_loss)
    plt.xlabel('Epoch', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.savefig(output_file_path + "training_loss.svg")
    plt.close()


def plot_cors(obs, pred, output_file_path):
    correlations = []
    vars = []
    for i in range(len(pred)):
        var = np.var(obs[i, :])
        vars.append(var)
        x = np.corrcoef(pred[i, :], obs[i, :])[0, 1]
        correlations.append(x)

    weighted_cor = np.dot(correlations, vars) / np.sum(vars)
    print('weighted_cor is {}'.format(weighted_cor))

    nan_cors = [value for value in correlations if math.isnan(value)]
    print("number of NaN values: %d" % len(nan_cors))
    correlations = [value for value in correlations if not math.isnan(value)]

 
    plt.clf()
    plt.hist(correlations, bins=30)
    plt.axvline(np.mean(correlations), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(0, color='k', linestyle='solid', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {%f}" % np.mean(correlations))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("correlation")
    plt.savefig(output_file_path + "basset_cor_hist.svg")
    plt.close()

    return correlations


def plot_piechart(correlations, eval_labels, output_file_path):
    ind_collection = []
    Q0_idx = []
    Q1_idx = []
    Q2_idx = []
    Q3_idx = []
    Q4_idx = []
    Q5_idx = []
    ind_collection.append(Q0_idx)
    ind_collection.append(Q1_idx)
    ind_collection.append(Q2_idx)
    ind_collection.append(Q3_idx)
    ind_collection.append(Q4_idx)
    ind_collection.append(Q5_idx)
    
    for i, x in enumerate(correlations):
        if x > 0.75:
            Q1_idx.append(i)
            if x > 0.9:
                Q0_idx.append(i)
        elif x > 0.5 and x <= 0.75:
            Q2_idx.append(i)
        elif x > 0.25 and x <= 0.5:
            Q3_idx.append(i)
        elif x > 0 and x <= 0.25:
            Q4_idx.append(i)
        elif x < 0:
            Q5_idx.append(i)
        
    # pie chart of correlations distribution
    pie_labels = "cor>0.75", "0.5<cor<0.75", "0.25<cor<0.5", "0<cor<0.25", 'cor<0'
    sizes = [len(Q1_idx), len(Q2_idx), len(Q3_idx), len(Q4_idx), len(Q5_idx)]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red']
    explode = (0.1, 0, 0, 0, 0)  # explode 1st slice
    plt.pie(sizes, explode=explode, labels=pie_labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('correlation_pie')
    plt.savefig(output_file_path + "basset_cor_pie.svg")
    plt.close()
    
    # Plot relation between SD/IQR vs prediction performance
    Q0 = eval_labels[Q0_idx]
    Q1 = eval_labels[Q1_idx]
    Q2 = eval_labels[Q2_idx]
    Q3 = eval_labels[Q3_idx]
    Q4 = eval_labels[Q4_idx]
    Q5 = eval_labels[Q5_idx]
    
    sd1 = np.std(Q1, axis=1)
    sd2 = np.std(Q2, axis=1)
    sd3 = np.std(Q3, axis=1)
    sd4 = np.std(Q4, axis=1)
    sd5 = np.std(Q5, axis=1)
    
    qr1 = np.percentile(Q1, 75, axis=1) - np.percentile(Q1, 25, axis=1)
    qr2 = np.percentile(Q2, 75, axis=1) - np.percentile(Q2, 25, axis=1)
    qr3 = np.percentile(Q3, 75, axis=1) - np.percentile(Q3, 25, axis=1)
    qr4 = np.percentile(Q4, 75, axis=1) - np.percentile(Q4, 25, axis=1)
    qr5 = np.percentile(Q5, 75, axis=1) - np.percentile(Q5, 25, axis=1)
    
    mean_sds = []
    mean_sd1 = np.mean(sd1)
    mean_sd2 = np.mean(sd2)
    mean_sd3 = np.mean(sd3)
    mean_sd4 = np.mean(sd4)
    mean_sd5 = np.mean(sd5)
    mean_sds.append(mean_sd1)
    mean_sds.append(mean_sd2)
    mean_sds.append(mean_sd3)
    mean_sds.append(mean_sd4)
    mean_sds.append(mean_sd5)
    print('1st sd: {0}, 2nd sd: {1}, 3rd sd: {2}, 4th sd: {3}'.format(mean_sd1, mean_sd2, mean_sd3, mean_sd4))
    
    mean_qrs = []
    mean_qr1 = np.mean(qr1)
    mean_qr2 = np.mean(qr2)
    mean_qr3 = np.mean(qr3)
    mean_qr4 = np.mean(qr4)
    mean_qr5 = np.mean(qr5)
    mean_qrs.append(mean_qr1)
    mean_qrs.append(mean_qr2)
    mean_qrs.append(mean_qr3)
    mean_qrs.append(mean_qr4)
    mean_qrs.append(mean_qr5)
    print('1st qr: {0}, 2nd qr: {1}, 3rd qr: {2}, 4th qr: {3}'.format(mean_qr1, mean_qr2, mean_qr3, mean_qr4))
    
    x_axis = np.arange(5)
    width = 0.3
    xticks = ["cor>0.75", "0.5<cor<0.75", "0.25<cor<0.5", "0<cor<0.25", 'cor<0']
    plt.figure(figsize=(16, 9))
    plt.bar(x_axis, mean_sds, width, color='#fc8d91', edgecolor='none', label='standard deviation')
    plt.bar(x_axis + width, mean_qrs, width, color='#f7d00e', edgecolor='none', label='interquartile range')
    plt.xticks(x_axis + width, xticks, fontsize=16)
    plt.title('Comparison among good and bad peaks')
    plt.xlabel('peaks class', fontsize=18)
    plt.ylabel('average', fontsize=18)
    plt.legend()
    plt.savefig(output_file_path + "basset_SD_IQR.svg")
    plt.close()
    
    return ind_collection

def plot_corr_variance(labels, correlations, output_file_path):
    #compute variance:
    variance = np.var(labels, axis=1)

    #plot scatterplot of variance-correlations
    plt.figure(figsize=(16, 9))
    plt.scatter(variance, correlations)
    plt.xlabel('Peak variance', fontsize=18)
    plt.ylabel('Prediction-ground truth correlation', fontsize=18)
    plt.savefig(output_file_path + "variance_correlation_plot.svg")
    plt.close()

    #plot 2D scatterplot of variance-correlations
    plt.figure(figsize=(16, 9))
    plt.hist2d(variance, correlations, bins=100)
    plt.xlabel('Peak variance', fontsize=18)
    plt.ylabel('Prediction-ground truth correlation', fontsize=18)
    plt.colorbar()
    plt.savefig(output_file_path + "variance_correlation_hist2D.svg")
    plt.close()

    #plot 2D log transformed scatterplot of variance-correlations
    plt.figure(figsize=(16, 9))
    plt.hist2d(variance, correlations, bins=100, norm=LogNorm())
    plt.xlabel('Peak variance', fontsize=18)
    plt.ylabel('Prediction-ground truth correlation', fontsize=18)
    plt.colorbar()
    plt.savefig(output_file_path + "variance_correlation_loghist2d.svg")
    plt.close()


# plot some predictions vs ground_truth on test set
def plot_random_predictions(eval_labels, predictions, correlations, ind_collection, eval_names, output_file_path):
    for n in range(3):
        mum_plt_row = 1
        mum_plt_col = 1
        num_plt = mum_plt_row * mum_plt_col
        # 3 plots for each correlation categories
        for k in range(len(ind_collection) - 1):
            idx = random.sample(ind_collection[k + 1], num_plt)
    
            y_samples_eval = eval_labels[idx]
            predicted_classes = predictions[idx]
            sample_names = eval_names[idx]
    
            # Plot
            x_axis = np.arange(81)
            xticks = ['B.Fem.Sp', 'B.Fo.Sp', 'B.FrE.BM', 'B.GC.CB.Sp', 'B.GC.CC.Sp', 'B.MZ.Sp', 'B.PB.Sp', 'B.PC.BM',
                      'B.PC.Sp', 'B.Sp',
                      'B.T1.Sp', 'B.T2.Sp', 'B.T3.Sp', 'B.mem.Sp', 'B1b.PC', 'DC.4+.Sp', 'DC.8+.Sp', 'DC.pDC.Sp',
                      'GN.BM',
                      'GN.Sp',
                      'GN.Thio.PC', 'ILC2.SI', 'ILC3.CCR6+.SI', 'ILC3.NKp46+.SI', 'ILC3.NKp46-CCR6-.SI', 'LTHSC.34+.BM',
                      'LTHSC.34-.BM', 'MF.102+480+.PC', 'MF.226+II+480lo.PC', 'MF.Alv.Lu',
                      'MF.Fem.PC', 'MF.PC', 'MF.RP.Sp', 'MF.microglia.CNS', 'MF.pIC.Alv.Lu', 'MMP3.48+.BM',
                      'MMP4.135+.BM',
                      'Mo.6C+II-.Bl', 'Mo.6C-II-.Bl', 'NK.27+11b+.BM',
                      'NK.27+11b+.Sp', 'NK.27+11b-.BM', 'NK.27+11b-.Sp', 'NK.27-11b+.BM', 'NK.27-11b+.Sp', 'NKT.Sp',
                      'NKT.Sp.LPS.18hr', 'NKT.Sp.LPS.3d', 'NKT.Sp.LPS.3hr', 'STHSC.150-.BM',
                      'T.4.Nve.Fem.Sp', 'T.4.Nve.Sp', 'T.4.Sp.aCD3+CD40.18hr', 'T.4.Th', 'T.8.Nve.Sp', 'T.8.Th',
                      'T.DN4.Th',
                      'T.DP.Th', 'T.ISP.Th', 'T8.IEL.LCMV.d7.Gut',
                      'T8.MP.LCMV.d7.Sp', 'T8.TE.LCMV.d7.Sp', 'T8.TN.P14.Sp', 'T8.Tcm.LCMV.d180.Sp',
                      'T8.Tem.LCMV.d180.Sp',
                      'Tgd.Sp', 'Tgd.g1.1+d1.24a+.Th', 'Tgd.g1.1+d1.LN', 'Tgd.g2+d1.24a+.Th', 'Tgd.g2+d1.LN',
                      'Tgd.g2+d17.24a+.Th', 'Tgd.g2+d17.LN', 'Treg.4.25hi.Sp', 'Treg.4.FP3+.Nrplo.Co', 'preT.DN1.Th',
                      'preT.DN2a.Th', 'preT.DN2b.Th', 'preT.DN3.Th', 'proB.CLP.BM', 'proB.FrA.BM',
                      'proB.FrBC.BM']
    
            def minmax_scale(pred, true):
                subtracted = pred - min(pred)
                max_pred = max(subtracted)
                max_true = max(true)
    
                scaled = max_true * subtracted / max_pred
                return scaled
    
    
            plt.figure(1)
            width = 0.35
            for i in range(num_plt):
                plt.figure(figsize=(16, 9))
                plt.subplot(mum_plt_row, mum_plt_col, i + 1)
                plt.bar(x_axis, y_samples_eval[i], width, color='#f99fa1', edgecolor='none', label='true activity')
                plt.bar(x_axis + width, minmax_scale(predicted_classes[i], y_samples_eval[i]), width, color='#014ead',
                        edgecolor='none', label='prediction')
                plt.xticks(x_axis + width, xticks, rotation='vertical', fontsize=9)
                plt.title('{0}, correlation = {1:.3f}'.format(sample_names[i].decode('utf-8'),
                                                                             correlations[idx[i]]))
                plt.xlabel('cell type', fontsize=12)
                plt.ylabel('normalized activity', fontsize=12)
                plt.legend()
    
    
            fig = plt.gcf()
            fig.tight_layout()
    
            plt.savefig(output_file_path + "basset_cor_q{0}{1}.svg".format(k, n + 1), bbox_inches='tight')
    
            plt.close()
            
def plot_filt_corr(filt_pred, labels, correlations, output_file_path):  
    filt_corr = []
    corr_change = []
    n_combos = filt_pred.shape[1]
    
    for i in range(len(filt_pred)):
        pred = filt_pred[i,:,:]
        label = labels[i]
        corr_original = np.full(n_combos, correlations[i])

        #compute correlation between label and each prediction
        def pearson_corr(pred, label):
            return np.corrcoef(pred, label)[0,1]
        corr = np.apply_along_axis(pearson_corr, 1, pred, label)
        filt_corr.append(corr) 
        #print("Sample correlation shape: %s" % corr.shape)
        
        #compute difference in correlation between original model and leave-one-filter-out results
        change = np.square(corr-corr_original)
        corr_change.append(change)
        #print("Change in correlation: %s" % change.shape)
        
    #convert filt_corr and corr_change from list to array
    filt_corr = np.stack(filt_corr, axis=0)
    corr_change = np.stack(corr_change, axis=0)

    # plot histogram of correlation values of all models
    plt.clf()
    plt.hist(filt_corr.flatten(), bins=30)
    plt.axvline(np.mean(filt_corr), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {1:.2f}".format(np.mean(filt_corr)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("Correlation")
    plt.savefig(output_file_path + "filtcorr_hist.svg")
    plt.close()
    
    # plot histogram of changes in predictions
    corr_change_filter = np.mean(corr_change, axis=0)
    plt.clf()
    plt.hist(corr_change_filter.flatten(), bins=30)
    plt.axvline(np.mean(corr_change_filter), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {1:.2f}".format(np.mean(corr_change)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("Correlation")
    plt.savefig(output_file_path + "corr_change_hist.svg")
    plt.close()
    
    #plot bar graph of correlation change
    plt.clf()
    plt.bar(np.arange(n_combos), corr_change_filter)
    plt.title("Influence of filters on model predictions")
    plt.ylabel("Influence")
    plt.xlabel("Filter")
    plt.savefig(output_file_path + "corrchange_bar_graph.svg")
    plt.close()
    
    print("Shape of filter-wise correlations:" )
    print(filt_corr.shape)
    print("Shape of filter influence:")
    print(corr_change.shape)
    
    return filt_corr, corr_change, corr_change_filter
    
def plot_filt_infl(pred, filt_pred, output_file_path, cell_labels=None):
    n_combos = filt_pred.shape[1]

    #expand pred array to be nx300x81
    pred = np.expand_dims(pred, 1)
    pred = np.repeat(pred, n_combos, axis=1)
    
    #compute the sum of squares of differences between pred and filt_pred; result 300x81 array of influences
    #infl_by_OCR = np.square(filt_pred - pred)
    infl_by_OCR = filt_pred - pred
    infl = np.mean(infl_by_OCR, axis=0).squeeze()
    
    # plot histogram
    plt.clf()
    plt.hist(infl.flatten(), bins=30)
    plt.axvline(np.mean(infl), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {1:.2f}".format(np.mean(infl)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("")
    plt.savefig(output_file_path + "filt_infl_hist.svg")
    plt.close()

    # plot heatmap
    plt.clf()
    sns.heatmap(np.log(infl))
    plt.title("Influence of filters on each cell type prediction")
    plt.ylabel("Filter")
    plt.xlabel("Cell Type")

    if not cell_labels:
        cell_labels = ['B.Fem.Sp', 'B.Fo.Sp', 'B.FrE.BM', 'B.GC.CB.Sp', 'B.GC.CC.Sp', 'B.MZ.Sp', 'B.PB.Sp', 'B.PC.BM',
                          'B.PC.Sp', 'B.Sp',
                          'B.T1.Sp', 'B.T2.Sp', 'B.T3.Sp', 'B.mem.Sp', 'B1b.PC', 'DC.4+.Sp', 'DC.8+.Sp', 'DC.pDC.Sp',
                          'GN.BM',
                          'GN.Sp',
                          'GN.Thio.PC', 'ILC2.SI', 'ILC3.CCR6+.SI', 'ILC3.NKp46+.SI', 'ILC3.NKp46-CCR6-.SI', 'LTHSC.34+.BM',
                          'LTHSC.34-.BM', 'MF.102+480+.PC', 'MF.226+II+480lo.PC', 'MF.Alv.Lu',
                          'MF.Fem.PC', 'MF.PC', 'MF.RP.Sp', 'MF.microglia.CNS', 'MF.pIC.Alv.Lu', 'MMP3.48+.BM',
                          'MMP4.135+.BM',
                          'Mo.6C+II-.Bl', 'Mo.6C-II-.Bl', 'NK.27+11b+.BM',
                          'NK.27+11b+.Sp', 'NK.27+11b-.BM', 'NK.27+11b-.Sp', 'NK.27-11b+.BM', 'NK.27-11b+.Sp', 'NKT.Sp',
                          'NKT.Sp.LPS.18hr', 'NKT.Sp.LPS.3d', 'NKT.Sp.LPS.3hr', 'STHSC.150-.BM',
                          'T.4.Nve.Fem.Sp', 'T.4.Nve.Sp', 'T.4.Sp.aCD3+CD40.18hr', 'T.4.Th', 'T.8.Nve.Sp', 'T.8.Th',
                          'T.DN4.Th',
                          'T.DP.Th', 'T.ISP.Th', 'T8.IEL.LCMV.d7.Gut',
                          'T8.MP.LCMV.d7.Sp', 'T8.TE.LCMV.d7.Sp', 'T8.TN.P14.Sp', 'T8.Tcm.LCMV.d180.Sp',
                          'T8.Tem.LCMV.d180.Sp',
                          'Tgd.Sp', 'Tgd.g1.1+d1.24a+.Th', 'Tgd.g1.1+d1.LN', 'Tgd.g2+d1.24a+.Th', 'Tgd.g2+d1.LN',
                          'Tgd.g2+d17.24a+.Th', 'Tgd.g2+d17.LN', 'Treg.4.25hi.Sp', 'Treg.4.FP3+.Nrplo.Co', 'preT.DN1.Th',
                          'preT.DN2a.Th', 'preT.DN2b.Th', 'preT.DN3.Th', 'proB.CLP.BM', 'proB.FrA.BM',
                          'proB.FrBC.BM']
    plt.xticks(np.arange(len(cell_labels)), cell_labels, rotation='vertical', fontsize=3.0)
    plt.savefig(output_file_path + "sns_filt_infl_heatmap.svg")
    plt.close()

    return infl, infl_by_OCR
  

def get_memes(activations, sequences, y, output_file_path):
    #find the threshold value for activation
    activation_threshold = 0.5*np.amax(activations, axis=(0,2))

    #pad sequences:
    npad = ((0, 0), (0, 0), (9, 9))
    sequences = np.pad(sequences, pad_width=npad, mode='constant', constant_values=0)
    
    pwm = np.zeros((300, 4, 19))
    pfm = np.zeros((300, 4, 19))
    nsamples = activations.shape[0]
    
    OCR_matrix = np.zeros((300, y.shape[0]))
    activation_indices = []
    activated_OCRs = np.zeros((300, y.shape[1]))
    n_activated_OCRs = np.zeros(300)
    total_seq = np.zeros(300)
    
    for i in range(300):
        #create list to store 19 bp sequences that activated filter
        act_seqs_list = []
        act_OCRs_tmp = []
        for j in range(nsamples):
            # find all indices where filter is activated
            indices = np.where(activations[j,i,:] > activation_threshold[i])

            #save ground truth peak heights of OCRs activated by each filter  
            if indices[0].shape[0]>0:
                act_OCRs_tmp.append(y[j, :])
                OCR_matrix[i, j] = 1

            for start in indices[0]:
                activation_indices.append(start)
                end = start+19
                act_seqs_list.append(sequences[j,:,start:end])#*activations[j, i, start])

        #convert act_seqs from list to array
        if act_seqs_list: 
          act_seqs = np.stack(act_seqs_list)
          pwm_tmp = np.sum(act_seqs, axis=0)
          pfm_tmp=pwm_tmp
          total = np.sum(pwm_tmp, axis=0)
          pwm_tmp = np.nan_to_num(pwm_tmp/total)
          
          #permute pwm from A, T, G, C order to A, C, G, T order
          order = [0, 3, 2, 1]
          pwm[i,:,:] = pwm_tmp[order, :]
          pfm[i,:,:] = pfm_tmp[order, :]
          
          #store total number of sequences that activated that filter
          total_seq[i] = len(act_seqs_list)

          #save mean OCR activation
          act_OCRs_tmp = np.stack(act_OCRs_tmp)
          activated_OCRs[i, :] = np.mean(act_OCRs_tmp, axis=0)

          #save the number of activated OCRs
          n_activated_OCRs[i] = act_OCRs_tmp.shape[0]
    
    
    activated_OCRs = np.stack(activated_OCRs)

    #write motifs to meme format
    #PWM file:
    meme_file = open(output_file_path + "filter_motifs_pwm.meme", 'w')
    meme_file.write("MEME version 4 \n")

    #PFM file:
    meme_file_pfm = open(output_file_path + "filter_motifs_pfm.meme", 'w')
    meme_file_pfm.write("MEME version 4 \n")

    for i in range(0, 300):
        if np.sum(pwm[i,:,:]) >0:
          meme_file.write("\n")
          meme_file.write("MOTIF filter%s \n" % i)
          meme_file.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i,:,:], axis=0)))

          meme_file_pfm.write("\n")
          meme_file_pfm.write("MOTIF filter%s \n" % i)
          meme_file_pfm.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i,:,:], axis=0)))

          for j in range(0, 19):
              if np.sum(pwm[i,:,j]) > 0:
                meme_file.write(str(pwm[i,0,j]) + "\t" + str(pwm[i,1,j]) + "\t" + str(pwm[i,2,j]) + "\t" + str(pwm[i,3,j]) + "\n")
                meme_file_pfm.write(str(pfm[i,0,j]) + "\t" + str(pfm[i,1,j]) + "\t" + str(pfm[i,2,j]) + "\t" + str(pfm[i,3,j]) + "\n")
      
    meme_file.close()
    meme_file_pfm.close()
    
    #plot indices of first position in sequence that activates the filters
    activation_indices_array = np.stack(activation_indices)
    
    plt.clf()
    plt.hist(activation_indices_array.flatten(), bins=260)
    plt.title("histogram of position indices.")
    plt.ylabel("Frequency")
    plt.xlabel("Position")
    plt.savefig(output_file_path + "position_hist.svg")
    plt.close()
    
    #plot total sequences that activated each filter
    total_seq_array = np.stack(total_seq)
    
    plt.clf()
    plt.bar(np.arange(300), total_seq_array)
    plt.title("Number of sequences activating each filter")
    plt.ylabel("N sequences")
    plt.xlabel("Filter")
    plt.savefig(output_file_path + "nseqs_bar_graph.svg")
    plt.close()
    
    return pwm, activation_indices_array, total_seq_array, activated_OCRs, n_activated_OCRs, OCR_matrix

#get layer 2 motifs
def get_memes2(activations, sequences, y, output_file_path):
    #find the threshold value for activation
    activation_threshold = 0.5*np.amax(activations, axis=(0,2))

    #pad sequences:
    npad = ((0, 0), (0, 0), (26, 26))
    sequences = np.pad(sequences, pad_width=npad, mode='constant', constant_values=0)
    
    pwm = np.zeros((200, 4, 51))
    pfm = np.zeros((200, 4, 51))
    nsamples = activations.shape[0]

    for i in range(200):
        #create list to store 19 bp sequences that activated filter
        act_seqs_list = []
        act_OCRs_tmp = []
        for j in range(nsamples):
            # find all indices where filter is activated
            indices = np.where(activations[j,i,:] > activation_threshold[i])

            for start in indices[0]:
                start = start*3
                end = start+51
                act_seqs_list.append(sequences[j,:,start:end]) #*activations[j, i, start]

        #convert act_seqs from list to array
        if act_seqs_list:
          act_seqs = np.stack(act_seqs_list)
          pwm_tmp = np.sum(act_seqs, axis=0)
          pfm_tmp=pwm_tmp
          total = np.sum(pwm_tmp, axis=0)
          pwm_tmp = np.nan_to_num(pwm_tmp/total)
          
          #permute pwm from A, T, G, C order to A, C, G, T order
          order = [0, 3, 2, 1]
          pwm[i,:,:] = pwm_tmp[order, :]
          pfm[i,:,:] = pfm_tmp[order, :]
          
          #store total number of sequences that activated that filter
          #total_seq[i] = len(act_seqs_list)

          #save mean OCR activation
          #act_OCRs_tmp = np.stack(act_OCRs_tmp)
          #activated_OCRs[i, :] = np.mean(act_OCRs_tmp, axis=0)

          #save the number of activated OCRs
          #n_activated_OCRs[i] = act_OCRs_tmp.shape[0]
    
    
    #activated_OCRs = np.stack(activated_OCRs)

    #write motifs to meme format
    #PWM file:
    meme_file = open(output_file_path + "layer2_filter_motifs_pwm.meme", 'w')
    meme_file.write("MEME version 4 \n")

    #PFM file:
    meme_file_pfm = open(output_file_path + "layer2_filter_motifs_pfm.meme", 'w')
    meme_file_pfm.write("MEME version 4 \n")

    for i in range(0, 200):
        if np.sum(pwm[i,:,:]) >0:
          meme_file.write("\n")
          meme_file.write("MOTIF filter%s \n" % i)
          meme_file.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i,:,:], axis=0)))

          meme_file_pfm.write("\n")
          meme_file_pfm.write("MOTIF filter%s \n" % i)
          meme_file_pfm.write("letter-probability matrix: alength= 4 w= %d \n" % np.count_nonzero(np.sum(pwm[i,:,:], axis=0)))

          for j in range(0, 51):
              if np.sum(pwm[i,:,j]) > 0:
                meme_file.write(str(pwm[i,0,j]) + "\t" + str(pwm[i,1,j]) + "\t" + str(pwm[i,2,j]) + "\t" + str(pwm[i,3,j]) + "\n")
                meme_file_pfm.write(str(pfm[i,0,j]) + "\t" + str(pfm[i,1,j]) + "\t" + str(pfm[i,2,j]) + "\t" + str(pfm[i,3,j]) + "\n")
      
    meme_file.close()
    meme_file_pfm.close()



#get cell-wise correlations
def plot_cell_cors(obs, pred, cell_labels, output_file_path):
    correlations = []
    for i in range(pred.shape[1]):
        x = np.corrcoef(pred[:, i], obs[:, i])[0, 1]
        correlations.append(x)

    #plot cell-wise correlation histogram
    plt.clf()
    plt.hist(correlations, bins=30)
    plt.axvline(np.mean(correlations), color='k', linestyle='dashed', linewidth=2)
    try:
        plt.title("histogram of correlation.  Avg cor = {1:.2f}".format(np.mean(correlations)))
    except Exception as e:
        print("could not set the title for graph")
        print(e)
    plt.ylabel("Frequency")
    plt.xlabel("Correlation")
    plt.savefig(output_file_path + "basset_cell_wise_cor_hist.svg")
    plt.close()
    
    #plot cell-wise correlations by cell type
    plt.clf()
    plt.bar(np.arange(len(cell_labels.shape)), correlations)
    plt.title("Correlations by Cell Type")
    plt.ylabel("Correlation")
    plt.xlabel("Cell Type")
    plt.xticks(np.arange(len(cell_labels)), cell_labels, rotation='vertical', fontsize=3.5)
    plt.savefig(output_file_path + "cellwise_cor_bargraph.svg")
    plt.close()
    
    return correlations

#convert mouse model predictions to human cell predictions
def mouse2human(mouse_predictions, mouse_cell_types, map, method='average'):
    
    human_cells = np.unique(map[:,1])
    
    human_predictions = np.zeros((mouse_predictions.shape[0], human_cells.shape[0]))
    
    for i, celltype in enumerate(human_cells):
        matches = map[np.where(map[:,1] == celltype)][:,0]
        idx = np.in1d(mouse_cell_types[:,1], matches).nonzero()
        human_predictions[:, i] = np.mean(mouse_predictions[:, idx], axis=2).squeeze()
   
    return human_predictions, human_cells