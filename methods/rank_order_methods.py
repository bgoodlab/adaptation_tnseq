import numpy as np
from data.wu_data import *

import scipy.ndimage.filters as ndif
def running_mean_uniform_filter1d(x, N):
    return ndif.uniform_filter1d(x, N, mode='reflect', origin=0)

def process_train_test_mice(bac, bc, train_mice_days_tuple, test_mice_days_tuple):
    ### ASSUMES train_day0 = 0
    train_mice, train_day0, train_day1 = train_mice_days_tuple
    train_dt = train_day1 - train_day0
    
    test_mice, test_day0, test_day1 = test_mice_days_tuple
    test_dt = test_day1 - test_day0
    
    if test_day0 == 0 and train_day0 == 0:
        day0_train = read_dict[bac][bc][0:18].sum(axis=0)
        barcode_indices = day0_train > 50
        day0_train = day0_train/day0_train.sum()

        day0_test_reads = read_dict[bac][bc][18:23].sum(axis=0)
        day0_test = day0_test_reads/day0_test_reads.sum()
    else:
        barcode_indices = read_dict[bac][bc][-1] > 100
        
        if train_day0 == 0:
            day0_train = read_dict[bac][bc][-1]
            barcode_indices = day0_train > 100
            day0_train = freq_dict[bac][bc][-1]
        else:
            train_ids = [MX[(mouse, train_day0)] for mouse in train_mice]
            day0_train = freq_dict[bac][bc][train_ids]
        
        if type(test_mice) == int:
            day0_test = freq_dict[bac][bc][MX[(test_mice, test_day0)]]
        else:
            day0_test = freq_dict[bac][bc][[MX[(mouse, test_day0)]for mouse in test_mice]]
    
    barcode_indices = (freq_dict[bac][bc][-1] > np.max([1e-6, 100 / read_counts[bac][bc][-1]]) )
    train_ids = [MX[(mouse, train_day1)] for mouse in train_mice]
    if type(test_mice) == int:
        test_ids = MX[(test_mice, test_day1)]
    else:
        test_ids = [MX[(mouse, test_day1)] for mouse in test_mice]
    
    return barcode_indices, (train_ids, day0_train, train_dt), (test_ids, day0_test, test_dt)

def select_fit_barcodes(bac, bc, day0, day4, dt, barcode_indices, binspace, threshold):
    where_finite, ref_lfc, emp_lfc, null_lfc = calc_emp_and_null_lfc(day4[:, barcode_indices], day0[barcode_indices])
    
    emp_density, bins = np.histogram(emp_lfc / dt, bins=binspace)
    null_density, bins = np.histogram(null_lfc / dt, bins=binspace)

    null_ccdf = np.sum(null_density) - np.cumsum(null_density) 
    emp_ccdf = np.sum(emp_density) - np.cumsum(emp_density)
    excess_fit = np.where( null_ccdf / emp_ccdf < threshold)[0][0]  ## 95% of barcodes are genuinely fit
    lfc_cutoff = bins[excess_fit]

    finite_barcodes = barcodes[bac][bc][barcode_indices][where_finite]
    fit_barcode_indices = emp_lfc > lfc_cutoff
    fit_barcodes = finite_barcodes[fit_barcode_indices]
    fit_indices = np.array([barcode in fit_barcodes for barcode in barcodes[bac][bc]])
    
    return fit_indices

def calculate_barcode_lfc(day0, day4, dt, fit_indices):
    return np.log( day4[fit_indices] / day0[fit_indices] ) / dt

def bootstrap_barcode_freq(freq_array):
    mean_freqs = np.mean( freq_array, axis=0)

    bootstrap_dist = np.zeros(mean_freqs.shape[0])
    num_mice = freq_array.shape[0]
    mice_range = np.arange(num_mice)

    for j in range(mean_freqs.shape[0]):
        bootstrap_sample = np.random.choice( mice_range,  num_mice, replace=True)
        bootstrap_freqs = freq_array[bootstrap_sample, j]
        bootstrap_dist[j] = np.mean(bootstrap_freqs)

    return mean_freqs, bootstrap_dist

def calc_emp_and_null_lfc(freq_array, input_array):
    
    mean_freqs, bootstrap_freqs = bootstrap_barcode_freq(freq_array)
    
    emp_fold_changes = mean_freqs / input_array    
    where_finite = (emp_fold_changes > 0)
    emp_lfc = np.log(emp_fold_changes[where_finite])
    
    binspace = np.linspace(-2, 2, 201)
    density, bins = np.histogram(emp_lfc, bins=binspace)
    ref_lfc = bins[np.argmax(density)]
#     ref_lfc = np.mean(emp_lfc)

    null_fold_changes = np.exp(ref_lfc) + (bootstrap_freqs - mean_freqs) / input_array
    null_lfc = np.log(null_fold_changes[where_finite])
    
    return where_finite, ref_lfc, emp_lfc, null_lfc

### 
def find_fit_gene_indices(bac, bc, barcode_indices, mouse, day0, day1):
    barcode_positions = barcodes[bac][bc][barcode_indices]

    gene_dict = {}
    multi_genes_count = 0
    for barcode_pos in barcode_positions:
        if barcode_pos in barcode_gene_dict[bac]:
            genes = barcode_gene_dict[bac][barcode_pos]
            if len(genes) > 1:
                multi_genes_count += 1
            gene = genes[0]
            if gene not in gene_dict:
                gene_dict[gene] = set(gene_barcode_dict[bac][gene][0]) ## gene_indices
#             gene_positions = gene_barcode_dict[bac][gene][1]
#             gene_indices = gene_barcode_dict[bac][gene][0]
#             try:
#                 gene_dict[gene].remove( gene_indices[list(gene_positions).index(barcode_pos)] )
#             except:
#                 pass
        else:
            pass
            
    if day0 == 0:
        D0 = read_counts[bac][bc][-1]
        m0_reads = read_dict[bac]['all'][-1]
    else:
        D0 = read_counts[bac][bc][MX[(mouse, day0)]]
        m0_reads = read_dict[bac]['all'][MX[(mouse, day0)]]  
    
    D1 = read_counts[bac][bc][MX[(mouse, day1)]]
    m1_reads = read_dict[bac]['all'][MX[(mouse, day1)]]
    
    gene_bc_indices = []
    for gene, gene_indices in gene_dict.items():
        gene_bc_indices += gene_indices
        
    m0_freqs = m0_reads[gene_bc_indices] / D0
    m1_freqs = m1_reads[gene_bc_indices] / D1
    
    return m0_freqs, m1_freqs

def find_fit_barcode_gene_other_barcodes(bac, bc, barcode_indices, mouse, day0, day1):
    barcode_positions = barcodes[bac][bc][barcode_indices]

    gene_dict = {}
    multi_genes_count = 0
    for barcode_pos in barcode_positions:
        if barcode_pos in barcode_gene_dict[bac]:
            genes = barcode_gene_dict[bac][barcode_pos]
            if len(genes) > 1:
                multi_genes_count += 1
            gene = genes[0]
            if gene not in gene_dict:
                gene_dict[gene] = set(gene_barcode_dict[bac][gene][0]) ## gene_indices
#             gene_positions = gene_barcode_dict[bac][gene][1]
#             gene_indices = gene_barcode_dict[bac][gene][0]
#             try:
#                 gene_dict[gene].remove( gene_indices[list(gene_positions).index(barcode_pos)] )
#             except:
#                 pass
        else:
            pass
            
    if day0 == 0:
        D0 = read_counts[bac][bc][-1]
        m0_reads = read_dict[bac]['all'][-1]
    else:
        D0 = read_counts[bac][bc][MX[(mouse, day0)]]
        m0_reads = read_dict[bac]['all'][MX[(mouse, day0)]]  
    
    D1 = read_counts[bac][bc][MX[(mouse, day1)]]
    m1_reads = read_dict[bac]['all'][MX[(mouse, day1)]]
    
    fit_gene_freq_dict_day0 = {'Inter':0}
    fit_gene_freq_dict_day1 = {'Inter':0}
    
    for gene, gene_other_bc_indices in gene_dict.items():
        try:
            gene_reads_day0 = m0_reads[list(gene_other_bc_indices)]
            where_big = (gene_reads_day0 > 100)
            
            
            gene_freq_day0 = m0_reads[list(gene_other_bc_indices)][where_big].sum() / D0
            gene_freq_day1 = m1_reads[list(gene_other_bc_indices)][where_big].sum() / D1
        except:
            gene_freq_day0 = 0
            gene_freq_day1 = 0
        
        fit_gene_freq_dict_day0[gene] = gene_freq_day0
        fit_gene_freq_dict_day1[gene] = gene_freq_day1
        
    
    return gene_dict, fit_gene_freq_dict_day0, fit_gene_freq_dict_day1