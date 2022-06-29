import sys; sys.path.insert(0, '../')
from wu_barcodes.config import data_dir

import numpy as np
import pickle
import itertools

diet_mice_map = {'HF': [6, 7, 8, 9, 10],
         'LF': [16, 18, 19],
         'HLH': [1, 2, 3, 5],
         'LHL': [11, 12, 13, 15],
         'all': [6, 7, 8, 9, 10, 16, 18, 19, 1, 2, 3, 5, 11, 12, 13, 15],
         'H4': [6, 7, 8, 9, 10, 1, 2, 3, 5],
         'H10': [6, 7, 8, 9, 10, 11, 12, 13, 15],
         'H16': [6, 7, 8, 9, 10, 1, 2, 3, 5],
         'L4': [16, 18, 19, 11, 12, 13, 15],
         'L10': [16, 18, 19, 1, 2, 3, 5],
         'L16':  [16, 18, 19, 11, 12, 13, 15],
        }

diet_colors = {'HF':'red', 'LF':'blue', 'HLH':'orange', 'LHL':'purple'}

switch_diets = {'HF':'LF', 'LF':'HF'}
bacteria = ['BWH2',  'Bovatus', 'BtVPI', 'Bt7330']

mouse2diet = {}
for diet in ['HF', 'HLH', 'LF', 'LHL']:
    for mouse in diet_mice_map[diet]:
        mouse2diet[mouse] = diet

days = [4, 10, 16]

inputs = list(range(0, 23))

mouse_columns = []
for i in range(1, 24):
    mouse_columns.append( (i, 0) ) # input replicates
for day in days:
    for diet in ['HF', 'HLH', 'LF', 'LHL']:
        mouse_nums = diet_mice_map[diet]
        for num in mouse_nums:
            mouse_columns.append( (num, day) )
mouse_columns.append((0,0))

MX = {mouse: i for i, mouse in enumerate(mouse_columns)}

barcodes = {bac: {} for bac in bacteria}
read_dict = {bac: {} for bac in bacteria}
read_counts = {bac: {} for bac in bacteria}
freq_dict = {bac: {} for bac in bacteria}
for bac in bacteria:
    for diet in ['all', 'HF', 'LF', 'HL']:
        if bac == 'BtVPI_mainchromosome':
            bac = 'BtVPI'

        if diet == 'all':
            barcodes[bac][diet] = np.load("{}/{}_master_barcodes.npy".format(str(data_dir), bac)).astype('int')
            read_dict[bac][diet] = np.load("{}/{}_master_reads.npy".format(str(data_dir), bac)).astype('int')
            read_counts[bac][diet] = np.load("{}/{}_master_counts.npy".format(str(data_dir), bac)).astype('int')
            freq_dict[bac][diet] = np.load("{}/{}_master_freqs.npy".format(str(data_dir), bac)).astype('float')
        else:
            barcodes[bac][diet] = np.load("{}/{}_{}x_barcodes.npy".format(str(data_dir), bac, diet)).astype('int')
            read_dict[bac][diet] = np.load("{}/{}_{}x_reads.npy".format(str(data_dir), bac, diet)).astype('int')
            read_counts[bac][diet] = np.load("{}/{}_{}x_counts.npy".format(str(data_dir), bac, diet)).astype('int')
            freq_dict[bac][diet] = np.load("{}/{}_{}x_freqs.npy".format(str(data_dir), bac, diet)).astype('float')

def bin_reads(lst_of_read_counts): #expects lst_of_freqs to be counts
    arr = np.zeros(np.max(lst_of_read_counts) + 1, dtype = 'int')
    for read_counts in lst_of_read_counts:
        arr[read_counts] += 1
    return arr


with open('{}/master_gene_barcode_dict.pkl'.format(str(data_dir)), 'rb') as f:
    gene_barcode_dict = pickle.load(f)
with open('{}/master_barcode_gene_dict.pkl'.format(str(data_dir)), 'rb') as f:
    barcode_gene_dict = pickle.load(f)
with open('{}/gene_coords_dict.pkl'.format(str(data_dir)), 'rb') as f:
    gene_coords_dict = pickle.load(f)
with open('{}/wu_genes.pkl'.format(str(data_dir)), 'rb') as f:
    wu_genes = pickle.load(f)
with open('{}/vitro_array.pkl'.format(str(data_dir)), 'rb') as f:
    vitro_array = pickle.load(f)
with open('{}/vitro_col_dict.pkl'.format(str(data_dir)), 'rb') as f:
    vitro_col_ids = pickle.load(f)



def find_expanding_barcodes(init_freqs, fin_freqs):
    """
    Finds barcodes that expand across all mice
    from day 0 to day x.

    Returns bool array for all barcodes.
    """
    delta_freqs = fin_freqs - init_freqs
    bool_delta_freqs = delta_freqs > 0
    fit_indices = bool_delta_freqs.prod(axis = 0) == 1

    return fit_indices

def subset_arrays(bac, bc, mice, input_set=None):
    """
    Subset barcode frequency arrays on a subset of mice, days

    Returns list of arrays by day
    """
    sub_arrays = []
    for day in [0, 4, 10, 16]:
        if day == 0:
            if np.any(input_set):
                random_input_ids = np.array(input_set)
                row_ids = [MX[(i, 0)] for i in input_set]
            else:
                # random_input_ids = np.random.choice(np.arange(23), 10, replace=False)
                random_input_ids = np.arange(1, 23, 2)
                row_ids = [MX[(i,0)] for i in random_input_ids]
            reads = read_dict[bac][bc][row_ids].sum(axis=0)
            freqs = (reads / reads.sum()).reshape(-1, reads.shape[0])
        else:
            row_ids = [MX[(mouse, day)] for mouse in mice]
            freqs = freq_dict[bac][bc][row_ids]
        sub_arrays.append( freqs )
    return sub_arrays, random_input_ids

# with open('data/read_counts.pkl', 'rb') as f:
#     read_counts = pickle.load(f)

# with open('data/read_dict.pkl', 'rb') as f:
#     read_dict = pickle.load(f)

# with open('data/freq_dict.pkl', 'rb') as f:
#     freq_dict = pickle.load(f)

# with open('data/read_lists.pkl', 'rb') as f:
#     read_lists = pickle.load(f)

# with open('data/freq_lists.pkl', 'rb') as f:
#     freq_lists = pickle.load(f)

# with open('data/del_barcodes.pkl', 'rb') as f:
#     del_barcodes = pickle.load(f)

# with open('data/ben_barcodes.pkl', 'rb') as f:
#     ben_barcodes = pickle.load(f)

# def bin_reads(lst_of_read_counts): #expects lst_of_freqs to be counts
#     lst = np.zeros(np.max(lst_of_read_counts) + 1, dtype = 'int')
#     for read_counts in lst_of_read_counts:
#         lst[read_counts] += 1
#     return lst

# def generate_pairwise(bacteria, mouse1, mouse2, freq_range = (20, 30)):
#     lst_of_freqs = []
#     freq_bins = {}
#     m1_dict = read_dict[bacteria][mouse1]
#     m2_dict = read_dict[bacteria][mouse2]
#     for barcode, freq in m1_dict.items():
#         if freq_range[0] <= freq <= freq_range[1]:
#             freq2 = m2_dict[barcode]
#             lst_of_freqs.append(freq2)
#             if freq2 not in freq_bins:
#                 freq_bins[freq2] = 0
#             freq_bins[freq2] += 1
#     lst = sorted(list(freq_bins.items()))
#     arr = np.array(lst)
#     freq_bins = arr[::,0]
#     bin_counts = arr[::,1]
#     return np.array(lst_of_freqs), freq_bins, bin_counts
