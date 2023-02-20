## pickled data and functions used by almost all plot scripts
root_name = 'tnseq_adaptation'
import os, sys; ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)

from methods.config import *
import numpy as np
import pickle

from scipy.ndimage import uniform_filter1d
def running_mean_uniform_filter1d(x, N):
    return uniform_filter1d(x, N, mode='reflect', origin=0)

if True:
    with open(f'{data_dir}/bac_lineage_positions.pkl', 'rb') as f:
        bac_lineage_positions = pickle.load(f)

    with open(f'{data_dir}/bac_row_ids.pkl', 'rb') as f:
        bac_row_ids = pickle.load(f)

    with open(f'{data_dir}/bac_read_arrays.pkl', 'rb') as f:
        bac_read_arrays = pickle.load(f)

    with open(f'{data_dir}/bac_input_arrays.pkl', 'rb') as f:
        bac_input_arrays = pickle.load(f)

    with open(f'{data_dir}/gene_meta_dict.pkl', 'rb') as f:
        bac_gene_meta_dict = pickle.load(f)

    with open(f'{data_dir}/lineage_gene_map.pkl', 'rb') as f:
        bac_lineage_gene_map = pickle.load(f)

    with open(f'{data_dir}/nonwu_indices.pkl', 'rb') as f:
        bac_nonwu_indices = pickle.load(f)

    with open(f'{data_dir}/wu_genes.pkl', 'rb') as f:
        wu_genes = pickle.load(f)

def get_read_arrays(bac, mice, t0, t1, split_day0=False, notWu=None):
    ''' Expects t1 > t0 '''
    if type(mice) == int:
        mice = [mice]

    row_ids = bac_row_ids[bac]
    if not np.any(notWu):
        notWu = bac_nonwu_indices[bac]
    read_array = bac_read_arrays[bac][:, notWu]
    input_array = bac_input_arrays[bac][:, notWu]

    if t0 == 0:
        if split_day0 is False:
            reads0 = read_array[0]
        else:
            split = (split_day0-1) % 2
            reads0 = input_array[split::2].sum(axis=0)
    else:
        reads0 = read_array[ [row_ids[(mouse, t0)] for mouse in mice] ].sum(axis=0)

    reads1 = read_array[ [row_ids[(mouse, t1)] for mouse in mice] ].sum(axis=0)
    return reads0, reads1

def get_freq_arrays(bac, mice, t0, t1, split_day0=False, notWu=None):
    ''' Expects t1 > t0 '''
    if type(mice) == int:
        mice = [mice]

    row_ids = bac_row_ids[bac]
    if not np.any(notWu):
        notWu = bac_nonwu_indices[bac]
    read_array = bac_read_arrays[bac][:, notWu]
    input_array = bac_input_arrays[bac][:, notWu]

    if t0 == 0:
        if split_day0 is False:
            reads0 = read_array[0]
        else:
            split = (split_day0-1) % 2
            reads0 = input_array[split::2].sum(axis=0)
    else:
        reads0 = read_array[ [row_ids[(mouse, t0)] for mouse in mice] ].sum(axis=0)

    reads1 = read_array[ [row_ids[(mouse, t1)] for mouse in mice] ].sum(axis=0)

    freqs0, freqs1 = maxmin_freqs(reads0, reads0.sum(), reads1, reads1.sum())
    return freqs0, freqs1

def calc_gene_complement_freqs(bac, sorted_indices, freqs0, freqs1, nearest_neighbors=False, notWu=None, cutoff=1e-6):
    """ Assuming single mouse.
    For sorted lineage indices, calculate gene complement frequency"""
    if not np.any(notWu):
        notWu = bac_nonwu_indices[bac]
    lineage_positions = bac_lineage_positions[bac][notWu]
    position_to_index_map = {pos: i for i, pos in enumerate(lineage_positions)}

    sorted_positions = lineage_positions[sorted_indices]

    gene_complement_f0 = np.ma.zeros(len(sorted_indices))
    gene_complement_f0.mask = np.full(len(sorted_indices), False)
    gene_complement_f1 = np.ma.zeros(len(sorted_indices))
    gene_complement_f1.mask = np.full(len(sorted_indices), False)

    for i in range(len(sorted_indices)):
        lineage_position = sorted_positions[i]
        lineage_index = sorted_indices[i]

        try:
            gene = bac_lineage_gene_map[bac][lineage_position][0]
            gene_bc_positions = set(bac_gene_meta_dict[bac][gene]['positions'])
            gene_bc_indices = [ position_to_index_map[pos] for pos in gene_bc_positions ]
            if len(gene_bc_indices) == 1: #
                # gene_complement_f0[i] = freqs0[lineage_index]
                # gene_complement_f1[i] = freqs1[lineage_index]
                gene_complement_f0.mask[i] = True
                gene_complement_f1.mask[i] = True
                continue
        except: #not found in gene
            gene_complement_f0.mask[i] = True
            gene_complement_f1.mask[i] = True
            # gene_complement_f0.mask[i] = True
            # gene_complement_f1.mask[i] = True
            continue

        if nearest_neighbors:
            f0, f1 = 0, 0
            if lineage_index - 1 in gene_bc_indices:
                f0 += freqs0[lineage_index - 1]
                f1 += freqs1[lineage_index - 1]
            if lineage_index + 1 in gene_bc_indices:
                f0 += freqs0[lineage_index + 1]
                f1 += freqs1[lineage_index + 1]
            gene_complement_f0[i] = f0
            gene_complement_f1[i] = f1
        else:
            where_small = (freqs0[gene_bc_indices] < cutoff)
            bigfreqs0 = np.copy( freqs0[gene_bc_indices] )
            bigfreqs0[where_small] = 0
            bigfreqs1 = np.copy( freqs1[gene_bc_indices] )
            bigfreqs1[where_small] = 0

            # freqs0[lineage_index] = freqs0[lineage_index]
            # freqs1[lineage_index] = freqs1[lineage_index]

            # gene_complement_f0[i] = freqs0[lineage_index]
            # gene_complement_f1[i] = freqs1[lineage_index]

            gene_complement_f0[i] =  bigfreqs0.sum() - freqs0[lineage_index]
            gene_complement_f1[i] = bigfreqs1.sum() - freqs1[lineage_index]
    return gene_complement_f0, gene_complement_f1

def calc_gene_complement_multiple_freqs(bac, sorted_indices, freqs0, freqs1, nearest_neighbors=False):
    """ For multiple_mice
    For sorted lineage indices, calculate gene complement frequency"""
    notWu = bac_nonwu_indices[bac]
    lineage_positions = bac_lineage_positions[bac][notWu]
    position_to_index_map = {pos: i for i, pos in enumerate(lineage_positions)}

    sorted_positions = lineage_positions[sorted_indices]

    gene_complement_f0 = np.ma.zeros( (freqs0.shape[0], len(sorted_indices)) )
    gene_complement_f0.mask = np.full((freqs0.shape[0], len(sorted_indices)), False)
    gene_complement_f1 = np.ma.zeros((freqs0.shape[0], len(sorted_indices)))
    gene_complement_f1.mask = np.full((freqs0.shape[0], len(sorted_indices)), False)

    for i in range(len(sorted_indices)):
        lineage_position = sorted_positions[i]
        lineage_index = sorted_indices[i]
        try:
            gene = bac_lineage_gene_map[bac][lineage_position][0]
            gene_bc_positions = set(bac_gene_meta_dict[bac][gene]['positions'])
            gene_bc_indices = [ position_to_index_map[pos] for pos in gene_bc_positions ]
            if len(gene_bc_indices) == 1: #
                gene_complement_f0[i] = freqs0[:, lineage_index]
                gene_complement_f1[i] = freqs1[:, lineage_index]
                continue
        except: #not found in gene
            gene_complement_f0.mask[:, i] = True
            gene_complement_f1.mask[:, i] = True
            continue

        if nearest_neighbors:
            f0, f1 = 0, 0
            if lineage_index - 1 in gene_bc_indices:
                f0 += freqs0[:, lineage_index - 1]
                f1 += freqs1[:, lineage_index - 1]
            if lineage_index + 1 in gene_bc_indices:
                f0 += freqs0[:, lineage_index + 1]
                f1 += freqs1[:, lineage_index + 1]
            gene_complement_f0[:, i] = f0
            gene_complement_f1[:, i] = f1
        else:
            where_small = (freqs0[:, gene_bc_indices] < 1e-6)
            bigfreqs0 = np.copy( freqs0[:, gene_bc_indices] )
            bigfreqs0[where_small] = 0
            bigfreqs1 = np.copy( freqs1[:, gene_bc_indices] )
            bigfreqs1[where_small] = 0

            gene_complement_f0[:, i] =  bigfreqs0.sum(axis=1) - freqs0[:, lineage_index]
            gene_complement_f1[:, i] = bigfreqs1.sum(axis=1) - freqs1[:, lineage_index]
    return gene_complement_f0, gene_complement_f1

def filter_lineages(reads0, reads1, min_reads=5, threshold='MaxOr'):
    D0, D1 = reads0.sum(), reads1.sum()
    if np.any(threshold):
        freqs = threshold
        # threshold = ((freqs*D1 > min_reads) + (freqs*D0 > min_reads))>0
        threshold = freqs * np.min((D0, D1)) > min_reads
    elif threshold=='MaxOr':
        maxfreqs = np.max([reads0/D0, reads1/D1], axis=0)
        threshold = ((maxfreqs*D1 > min_reads) + (maxfreqs*D0 > min_reads))>0
    elif threshold=='MaxAnd':
        maxfreqs = np.max([reads0/D0, reads1/D1], axis=0)
        threshold = ((maxfreqs*D1 > min_reads) * (maxfreqs*D0 > min_reads))>0
    elif threshold=='AvgOr':
        avgfreqs = (reads0+reads1)/(D0+D1)
        threshold = ((avgfreqs*D1 > min_reads) + (avgfreqs*D0 > min_reads))>0
    elif threshold=='AvgAnd':
        avgfreqs = (reads0+reads1)/(D0+D1)
        threshold = ((avgfreqs*D1 > min_reads) * (avgfreqs*D0 > min_reads))>0
    elif threshold=='MinOr':
        minfreqs = np.min([reads0/D0, reads1/D1], axis=0)
        threshold = ((minfreqs*D1 > min_reads) + (minfreqs*D0 > min_reads))>0
    elif threshold=='MinAnd':
        minfreqs = np.min([reads0/D0, reads1/D1], axis=0)
        threshold = ((minfreqs*D1 > min_reads) * (minfreqs*D0 > min_reads))>0
    return threshold

def filter_pairs(set1, set2, min_reads=(5,5), threshold='MaxOr', union=False):
    ''' set1, set2 are (summed) read sets (reads0, reads1)'''
    if type(min_reads) is int:
        min_reads = (min_reads, min_reads)

    set1_filter = filter_lineages(*set1, min_reads=min_reads[0], threshold=threshold)
    set2_filter = filter_lineages(*set2, min_reads=min_reads[1], threshold=threshold)
    if union:
        return (set1_filter + set2_filter) > 0
    else: #intersection
        return (set1_filter * set2_filter) > 0

def maxmin_freqs(freqs0, depth0, freqs1, depth1):
    overD0 = np.full(freqs0.shape[-1], 1 / depth0)
    overD1 = np.full(freqs1.shape[-1], 1 / depth1)

    min_f0 = np.min([freqs1, overD0], axis=0)
    maxmin_f0 = np.max([freqs0, min_f0], axis=0)

    min_f1 = np.min([freqs0, overD1], axis=0)
    maxmin_f1 = np.max([freqs1, min_f1], axis=0)

    return maxmin_f0, maxmin_f1

def generate_pair_freqs(bac, m1_tuple, m2_tuple, d0_cutoff=1e-6):
    if 0 in m1_tuple[1:] and 0 in m2_tuple[1:]:
        discovery_reads0, discovery_reads1 = get_read_arrays(bac, *m1_tuple, split_day0=1)
        validation_reads0, validation_reads1 = get_read_arrays(bac, *m2_tuple, split_day0=2)
    else:
        discovery_reads0, discovery_reads1 = get_read_arrays(bac, *m1_tuple)
        validation_reads0, validation_reads1 = get_read_arrays(bac, *m2_tuple)

    discovery_freqs0 = (discovery_reads0 / discovery_reads0.sum())
    discovery_freqs1 = (discovery_reads1 / discovery_reads1.sum())
    discovery_D0 = discovery_reads0.sum()
    discovery_D1 = discovery_reads1.sum()

    validation_freqs0 = (validation_reads0 / validation_reads0.sum())
    validation_freqs1 = (validation_reads1 / validation_reads1.sum())
    validation_D0 = validation_reads0.sum()
    validation_D1 = validation_reads1.sum()

    d0_reads = bac_read_arrays[bac][0][bac_nonwu_indices[bac]]
    d0_freqs = d0_reads / d0_reads.sum()

    max_freqs = np.max([discovery_freqs0, discovery_freqs1], axis=0)
    paired_filter = filter_pairs((discovery_reads0, discovery_reads1), (validation_reads0, validation_reads1),
                                    min_reads=5, threshold=max_freqs)

    d_f0, d_f1 = maxmin_freqs(discovery_freqs0, discovery_D0, discovery_freqs1, discovery_D1)
    v_f0, v_f1 = maxmin_freqs(validation_freqs0, validation_D0, validation_freqs1, validation_D1)
    lineage_bool = (d0_freqs > d0_cutoff) * paired_filter

    return (d_f0, d_f1), (v_f0, v_f1), lineage_bool

def calc_lfc_array(freqs0, freqs1, dt, valid=None):
    if np.any(valid):
        mask = ~valid
        freqs0, freqs1 = np.ma.masked_array(freqs0, mask), np.ma.masked_array(freqs1, mask)
        return np.ma.log( freqs1[valid]/freqs0[valid] ) / dt
    else:
        mask = ((freqs0 == 0) + (freqs1 == 0)) > 0
        freqs0, freqs1 = np.ma.masked_array(freqs0, mask), np.ma.masked_array(freqs1, mask)
        return np.ma.log( freqs1/freqs0 ) / dt

def calc_coarse_grained_lfc_array(freqs0, D0, freqs1, D1, dt, coarse_grain=100):
    cg_freqs0 = running_mean_uniform_filter1d(freqs0, coarse_grain)
    cg_freqs1 = running_mean_uniform_filter1d(freqs1, coarse_grain)

    cg_f0, cg_f1 = maxmin_freqs(cg_freqs0, D0, cg_freqs1, D1)

    return calc_lfc_array(cg_f0, cg_f1, dt)

def rank_barcodes(freqs0, freqs1, dt=1, valid=None):
    fitnesses = calc_lfc_array(freqs0, freqs1, dt, valid=valid)
    sorted_indices = np.argsort(fitnesses)[::-1] # fit to least fit
    return sorted_indices, fitnesses[sorted_indices]

