root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
##
from methods.config import *
import matplotlib as mpl
import matplotlib.pyplot as plt
if True:
    mpl.rcParams['lines.linewidth'] = 0.75
    mpl.rcParams['font.family'] = 'Helvetica'
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = 6
    mpl.rcParams['ytick.labelsize'] = 6
    mpl.rcParams['figure.facecolor'] = (1,1,1,1) #white
    mpl.rcParams['figure.edgecolor'] = (1,1,1,1) #white
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['legend.fontsize'] = 6
import methods.ax_methods as ax_methods
import methods.fitness_profiling as fp
import methods.shared as shared
import numpy as np; rnd = np.random.default_rng()
import pickle

bacteria = ['BtVPI', 'BWH2']
fitness_profiles_meta = {}

for bac in bacteria:
    print(f'Generating fitness profiles for {bac}...')
    row_ids = shared.bac_row_ids[bac]
    notWu = shared.bac_nonwu_indices[bac]
    read_array = shared.bac_read_arrays[bac]
    input_array = shared.bac_input_arrays[bac]
    day0_reads = read_array[0]
    day0_freqs = read_array[0] / read_array[0].sum()
    gene_lineage_map = shared.bac_gene_meta_dict[bac]
    read_arrays_meta = {'input_array':input_array,
                        'read_array':read_array,
                        'row_ids':row_ids}

    # Get genes and their Tn lineage indices
    genes = [gene for gene in shared.bac_gene_meta_dict[bac].keys()]
    gene_lineage_indices_map = fp.call_conservative_gene_indices(genes, gene_lineage_map)
    candidate_lineage_bool = (day0_freqs > -1)

    # Define environments, replicates in each environment,
    ENV_ALL_REPLICATES = fp.make_env_replicate_tuples(bac)
    ENV_S1_REPLICATES = fp.make_env_replicate_tuples(bac, split=1)
    ENV_S2_REPLICATES = fp.make_env_replicate_tuples(bac, split=2)
    ENV_LABELS = fp.make_env_labels(ENV_ALL_REPLICATES)

    ## S1 - S2
    for rep1, rep2 in zip(ENV_S1_REPLICATES, ENV_S2_REPLICATES):
        if np.any( set(rep1[2]).intersection(rep2[2])):
            print(f'Shared replicates! {rep1[0]}')

    max_removed = 0
    ## average profiles
    f0_arr, D0_arr, f1_arr, D1_arr = fp.get_env_frequency_data(ENV_ALL_REPLICATES, read_arrays_meta)
    S1_f0_arr, S1_D0_arr, S1_f1_arr, S1_D1_arr = fp.get_env_frequency_data(ENV_S1_REPLICATES, read_arrays_meta)
    S2_f0_arr, S2_D0_arr, S2_f1_arr, S2_D1_arr = fp.get_env_frequency_data(ENV_S2_REPLICATES, read_arrays_meta)

    print(f'>> Generating fitness profiles over all data...')
    gene_average_loo_profiles = fp.get_leave_one_out_distributions(gene_lineage_indices_map, f0_arr, f1_arr, D0_arr, D1_arr)
    gene_average_profiles = fp.make_pared_gene_lfcs_dict(f0_arr, f1_arr, D0_arr, D1_arr, gene_average_loo_profiles, gene_lineage_map, max_removed=max_removed)
    lineage_average_profiles, _ = fp.generate_mouse_replicate_profiles(ENV_ALL_REPLICATES, candidate_lineage_bool, read_arrays_meta, empirical=False)
    gene_replicate_profiles = fp.generate_gene_replicate_profiles(genes, gene_average_profiles, f0_arr, f1_arr, D0_arr, D1_arr)

    print(f'>> Generating fitness profiles over replicate set 1 (S1) data...')
    gene_S1_loo_profiles = fp.get_leave_one_out_distributions(gene_lineage_indices_map, S1_f0_arr, S1_f1_arr, S1_D0_arr, S1_D1_arr)
    gene_S1_profiles = fp.make_pared_gene_lfcs_dict(S1_f0_arr, S1_f1_arr, S1_D0_arr, S1_D1_arr, gene_S1_loo_profiles, gene_lineage_map, max_removed=max_removed)
    lineage_S1_profiles, _ = fp.generate_mouse_replicate_profiles(ENV_S1_REPLICATES, candidate_lineage_bool, read_arrays_meta, empirical=False)

    print(f'>> Generating fitness profiles over replicate set 2 (S2) data...')
    gene_S2_loo_profiles = fp.get_leave_one_out_distributions(gene_lineage_indices_map, S2_f0_arr, S2_f1_arr, S2_D0_arr, S2_D1_arr)
    gene_S2_profiles = fp.make_pared_gene_lfcs_dict(S2_f0_arr, S2_f1_arr, S2_D0_arr, S2_D1_arr, gene_S2_loo_profiles, gene_lineage_map, max_removed=max_removed)
    lineage_S2_profiles, _ = fp.generate_mouse_replicate_profiles(ENV_S2_REPLICATES, candidate_lineage_bool, read_arrays_meta, empirical=False)
    #
    bac_fitness_profiles_meta = {}
    bac_fitness_profiles_meta['gene_profile_data'] = gene_average_profiles
    bac_fitness_profiles_meta['gene_profile_S1_data'] = gene_S1_profiles
    bac_fitness_profiles_meta['gene_profile_S2_data'] = gene_S2_profiles
    bac_fitness_profiles_meta['gene_profile_replicate_data'] = gene_replicate_profiles
    bac_fitness_profiles_meta['lineage_profile_data'] = lineage_average_profiles
    bac_fitness_profiles_meta['lineage_profile_S1_data'] = lineage_S1_profiles
    bac_fitness_profiles_meta['lineage_profile_S2_data'] = lineage_S2_profiles
    all_freq_arrays = [f0_arr, D0_arr, f1_arr, D1_arr]
    S1_freq_arrays = [S1_f0_arr, S1_D0_arr, S1_f1_arr, S1_D1_arr]
    S2_freq_arrays = [S1_f0_arr, S1_D0_arr, S1_f1_arr, S1_D1_arr]
    bac_fitness_profiles_meta['all_freq_arrays'] = all_freq_arrays
    bac_fitness_profiles_meta['S1_freq_arrays'] = S1_freq_arrays
    bac_fitness_profiles_meta['S2_freq_arrays'] = S2_freq_arrays
    print('\n')

    with open(f'{data_dir}/{bac}_fitness_profiling_data.pkl', 'wb') as f:
        pickle.dump(bac_fitness_profiles_meta, f)

#     fitness_profiles_meta[bac] = bac_fitness_profiles_meta
#
# with open(f'{data_dir}/fitness_profiling_data.pkl', 'wb') as f:
#     pickle.dump(fitness_profiles_meta, f)