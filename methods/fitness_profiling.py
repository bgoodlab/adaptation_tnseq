import numpy as np; rnd = np.random.default_rng()
import copy
import scipy

BWH2_MEDIA = ['Arabinose', 'Glucose', 'TYG', 'WAX', 'Xylose']
BWH2_LENGTH = 7084828
BTVPI_MAIN_LENGTH = 6260361
BTVPI_PLASMID_LENGTH = 33038
BTVPI_LENGTH = BTVPI_MAIN_LENGTH + BTVPI_PLASMID_LENGTH
BT7330_LENGTH = 6487685
BOVATUS_LENGTH = 6472489
GENOME_LENGTHS = {'BWH2':BWH2_LENGTH, 'BtVPI':BTVPI_LENGTH, 'Bt7330':BT7330_LENGTH, 'Bovatus':BOVATUS_LENGTH}

#%%
fitness_browser_data = {'BT3705':{'Heparin': (-0.4*np.log(2), -0.1*np.log(2)),
                                  'Chondroitin sulfate': (0.1*np.log(2), 0.1*np.log(2)),
                                  'Corn starch': (-1.9*np.log(2), -2.9*np.log(2), -3.0*np.log(2)),
                                  'Amylopectin': (-2.6*np.log(2), -2.8*np.log(2), -2.8*np.log(2), -2.4*np.log(2)),
                                  'Amylose': (-1.2*np.log(2), -1.4*np.log(2)),
                                  'Arabinose':(0.1*np.log(2), 0.0*np.log(2)),
                                  'Fructose': (0.2*np.log(2), -0.0*np.log(2)),
                                  'Galactose':(0.1*np.log(2), -0.1*np.log(2)),
                                  'Glucose':(0.0, 0.0,
                                             0.1*np.log(2), 0.1*np.log(2),
                                             0.0*np.log(2), 0.1*np.log(2),
                                             0.0, -0.1*np.log(2),
                                             0.1*np.log(2), 0.0)},
                        'BT0238':{'Heparin': (-3.2*np.log(2), -3.7*np.log(2)),
                                  'Chondroitin sulfate': (-2.6*np.log(2), -2.6*np.log(2)),
                                  'Corn starch': (-0.4*np.log(2), +0.7*np.log(2), +0.1*np.log(2)),
                                  'Amylopectin':(-0.1*np.log(2), -0.2*np.log(2), 0.0*np.log(2), -0.1*np.log(2)),
                                  'Amylose': (-0.1*np.log(2), +0.6*np.log(2)),
                                  'Arabinose':(0.6*np.log(2), 1.0*np.log(2)),
                                  'Fructose': (0.2*np.log(2), 0.1*np.log(2)),
                                  'Galactose':(0.4*np.log(2), 0.6*np.log(2)),
                                  'Glucose':(0.1*np.log(2), 0.2*np.log(2),
                                             0.5*np.log(2), 0.7*np.log(2),
                                             0.6*np.log(2), 0.6*np.log(2),
                                             0.0*np.log(2), 0.4*np.log(2),
                                             0.3*np.log(2), -0.3*np.log(2))}
                        }

# Define genes (which Tn lineages qualify as part of gene)
## Here, defined as any Tn within (start - 100, end) of gene
def call_conservative_gene_indices(lst_of_genes, gene_lineage_map):
    gene_lineage_indices = {}

    for gene in lst_of_genes:
        coords, strand = gene_lineage_map[gene]['coords']
        gene_indices = list(np.copy(gene_lineage_map[gene]['indices']))

        # gene_indices = [idx for idx in gene_indices if day0_freqs[idx] > 10**-5.5]
        # for i, idx in zip(np.arange(len(gene_indices)), np.copy(gene_indices)):
        #     pos = gene_lineage_map[gene]['positions'][i]
        #     if strand == '+':
        #         rel_pos = (pos - coords[0]) / (coords[1]-coords[0])
        #     if strand == '-':
        #         rel_pos = (coords[1] - pos) / (coords[1]-coords[0])
        #     if rel_pos < 0.1 or rel_pos > 0.9:
        #         gene_indices.remove(idx)
        #         continue
        for i, idx in zip(np.arange(len(gene_indices)), np.copy(gene_indices)):
            pos = gene_lineage_map[gene]['positions'][i]
            if strand == '+':
                rel_pos = (pos - (coords[0]-100)) / ( coords[1] - (coords[0]-100) )
            if strand == '-':
                rel_pos = ((coords[1]+100) - pos) / ( (coords[1]+100) -coords[0] )
            if rel_pos < 0 or rel_pos > 1:
                gene_indices.remove(idx)
                continue

        if len(gene_indices) == 0:
            continue
        gene_lineage_indices[gene] = gene_indices
    return gene_lineage_indices

# Get arrays for fitness profiling
def split_input_pools(input_array, n_pools, day0_split=None):
    input_sizes = input_array.sum(axis=1)
    input_argsort = np.argsort(input_sizes)[::-1] #sort arrays from biggest to smallest
    if day0_split is None:
        sampled_input_array = np.copy(input_array[input_argsort])
    elif day0_split in [1,2]:
        sampled_input_array = np.copy(input_array[input_argsort][day0_split%2::2])
    else:
        print(f'Error in day0_split value! {day0_split} (should be (None, 1, 2)')

    split_pools = np.zeros((n_pools, sampled_input_array.shape[1]))

    for i in range( sampled_input_array.shape[0] ): #add input library to current smallest pool
        smallest_pool_index = np.argmin(split_pools.sum(axis=1))
        split_pools[smallest_pool_index] += sampled_input_array[i]
    # alterantive SNAKE DRAFT deprecated
    # for i in range( sampled_input_array.shape[0] ): # SNAKE DRAFT!!
    #     if i//n_pools % 2 == 0: # forward
    #         index = i%n_pools
    #     else: # reverse
    #         index = -(1+i%n_pools)
    #     print(index)
    #     split_pools[index] += sampled_input_array[i]

    return split_pools

def get_r0_r1_arrays(diet, d0, d1, mice, read_arrays_meta, day0_split=None):
    input_array = read_arrays_meta['input_array']
    read_array = read_arrays_meta['read_array']
    row_ids = read_arrays_meta['row_ids']
    # input_array = day 0 inputs, read_array = array of (t>0) read libraries,

    if diet in BWH2_MEDIA or d0 == 0: # DAY 0
        r0_replicates = split_input_pools(input_array, len(mice), day0_split=day0_split)
    else:
        r0_replicates = np.array(read_array[[row_ids[(mouse, d0)] for mouse in mice]])

    if diet in BWH2_MEDIA:
        r1_replicates = np.array(read_array[[row_ids[(diet, i)] for i in mice]]) #mice=replicates
    else:
        r1_replicates = np.array(read_array[[row_ids[(mouse, d1)] for mouse in mice]])

    return r0_replicates, r1_replicates

def get_env_frequency_data(env_replicates, read_arrays_meta, day0_split=None):
    read_array = read_arrays_meta['read_array']
    f1_arr = np.zeros( (len(env_replicates), read_array.shape[1]) )
    f2_arr = np.zeros( (len(env_replicates), read_array.shape[1]) )
    D1_arr = np.zeros( len(env_replicates) )
    D2_arr = np.zeros( len(env_replicates) )

    for e, (diet, (d1, d2), mice) in enumerate(env_replicates):
        ## get relevant read arrays
        r1_replicates, r2_replicates = get_r0_r1_arrays(diet, d1, d2, mice, read_arrays_meta, day0_split=day0_split)
        f1 = r1_replicates.sum(axis=0) / r1_replicates.sum()
        f2 = r2_replicates.sum(axis=0) / r2_replicates.sum()

        f1_arr[e] = f1
        f2_arr[e] = f2
        D1_arr[e] = r1_replicates.sum()
        D2_arr[e] = r2_replicates.sum()

    return f1_arr, D1_arr, f2_arr, D2_arr

# # Get leave-one-out distributions of gene fitnesses
def maxmin_freqs_over_env(freq0, depth0, freq1, depth1):
    overD0 = 1/depth0
    overD1 = 1/depth1

    min_f0 = np.min([freq1, overD0], axis=0)
    maxmin_f0 = np.max([freq0, min_f0], axis=0)

    min_f1 = np.min([freq0, overD1], axis=0)
    maxmin_f1 = np.max([freq1, min_f1], axis=0)

    where_both_zero = (maxmin_f0 + maxmin_f1) == 0
    maxmin_f0 = np.ma.masked_array(maxmin_f0, where_both_zero)
    maxmin_f1 = np.ma.masked_array(maxmin_f1, where_both_zero)

    return maxmin_f0, maxmin_f1

def maxmin_freqs(freqs0, depth0, freqs1, depth1):
    overD0 = np.full(freqs0.shape[-1], 1 / depth0)
    overD1 = np.full(freqs1.shape[-1], 1 / depth1)

    min_f0 = np.min([freqs1, overD0], axis=0)
    maxmin_f0 = np.max([freqs0, min_f0], axis=0)

    min_f1 = np.min([freqs0, overD1], axis=0)
    maxmin_f1 = np.max([freqs1, min_f1], axis=0)

    return maxmin_f0, maxmin_f1

def calc_cg_fitness_leave_one_in(f0, f1, D0, D1, maxmin=True):
    """ Expects f0, f1 as array of frequencies, single dimension """
    if f0.ndim == 1:
        lfcs = np.zeros(f0.shape[0])
        for i in range(lfcs.shape[0]):
            f0_xi = f0[i]
            f1_xi = f1[i]
            if maxmin:
                f0_xi, f1_xi = maxmin_freqs(f0_xi, D0, f1_xi, D1)

            with np.errstate(divide='ignore'):
                lfcs[i]  = np.log(f1_xi / f0_xi)
    elif f0.ndim == 2: #multiple environments, first dim = env, second dim = lineage
        lfcs = np.ma.zeros(f0.shape)
        lfcs.mask = np.full(f0.shape, False)
        for i in range(lfcs.shape[1]):
            f0_xi = f0[:, i]
            f1_xi = f1[:, i]
            if maxmin:
                f0_xi, f1_xi = maxmin_freqs_over_env(f0_xi, D0, f1_xi, D1)

            with np.errstate(divide='ignore'):
                lfcs[:, i]  = np.log(f1_xi / f0_xi)
            lfcs.mask[:, i] = f0_xi.mask

    return lfcs

def calc_cg_fitness_leave_one_out(f0, f1, D0, D1, maxmin=True):
    """ Expects f0, f1 as array of frequencies, single dimension """
    if f0.ndim == 1:
        lfcs = np.zeros(f0.shape[0])
        for i in range(lfcs.shape[0]):
            f0_xi = f0[:i].sum() + f0[i+1:].sum()
            f1_xi = f1[:i].sum() + f1[i+1:].sum()
            if maxmin:
                f0_xi, f1_xi = maxmin_freqs(f0_xi, D0, f1_xi, D1) #maxmin
            with np.errstate(divide='ignore'):
                lfcs[i]  = np.log(f1_xi / f0_xi)
    elif f0.ndim == 2: #multiple environments, first dim = env, second dim = lineage
        lfcs = np.ma.zeros(f0.shape)
        lfcs.mask = np.full(f0.shape, False)
        for i in range(lfcs.shape[1]):
            f0_xi = f0[:, :i].sum(axis=1) + f0[:, i+1:].sum(axis=1)
            f1_xi = f1[:, :i].sum(axis=1) + f1[:, i+1:].sum(axis=1)

            if maxmin:
                f0_xi, f1_xi = maxmin_freqs_over_env(f0_xi, D0, f1_xi, D1)

            with np.errstate(divide='ignore'):
                lfcs[:, i]  = np.log(f1_xi / f0_xi)
            lfcs.mask[:, i] = f0_xi.mask

    return lfcs

def get_leave_one_in_distributions(gene_lineage_indices_map, freqs0, freq, D0_arr, D1_arr):
    gene_loi_lfcs = {}

    if freqs0.ndim == 1:
        for gene, indices in gene_lineage_indices_map.items():
            f0 = freqs0[indices]
            f1 = freq[indices]

            gene_leave_one_in_lfcs = calc_cg_fitness_leave_one_in(f0, f1, D0_arr, D1_arr)
            gene_loi_lfcs[gene] = gene_leave_one_in_lfcs

    elif freqs0.ndim == 2:
        for gene, indices in gene_lineage_indices_map.items():
            f0 = freqs0[:, indices]
            f1 = freq[:, indices]

            gene_leave_one_in_lfcs = calc_cg_fitness_leave_one_in(f0, f1, D0_arr, D1_arr)
            gene_loi_lfcs[gene] = gene_leave_one_in_lfcs

    return gene_loi_lfcs

def get_leave_one_out_distributions(gene_lineage_indices_map, freqs0, freq, D0_arr, D1_arr):
    gene_loo_lfcs = {}

    if freqs0.ndim == 1:
        for gene, indices in gene_lineage_indices_map.items():
            f0 = freqs0[indices]
            f1 = freq[indices]

            gene_leave_one_out_lfcs = calc_cg_fitness_leave_one_out(f0, f1, D0_arr, D1_arr)
            gene_loo_lfcs[gene] = [indices, gene_leave_one_out_lfcs]

    elif freqs0.ndim == 2:
        for gene, indices in gene_lineage_indices_map.items():
            f0 = freqs0[:, indices]
            f1 = freq[:, indices]

            gene_leave_one_out_lfcs = calc_cg_fitness_leave_one_out(f0, f1, D0_arr, D1_arr)
            gene_loo_lfcs[gene] = [indices, gene_leave_one_out_lfcs]

    return gene_loo_lfcs


# Pass through genes and screen out likely secondary adaptive mutants
def get_outlier_from_kmeans(loo_lfcs):
    try:
        clusters, score = scipy.cluster.vq.kmeans(loo_lfcs.transpose(), 2)
    except:
        return None
    if len(clusters) != 2:
        # print('More than 2 clusters!')
        return None

    cluster_labels = [[], []]

    for idx, loo_lfc in enumerate(loo_lfcs.transpose()):
        z = np.argmin( np.sum(np.abs(loo_lfc - clusters), axis=1) )
        cluster_labels[z].append(idx)

    if len(cluster_labels[0]) == 1:
        idx = cluster_labels[0][0]
        loo_lfc = loo_lfcs[:, cluster_labels[0][0]]
        other_lfc = clusters[1]
        if np.max( np.abs(loo_lfc-other_lfc) ) > 1:
            return idx
        else:
            return None

    if len(cluster_labels[1]) == 1:
        idx = cluster_labels[1][0]
        loo_lfc = loo_lfcs[:, cluster_labels[1][0]]
        other_lfc = clusters[0]
        if np.max( np.abs(loo_lfc-other_lfc) ) > 1:
            return idx
        else:
            return None

    else:
        return None

def remove_secondary_mutant_lineages(gene_loo_lfcs, freqs0, freq, D0_arr, D1_arr, gene=None, max_removed=1):
    indices = list(range(gene_loo_lfcs.shape[1]))

    loo_lfcs = np.copy(gene_loo_lfcs)
    loo_lfcs[np.isnan(loo_lfcs)] = 0

    secondary_idx = get_outlier_from_kmeans(loo_lfcs)
    secondary_mutations = []
    while secondary_idx != None and loo_lfcs.shape[1] > 3 and len(secondary_mutations) < max_removed:#np.min([3, len(indices)//5 + 2]):
        secondary_mutations.append(secondary_idx)
        subindices = [index for index in indices if index not in secondary_mutations] # INDEXING PROBLEM...
        f0 = freqs0[:, subindices]
        f1 = freq[:, subindices]

        loo_lfcs = calc_cg_fitness_leave_one_out(f0, f1, D0_arr, D1_arr)
        loo_lfcs[np.isnan(loo_lfcs)] = 0
        # loo_lfcs[loo_lfcs == np.inf] = 10
        # loo_lfcs[loo_lfcs == -np.inf] = -10
        secondary_idx = get_outlier_from_kmeans(loo_lfcs)
        if secondary_idx:
            secondary_idx = subindices[secondary_idx]


    if len(secondary_mutations) > 0:
        included_indices = subindices
        excluded_indices = secondary_mutations
    else:
        included_indices = np.copy(indices)
        excluded_indices = []

    if len(included_indices) == 0:
        print(gene)
    f0 = freqs0[:, included_indices]
    f1 = freq[:, included_indices]
    loo_lfcs = calc_cg_fitness_leave_one_out(f0, f1, D0_arr, D1_arr)
    return loo_lfcs, included_indices, excluded_indices

def make_pared_gene_lfcs_dict(freqs0, freq, D0_arr, D1_arr, gene_loo_lfcs, gene_lineage_map, max_removed=1):
    gene_pared_lfcs = {gene:{} for gene in gene_loo_lfcs.keys()}
    for g, (gene, [indices, loo_lfcs]) in enumerate(gene_loo_lfcs.items()):
        indices = np.array(indices, dtype='int')
        f0, f1 = freqs0[:, indices], freq[:, indices]
        all_indices = gene_lineage_map[gene]['indices']
        gene_pared_lfcs[gene]['all_indices'] = all_indices

        loo_lfcs, included, excluded = remove_secondary_mutant_lineages(loo_lfcs, f0, f1, D0_arr, D1_arr, gene=gene, max_removed=max_removed)
        included_indices = indices[included]
        excluded_indices = indices[excluded]

        gene_pared_lfcs[gene]['loo_lfcs'] = loo_lfcs
        gene_pared_lfcs[gene]['included_indices'] = included_indices
        gene_pared_lfcs[gene]['excluded_indices'] = excluded_indices
        gene_pared_lfcs[gene]['included_f0'] = f0[:, included]
        gene_pared_lfcs[gene]['included_f1'] = f1[:, included]

        # all_f0, all_f1 = shared.maxmin_freqs(f0[:, included].sum(axis=1), D0_arr, f1[:, included].sum(axis=1), D1_arr)
        all_f0, all_f1 = f0[:, included].sum(axis=1), f1[:, included].sum(axis=1)

        f0_included, f1_included = f0[:, included], f1[:, included]
        loi_lfcs = calc_cg_fitness_leave_one_in(f0_included, f1_included, D0_arr, D1_arr)

        all_f0, all_f1 = maxmin_freqs_over_env(all_f0, D0_arr, all_f1, D1_arr)

        with np.errstate(divide='ignore'):
            gene_pared_lfcs[gene]['lfc'] = np.log(all_f1 / all_f0)
        gene_pared_lfcs[gene]['loi_lfcs'] = loi_lfcs
        if g % 500 == 0:
            print(f"\r>> Completed... {g} genes.", end='', flush=True)
    print(f'\rComplete! {" "*20}')
    return gene_pared_lfcs

def make_env_replicate_tuples(bac, split=None):
    if split is None: # tuple is (label, (t0, t1), [replicate set])
        replicate_tuples = [('LF', (0,4), [11, 12, 13, 15, 16, 18, 19]),
                  ('LF', (4, 10), [16,18,19]),
                  ('HLH', (4, 10), [1,2,3,5]),
                  ('LF', (10, 16), [16,18,19]),
                  ('LHL', (10, 16), [11,12,13,15]),
                  ('HF', (0,4), [1,2,3,5,6,7,8,9,10]),
                  ('HF', (4,10), [6,7,8,9,10]),
                  ('LHL', (4,10), [11,12,13,15]),
                  ('HF', (10, 16), [7,8,9,10]),
                  ('HLH', (10, 16), [1,2,3,5])]
        vitro_reps = [1, 2] #BWH2
    elif split == 1: #split such that there are no shared timepoints in across environments!
        replicate_tuples = [('LF', (0,4), [11, 12, 19]),
                  ('LF', (4, 10), [16,18]),
                  ('HLH', (4, 10), [3,5]),
                  ('LF', (10, 16), [19]),
                  ('LHL', (10, 16), [11,12]),
                  ('HF', (0,4), [1,2,9,10]),
                  ('HF', (4,10), [6,7,8]),
                  ('LHL', (4,10), [13,15]),
                  ('HF', (10, 16), [9,10]),
                  ('HLH', (10, 16), [1,2])]
        vitro_reps = [1] #BWH2
    elif split == 2:
        replicate_tuples = [('LF', (0,4), [13, 15, 16, 18]),
                  ('LF', (4, 10), [19]),
                  ('HLH', (4, 10), [1,2]),
                  ('LF', (10, 16), [16, 18]),
                  ('LHL', (10, 16), [13,15]),
                  ('HF', (0,4), [3,5,6,7,8]),
                  ('HF', (4,10), [9,10]),
                  ('LHL', (4,10), [11,12]),
                  ('HF', (10, 16), [6,7,8]),
                  ('HLH', (10, 16), [3,5])]
        vitro_reps = [2] #BWH2
    if bac == 'BWH2':
        for medium in BWH2_MEDIA:
            replicate_tuples.append( (medium, (0,1), vitro_reps))
    return replicate_tuples

def make_env_labels(replicate_tuples):
    labels = []
    for env_tuple in replicate_tuples:
        env = env_tuple[0]
        t0, t1 = env_tuple[1]
        if env in BWH2_MEDIA:
            label = env
        else:
            label = f'{env} {t0}-{t1}'
        labels.append(label)
    return labels


# # Get individual lineage fitness profiles
def generate_environment_lfc_arrays(array_of_f1, array_of_D1, array_of_f2, array_of_D2, err_type='empirical'):
    fitnesses = []
    expected_rep_errs = []
    for (f1, D1, f2, D2) in zip(array_of_f1, array_of_D1, array_of_f2, array_of_D2):
        f1_mm, f2_mm = maxmin_freqs(f1, D1, f2, D2)

        fitness = np.log(f2_mm / f1_mm)
        expected_err = calc_replicate_fitness_err(f1, D1, f2, D2)

        fitnesses.append(fitness)
        expected_rep_errs.append(expected_err)

    fitnesses = np.array(fitnesses)
    if err_type == 'empirical':
        errs = np.std(fitnesses, axis=0, ddof=1)
    elif err_type == 'expected':
        errs = 1/len(expected_rep_errs) * np.sqrt( np.sum( [err**2 for err in expected_rep_errs] ) )
    return fitnesses, errs

def calc_replicate_fitness_err(f0, D0, f1, D1, log=True):
    if log:
        return np.sqrt( 1/(f0*D0) + 1/(f1*D1) )
    else:
        return np.sqrt( f1/f0**2 / D1 + f1**2/f0**3 / D0 )

def calc_paired_replicate_fitness_err(fitnes, err1, fitness2, err2, kappa):
    # calculates conservatively: uses empirical estimate of mean lineage error
    # if larger than inferred estimate based on (crude) noise model
    return np.max( [np.sqrt( 1/2*(fitnes-fitness2)**2 ), kappa / 2 * np.sqrt(err1**2 + err2**2)] )

def calc_replicate_fitness_errs(fitness_err_pairs, kappa):
    # calculates conservatively: uses empirical estimate of mean lineage error
    # if larger than inferred estimate based on (crude) noise model
    n_reps = len(fitness_err_pairs)

    fitnesses = np.array([fitness for fitness, err in fitness_err_pairs])
    errs = np.array([err for fitness, err in fitness_err_pairs])

    empirical_err = np.std(fitnesses, axis=0, ddof=1)
    inferred_errs = np.sqrt( np.sum(errs**2, axis=0) )

    return np.max( [empirical_err, kappa/n_reps * inferred_errs], axis=0 )

def generate_mouse_replicate_profiles(env_replicates, lineage_bool, read_arrays_meta, empirical=True):
    lineage_profiles_data = []
    lineage_profile_replicates = []

    for e, (diet, (d1, d2), mice) in enumerate(env_replicates):
        ## get relevant read arrays
        r1_replicates, r2_replicates = get_r0_r1_arrays(diet, d1, d2, mice, read_arrays_meta)

        ## calculate fitnesses
        fitnesses = np.zeros((r1_replicates.shape[0], lineage_bool.sum()))
        errs = np.zeros((r1_replicates.shape[0], lineage_bool.sum()))
        n_mice = np.full(lineage_bool.sum(), len(mice))
        for rep, (r1, r2) in enumerate(zip(r1_replicates, r2_replicates)):
            # D1, D2 = r1.sum(), r2.sum()
            D1 = r1.sum()
            D2 = r2.sum()
            f1, f2 = r1[lineage_bool] / D1, r2[lineage_bool] / D2

            f1_mm, f2_mm = maxmin_freqs(f1, D1, f2, D2)

            fitness = np.log(f2_mm / f1_mm)
            err = calc_replicate_fitness_err(f1_mm, D1, f2_mm, D2, log=True)

            isnan = np.isnan(fitness)
            # fitness[isnan] = 0
            # err[isnan] = 0
            n_mice[isnan] -= 1

            fitnesses[rep] = fitness
            errs[rep] = err

        ##
        masked_fitnesses =  np.ma.masked_array(fitnesses, np.isnan(fitnesses))
        mean_fitnesses = np.sum(masked_fitnesses, axis=0) / n_mice
        masked_errs = np.ma.masked_array(errs, np.isnan(errs))
        empirical_errs = 1/np.sqrt(n_mice-1) * np.sqrt(np.sum((masked_fitnesses-mean_fitnesses)**2, axis=0))

        inferred_errs = 1/np.sqrt(n_mice)*np.sqrt(np.sum(masked_errs**2, axis=0) )

        if empirical:
            normed_errs = empirical_errs
        else:
            # kappa = kappa_dict[(diet, d0, d1)]
            kappa = 1
            normed_errs = kappa  * inferred_errs

        # errs = np.max([empirical_errs, normed_errs], axis=0)
        # normed_errs[ normed_errs < 0.1] = 0.1
        errs = normed_errs

        # normed_errs = np.einsum('ij, j->ij', (fitnesses - np.mean(fitnesses, axis=0)), np.array(inferred_errs)**-1) / kappa
        lineage_profile_replicates.append(fitnesses)
        lineage_profiles_data.append( [mean_fitnesses, errs, n_mice] )

    lineage_profiles_data = np.swapaxes(lineage_profiles_data, 1, 0)
    lineage_profiles_data = np.swapaxes(lineage_profiles_data, 1, 2)
    return lineage_profiles_data, lineage_profile_replicates


### Distance measures between profiles

def distance(profile1, profile2):
    min_envs = profile1.shape[0]//2 + profile1.shape[0]%2
    if type(profile1) == np.ma.core.MaskedArray:
        paired_mask = np.any(profile1.mask + profile2.mask)
        if (paired_mask).sum() > profile1.shape[0]-min_envs: #if not measured in at least one profile in >half of envts in
        # if (paired_mask).sum() > 0:
            return np.inf
        else:
            return np.sqrt( np.sum((profile1[paired_mask]-profile2[paired_mask])**2) )
    else:
        return np.sqrt( np.sum((profile1-profile2)**2) )
    
def parallelized_distance(profile1, arr_of_profile2, arr_of_stds=None):
    if profile1.ndim == 2:
        distances = np.zeros( (profile1.shape[0], arr_of_profile2.shape[0]) )
        for r, prof1 in enumerate(profile1):
            distances[r] = np.sqrt( np.sum( (prof1-arr_of_profile2)**2, axis=1) )
    elif np.any(arr_of_stds):
        distances = np.sqrt( np.sum( ( (profile1-arr_of_profile2)/arr_of_stds )**2, axis=1) )
    else:
        distances = np.sqrt( np.sum( (profile1-arr_of_profile2)**2, axis=1) )
        
    return distances
def get_all_pairwise_distances(profiles):
    pairwise_distances = []
    for i in range(len(profiles)-1):
        profile_i = profiles[i]
        for j in range(i+1, len(profiles)):
            profile_j = profiles[j]
            pairwise_distances.append(distance(profile_i, profile_j))
    return pairwise_distances
def distance_rectangular_matrices(A, B):
    """ A, B should share same column dimension"""
    distances = np.zeros( (A.shape[0], B.shape[0]) )
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            distances[i,j] = np.sqrt( np.sum((a-b)**2) )
    return distances

# fitness profile plot formatting
def format_profile_ax(ax, labels, splits=(5,10)):
    ax.set_xlim(ax.get_xlim())
    ax.axvspan(ax.get_xlim()[0], splits[0]-0.5, color='blue', alpha=0.1, lw=0)
    ax.axvspan(splits[0]-0.5, splits[1]-0.5, color='red', alpha=0.1, lw=0)
    ax.set_xticks( np.arange(len(labels) ) )
    ax.set_xticklabels( labels, rotation=+45, fontsize=8, ha='right')
    ax.axhline(1, color='black', linestyle='dashed', zorder=0)
def make_genome_colorbar(ax, genome_length):
    ax.set_xlim(0, 1)
    ax.set_ylim(-10**5, genome_length+10**5)
    # ax.axvspan(*ax.get_xlim(), color='grey', alpha=0.5)
    ax.fill_between([0.2, 0.8], y1=0, y2=genome_length, color='grey', alpha=0.5, lw=0)
    # ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel('Genome position (Mb)', fontsize=6, labelpad=2)
    ax.set_yticks( np.arange(10**6, genome_length, 10**6) )
    ax.set_yticklabels( np.arange(1, genome_length // 10**6 + 1,  dtype=int))

### gene replicate profiling

def make_gene_split_profiles(gene_profile_data, f0_arr, f1_arr, D0_arr, D1_arr, exclude_indices=[]):
    include_indices = list( gene_profile_data['included_indices'] )
    for idx in exclude_indices:
        if idx in include_indices:
            include_indices.remove(idx)

    gene_lineage_f0 = f0_arr[:, include_indices]
    gene_lineage_f1 = f1_arr[:, include_indices]
    all_f0, all_f1 = gene_lineage_f0.sum(axis=1), gene_lineage_f1.sum(axis=1)
    mm_f0, mm_f1 = maxmin_freqs_over_env(all_f0, D0_arr, all_f1, D1_arr)

    index_permutation = rnd.permutation( len(include_indices) )
    S1 = index_permutation[::2]
    S2 = index_permutation[1::2]

    S1_f0, S1_f1 = gene_lineage_f0[:, S1].sum(axis=1), gene_lineage_f1[:, S1].sum(axis=1)
    S1_mm_f0, S1_mm_f1 = maxmin_freqs_over_env(S1_f0, D0_arr, S1_f1, D1_arr)

    S2_f0, S2_f1 = gene_lineage_f0[:, S2].sum(axis=1), gene_lineage_f1[:, S2].sum(axis=1)
    S2_mm_f0, S2_mm_f1 = maxmin_freqs_over_env(S2_f0, D0_arr, S2_f1, D1_arr)
    

    all_profile = np.log(mm_f1/mm_f0)
    S1_profile = np.log(S1_mm_f1/S1_mm_f0)
    S2_profile = np.log(S2_mm_f1/S2_mm_f0)
    
    all_profile = np.ma.masked_array(all_profile, mm_f1+mm_f0 == 0)
    S1_profile = np.ma.masked_array(S1_profile, S1_mm_f1+S1_mm_f0 == 0)
    S2_profile = np.ma.masked_array(S2_profile, S2_mm_f1+S2_mm_f0 == 0)


    return all_profile, S1_profile, S2_profile

def generate_gene_replicate_profiles(gene_lst, all_gene_profile_data, f0_arr, f1_arr, D0_arr, D1_arr, exclude_indices=[]):
    gene_replicate_profiles = {}

    for gene in gene_lst:
        try:
            gene_profile_data = all_gene_profile_data[gene]
        except:
            continue
        include_indices = list(gene_profile_data['included_indices'])
        for idx in exclude_indices:
            if idx in include_indices:
                include_indices.remove(idx)

        if len(include_indices) < 2:
            continue

        all_profile, S1_profile, S2_profile = make_gene_split_profiles(gene_profile_data, f0_arr, f1_arr, D0_arr, D1_arr)

        replicate_distance = distance(S1_profile, S2_profile)
        gene_replicate_profiles[gene] = [all_profile, S1_profile, S2_profile, replicate_distance]
    return gene_replicate_profiles

def filter_gene_replicate_profiles(gene_replicate_profiles, all_gene_profile_data, f0_arr, f1_arr, D0_arr, D1_arr, lineage_gene_map, exclude_indices=[], exclude_positions=[]):
    filtered_gene_replicate_profiles = copy.copy(gene_replicate_profiles)

    for index, genome_position in zip(exclude_indices,exclude_positions):
        if genome_position in lineage_gene_map:
            position_genes = lineage_gene_map[genome_position]
        else:
            continue

        for gene in position_genes:
            gene_profile_data = all_gene_profile_data[gene]
            include_indices = list(gene_profile_data['included_indices'])
            for idx in exclude_indices:
                if idx in include_indices:
                    include_indices.remove(idx)
                else:
                    continue

            if len(include_indices) < 2 and gene in filtered_gene_replicate_profiles:
                del filtered_gene_replicate_profiles[gene]

            if len(include_indices) < 2 and gene not in filtered_gene_replicate_profiles:
                continue

            all_profile, S1_profile, S2_profile = make_gene_split_profiles(gene_profile_data, f0_arr, f1_arr, D0_arr, D1_arr, exclude_indices=exclude_indices)
            rep_distance = distance(S1_profile, S2_profile)
            filtered_gene_replicate_profiles[gene] = [all_profile, S1_profile, S2_profile, rep_distance]
    return filtered_gene_replicate_profiles


###
def calculate_cluster_bands(rep1_profiles, rep2_profiles, percentiles=(25,75)):
    # concatenated_profiles = np.concatenate([rep1_profiles, rep2_profiles], axis=0)

    concatenated_profiles = []
    for p1, p2 in zip(rep1_profiles, rep2_profiles):
        concatenated_profiles.append( np.median([p1, p2], axis=0) )

    if len(rep1_profiles) < 10:
        lower_profile = np.percentile(concatenated_profiles, 0, axis=0)
        upper_profile = np.percentile(concatenated_profiles, 100, axis=0)

    else:
        lower_profile = np.percentile(concatenated_profiles, percentiles[0], axis=0)
        upper_profile = np.percentile(concatenated_profiles, percentiles[1], axis=0)

    # mean_profile = np.mean(concatenated_profiles, axis=0)
    # std_profile = np.std(concatenated_profiles, axis=0)
    # lower_profile = mean_profile - std_profile
    # upper_profile = mean_profile + std_profile

    return lower_profile, upper_profile
