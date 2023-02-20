root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared
import scipy.optimize
import itertools

###
def generate_empirical_ordered_fitnesses(bac, mice, t=4, min_freq=-1):
    nonWu = shared.bac_nonwu_indices[bac]
    read_arrays = shared.bac_read_arrays[bac][:, nonWu]
    row_ids = shared.bac_row_ids[bac]
    day0_freqs = read_arrays[0] / read_arrays[0].sum()

    mice_reads = read_arrays[ [row_ids[(mouse, t)] for mouse in mice ] ]
    avg_freqs = mice_reads.sum(axis=0) / mice_reads.sum()

    f0, f1 = shared.maxmin_freqs(day0_freqs, read_arrays[0].sum(), avg_freqs, mice_reads.sum())
    lineage_indices = day0_freqs > min_freq

    empirical_fitnesses = shared.calc_lfc_array(f0, f1, t)
    empirical_fitnesses[ empirical_fitnesses.mask ] = -1

    sorted_indices = np.argsort(empirical_fitnesses)[::-1]
    sorted_fitnesses = empirical_fitnesses[sorted_indices]
    sorted_indices = np.arange(lineage_indices.sum())[sorted_indices]

    return sorted_fitnesses, sorted_indices

### bootstrapped uncertainty estimators for downstream weighted regression
def bootstrap_mean(freqs, n_bootstrap=1000):
    bootstrap_samples = []
    z = np.arange(freqs.shape[-1])
    for i in range(n_bootstrap):
        bootstrap = rnd.choice(z, len(z), replace=True)
        bootstrap_samples.append( np.mean(freqs[bootstrap]) )
    return np.mean(freqs), np.std(bootstrap_samples)


def calculate_selection_g(f0, freqs, average=True, bootstrap=0):
    if not bootstrap:
        fold_change = freqs/f0
        with np.errstate(divide='ignore'):
            log_fold_change = np.log(fold_change)

        if average:
            numerator = np.mean( f0*fold_change*(fold_change-1)/log_fold_change )
        else:
            numerator = f0*fold_change*(fold_change-1)/log_fold_change
        return numerator
    else:
        fold_change = freqs/f0
        with np.errstate(divide='ignore'):
            log_fold_change = np.log(fold_change)

        numerator = np.mean( f0*fold_change*(fold_change-1)/log_fold_change )
        bootstrap_err = bootstrap_selection_g(f0, freqs, n_bootstrap=bootstrap)

        return numerator, bootstrap_err

def bootstrap_selection_g(f0, freqs, n_bootstrap=1000):
    z = np.arange(freqs.shape[-1])
    bootstrap_samples = []
    for i in range(n_bootstrap):
        bootstrap = rnd.choice(z, len(z), replace=True)
        bootstrap_freqs = freqs[bootstrap]
        bootstrap_sample = calculate_selection_g(f0[bootstrap], bootstrap_freqs, average=True, bootstrap=False)
        bootstrap_samples.append(bootstrap_sample)
    return(np.std(bootstrap_samples))


def calculate_freq_variance(freqs, average=True, bootstrap=0):
    if not bootstrap:
        if average:
            return np.mean( np.var(freqs, axis=0, ddof=1) )
        else:
            return np.var(freqs, axis=0, ddof=1)
    else:
        var = np.mean( np.var(freqs, axis=0, ddof=1) )
        bootstrap_err = bootstrap_freq_variance_err(freqs, n_boostrap=bootstrap)
        return var, bootstrap_err

def bootstrap_freq_variance_err(freqs, n_bootstrap=1000):
    z = np.arange(freqs.shape[-1])
    pop_var = []
    for i in range(n_bootstrap):
        bootstrap = rnd.choice(z, len(z), replace=True)
        bootstrap_freqs = freqs[:, bootstrap]
        bootstrap_var = calculate_freq_variance(bootstrap_freqs, average=True, bootstrap=False)
        pop_var.append(bootstrap_var)
    return(np.std(pop_var))


def calculate_delta_freq_variance(freqs1, freqs2, average=True, bootstrap=0):
    # return np.var( (freqs1 - freqs2), ddof=1 )
    if not bootstrap:
        if average:
            return np.mean( (freqs1-freqs2)**2 )
        else:
            return (freqs1-freqs2)**2
    else:
        var = np.mean( (freqs1-freqs2)**2 )
        bootstrap_err = bootstrap_delta_freq_variance_err(freqs1, freqs2, n_bootstrap=bootstrap)
        return var, bootstrap_err

def bootstrap_delta_freq_variance_err(freqs1, freqs2, n_bootstrap=1000):
    z = np.arange(freqs1.shape[-1])
    pop_var = []
    for i in range(n_bootstrap):
        bootstrap = rnd.choice(z, len(z), replace=True)
        bootstrap1, bootstrap2 = freqs1[bootstrap], freqs2[bootstrap]
        bootstrap_var = calculate_delta_freq_variance(bootstrap1, bootstrap2, average=True, bootstrap=False)
        pop_var.append(bootstrap_var)
    return(np.std(pop_var))

### lineage groupings
def group_lineages_across_freq_ranges(init_freqs, init_freq_ranges, mean_freqs, start=0, min_group_size=100, min_freq_ratio=0.5):
    lineage_indices = np.arange(init_freqs.shape[-1])

    lineage_group_indices= []
    for (freq_min, freq_max) in init_freq_ranges:
        lineage_bool = (init_freqs >= freq_min) * (init_freqs < freq_max)
        fr_lineage_bools = group_lineages(init_freqs[lineage_bool], mean_freqs[lineage_bool],
                                          start=start, min_group_size=min_group_size, min_freq_ratio=min_freq_ratio)

        fr_lineage_indices = lineage_indices[lineage_bool]
        for (i1, i2) in fr_lineage_bools:
            lineage_group_indices.append(fr_lineage_indices[i1:i2])
    return lineage_group_indices

def group_lineages(f0, mean_freqs, start=0, min_group_size=100, min_freq_ratio=0.5):
    lineage_bool_ranges = []
    fold_changes = mean_freqs / f0
    while start < fold_changes.shape[-1]:
        max_fc = fold_changes[start]
        if max_fc < 1:
            break
        try:
            end = np.where( fold_changes/max_fc > min_freq_ratio)[0][-1]
            if end - start < min_group_size:
                start += 1
                continue
        except:
            start += 1
            continue

        lineage_bool_ranges.append((start, end))
        start = end+1
    return lineage_bool_ranges

def calc_estimators_in_groups(init_freqs, rep_reads, D1_arr, group_lineage_indices, splits, bootstrap=1000):
    x1 = []
    x1_err = []
    x2 = []
    x2_err = []
    y = []
    y_err = []
    num_lineages = []

    var_freqs = np.einsum('ij, i->ij', rep_reads[:splits[0]], D1_arr[:splits[0]]**-1.)
    x1_freqs = rep_reads[splits[0]:splits[0]+splits[1]].sum(axis=0) / D1_arr[splits[0]:splits[0]+splits[1]].sum()
    x2_freqs = rep_reads[splits[0]+splits[1]:].sum(axis=0) / D1_arr[splits[0]+splits[1]:].sum()

    for group_indices in group_lineage_indices:
        freqs0 = init_freqs[group_indices]

        if splits[0] == 2:
            df_var, df_var_err = calculate_delta_freq_variance(var_freqs[0][group_indices], var_freqs[1][group_indices], bootstrap=bootstrap)
        else:
            df_var, df_var_err = calculate_freq_variance(var_freqs[:, group_indices], bootstrap=bootstrap)

        mean_freq, mean_err = bootstrap_mean(x2_freqs[group_indices], n_bootstrap=bootstrap)
        g_numerator, g_err = calculate_selection_g(freqs0, x1_freqs[group_indices], bootstrap=bootstrap)

        if np.isnan(mean_freq) or np.isnan(g_numerator) or np.isnan(df_var):
            continue

        x1.append(g_numerator)
        x1_err.append(g_err)
        x2.append(mean_freq)
        x2_err.append(mean_err)
        y.append(df_var)
        y_err.append(df_var_err)
        num_lineages.append( len(group_indices) )

    return [x1, x1_err, x2, x2_err, y, y_err], num_lineages

def fit_multiple_regression(x1, x2, y, y_err, weighted=True):
    def lin_2var(x, a, b):
        x1, x2 = x
        return a * x1 + b * x2

    if weighted:
        popt, pcov = scipy.optimize.curve_fit(lin_2var, [x1, x2], y, sigma=y_err, p0=(10**-5, 10**-5))
    else:
        popt, pcov = scipy.optimize.curve_fit(lin_2var, [x1, x2], y, p0=(10**-5, 10**-5))
    return popt, pcov

####
def generate_unique_permutations(splits=(2, 2, 1)):
    """Assumes splits[0] = y1. num_mice = sum(splits)"""

    n_mice = np.sum(splits)
    n_y1, n_x1, n_x2 = splits[0], splits[1], splits[2]
    permutations = []
    for i in range(n_mice - 1):
        for j in range(i+1, n_mice): # y1 pairs
            y1_pair = [i, j]
            x_lst = [k for k in range(n_mice) if k not in [i, j]]

            for x1_set in itertools.combinations(x_lst, n_x1):
                x2_set = [k for k in range(n_mice) if k not in list(x1_set) + y1_pair]
                order = y1_pair + list(x1_set) + list(x2_set)
                permutations.append(order)
    return permutations

def generate_drift_estimates(bac, mice, splits, permutations, min_group_size=50, min_freq_ratio=0.7, bootstrap=1000):
    if np.sum(splits) != len(mice):
        print("Error! sum(splits) must = num. mice")

    grouped_weighted_drift_dict = {}
    day = 4
    group_read_range_factor = 1.2
    print(f'Inferring {bac} Ntau')
    if 'load empirical bac' != '':
        nonWu = shared.bac_nonwu_indices[bac]
        read_array = shared.bac_read_arrays[bac][:, nonWu]
        row_ids = shared.bac_row_ids[bac]
        D_lst = read_array[[row_ids[(mouse, day)] for mouse in mice]].sum(axis=1)
        init_reads = read_array[0]
        init_freqs = read_array[0] / read_array[0].sum()

        read_ranges = []
        read_min = 100  #
        while read_min < np.max(init_reads):
            read_max = read_min * group_read_range_factor
            read_ranges.append((read_min, read_max))
            read_min = read_max
        init_freq_ranges = np.array(read_ranges) / init_reads.sum()

    if 'generate frequencies and groups' != '':
        mice_reads = read_array[[row_ids[(mouse, day)] for mouse in mice]]
        D_lst = mice_reads.sum(axis=1)
        sorted_fitnesses, sorted_indices = generate_empirical_ordered_fitnesses(bac, mice, day)
        mask = init_freqs[sorted_indices] > 10**-6
        mice_reads = mice_reads[:, sorted_indices][:, mask]
        init_freqs = init_freqs[sorted_indices][mask]

        mean_freqs = mice_reads.sum(axis=0) / D_lst.sum()
        lineage_group_indices = group_lineages_across_freq_ranges(init_freqs, init_freq_ranges, mean_freqs,
                                                                  min_group_size=min_group_size,
                                                                  min_freq_ratio=min_freq_ratio)

    n_permutations = len(permutations)
    for z, order in enumerate(permutations):
        print(f'Starting permutation {z}, {z/n_permutations * 100:.0f}%', end='\r')
        reads = mice_reads[order]
        D_arr = D_lst[order]

        ### generate grouped lineages
        estimators, total_lineages = calc_estimators_in_groups(init_freqs, reads, D_arr,
                                                               lineage_group_indices, splits, bootstrap=bootstrap)
        tot_lineages = np.sum(total_lineages)
        x1, x1_err, x2, x2_err, y, y_err = estimators
        num_groups = len(x1)

        if num_groups < 3:
            popt, pcov = np.full(2, np.nan), np.full((2, 2), np.nan)
            grouped_weighted_drift_dict[tuple(order)] = (tot_lineages, -1, popt, pcov)
            continue
        else:
            popt, pcov = fit_multiple_regression(x1, x2, y, y_err, weighted=True)
            grouped_weighted_drift_dict[tuple(order)] = (tot_lineages, num_groups, popt, pcov)
    print('\nComplete!')
    return grouped_weighted_drift_dict

def make_sfig(fig, ax, mice, drift_dict):
    index_mouse_map = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7, 6:8, 7:9, 8:10}
    slopes = {mouse: [] for mouse in range(len(mice))}

    for key, fit in drift_dict.items():
        if fit[2][0] < 0:
            slopes[key[0]].append(1e-8)
            slopes[key[1]].append(1e-8)
        else:
            slopes[key[0]].append(fit[2][0])
            slopes[key[1]].append(fit[2][0])

    for mouse in range(9):
        mouse_slopes = slopes[mouse]

        mouse_Ntau = 8*np.array(mouse_slopes)**-1.
        median = np.median(mouse_Ntau)
        iqr_25 = np.percentile(mouse_Ntau, 25)
        iqr_75 = np.percentile(mouse_Ntau, 75)
        ax.errorbar( [mouse+1], [median], yerr=[[median-iqr_25],[iqr_75-median]], color=KELLY_COLORS[0], marker='o', mec='black')

        outliers = ((mouse_Ntau > iqr_75) + (mouse_Ntau < iqr_25))>0
        ax.scatter([mouse+1]*outliers.sum(), mouse_Ntau[outliers], color=KELLY_COLORS[0], marker='.', s=5)

        seq_depth = shared.bac_read_arrays[bac][shared.bac_row_ids[bac][(index_mouse_map[mouse],4)]][shared.bac_nonwu_indices[bac]].sum()
        ax.scatter(mouse+1, seq_depth, color='grey', marker='*')
    return

def format_sfig(ax):
    ax.set_yscale('log')
    ax.set_xlabel(r'HF/HS mouse')
    ax.set_ylabel(r'inferred $N\tau$ (days)')
    # ax.set_ylim(10**5, 2*10**8)

if __name__ == '__main__':
    bac = 'BWH2'

    ## DEMO PARAMS
    n_bootstrap = 100
    n_permutations = 100

    mice = [1, 2, 3, 5, 6, 7, 8, 9, 10] ## HF/HS
    splits = (2, 4, 3) #num mice to estimate (y1, x1, x2). y1 should = 2

    permutations = generate_unique_permutations(splits)
    drift_dict = generate_drift_estimates(bac, mice, splits,
                                          rnd.permutation(permutations[:])[:n_permutations], bootstrap=n_bootstrap)

    fig, ax = plt.subplots(figsize=(3, 2))
    make_sfig(fig, ax, mice, drift_dict)
    format_sfig(ax)

    fig.savefig(f'{plot_dir}/sfig17B_drift_BWH2_inference.pdf')
