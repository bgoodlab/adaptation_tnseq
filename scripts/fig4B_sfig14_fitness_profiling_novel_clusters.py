# path config
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
np.seterr(invalid='ignore')
np.seterr(all='ignore')
import pickle
from sklearn.cluster import AgglomerativeClustering as AgClustering

def get_profile_data(bac):
    with open(f'{data_dir}/{bac}_fitness_profiling_data.pkl', 'rb') as f:
        fitness_profiling_data = pickle.load(f)

    gene_average_profiles = fitness_profiling_data['gene_profile_data']
    gene_S1_profiles = fitness_profiling_data['gene_profile_S1_data']
    gene_S2_profiles = fitness_profiling_data['gene_profile_S2_data']
    gene_replicate_profiles = fitness_profiling_data['gene_profile_replicate_data']
    gene_profile_data = [gene_average_profiles, gene_S1_profiles, gene_S2_profiles, gene_replicate_profiles]

    lineage_average_profiles = fitness_profiling_data['lineage_profile_data']
    lineage_S1_profiles = fitness_profiling_data['lineage_profile_S1_data']
    lineage_S2_profiles = fitness_profiling_data['lineage_profile_S2_data']
    lineage_profile_data = [lineage_average_profiles, lineage_S1_profiles, lineage_S2_profiles]

    all_freq_arrays = fitness_profiling_data['all_freq_arrays']
    S1_freq_arrays = fitness_profiling_data['S1_freq_arrays']
    S2_freq_arrays = fitness_profiling_data['S2_freq_arrays']

    freq_data = [all_freq_arrays, S1_freq_arrays, S2_freq_arrays]
    # [f0_arr, D0_arr, f1_arr, D1_arr] = all_freq_arrays
    # [S1_f0_arr, S1_D0_arr, S1_f1_arr, S1_D1_arr] = S1_freq_arrays
    # [S2_f0_arr, S2_D0_arr, S2_f1_arr, S2_D1_arr] = S1_freq_arrays

    return gene_profile_data, lineage_profile_data, freq_data

def get_seed_and_candidate_data(bac, lineage_profile_data):
    """
    Get data (indices, positions, fitness profiles) for cluster seeding and other candidate lineages
    """
    lineage_average_profiles, lineage_S1_profiles, lineage_S2_profiles = lineage_profile_data

    seed_indices = []
    seed_positions = []
    for i, (prof, n_rep) in enumerate(zip(lineage_average_profiles[0], lineage_average_profiles[2])):
        # if measured in all environments, and strongly beneficial in vivo...
        if (n_rep > 0).sum() == len(n_rep) and np.max(prof[:-5]) > np.log(10):
            seed_indices.append(i)
            position = shared.bac_lineage_positions[bac][i]
            seed_positions.append(position)

    seed_indices = np.array(seed_indices)
    seed_positions = np.array(seed_positions)
    seed_profiles = np.array([lineage_average_profiles[0, index] for index in seed_indices])

    candidate_indices = []
    candidate_positions = []
    for i, (prof, n_rep) in enumerate(zip(lineage_average_profiles[0], lineage_average_profiles[2])):
        if (n_rep > 0).sum() == len(n_rep):
            candidate_indices.append(i)

            position = shared.bac_lineage_positions[bac][i]
            candidate_positions.append(position)
    candidate_profiles = np.array([lineage_average_profiles[0, index] for index in candidate_indices])

    print(f'seed lineage counts: {len(seed_indices)}')
    print(f'candidate lineage counts: {len(candidate_indices)}')

    seed_data = [seed_indices, seed_positions, seed_profiles]
    candidate_data = [candidate_indices, candidate_positions, candidate_profiles]

    return seed_data, candidate_data

def cluster_profiles(seed_profiles, cutoff=4):
    print('Clustering... ', end="")
    clustering = AgClustering(n_clusters=None, linkage='average',
                          distance_threshold=cutoff).fit(seed_profiles)
    print('complete!')
    return clustering

def get_clusters(clustering, seed_indices, seed_positions, seed_profiles):
    clusters = {}
    seed_cluster_ids = clustering.fit_predict(seed_profiles)
    index_to_cluster_map = {}
    for z, (cluster_id, seed_index, seed_position, seed_profile) in enumerate(zip(seed_cluster_ids, seed_indices, seed_positions, seed_profiles)):
        if cluster_id not in clusters:
            clusters[cluster_id] = [[], [], []]
        clusters[cluster_id][0].append(seed_index)
        clusters[cluster_id][1].append(seed_position)
        clusters[cluster_id][2].append(seed_profile)
        index_to_cluster_map[seed_index] = cluster_id
    cluster_sizes = [len(cluster[0]) for cluster_id, cluster in clusters.items()]
    print(f"After seeding: N_clusters = {len(clusters)}, MEDIAN {int(np.median(cluster_sizes))}, MIN {np.min(cluster_sizes)}, MAX {np.max(cluster_sizes)}")
    return clusters, index_to_cluster_map

def expand_clusters(clusters, index_to_cluster_map, candidate_indices, candidate_positions, candidate_profiles, cutoff=4):
    cluster_mean_profiles = []
    for cluster_id, (cluster_indices, cluster_positions, cluster_profiles) in sorted(clusters.items()):
        cluster_mean = np.mean(cluster_profiles, axis=0)
        cluster_mean_profiles.append(cluster_mean)

    print(f'Generating cluster distances for all lineages... ', end='')
    lineage_cluster_distances = fp.distance_rectangular_matrices(candidate_profiles, np.array(cluster_mean_profiles))
    print(f'complete!')

    print(f'Clustering all candidate lineages... ', end='')
    cluster_ids = np.argmin(lineage_cluster_distances, axis=1)
    lineage_counts = 0
    for i, cluster_id in enumerate(cluster_ids):
        lineage_idx = candidate_indices[i]
        score = lineage_cluster_distances[i][cluster_id]

        if lineage_idx in index_to_cluster_map:
            lineage_counts += 1
            continue
        if score > cutoff:
            continue

        else:
            lineage_profile = candidate_profiles[i]
            lineage_position = candidate_positions[i]
            index_to_cluster_map[lineage_idx] = cluster_id

            clusters[cluster_id][0].append(lineage_idx)
            clusters[cluster_id][1].append(lineage_position)
            clusters[cluster_id][2].append(lineage_profile)

            lineage_counts += 1
    print(f'complete!')
    cluster_sizes = [len(cluster[0]) for cluster_id, cluster in clusters.items()]
    print(f"After augmenting: N_clusters = {len(clusters)}, MEDIAN {int(np.median(cluster_sizes))}, MIN {np.min(cluster_sizes)}, MAX {np.max(cluster_sizes)}")

    return clusters, index_to_cluster_map

def filter_clusters(clusters, bac): ### relevant for testing
    """ Check clusters that are singletons, or ultiple gene-hits """
    singletons = 0
    two_hit_genes = 0
    three_hit_genes = 0
    four_hit_genes = []
    filtered_cluster_ids = []
    for cluster_id, (cluster_indices, cluster_positions, cluster_profiles) in sorted(clusters.items()):
        if len(cluster_indices) < 2:
            singletons +=1
            continue

        cluster_genes = {}
        for position in cluster_positions:
            try:
                index_genes = shared.bac_lineage_gene_map[bac][position]
                for gene in index_genes:
                    if gene not in cluster_genes:
                        cluster_genes[gene] = 0
                    cluster_genes[gene] += 1
            except:
                pass
        gene_counts = np.array( [count for (gene, count) in cluster_genes.items()] )

        if len(cluster_indices) <= 10 and np.any(gene_counts) and np.max(gene_counts) > 1:
            two_hit_genes += 1
            continue
        if len(cluster_indices) <= 100 and np.any(gene_counts) and np.max(gene_counts) > 2:
            three_hit_genes += 1
            continue
        if len(cluster_indices) <= 1000 and np.any(gene_counts) and np.max(gene_counts) > 3:
            four_hit_genes.append(np.max(gene_counts))
            continue

        filtered_cluster_ids.append(cluster_id)

    print(f"Viable clusters {len(filtered_cluster_ids)}\n",
          f"REMOVED {len(clusters) - len(filtered_cluster_ids)}\n"
          f"SINGLETONS {singletons}, two_hit_genes {two_hit_genes}, three_hit_genes {three_hit_genes}, four_hit_genes {four_hit_genes}")
    filtered_cluster_ids.sort()
    return filtered_cluster_ids

def generate_clusters(seed_data, candidate_data, cutoff=4):
    """ Generate hierarchical clusters from seeds, augment with candidate lineages"""
    seed_indices, seed_positions, seed_profiles = seed_data
    candidate_indices, candidate_positions, candidate_profiles = candidate_data

    clustering = cluster_profiles(seed_profiles, cutoff=cutoff)
    clusters, index_to_cluster_map = get_clusters(clustering, *seed_data)
    clusters, index_to_cluster_map =  expand_clusters(clusters, index_to_cluster_map, *candidate_data, cutoff=cutoff)
    # filtered_cluster_ids = filter_clusters(clusters, bac)

    return clusters, index_to_cluster_map

def find_cluster_ids_of_tns(tns_of_interest, index_to_cluster_map): #different implementations of clustering may lead to different cluster_ids for same cluster
    return [index_to_cluster_map[tn_idx] for tn_idx in tns_of_interest]

def get_pared_gene_data(bac, freq_data):
    [f0_arr, D0_arr, f1_arr, D1_arr] = freq_data[0]

    # Get genes and their Tn lineage indices
    genes = [gene for gene in shared.bac_gene_meta_dict[bac].keys()]
    gene_lineage_indices_map = fp.call_conservative_gene_indices(genes, gene_lineage_map)

    # calculate leave one out profiles
    gene_loo_lfcs = fp.get_leave_one_out_distributions(gene_lineage_indices_map, f0_arr, f1_arr, D0_arr, D1_arr)
    pared_gene_profiles = fp.make_pared_gene_lfcs_dict(f0_arr, f1_arr, D0_arr, D1_arr, gene_loo_lfcs,
                                                       shared.bac_gene_meta_dict[bac], max_removed=1)
    pared_gene_replicate_profiles = fp.generate_gene_replicate_profiles(genes, pared_gene_profiles, f0_arr, f1_arr,
                                                                        D0_arr, D1_arr)
    gene_loo_lfcs.clear()

    finite_indices = 0
    for gene, gene_data in pared_gene_profiles.items():
        n_indices = len(gene_data['all_indices'])
        finite_indices += (n_indices > 1)

    finite_measure = 0
    high_quality_genes = []
    for gene, gene_data in pared_gene_replicate_profiles.items():
        p1, p2 = gene_data[1], gene_data[2]

        well_measured = (((p1.mask + p2.mask) > 0).sum() <= 0)

        if well_measured:
            finite_measure += 1
            high_quality_genes.append(gene)

    high_quality_genes = np.array(high_quality_genes)
    return pared_gene_profiles, pared_gene_replicate_profiles, high_quality_genes

def check_clusters_for_novelty(freq_data, pared_gene_data, cluster_ids_of_interest):
    [f0_arr, D0_arr, f1_arr, D1_arr] = freq_data[0]
    cluster_novelty_scores = []

    pared_gene_profiles, pared_gene_replicate_profiles, high_quality_genes = pared_gene_data

    for cluster_id, cluster_data in clusters.items():
        if cluster_id not in cluster_ids_of_interest:
            continue
        cluster_indices, cluster_positions = cluster_data[:2]
        cluster_profiles = np.array(cluster_data[2])
        mean_cluster_profile = np.mean(cluster_profiles, axis=0)

        filtered_gene_profile_data = fp.filter_gene_replicate_profiles(pared_gene_replicate_profiles,
                                                                       pared_gene_profiles,
                                                                       f0_arr, f1_arr, D0_arr, D1_arr,
                                                                       shared.bac_lineage_gene_map[bac],
                                                                       exclude_indices=cluster_indices,
                                                                       exclude_positions=cluster_positions)

        gene_profiles = []
        gene_S1_profiles = []
        gene_S2_profiles = []
        gene_distances = []

        for gene in high_quality_genes:
            gene_profile = filtered_gene_profile_data[gene][0]
            gene_profiles.append(gene_profile)

            S1_mask = filtered_gene_profile_data[gene][1].mask
            S2_mask = filtered_gene_profile_data[gene][2].mask

            S1_profile = np.ma.masked_array(filtered_gene_profile_data[gene][1], S1_mask)
            S2_profile = np.ma.masked_array(filtered_gene_profile_data[gene][2], S2_mask)

            gene_S1_profiles.append(S1_profile)
            gene_S2_profiles.append(S2_profile)

            median_profile = np.ma.masked_array(1 / 2 * (S1_profile + S2_profile), (S1_mask + S2_mask) > 0)
            distance = fp.distance(mean_cluster_profile, median_profile)
            if distance == np.isnan(distance):
                print('ah!')

            gene_distances.append(fp.distance(mean_cluster_profile, median_profile))

        gene_profiles = np.array(gene_profiles)
        gene_S1_profiles = np.array(gene_S1_profiles)
        gene_S2_profiles = np.array(gene_S2_profiles)
        gene_distances = np.array(gene_distances)

        argsort_gene_distances = np.argsort(gene_distances)

        sorted_distances = gene_distances[argsort_gene_distances]
        min_gene_distance = sorted_distances[0]
        sorted_genes = high_quality_genes[argsort_gene_distances]
        sorted_gene_profiles = gene_profiles[argsort_gene_distances]
        sorted_gene_S1_profiles = gene_S1_profiles[argsort_gene_distances]
        sorted_gene_S2_profiles = gene_S2_profiles[argsort_gene_distances]

        cluster_novelty_scores.append((cluster_id, min_gene_distance, sorted_genes[:10], \
                                       sorted_distances[:10], sorted_gene_profiles[:10], \
                                       sorted_gene_S1_profiles[:10], sorted_gene_S2_profiles[:10]))

    sorted_cluster_novelty_scores = sorted(cluster_novelty_scores, key=lambda x: x[1], reverse=True)
    novelty_cluster_items = {cluster_item[0]: cluster_item for cluster_item in sorted_cluster_novelty_scores}
    return sorted_cluster_novelty_scores, novelty_cluster_items

def make_fig4_novel(fig, gs, bac, lineage_profile_data, clusters, plot_cluster_ids, novelty_cluster_items):
    lineage_average_profiles, lineage_S1_profiles, lineage_S2_profiles = lineage_profile_data


    global_ax = fig.add_subplot(gs[:])
    ax_methods.turn_off_ax(global_ax)
    global_ax.set_ylabel('Fold change', labelpad=10, fontsize=12)
    ENV_ALL_REPLICATES = fp.make_env_replicate_tuples(bac)
    ENV_LABELS = fp.make_env_labels(ENV_ALL_REPLICATES)

    for row, cluster_id in enumerate(plot_cluster_ids):
        cluster_data = clusters[cluster_id]
        cluster_indices, cluster_positions = cluster_data[:2]
        cluster_profiles = np.array(cluster_data[2])
        cluster_S1_profiles = lineage_S1_profiles[0, cluster_indices]
        cluster_S2_profiles = lineage_S2_profiles[0, cluster_indices]
        n_env = cluster_profiles.shape[1]

        ax = fig.add_subplot(gs[row, 0])
        genome_ax = fig.add_subplot(gs[row, 2])
        fp.make_genome_colorbar(genome_ax, fp.BWH2_LENGTH)
        genome_ax.set_ylabel('Genome position (Mb)', fontsize=8)
        genome_ax.yaxis.set_tick_params(labelsize=7)

        within_cluster_scores = fp.parallelized_distance(np.mean(cluster_profiles, axis=0), cluster_profiles)
        argsort = np.argsort(within_cluster_scores)
        for z_sort, idx in enumerate(argsort[:]):
            profile = cluster_profiles[z_sort]
            position = cluster_positions[z_sort]

            if z_sort <= 4:
                label = 'Tn lineage' if z_sort == 0 else ''
                ax.plot(np.exp(profile), color=KELLY_COLORS[z_sort % len(KELLY_COLORS)], label=label)
            genome_ax.axhline(position, color=KELLY_COLORS[z_sort % len(KELLY_COLORS)], lw=1)

        lower_profile, upper_profile = fp.calculate_cluster_bands(cluster_S1_profiles, cluster_S2_profiles)
        ax.fill_between(np.arange(n_env), np.exp(lower_profile),
                        np.exp(upper_profile), color='green', alpha=0.5, lw=0, zorder=0)

        cluster_item = novelty_cluster_items[cluster_id]
        min_gene_score, sorted_genes, sorted_scores = cluster_item[1:4]
        nearby_gene_profiles, nearby_gene_S1_profiles, nearby_gene_S2_profiles = cluster_item[4:]

        for l in range(1):
            similar_gene_profile = np.copy(nearby_gene_profiles[l])
            similar_gene_S1_profile = np.copy(nearby_gene_S1_profiles[l])
            similar_gene_S2_profile = np.copy(nearby_gene_S2_profiles[l])

            similar_gene_profile[similar_gene_profile == 1] = 0
            similar_gene_S1_profile[similar_gene_S1_profile == 1] = 0
            similar_gene_S2_profile[similar_gene_S2_profile == 1] = 0

            min_similar_gene_profile = np.min([similar_gene_S1_profile, similar_gene_S2_profile], axis=0)
            max_similar_gene_profile = np.max([similar_gene_S1_profile, similar_gene_S2_profile], axis=0)
            ax.plot(np.exp(similar_gene_profile), color='black', marker='.',
                    linestyle='solid', markersize=5, label='Most similar\ngene knockout')
            ax.fill_between(np.arange(similar_gene_profile.shape[0]), np.exp(min_similar_gene_profile),
                            np.exp(max_similar_gene_profile), alpha=0.2, color='black', lw=0)

            nearby_gene = sorted_genes[l]
            gene_start, gene_stop = gene_lineage_map[nearby_gene]['coords'][0]
            genome_ax.axhspan(ymin=gene_start - 10 ** 5, ymax=gene_stop + 10 ** 5, color='black', lw=0, zorder=0)

        if 'formatting' != '':
            fp.format_profile_ax(ax, ENV_LABELS)

            ax.set_yscale('log')
            ax.set_ylim(0.2 * 10 ** -1, 10 ** 3)
            ax.set_yticks([10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2])
            ax.set_yticklabels([r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"], fontsize=7)

            ax.tick_params(axis='y', which='minor', bottom=True)

            if row == 0:
                ax.set_xticklabels([])
                ax.legend(loc=1, fontsize=7, frameon=False, ncol=1)

def make_sfig_novel(fig, gs, bac, lineage_profile_data, clusters, plot_cluster_ids, novelty_cluster_items):
    lineage_average_profiles, lineage_S1_profiles, lineage_S2_profiles = lineage_profile_data

    global_ax = fig.add_subplot(gs[:])
    ax_methods.turn_off_ax(global_ax)
    global_ax.set_ylabel('Fold change', labelpad=10, fontsize=12)
    ENV_ALL_REPLICATES = fp.make_env_replicate_tuples(bac)
    ENV_LABELS = fp.make_env_labels(ENV_ALL_REPLICATES)

    for z, cluster_id in enumerate(plot_cluster_ids):
        cluster_data = clusters[cluster_id]
        cluster_indices, cluster_positions = cluster_data[:2]
        cluster_profiles = np.array(cluster_data[2])
        n_env = cluster_profiles.shape[1]
        cluster_S1_profiles = lineage_S1_profiles[0, cluster_indices]
        cluster_S2_profiles = lineage_S2_profiles[0, cluster_indices]

        lower_profile, upper_profile = fp.calculate_cluster_bands(cluster_S1_profiles, cluster_S2_profiles)
        ax = fig.add_subplot(gs[z // 2, 3 * (z % 2)])

        ax.fill_between(np.arange(n_env), np.exp(lower_profile),
                        np.exp(upper_profile), color='green', alpha=0.5, lw=0, zorder=0)

        genome_ax = fig.add_subplot(gs[z // 2, 3 * (z % 2) + 1])
        fp.make_genome_colorbar(genome_ax, fp.BWH2_LENGTH)
        genome_ax.set_ylabel('')

        within_cluster_scores = fp.parallelized_distance(np.mean(cluster_profiles, axis=0), cluster_profiles)
        argsort = np.argsort(within_cluster_scores)
        for z_sort, idx in enumerate(argsort[:]):
            profile = cluster_profiles[z_sort]
            position = cluster_positions[z_sort]

            if z_sort <= 4: # plot first 5 lineages, regardless of distances
                label = 'Tn lineage' if z_sort == 0 else ''
                ax.plot(np.exp(profile), color=KELLY_COLORS[z_sort % len(KELLY_COLORS)], label=label)

            genome_ax.axhline(position, color=KELLY_COLORS[z_sort % len(KELLY_COLORS)], lw=1)
        ax.set_yscale('log')

        cluster_item = novelty_cluster_items[cluster_id]
        min_gene_score, sorted_genes, sorted_scores = cluster_item[1:4]
        nearby_gene_profiles, nearby_gene_S1_profiles, nearby_gene_S2_profiles = cluster_item[4:]

        for l in range(1):
            similar_gene_profile = np.copy(nearby_gene_profiles[l])
            similar_gene_S1_profile = np.copy(nearby_gene_S1_profiles[l])
            similar_gene_S2_profile = np.copy(nearby_gene_S2_profiles[l])

            similar_gene_profile[similar_gene_profile == 1] = 0
            similar_gene_S1_profile[similar_gene_S1_profile == 1] = 0
            similar_gene_S2_profile[similar_gene_S2_profile == 1] = 0

            min_similar_gene_profile = np.min([similar_gene_S1_profile, similar_gene_S2_profile], axis=0)
            max_similar_gene_profile = np.max([similar_gene_S1_profile, similar_gene_S2_profile], axis=0)
            ax.plot(np.exp(similar_gene_profile), color='black', marker='.',
                    linestyle='solid', markersize=5, label='Most similar\ngene knockout')
            ax.fill_between(np.arange(similar_gene_profile.shape[0]), np.exp(min_similar_gene_profile),
                            np.exp(max_similar_gene_profile), alpha=0.2, color='black', lw=0)

            nearby_gene = sorted_genes[0]
            gene_start, gene_stop = gene_lineage_map[nearby_gene]['coords'][0]
            genome_ax.axhspan(ymin=gene_start - 10 ** 5, ymax=gene_stop + 10 ** 5, color='black', lw=0, zorder=0)

        fp.format_profile_ax(ax, ENV_LABELS)
        if z // 2 != 3:
            ax.set_xticklabels([])
        if z == 0:
            ax.legend(loc=1, fontsize=6.5, frameon=False)

if __name__ == '__main__':
    bac = 'BWH2'
    row_ids = shared.bac_row_ids[bac]
    read_array = shared.bac_read_arrays[bac]
    input_array = shared.bac_input_arrays[bac]
    day0_reads = read_array[0]
    day0_freqs = read_array[0] / read_array[0].sum()
    gene_lineage_map = shared.bac_gene_meta_dict[bac]
    read_arrays_meta = {'input_array': input_array,
                        'read_array': read_array,
                        'row_ids': row_ids}

    # get profile data
    gene_profile_data, lineage_profile_data, freq_data = get_profile_data(bac)

    seed_data, candidate_data = get_seed_and_candidate_data(bac, lineage_profile_data)
    clusters, index_to_cluster_map = generate_clusters(seed_data, candidate_data)

    # plot_cluster_ids = [58, 62, 35, 34, 29, 51, 74, 16, 25, 17]
    plot_cluster_tn_indices = [78472, 62235, 4599, 3854, 100180, 2122, 8058, 5281, 14192, 8328]
    plot_cluster_ids = find_cluster_ids_of_tns(plot_cluster_tn_indices, index_to_cluster_map)

    pared_gene_profiles, pared_gene_replicate_profiles, high_quality_genes = get_pared_gene_data(bac, freq_data)
    pared_gene_data = [pared_gene_profiles, pared_gene_replicate_profiles, high_quality_genes]

    print("Checking clusters for novelty...")
    sorted_cluster_novelty_scores, novelty_cluster_items = check_clusters_for_novelty(freq_data, pared_gene_data, plot_cluster_ids)

    fig = plt.figure(figsize=(4, 3))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=3, width_ratios=(1.5, 0.06, 0.15), figure=fig, wspace=0.1)
    make_fig4_novel(fig, gs, bac, lineage_profile_data, clusters, plot_cluster_ids[-2:], novelty_cluster_items)
    fig.savefig(f'{plot_dir}/fig4_{bac}_novel.pdf')
    #
    fig = plt.figure(figsize=(8, 6))
    gs = mpl.gridspec.GridSpec(nrows=4, ncols=5, width_ratios=(1, 0.1, 0.01, 1, 0.1), figure=fig)
    make_sfig_novel(fig, gs, bac, lineage_profile_data, clusters, plot_cluster_ids[:-2], novelty_cluster_items)
    fig.savefig(f'{plot_dir}/sfig14_{bac}_novel.pdf')