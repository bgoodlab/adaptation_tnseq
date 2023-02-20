root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
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

gene_descriptions = {'BT3705': 'BT3705 \nstarch metabolism (PUL 66)',
                     'BT0238': 'BT0238\nsulfated host glycan metabolism',
                     'BT1052':'BT1052: anti-sigma factor in PUL 14 (mucin)',
                     'BT1272':'BT1272: GntR-family transcription factor',
                     'BT3387':'BT3387: putative polysaccharide deacetylase',
                     'BT3388':f'BT3387-3388: Polysaccharide deacetylase,\n{" "*25}glycosyltransferase',
                     'BT4174':'BT4174: hypothetical protein in PUL 77',
                     'BT4248':'BT4248: anti-sigma factor in PUL 78 (mucin)',
                     'BT4332':"BT4332: 3'-5' exonuclease",
                     'BT4719':'BT4719: NigD-like protein',
                     'BT3700-3705':"BT3700-3705: PUL 66 (starch)",
                     'BT1765': f'BT1765: glycoside hydrolase\n{" "*15}in PUL 22 (host glycan)'
                     }

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

def specify_candidate_lineages(bac, lineage_profile_data):
    """
    Get data (indices, positions, fitness profiles) for lineages that are "well-measured"
    """
    lineage_average_profiles, lineage_S1_profiles, lineage_S2_profiles = lineage_profile_data

    candidate_indices = []
    candidate_positions = []
    for i, (prof, n_rep) in enumerate(zip(lineage_average_profiles[0], lineage_average_profiles[2])):
        if (n_rep > 0).sum() >= len(n_rep): # only require that measured across all environments
            candidate_indices.append(i)
            position = shared.bac_lineage_positions[bac][i]
            candidate_positions.append(position)

    candidate_indices = np.array(candidate_indices)
    candidate_positions = np.array(candidate_positions)
    candidate_profiles = np.array([lineage_average_profiles[0, index] for index in candidate_indices])
    candidate_S1_profiles = np.array([lineage_S1_profiles[0, index] for index in candidate_indices])
    candidate_S2_profiles = np.array([lineage_S2_profiles[0, index] for index in candidate_indices])

    candidate_data = [candidate_indices, candidate_positions, candidate_profiles, candidate_S1_profiles, candidate_S2_profiles]

    return candidate_data

def remove_lineages_in_same_gene(bac, lineages_in_cluster, ignore_score=2):
    ''' lineages_in_cluster = list of [(index, position, score)]'''
    cluster_genes = {}
    for z, index_tuple in enumerate(lineages_in_cluster):
        index, position, score = index_tuple
        if score > ignore_score: # not actually part of cluster
            continue
        genome_position = shared.bac_lineage_positions[bac][index]
        if genome_position in shared.bac_lineage_gene_map[bac]:
            lineage_genes = shared.bac_lineage_gene_map[bac][genome_position]
            for gene in lineage_genes:
                if gene not in cluster_genes:
                    cluster_genes[gene] = []
                cluster_genes[gene].append(z)

    exclude_indices = []
    for gene, z_lst in cluster_genes.items():
        if len(z_lst) > 1: # multiple lineages in the gene found in same cluster!
            exclude_indices.extend(z_lst)

    new_cluster_indices = [index_tuple for z, index_tuple in enumerate(lineages_in_cluster) if z not in exclude_indices]
    sorted_indices = sorted(new_cluster_indices, key=lambda x:x[2]) # sort on score
    return sorted_indices

def find_lineages_coclustering_with_genes(gene_lst, gene_profile_data, candidate_profile_data, score_cutoff=2, min_distance=10**5):
    gene_average_profiles = gene_profile_data[0]
    candidate_indices, candidate_positions, candidate_profiles = candidate_profile_data[:3]

    gene_coclustering_lineages_map = {gene: [] for gene in plot_gene_lst}
    for g, gene in enumerate(gene_lst):
        gene_start, gene_stop = gene_lineage_map[gene]['coords'][0] # gene data
        gene_profile = gene_average_profiles[gene]['lfc']

        distant_lineage_bool = ((candidate_positions < gene_start - min_distance) + (
                candidate_positions > gene_stop + min_distance)) > 0
        distant_indices = candidate_indices[distant_lineage_bool]
        distant_positions = candidate_positions[distant_lineage_bool]
        distant_profiles = candidate_profiles[distant_lineage_bool]
        candidate_scores = fp.parallelized_distance(gene_profile, distant_profiles)
        argsort_scores = np.argsort(candidate_scores) # sort candidate lineages by score

        cocluster_indices = []
        for argsort, i in enumerate(argsort_scores[:]):
            score = candidate_scores[i]
            index = distant_indices[i]
            position = distant_positions[i]

            if (score > score_cutoff and argsort > 4):
                continue
            elif score > score_cutoff and argsort <= 5:
                cocluster_indices.append((index, position, score))
            elif score < score_cutoff:
                cocluster_indices.append((index, position, score))

        cocluster_indices = remove_lineages_in_same_gene(bac, cocluster_indices)
        gene_coclustering_lineages_map[gene] = cocluster_indices

    return gene_coclustering_lineages_map

def make_fig4_geneko_like(fig, gs, bac, plot_gene_lst, gene_data, lineage_data, freq_data, gene_coclustering_lineages_map, score_cutoff=2):
    gene_average_profiles, gene_S1_profiles, gene_S2_profiles, gene_replicate_profiles = gene_data
    lineage_average_profiles, lineage_S1_profiles, lineage_S2_profiles = lineage_data
    [f0_arr, D0_arr, f1_arr, D1_arr] = freq_data[0]
    ENV_ALL_REPLICATES = fp.make_env_replicate_tuples(bac)
    ENV_LABELS = fp.make_env_labels(ENV_ALL_REPLICATES)

    global_ax = fig.add_subplot(gs[:])
    ax_methods.turn_off_ax(global_ax)
    global_ax.set_ylabel('Fold change', labelpad=10, fontsize=12)

    for row, gene in enumerate(plot_gene_lst):
        ax = fig.add_subplot(gs[row, 0])
        fitness_browser_ax = fig.add_subplot(gs[row, 1])
        genome_ax = fig.add_subplot(gs[row, 3])

        gene_start, gene_stop = gene_lineage_map[gene]['coords'][0]
        fp.make_genome_colorbar(genome_ax, fp.GENOME_LENGTHS[bac])
        genome_ax.axhspan(ymin=gene_start - 10 ** 5, ymax=gene_stop + 10 ** 5, color='black', lw=0, zorder=10 ** 4)
        genome_ax.set_ylabel('Genome position (Mb)', fontsize=8)
        genome_ax.yaxis.set_tick_params(labelsize=7)

        ### Plot gene profiles
        ### plot gene-replicate averaged fitness profiles
        gene_profile = gene_average_profiles[gene]['lfc']
        n_env = gene_profile.shape[0]
        gene_rep1_profile = gene_replicate_profiles[gene][1]
        gene_rep2_profile = gene_replicate_profiles[gene][2]
        gene_min_profile = np.min([gene_rep1_profile, gene_rep2_profile], axis=0)
        gene_max_profile = np.max([gene_rep1_profile, gene_rep2_profile], axis=0)

        ax.plot(np.exp(gene_profile), zorder=100, color='black', marker='.', linestyle='solid', markersize=5,
                label='Gene-knockout')
        ax.fill_between(np.arange(n_env), np.exp(gene_min_profile), np.exp(gene_max_profile), alpha=0.2, color='black',
                        lw=0, zorder=99)
        ax.text(0.05, 0.05, f'{gene_descriptions[gene]}', fontsize=9, transform=ax.transAxes, va='bottom')
        ax.set_yscale('log')
        ax.axhline(1, color='black')
        ax.set_ylim(0.3 * 10 ** -1, 4 * 10 ** 1)
        #     ax.tick_params(axis='both', which='minor', labelsize=8)
        ax.set_yticks([10 ** -1, 10 ** 0, 10 ** 1])
        ax.set_yticklabels([r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"], fontsize=7)
        #     ax.yaxis.set_tick_params(labelsize=8)

        fp.format_profile_ax(ax, ENV_LABELS)
        cocluster_index_tuples = gene_coclustering_lineages_map[gene]

        cocluster_indices = [index_tup[0] for index_tup in cocluster_index_tuples]
        cocluster_positions = [index_tup[1] for index_tup in cocluster_index_tuples]
        cocluster_scores = [index_tup[2] for index_tup in cocluster_index_tuples]

        cocluster_profiles = lineage_average_profiles[0, cocluster_indices]
        for z, (position, score, profile) in enumerate(zip(cocluster_positions, cocluster_scores, cocluster_profiles)):
            if score >score_cutoff and z <= 4:
                continue
                ls = 'dashed'
            if score <=score_cutoff and z <= 4:
                ls = 'solid'

            genome_ax.axhline(position, color=shared.get_kelly_color(z), linestyle=ls, lw=1, alpha=1)

            if z <= 4:
                label = 'Unrelated Tn lineage' if z == 0 else ''
                ax.plot(np.exp(profile), color=shared.get_kelly_color(z), linestyle=ls, label=label)

        cocluster_tight_indices = [index_tup[0] for index_tup in cocluster_index_tuples if index_tup[2] <score_cutoff]
        cocluster_tight_positions = [index_tup[1] for index_tup in cocluster_index_tuples if index_tup[2] <score_cutoff]
        cocluster_lineage_S1_profiles = lineage_S1_profiles[0, cocluster_tight_indices]
        cocluster_lineage_S2_profiles = lineage_S2_profiles[0, cocluster_tight_indices]
        cocluster_lower_profile, cocluster_upper_profile = fp.calculate_cluster_bands(cocluster_lineage_S1_profiles,
                                                                                      cocluster_lineage_S2_profiles)
        ax.fill_between(np.arange(n_env), np.exp(cocluster_lower_profile), np.exp(cocluster_upper_profile),
                        color='green', alpha=0.5, lw=0, zorder=0)

        if row == 0:
            ax.set_xticklabels([])
            ax.legend(loc=2, fontsize=7, frameon=False)

        x_counter = 0
        scatter_x, scatter_y = [], []
        line_x, line_y = [], []
        for z, (env_label, media) in enumerate(zip(['Starch', 'Host glycans', 'Monosaccharides'],
                                                   [('Corn starch', 'Amylopectin', 'Amylose'),
                                                    ('Heparin', 'Chondroitin sulfate'),
                                                    ('Arabinose', 'Fructose', 'Galactose', 'Glucose')])):
            for m, medium in enumerate(media):
                line_x.append(x_counter)
                line_y.append(np.mean(fp.fitness_browser_data[gene][medium]))
                for r, rep in enumerate(fp.fitness_browser_data[gene][medium]):
                    scatter_x.append(x_counter + rnd.normal(0, 0.1))
                    scatter_y.append(rep)
                x_counter += 1
            if z % 2 == 1:
                fitness_browser_ax.axvspan(xmin=x_counter - len(media) - 0.5, xmax=x_counter - 0.5,
                                           color='grey', alpha=0.3, lw=0)

        fitness_browser_ax.plot(line_x, np.exp(line_y), color='black', lw=1)
        fitness_browser_ax.scatter(scatter_x, np.exp(scatter_y), facecolors='none', edgecolors='black', s=7)
        fitness_browser_ax.set_ylim(ax.get_ylim())
        fitness_browser_ax.set_xlim(fitness_browser_ax.get_xlim()[0] - 0.5, fitness_browser_ax.get_xlim()[1])
        fitness_browser_ax.set_yscale('log')
        fitness_browser_ax.set_yticklabels([])
        fitness_browser_ax.set_xticks([1, 3.5, 6.5])
        fitness_browser_ax.set_xticklabels([])
        fitness_browser_ax.axhline(1, color='black', linestyle='dashed', zorder=0)

        if row == 1:
            fitness_browser_ax.set_xticklabels(['Starches', 'Host\nglycans', 'Simple\nsugars'],
                                               fontsize=8, rotation=45)

        cocluster_tight_indices = [index_tup[0] for index_tup in cocluster_index_tuples if index_tup[2] <score_cutoff]
        cocluster_tight_positions = [index_tup[1] for index_tup in cocluster_index_tuples if index_tup[2] <score_cutoff]
        excluded_indices = cocluster_tight_indices
        excluded_positions = cocluster_tight_positions

        # ene_replicate_profiles, all_gene_profile_data, f0_arr, f1_arr, D0_arr, D1_arr, lineage_gene_map, exclude_indices=[], exclude_positions=[]
        filtered_gene_replicate_profiles = \
            fp.filter_gene_replicate_profiles(gene_replicate_profiles, gene_average_profiles, f0_arr, f1_arr, D0_arr,
                                              D1_arr, shared.bac_lineage_gene_map[bac], excluded_indices,
                                              excluded_positions)

        filtered_genes = np.array([gene for gene, profiles in filtered_gene_replicate_profiles.items()])
        filtered_profiles = np.array([profiles[0] for gene, profiles in filtered_gene_replicate_profiles.items()])
        filtered_S1_profiles = np.array([profiles[1] for gene, profiles in filtered_gene_replicate_profiles.items()])
        filtered_S2_profiles = np.array([profiles[2] for gene, profiles in filtered_gene_replicate_profiles.items()])
        gene_distances = fp.parallelized_distance(gene_profile, filtered_profiles)
        gene_distance_argsort = np.argsort(gene_distances)
        for i, idx in enumerate(gene_distance_argsort[:]):
            filtered_profile = filtered_profiles[idx]
            filtered_S1_profile = filtered_S1_profiles[idx]
            filtered_S2_profile = filtered_S2_profiles[idx]
            filtered_gene = filtered_genes[idx]

            if fp.distance(filtered_S1_profile, filtered_S2_profile) > 5:
                continue
            if gene_distances[idx] >score_cutoff:
                break

    plt.subplots_adjust(wspace=0.1)

def make_sfig_geneko_like(fig, gs, bac, plot_gene_lst, gene_data, lineage_data, freq_data, gene_coclustering_lineages_map, score_cutoff=2):
    gene_average_profiles, gene_S1_profiles, gene_S2_profiles, gene_replicate_profiles = gene_data
    lineage_average_profiles, lineage_S1_profiles, lineage_S2_profiles = lineage_data
    [f0_arr, D0_arr, f1_arr, D1_arr] = freq_data[0]
    ENV_ALL_REPLICATES = fp.make_env_replicate_tuples(bac)
    ENV_LABELS = fp.make_env_labels(ENV_ALL_REPLICATES)

    n_col = 2
    for g, gene in enumerate(plot_gene_lst):
        ax = fig.add_subplot(gs[g // n_col, 3 * (g % n_col)])
        genome_ax = fig.add_subplot(gs[g // n_col, 3 * (g % n_col) + 1])
        gene_start, gene_stop = gene_lineage_map[gene]['coords'][0]
        fp.make_genome_colorbar(genome_ax, fp.GENOME_LENGTHS[bac])
        genome_ax.axhspan(ymin=gene_start - 10 ** 5, ymax=gene_stop + 10 ** 5, color='black', lw=0, zorder=10 ** 4)

        gene_profile = gene_average_profiles[gene]['lfc']
        n_env = gene_profile.shape[0]
        gene_rep1_profile = gene_replicate_profiles[gene][1]
        gene_rep2_profile = gene_replicate_profiles[gene][2]
        gene_min_profile = np.min([gene_rep1_profile, gene_rep2_profile], axis=0)
        gene_max_profile = np.max([gene_rep1_profile, gene_rep2_profile], axis=0)

        ax.plot(np.exp(gene_profile), zorder=100, color='black', marker='.', linestyle='solid', markersize=5,
                label='Gene-knockout')
        ax.fill_between(np.arange(n_env), np.exp(gene_min_profile), np.exp(gene_max_profile), alpha=0.2, color='black',
                        lw=0, zorder=99)
        ax.text(0.05, 0.07, f'{gene_descriptions[gene]}', fontsize=9, transform=ax.transAxes, va='bottom')
        ax.set_yscale('log')
        ax.axhline(1, color='black')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(np.min([ymin, 5 * 10 ** -2]), np.max([ymax, 5 * 10 ** 1]))
        fp.format_profile_ax(ax, ENV_LABELS)
        if g // n_col != 3:
            ax.set_xticklabels([])


        cocluster_index_tuples = gene_coclustering_lineages_map[gene]

        cocluster_indices = [index_tup[0] for index_tup in cocluster_index_tuples]
        cocluster_positions = [index_tup[1] for index_tup in cocluster_index_tuples]
        cocluster_scores = [index_tup[2] for index_tup in cocluster_index_tuples]
        cocluster_profiles = lineage_average_profiles[0, cocluster_indices]
        for z, (position, score, profile) in enumerate(zip(cocluster_positions, cocluster_scores, cocluster_profiles)):
            if score > score_cutoff and z <= 4:
                ls = 'dashed'
            if score <= score_cutoff and z <= 4:
                ls = 'solid'
            genome_ax.axhline(position, color=shared.get_kelly_color(z), linestyle=ls, lw=1, alpha=1)
            if z <= 4:
                label = 'Unrelated Tn lineage' if z == 0 else ''
                ax.plot(np.exp(profile), color=shared.get_kelly_color(z), linestyle=ls, label=label)

        cocluster_tight_indices = [index_tup[0] for index_tup in cocluster_index_tuples if index_tup[2] < score_cutoff]
        cocluster_tight_positions = [index_tup[1] for index_tup in cocluster_index_tuples if index_tup[2] < score_cutoff]
        excluded_indices = cocluster_tight_indices
        excluded_positions = cocluster_tight_positions

        if len(cocluster_tight_indices) > 0:
            cocluster_lineage_S1_profiles = lineage_S1_profiles[0, cocluster_tight_indices]
            cocluster_lineage_S2_profiles = lineage_S2_profiles[0, cocluster_tight_indices]

            cocluster_lower_profile, cocluster_upper_profile = fp.calculate_cluster_bands(cocluster_lineage_S1_profiles,
                                                                                          cocluster_lineage_S2_profiles)
            ax.fill_between(np.arange(n_env), np.exp(cocluster_lower_profile), np.exp(cocluster_upper_profile),
                            color='green', alpha=0.5, lw=0, zorder=0)

        filtered_gene_replicate_profiles = \
            fp.filter_gene_replicate_profiles(gene_replicate_profiles, gene_average_profiles, f0_arr, f1_arr, D0_arr,
                                              D1_arr, shared.bac_lineage_gene_map[bac], excluded_indices,
                                              excluded_positions)

        filtered_genes = np.array([gene for gene, profiles in filtered_gene_replicate_profiles.items()])
        filtered_profiles = np.array([profiles[0] for gene, profiles in filtered_gene_replicate_profiles.items()])
        filtered_S1_profiles = np.array([profiles[1] for gene, profiles in filtered_gene_replicate_profiles.items()])
        filtered_S2_profiles = np.array([profiles[2] for gene, profiles in filtered_gene_replicate_profiles.items()])
        gene_distances = fp.parallelized_distance(gene_profile, filtered_profiles)
        gene_distance_argsort = np.argsort(gene_distances)
        for i, idx in enumerate(gene_distance_argsort[:]):
            filtered_S1_profile = filtered_S1_profiles[idx]
            filtered_S2_profile = filtered_S2_profiles[idx]
            filtered_gene = filtered_genes[idx]

            if fp.distance(filtered_S1_profile, filtered_S2_profile) > 5:
                continue
            if gene_distances[idx] > score_cutoff:
                break

        if g == 0:
            ax.set_xticklabels([])
            ax.legend(loc=2, fontsize=7, frameon=False)

    # PUL 66
    ax = fig.add_subplot(gs[3, 3])
    genome_ax = fig.add_subplot(gs[3, 4])
    fp.make_genome_colorbar(genome_ax, fp.GENOME_LENGTHS[bac])
    ax.text(0.05, 0.07, gene_descriptions['BT3700-3705'], transform=ax.transAxes, fontsize=9)

    for gene, color in zip(['BT3700', 'BT3701', 'BT3702', 'BT3703', 'BT3705'],
                           ['silver', 'darkgrey', 'grey', 'dimgrey', 'black']):
        gene_lfc = gene_average_profiles[gene]['lfc']
        gene_loo_lfcs = gene_average_profiles[gene]['loo_lfcs']
        gene_loi_lfcs = np.copy(gene_average_profiles[gene]['loi_lfcs'])
        n_env = gene_lfc.shape[0]

        gene_loi_lfcs[np.isnan(gene_loi_lfcs)] = np.log(1 / 10)
        gene_S1_profile, gene_S2_profile = gene_replicate_profiles[gene][1:3]

        gene_start, gene_stop = gene_lineage_map[gene]['coords'][0]
        genome_ax.axhspan(ymin=gene_start - 10 ** 5, ymax=gene_stop + 10 ** 5, color=color, lw=0, zorder=10 ** 4)
        ax.plot(np.exp(gene_lfc), color=color, marker='.')

    joined_indices = set()
    for gene in ['BT3700', 'BT3701', 'BT3702', 'BT3703', 'BT3705']:
        cocluster_index_tuples = gene_coclustering_lineages_map[gene]
        for z, (index, pos, score) in enumerate(cocluster_index_tuples):
            if score < score_cutoff:
                joined_indices.add(index)
                genome_ax.axhline(pos, color=shared.get_kelly_color(z))
    joined_indices = np.array(list(joined_indices))
    PUL66_lineage_S1_profiles = lineage_S1_profiles[0, joined_indices]
    PUL66_lineage_S2_profiles = lineage_S2_profiles[0, joined_indices]
    PUL66_lower_profile, PUL66_upper_profile = fp.calculate_cluster_bands(PUL66_lineage_S1_profiles,
                                                                          PUL66_lineage_S2_profiles)
    ax.fill_between(np.arange(n_env), np.exp(PUL66_lower_profile), np.exp(PUL66_upper_profile), color='green',
                    alpha=0.5, lw=0, zorder=0)

    ax.set_ylim(np.min([ymin, 5 * 10 ** -2]), 5 * 10 ** 1)
    ax.set_yscale('log')
    ax.axhline(1, color='black')
    fp.format_profile_ax(ax, ENV_LABELS)

if __name__ == '__main__':
    bac = 'BtVPI'
    row_ids = shared.bac_row_ids[bac]
    read_array = shared.bac_read_arrays[bac]
    input_array = shared.bac_input_arrays[bac]
    day0_reads = read_array[0]
    day0_freqs = read_array[0] / read_array[0].sum()
    gene_lineage_map = shared.bac_gene_meta_dict[bac]
    read_arrays_meta = {'input_array': input_array,
                        'read_array': read_array,
                        'row_ids': row_ids}
    plot_gene_lst = ['BT1052', 'BT1272', 'BT1765', 'BT3388', 'BT4174', 'BT4248',
                     'BT4719', 'BT0238', 'BT3700', 'BT3701', 'BT3702', 'BT3703', 'BT3705']

    # get profile data
    gene_profile_data, lineage_profile_data, freq_data = get_profile_data(bac)

    # specify candidate lineages
    candidate_data = specify_candidate_lineages(bac, lineage_profile_data)
    candidate_indices, candidate_positions = candidate_data[:2]
    candidate_profiles, candidate_S1_profiles, candidate_S2_profiles = candidate_data[2:]

    gene_coclustering_lineages_map = find_lineages_coclustering_with_genes(plot_gene_lst, gene_profile_data, candidate_data,
                                                                           score_cutoff=2, min_distance=10 ** 5)

    fig = plt.figure(figsize=(5, 3))
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=4, width_ratios=(1, 0.45, 0.05, 0.1), figure=fig)
    make_fig4_geneko_like(fig, gs, bac, ['BT3705', 'BT0238'], gene_profile_data, lineage_profile_data, freq_data, gene_coclustering_lineages_map, score_cutoff=2)
    fig.savefig(f'{plot_dir}/fig4_{bac}_ko_like.pdf')

    fig = plt.figure(figsize=(9, 8))
    gs = plt.GridSpec(nrows=4, ncols=5, width_ratios=(1, 0.1, 0.00, 1, 0.1), figure=fig)
    make_sfig_geneko_like(fig, gs, bac, plot_gene_lst[:7],
                          gene_profile_data, lineage_profile_data, freq_data, gene_coclustering_lineages_map, score_cutoff=2) # 7 genes + 3700-3705 PUL 66
    fig.savefig(f'{plot_dir}/sfig13_{bac}_ko_like.pdf')