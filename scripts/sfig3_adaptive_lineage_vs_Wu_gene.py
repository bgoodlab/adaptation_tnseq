root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared
import scipy.stats

def calc_wu_gene_freq_array(bac, diet, mice, t1=4, split_day=1): #assume t0 = 0
    t0_freqs = []
    t1_freqs = []

    t0_reads = shared.bac_input_arrays[bac][(split_day % 2)::2].sum(axis=0)
    t1_reads = shared.bac_read_arrays[bac][ [shared.bac_row_ids[bac][(mouse, t1)] for mouse in mice] ].sum(axis=0)

    for gene in shared.wu_genes[bac][diet]['del']:
        gene_barcode_indices = shared.bac_gene_meta_dict[bac][gene]['indices']
        if np.any(gene_barcode_indices):
            t0_freqs_g = t0_reads[gene_barcode_indices].sum() / t0_reads.sum()
            t0_freqs.append(t0_freqs_g)

            t1_freqs_g = t1_reads[gene_barcode_indices].sum() / t1_reads.sum()
            t1_freqs.append( t1_freqs_g )

    return  np.array(t0_freqs), np.array(t1_freqs)

def make_sfig(fig, outer):
    cutoff = 10**-6.5
    min_reads = 5
    discovery_d0 = 0
    validation_d0 = 0
    discovery_d1 = 4
    validation_d1 = 4
    discovery_dt = discovery_d1 - discovery_d0
    validation_dt = validation_d1 - validation_d0
    cg = 100

    bac_nfit = {'BWH2':10000, 'BtVPI':5000, 'Bovatus':5000, 'Bt7330':3000}
    for b, bac in enumerate(BACTERIA):
        ax = fig.add_subplot(outer[b])
        for (diet, discovery_mice, validation_mice) in [('HF', [6, 7, 8, 9, 10], [1, 2, 3, 5]),
                                                        ('LF', [11, 12, 13, 15], [16, 18, 19])]:

            wu_t0_discovery, wu_t1_discovery = calc_wu_gene_freq_array(bac, diet, validation_mice, t1=4, split_day=1)
            wu_discovery_lfcs = shared.calc_lfc_array(wu_t0_discovery, wu_t1_discovery, 4)
            # wu_sorted_indices = np.argsort(wu_discovery_lfcs)
            #
            # wu_t0_validate, wu_t1_validate = calc_wu_gene_freq_array(bac, diet, validation_mice, t1=validation_d1, split_day=2)
            # wu_validate_lfcs = shared.calc_lfc_array( wu_t0_validate, wu_t1_validate, validation_d1)

            if 'generate discovery and validation frequencies':
                discovery_reads0, discovery_reads1 = np.copy(
                    shared.get_read_arrays(bac, discovery_mice, discovery_d0, discovery_d1, split_day0=2))
                validation_reads0, validation_reads1 = np.copy(
                    shared.get_read_arrays(bac, validation_mice, validation_d0, validation_d1, split_day0=1))
                # discovery_reads0 -= shared.bac_input_arrays[bac][6][shared.bac_nonwu_indices[bac]]

                discovery_D0, discovery_D1 = discovery_reads0.sum(), discovery_reads1.sum()
                discovery_freqs0, discovery_freqs1 = discovery_reads0 / discovery_D0, discovery_reads1 / discovery_D1

                validation_D0, validation_D1 = validation_reads0.sum(), validation_reads1.sum()
                validation_freqs0, validation_freqs1 = validation_reads0 / validation_D0, validation_reads1 / validation_D1

                d0_reads = shared.bac_read_arrays[bac][0][shared.bac_nonwu_indices[bac]]
                d0_freqs = d0_reads / d0_reads.sum()
                d1_reads = (discovery_reads1 + validation_reads1)

                max_freqs = np.max([discovery_freqs0, discovery_freqs1], axis=0)
                d_valid = shared.filter_lineages(discovery_reads0, discovery_reads1, min_reads=min_reads,
                                                 threshold=max_freqs)
                v_valid = shared.filter_lineages(validation_reads0, validation_reads1, min_reads=min_reads,
                                                 threshold=max_freqs)

                filtered_bool = (d0_freqs > cutoff) * v_valid  # * d_valid
                lineage_indices = rnd.permutation(np.arange(d0_freqs.shape[-1])[filtered_bool])

                d_f0, d_f1 = shared.maxmin_freqs(discovery_freqs0, discovery_D0, discovery_freqs1, discovery_D1)
                v_f0, v_f1 = shared.maxmin_freqs(validation_freqs0, validation_D0, validation_freqs1, validation_D1)

            if 'generate sorted fitnesses':
                n_fit = len(lineage_indices)
                fit_indices, sorted_fitnesses = shared.rank_barcodes(d_f0[lineage_indices], d_f1[lineage_indices],
                                                                     discovery_dt)
                fit_indices = lineage_indices[fit_indices]

                # n_fit = filtered_bool.sum()
                v_f0_fit, v_f1_fit = v_f0[fit_indices][:n_fit], v_f1[fit_indices][:n_fit]
                d_freqs0_fit, d_freqs1_fit = discovery_freqs0[fit_indices][:n_fit], discovery_freqs1[fit_indices][
                                                                                    :n_fit]
                v_freqs0_fit, v_freqs1_fit = validation_freqs0[fit_indices][:n_fit], validation_freqs1[fit_indices][
                                                                                     :n_fit]
                validation_lfcs = shared.calc_lfc_array(v_f0_fit, v_f1_fit, validation_dt)

                cg_validate_lfcs = shared.calc_coarse_grained_lfc_array(v_freqs0_fit, validation_D0, v_freqs1_fit,
                                                                 validation_D1,
                                                                 validation_dt, coarse_grain=cg)

            cg_subset = cg_validate_lfcs[cg // 2:bac_nfit[bac]]
            wu_kernel = scipy.stats.gaussian_kde(wu_discovery_lfcs[~wu_discovery_lfcs.mask], bw_method=0.2)
            lineage_kernel = scipy.stats.gaussian_kde(cg_subset[~cg_subset.mask], bw_method=0.2)
            dx = 0.025
            if bac == 'Bt7330':
                x = np.arange(-1.0, 1.5, dx)
            else:
                x = np.arange(-1.0, 1.2, dx)
            x_avg = (x[1:] + x[:-1]) / 2
            ax.plot(x_avg, wu_kernel(x_avg), color=DIET_COLORS[diet], linestyle='dashed')
            ax.plot(x_avg, lineage_kernel(x_avg), color=DIET_COLORS[diet])
            # ax.plot(wu_discovery_lfcs[wu_sorted_indices])
            # ax.scatter(np.arange(wu_validate_lfcs.shapee[-1]), wu_validate_lfcs[wu_sorted_indices],
            #            s=3, color='violet', label='Wu genes', alpha=0.7, facecolor='none')

        ax.set_yticks([])
        ax.text(0.99, 0.85, BAC_FORMAL_NAMES[bac],
                transform=ax.transAxes, fontsize=8, horizontalalignment='right')
        if bac == 'BWH2':
            legend_markers = [ax_methods.make_marker_obj(linestyle='solid', color=DIET_COLORS['HF']),
                              ax_methods.make_marker_obj(linestyle='dashed', color=DIET_COLORS['HF']),
                              ax_methods.make_marker_obj(linestyle='solid', color=DIET_COLORS['LF']),
                              ax_methods.make_marker_obj(linestyle='dashed', color=DIET_COLORS['LF'])]
            ax.legend(legend_markers, ['HF ben. lineage', 'HF del. gene k.o.', 'LF ben. lineage', 'LF del. gene k.o.'],
                      loc=2)

    return

if __name__ == '__main__':
    fig = plt.figure(figsize=(4, 5.5))
    outer = mpl.gridspec.GridSpec(nrows=4, ncols=1, figure=fig)
    outer_ax = fig.add_subplot(outer[:])
    outer_ax.set_xlabel('Relative fitness (days$^{-1}$), days 0-4', labelpad=5, fontsize=8)
    outer_ax.set_ylabel('Fraction of lineages', labelpad=-2, fontsize=8)
    ax_methods.turn_off_ax(outer_ax)

    make_sfig(fig, outer)

    fig.savefig(f'{plot_dir}/sfig3_adaptive_lineages_vs_Wu_gene.pdf')
