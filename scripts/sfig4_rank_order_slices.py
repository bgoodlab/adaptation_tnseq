root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared

def make_sfig(fig, outer):
    lineage_col = fig.add_subplot(outer[:, 0])
    lineage_col.set_ylabel(r'Fraction of lineages', labelpad=5, fontsize=12)
    ax_methods.turn_off_ax(lineage_col)
    gene_col = fig.add_subplot(outer[:, 2])
    gene_col.set_ylabel(r'Fraction of gene complements', labelpad=5, fontsize=12)
    ax_methods.turn_off_ax(gene_col)

    k = 5
    norm = True
    thresholding = 'MaxOr'
    ### rank order curves
    if 'plot rank order curves, etc':
        min_reads = 5
        gene_nn = False
        cutoff = 10 ** -6.5
        cg = 100  # coarse_grain

        discovery_mice = [6, 7, 8, 9, 10]
        validation_mice = [1, 2, 3, 5]

        # discovery_mice = [11, 12, 13, 15]
        # validation_mice = [16, 18, 19]

        discovery_d0, discovery_d1 = 0, 4
        discovery_dt = discovery_d1 - discovery_d0
        validation_d0, validation_d1 = 0, 4
        validation_dt = validation_d1 - validation_d0

    for b, (bac, n_fit) in enumerate([('BWH2', 100000), ('Bovatus', 100000),
                                      ('BtVPI', 100000), ('Bt7330', 100000)]):
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

            filtered_bool = (d0_freqs > cutoff) * v_valid
            lineage_indices = rnd.permutation(np.arange(d0_freqs.shape[-1])[filtered_bool])

            d_f0, d_f1 = shared.maxmin_freqs(discovery_freqs0, discovery_D0, discovery_freqs1, discovery_D1)
            v_f0, v_f1 = shared.maxmin_freqs(validation_freqs0, validation_D0, validation_freqs1, validation_D1)

        if 'generate sorted fitnesses':
            fit_indices, sorted_fitnesses = shared.rank_barcodes(d_f0[lineage_indices], d_f1[lineage_indices],
                                                                 discovery_dt)
            fit_indices = lineage_indices[fit_indices]

            n_fit = filtered_bool.sum()
            v_f0_fit, v_f1_fit = v_f0[fit_indices][:n_fit], v_f1[fit_indices][:n_fit]
            # d_freqs0_fit, d_freqs1_fit = discovery_freqs0[fit_indices][:n_fit], discovery_freqs1[fit_indices][:n_fit]
            v_freqs0_fit, v_freqs1_fit = validation_freqs0[fit_indices][:n_fit], validation_freqs1[fit_indices][:n_fit]
            validation_lfcs = shared.calc_lfc_array(v_f0_fit, v_f1_fit, validation_dt)

            # cg_validate_lfcs = shared.calc_coarse_grained_lfc_array(v_freqs0_fit, validation_D0, v_freqs1_fit, validation_D1,
            #                                                  validation_dt, coarse_grain=cg)
            gene_complement_f0, gene_complement_f1 = shared.calc_gene_complement_freqs(bac, fit_indices[:n_fit],
                                                                                       validation_freqs0,
                                                                                       validation_freqs1,
                                                                                       nearest_neighbors=gene_nn,
                                                                                       cutoff=cutoff)
            gene_complement_lfcs = shared.calc_lfc_array(gene_complement_f0, gene_complement_f1, validation_dt)

            # cg_gene_complement_lfcs = shared.calc_coarse_grained_lfc_array(gene_complement_f0, validation_D0,
            #                                                         gene_complement_f1, validation_D1,
            #                                                         validation_dt, coarse_grain=cg)
        if bac == 'BWH2':
            rank_ranges = [(1, 200), (2000, 2200), (10000, 10200), (20000, 20200)]
        elif bac != 'Bt7330':
            rank_ranges = [(1, 200), (2000, 2200), (5000, 5200), (20000, 20200)]
        else:
            rank_ranges = [(1, 200), (1000, 1200), (2000, 2200), (6000, 6200)]

        cmap = [KELLY_COLORS[i] for i in [9, 3, 8, 4]]
        if 'rank order plot':
            lineage_ax = fig.add_subplot(outer[b, 0])
            gene_ax = fig.add_subplot(outer[b, 2])

            bins = np.linspace(-0.4, 1.65, 42)
            avg_bins = (bins[:-1] + bins[1:]) / 2
            for i, (rank0, rank1) in enumerate(rank_ranges):

                gene_valid = ~gene_complement_lfcs.mask[rank0:rank1]
                density, bins = np.histogram(validation_lfcs[rank0:rank1][gene_valid], bins=bins, density=True)
                gene_density, bins = np.histogram(gene_complement_lfcs[rank0:rank1][gene_valid], bins=bins,
                                                  density=True)
                cdf = np.cumsum(density) / density.sum()
                if rank0 >= 1000:
                    label_rank0 = f'{rank0 / 1000:.1f}k'
                    label_rank1 = f'{rank1 / 1000:.1f}k'
                else:
                    label_rank0 = rank0
                    label_rank1 = rank1
                lineage_ax.plot(avg_bins, density, color=cmap[i], label=f'{label_rank0}-{label_rank1}', zorder=5 - i)
                gene_ax.plot(avg_bins, gene_density, color=cmap[i], label=f'{label_rank0}-{label_rank1}', zorder=5 - i)

            gene_ax.legend(title=f'Ranks', frameon=False)
            lineage_ax.axvline(0, linestyle='dashed', color='black', zorder=0)
            gene_ax.axvline(0, linestyle='dashed', color='black', zorder=0)

        gene_ax.text(1.03, 0.5, f'{BAC_FORMAL_NAMES[bac]}',
                     transform=gene_ax.transAxes, fontsize=14, weight='bold')
        gene_ax.tick_params(axis='both', labelsize=10)
        lineage_ax.tick_params(axis='both', labelsize=10)
        gene_ax.set_xticks([0, 0.5, 1])
        lineage_ax.set_xticks([0, 0.5, 1])

if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 7))
    outer = mpl.gridspec.GridSpec(nrows=4, ncols=3, width_ratios=(1, 0.03, 1), figure=fig)
    outer_ax = fig.add_subplot(outer[:, :])
    ax_methods.turn_off_ax(outer_ax)
    outer_ax.set_xlabel(r'Relative fitness (days$^{-1}$) in validation, days 0-4', labelpad=7, fontsize=12)
    outer_ax.set_ylabel(r'Fraction of lineages', labelpad=5, fontsize=12)

    make_sfig(fig, outer)

    fig.savefig(f'{plot_dir}/sfig4_rank_order_slices.pdf')

