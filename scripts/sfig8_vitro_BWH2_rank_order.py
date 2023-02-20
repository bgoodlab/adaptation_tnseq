root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared

def make_sfig(fig, outer):
    outer_ax = fig.add_subplot(outer[:])
    outer_ax.set_ylabel('$\it{in~vitro}$ fold change', fontsize=10, labelpad=10)
    outer_ax.set_xlabel('day 0-4 fitness rank order in HF/HS mice', fontsize=10, labelpad=10)
    ax_methods.turn_off_ax(outer_ax)

    bac = 'BWH2'
    if 'plot rank order curves, etc':
        min_reads = 5
        gene_nn = False
        cutoff = 10 ** -6.5
        cg = 100  # coarse_grain

        discovery_mice = [1, 2, 3, 5, 6, 7, 8, 9, 10]

        discovery_d0, discovery_d1 = 0, 4
        discovery_dt = discovery_d1 - discovery_d0
        validation_d0, validation_d1 = 0, 4
        validation_dt = validation_d1 - validation_d0

    for m, medium in enumerate(MEDIA):
        ax_rank_order = fig.add_subplot(outer[m // 2, m % 2])
        for i, color in [(1, KELLY_COLORS[5]), (2, KELLY_COLORS[17])]:
            if 'generate discovery and validation frequencies':
                discovery_reads0, discovery_reads1 = np.copy(
                    shared.get_read_arrays(bac, discovery_mice, discovery_d0, discovery_d1, split_day0=1))
                validation_reads0, _ = np.copy(
                    shared.get_read_arrays(bac, [1], validation_d0, validation_d1, split_day0=2))
                validation_reads1 = shared.bac_read_arrays[bac][shared.bac_row_ids[bac][(medium, i)]][
                    shared.bac_nonwu_indices[bac]]

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
                fit_indices, sorted_fitnesses = shared.rank_barcodes(d_f0[lineage_indices], d_f1[lineage_indices],
                                                                     discovery_dt)
                fit_indices = lineage_indices[fit_indices]

                n_fit = filtered_bool.sum()
                v_f0_fit, v_f1_fit = v_f0[fit_indices][:n_fit], v_f1[fit_indices][:n_fit]
                d_freqs0_fit, d_freqs1_fit = discovery_freqs0[fit_indices][:n_fit], discovery_freqs1[fit_indices][
                                                                                    :n_fit]
                v_freqs0_fit, v_freqs1_fit = validation_freqs0[fit_indices][:n_fit], validation_freqs1[fit_indices][
                                                                                     :n_fit]
                validation_lfcs = shared.calc_lfc_array(v_f0_fit, v_f1_fit, 1)

                cg_validate_lfcs = shared.calc_coarse_grained_lfc_array(v_freqs0_fit, validation_D0, v_freqs1_fit,
                                                                 validation_D1,
                                                                 1, coarse_grain=cg)

            if 'plot rank order':
                ax_rank_order.plot(np.arange(n_fit)[cg // 2:-cg // 2], np.exp(cg_validate_lfcs[cg // 2:-cg // 2]),
                                   color=color)

        if True:  # formatting
            ax_rank_order.axhline(1, color='black', linestyle='dashed', zorder=0)
            # ax_rank_order.set_xlim(-n_fit/10, n_fit*(1+1/10))
            ax_rank_order.set_yscale('log')
            # ax_rank_order.set_ylim(4*10**-2, 2*10)
            ax_rank_order.text(0.97, 0.85, f'{BAC_FORMAL_NAMES[bac]} {medium}',
                               fontsize=8, transform=ax_rank_order.transAxes,
                               horizontalalignment='right')
            # ax_rank_order.set_xscale('log')
            # ax_rank_order.set_xticks([10**2, 10**3, 10**4])
            # ax_rank_order.set_xticklabels([r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'], fontsize=8)

            ax_rank_order.set_yticks([10 ** -1, 1, 10 ** 1])
            ax_rank_order.set_xticks([1, 20000, 40000, 60000])
            ax_rank_order.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'], fontsize=8)
            ax_rank_order.set_xticklabels(['1', '20k', '40k', '60k'], fontsize=8)

if __name__ == '__main__':
    fig = plt.figure(figsize=(5, 5))
    outer = mpl.gridspec.GridSpec(nrows=3, ncols=2, figure=fig)

    make_sfig(fig, outer)

    fig.savefig(f'{plot_dir}/sfig8_vitro_BWH2_rank_order.pdf')