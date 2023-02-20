root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared

def make_sfig(fig, outer):
    cg = 100  # coarse-grain

    all_ax_rank = fig.add_subplot(outer[:, 0])
    all_ax_rank.set_ylabel('Relative fitness in validation (days$^{-1}$)', fontsize=8, labelpad=3)
    ax_methods.turn_off_ax(all_ax_rank)

    HF_discovery = [6, 7, 8, 9, 10]
    LF_discovery = [11, 12, 13, 15]
    HF_validation = [1, 2, 3, 5]
    LF_validation = [16, 18, 19]

    discovery_d0, discovery_d1 = 0, 4
    validation_d0, validation_d1 = 0, 4
    discovery_dt = discovery_d1 - discovery_d0
    validation_dt = validation_d1 - validation_d0
    gene_nn = False
    min_reads = 5
    cutoff = 10 ** -6.5
    sorted_indices_dict = {}

    # discovery_validation_sets = [('HF/HS', [(HF_discovery, LF_validation, 4), (HF_discovery, HF_validation, 9)]),
    #                              ('LF/HPP', [(LF_discovery, HF_validation, 4), (LF_discovery, LF_validation, 9)])]

    HF_color = DIET_COLORS['HF']
    LF_color = DIET_COLORS['LF']
    discovery_validation_sets = [
        ('HF/HS', [(HF_discovery, LF_validation, LF_color), (HF_discovery, HF_validation, HF_color)]),
        ('LF/HPP', [(LF_discovery, HF_validation, HF_color), (LF_discovery, LF_validation, LF_color)])]

    for b, bac in enumerate(['BWH2', 'Bovatus', 'BtVPI', 'Bt7330']):
        row_ids = shared.bac_row_ids[bac]
        notWu = shared.bac_nonwu_indices[bac]
        read_array = shared.bac_read_arrays[bac][:, notWu]
        input_array = shared.bac_input_arrays[bac][:, notWu]

        input_set1 = np.arange(0, input_array.shape[0], 2)
        input_set2 = np.arange(1, input_array.shape[0], 2)
        joint_set = list(input_set1) + list(input_set2)

        d0_reads = input_array[joint_set].sum(axis=0)
        d0_freqs = d0_reads / d0_reads.sum()

        n_fit = 10000
        # for c, (discovery_mice, validation_mouse) in enumerate([(HF_discovery, HF_validation), (HF_discovery, LF_validation), (LF_discovery, LF_validation), (LF_discovery, HF_validation)]):
        for j, (discovery_diet, discovery_validation_set) in enumerate(discovery_validation_sets):
            ax_rank_order = fig.add_subplot(outer[2 * b, j])
            # for b, (bac, n_fit) in enumerate([('BWH2', 10000), ('Bovatus', 10000), ('BtVPI', 10000), ('Bt7330', 10000)]):
            for (discovery_mice, validation_mice, color) in discovery_validation_set:
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
                    # n_fit = len(lineage_indices)
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
                    gene_complement_f0, gene_complement_f1 = shared.calc_gene_complement_freqs(bac, fit_indices[:n_fit],
                                                                                               validation_freqs0,
                                                                                               validation_freqs1,
                                                                                               nearest_neighbors=gene_nn,
                                                                                               cutoff=cutoff)
                    cg_gene_complement_lfcs = shared.calc_coarse_grained_lfc_array(gene_complement_f0, validation_D0,
                                                                            gene_complement_f1, validation_D1,
                                                                            validation_dt, coarse_grain=cg)

                if 'rank order plot':
                    label = DIET_NAMES[mouse2diet[validation_mice[0]]]
                    ax_rank_order.plot(np.arange(cg_validate_lfcs.shape[-1])[cg // 2:-cg // 2],
                                       cg_validate_lfcs[cg // 2:-cg // 2], color=color, label=f'{label} validation')

            if 'format rank order plot':
                _, xticks, xtick_labels = RANK_ORDER_FORMATS['Bovatus']
                ax_rank_order.set_xticks(xticks)
                ax_rank_order.set_xticklabels(xtick_labels)
                ax_rank_order.set_xlim(-n_fit / 100, n_fit)
                ax_rank_order.set_ylim(-0.6, 1.4)
                y_ticks = [-0.4, 0, 0.4, 0.8, 1.2]
                if bac == 'Bt7330':
                    ax_rank_order.set_ylim(-0.6, 1.8)
                    y_ticks = [-0.4, 0, 0.4, 0.8, 1.2, 1.6]
                ax_rank_order.set_yticks(y_ticks, fontsize=40)
                ax_rank_order.axhline(0, color='black', linestyle='dashed')
                ax_rank_order.text(0.05, 0.82, f'{BAC_FORMAL_NAMES[bac]}', transform=ax_rank_order.transAxes)
                # ax_rank_order.text(-0.05, 1.05, panel_indices[bac][0], fontsize=8, transform=ax_rank_order.transAxes)
                if b == 3:
                    disco_diet = DIET_NAMES[mouse2diet[discovery_mice[0]]]
                    ax_rank_order.set_xlabel(f'Fitness rank\nin {disco_diet} discovery, days 0-4', fontsize=8,
                                             labelpad=2)
                if bac == 'BWH2':
                    ax_rank_order.legend(frameon=False)

if __name__ == '__main__':
    RANK_ORDER_FORMATS = {
        'BWH2': (20000, [1000, 5000, 10000, 15000, 19000], ['1000', '5000', '10000', '15000', '19000']),
        'Bovatus': (10000, [1000, 3000, 5000, 7000, 9000], ['1000', '3000', '5000', '7000', '9000']),
        'BtVPI': (10000, [1000, 3000, 5000, 7000, 9000], ['1000', '3000', '5000', '7000', '9000']),
        'Bt7330': (5000, [500, 1500, 2500, 3500, 4500], ['500', '1500', '2500', '3500', '4500'])}
    DIET_NAMES = {'HF': 'HF/HS', 'LF': 'LF/HPP', 'HLH': 'HF/HS', 'LHL': 'LF/HPP'}

    fig = plt.figure(figsize=(5, 5))
    outer = mpl.gridspec.GridSpec(nrows=7, ncols=2,
                                  height_ratios=[1, 0.1, 1, 0.1, 1, 0.1, 1], figure=fig)
    make_sfig(fig, outer)

    fig.savefig(f'{plot_dir}/sfig7_04_rank_order_switch_diets.pdf')