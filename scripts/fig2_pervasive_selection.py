root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared
import scipy.stats

def heat_map(ax, bac, m1, m2, day, day0_read_range):
    row_ids = shared.bac_row_ids[bac]
    notWu = shared.bac_nonwu_indices[bac]
    read_array = shared.bac_read_arrays[bac][:, notWu]

    d0_reads = read_array[0]

    m1_reads = np.copy( read_array[ row_ids[(m1, day)] ] )
    m2_reads = np.copy( read_array[ row_ids[(m2, day)] ] )

    day0_min, day0_max = day0_read_range
    lineage_indices = (d0_reads >= day0_min) * (d0_reads < day0_max) #* (m1_d4_reads > 0) * (m2_d4_reads > 0)

    ## plot empirical
    binspace = np.linspace(0, 80, 81)
    ax.hist2d(m1_reads[lineage_indices], m2_reads[lineage_indices], bins=binspace, norm=mpl.colors.LogNorm(0.1, 200), cmap=mpl.cm.Oranges)
    r, pval = scipy.stats.pearsonr(m1_reads[lineage_indices], m2_reads[lineage_indices])

    ax.set_xlabel('mouse {} day 4 reads'.format(m1))
    ax.set_ylabel('mouse {} day 4 reads'.format(m2))

    ax.set_xlim(0, 80)
    ax.set_ylim(0, 80)
    ax.axvspan( day0_min * m1_reads.sum() / d0_reads.sum(), day0_max * m1_reads.sum() / d0_reads.sum(),
                linewidth=0, color='grey', alpha=0.5, zorder=0)
    ax.axhspan( day0_min * m2_reads.sum() / d0_reads.sum(), day0_max * m2_reads.sum() / d0_reads.sum(), color='grey', alpha=0.5, zorder=0, linewidth=0)
    ax.plot( ax.get_xlim(), np.array(ax.get_xlim()) * m2_reads.sum()/m1_reads.sum(),
             color='black', linestyle='dashed')

def calc_coarse_grained_lfc_array(freqs0, D0, freqs1, D1, dt, coarse_grain=100):
    cg_freqs0 = shared.running_mean_uniform_filter1d(freqs0, coarse_grain)
    cg_freqs1 = shared.running_mean_uniform_filter1d(freqs1, coarse_grain)

    cg_f0, cg_f1 = shared.maxmin_freqs(cg_freqs0, D0, cg_freqs1, D1)

    with np.errstate(divide='ignore'):
        return np.log( cg_f1/cg_f0 ) / dt

RANK_ORDER_FORMATS = {'BWH2': (20000, [1000, 5000, 10000, 15000, 19000], ['1000', '5000', '10000', '15000', '19000']),
                      'Bovatus': (10000, [1000, 3000, 5000, 7000, 9000], ['1000', '3000', '5000', '7000', '9000']),
                      'BtVPI': (10000, [1000, 3000, 5000, 7000, 9000], ['1000', '3000', '5000', '7000', '9000']),
                      'Bt7330': (5000, [500, 1500, 2500, 3500, 4500], ['500', '1500', '2500', '3500', '4500'])}

def make_fig2(fig, outer):
    ### Joint distribution
    if 'plot joint distribution':
        bac, bc = 'BWH2', 'HL'
        m1, m2 = 6, 7
        day = 4

        row_ids = shared.bac_row_ids[bac]
        notWu = shared.bac_nonwu_indices[bac]
        read_array = shared.bac_read_arrays[bac][:, notWu]

        d0_reads = read_array[0]

        m1_reads = np.copy( read_array[ row_ids[(m1, day)] ] )
        m2_reads = np.copy( read_array[ row_ids[(m2, day)] ] )

        day0_read_range = np.array([10, 15])*d0_reads.sum()/(5*10**5)
        joint_ax = fig.add_subplot(outer[1, 0:2])
        heat_map(joint_ax, bac, m1, m2, day, day0_read_range)
        joint_ax.set_xticks([0, 20, 40, 60])
        joint_ax.set_yticks([0, 20, 40, 60])
        joint_ax.set_xlabel('Day 4 lineage counts in M1', fontsize=8, labelpad=3)
        joint_ax.set_ylabel('Day 4 lineage counts in M2', fontsize=8, labelpad=3)
        joint_ax.text(0.04, 0.9, '$\\bf{a}$', fontsize=8, transform=joint_ax.transAxes)
        joint_ax.text(0.1, 0.9, f'{BAC_FORMAL_NAMES[bac]} HF/HS', transform=joint_ax.transAxes, fontsize=8)
        joint_ax.tick_params(axis='both', labelsize=6)

        if 'colorbar':
            colorbar_spec = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=(0.00, 1), subplot_spec=outer[1, 2])
            colorbar_ax = fig.add_subplot(colorbar_spec[1])
            norm = mpl.colors.LogNorm(0.1, 200)
            _ = mpl.colorbar.ColorbarBase(colorbar_ax, cmap=mpl.cm.Oranges, norm=norm)
            colorbar_ax.tick_params(axis='both', direction='out', labelsize=6)
            colorbar_ax.text(-0.15, 1.03, '# lineages', transform=colorbar_ax.transAxes,fontsize=8)
            colorbar_ax.set_ylim(1, 200)
            colorbar_ax.set_yticks([1, 10, 100])
            colorbar_ax.set_yticklabels(['1', '10', '100'])

    ### schematic
    if 'plot cross-validation schematic':
        schematic_ax = fig.add_subplot(outer[0:3,  4:])
        im = mpl.image.imread(f'{plot_dir}/fig2_schematic.png')
        schematic_ax.imshow(im)
        ax_methods.turn_off_ax(schematic_ax)
        schematic_ax.set_xticks([])
        schematic_ax.set_yticks([])
        schematic_ax.text(-0.02, 0.95, '$\\bf{b}$', transform=schematic_ax.transAxes)

    all_ax_rank = fig.add_subplot(outer[3:, 0:4])
    all_ax_rank.set_ylabel('Relative fitness in validation (days$^{-1}$)', fontsize=8, labelpad=5)
    all_ax_rank.set_xlabel('Fitness rank in discovery, days 0-4', fontsize=8, labelpad=5)
    ax_methods.turn_off_ax(all_ax_rank)

    all_ax_gene = fig.add_subplot(outer[3:, 5:6])
    all_ax_gene.set_ylabel('Average relative fitness', fontsize=8, labelpad=5)
    all_ax_gene.set_xlabel('Gene complement', fontsize=8, labelpad=5)
    ax_methods.turn_off_ax(all_ax_gene)

    all_ax_temporal = fig.add_subplot(outer[3:, 7:8])
    all_ax_temporal.set_ylabel('Coarse-grained fold change, $f(t)/f(0)$', fontsize=8, labelpad=5)
    all_ax_temporal.set_xlabel('Days', fontsize=8, labelpad=5)
    ax_methods.turn_off_ax(all_ax_temporal)


    ### rank order curves
    if 'plot rank order curves, etc':
        min_reads = 5
        gene_nn = False
        cutoff = 10**-6.5
        cg = 100 #coarse_grain

        discovery_mice = [6, 7, 8, 9, 10]
        validation_mice = [1, 2, 3, 5]

        discovery_d0, discovery_d1 = 0, 4
        discovery_dt = discovery_d1 - discovery_d0
        validation_d0, validation_d1 = 0, 4
        validation_dt = validation_d1 - validation_d0
        panel_indices = {'BWH2':['$\\bf{c}$', '$\\bf{d}$', '$\\bf{e}$'],
                     'BtVPI':['$\\bf{f}$', '$\\bf{g}$', '$\\bf{h}$'],
                     'Bovatus':['$\\bf{f}$', '$\\bf{g}$', '$\\bf{h}$'],
                         'Bt7330':['$\\bf{f}$', '$\\bf{g}$', '$\\bf{h}$']}

    for b, (bac, n_show) in enumerate([('BWH2', 20000), ('BtVPI', 20000)]):
        if 'generate discovery and validation frequencies':
            discovery_reads0, discovery_reads1 = np.copy(shared.get_read_arrays(bac, discovery_mice, 0, 4, split_day0=2))
            validation_reads0, validation_reads1 = np.copy(shared.get_read_arrays(bac, validation_mice, 0, 4, split_day0=1))
            # discovery_reads0 -= shared.bac_input_arrays[bac][6][shared.bac_nonwu_indices[bac]]

            discovery_D0, discovery_D1 = discovery_reads0.sum(), discovery_reads1.sum()
            discovery_freqs0, discovery_freqs1 = discovery_reads0 / discovery_D0, discovery_reads1 / discovery_D1

            validation_D0, validation_D1 = validation_reads0.sum(), validation_reads1.sum()
            validation_freqs0, validation_freqs1 = validation_reads0 /validation_D0, validation_reads1 /validation_D1

            d0_reads = shared.bac_read_arrays[bac][0][shared.bac_nonwu_indices[bac]]
            d0_freqs = d0_reads / d0_reads.sum()
            d1_reads = (discovery_reads1 + validation_reads1)

            max_freqs = np.max([discovery_freqs0, discovery_freqs1], axis=0)
            d_valid = shared.filter_lineages(discovery_reads0, discovery_reads1, min_reads=min_reads, threshold=max_freqs)
            v_valid = shared.filter_lineages(validation_reads0, validation_reads1, min_reads=min_reads, threshold=max_freqs)

            filtered_bool =  (d0_freqs > cutoff) * v_valid #* d_valid
            lineage_indices = rnd.permutation( np.arange(d0_freqs.shape[-1])[filtered_bool] )

            d_f0, d_f1 = shared.maxmin_freqs(discovery_freqs0, discovery_D0, discovery_freqs1, discovery_D1)
            v_f0, v_f1 = shared.maxmin_freqs(validation_freqs0, validation_D0, validation_freqs1, validation_D1)

        if 'generate sorted fitnesses':
            fit_indices, sorted_fitnesses = shared.rank_barcodes(d_f0[lineage_indices], d_f1[lineage_indices], discovery_dt)
            fit_indices = lineage_indices[fit_indices]

            v_f0_fit, v_f1_fit = v_f0[fit_indices][:n_show], v_f1[fit_indices][:n_show]
            # d_freqs0_fit, d_freqs1_fit = discovery_freqs0[fit_indices][:n_show], discovery_freqs1[fit_indices][:n_show]
            v_freqs0_fit, v_freqs1_fit = validation_freqs0[fit_indices][:n_show], validation_freqs1[fit_indices][:n_show]
            validation_lfcs = shared.calc_lfc_array(v_f0_fit, v_f1_fit, validation_dt)

            cg_validate_lfcs = calc_coarse_grained_lfc_array(v_freqs0_fit, validation_D0, v_freqs1_fit, validation_D1,
                                                             validation_dt, coarse_grain=cg)
            gene_complement_f0, gene_complement_f1 = shared.calc_gene_complement_freqs(bac, fit_indices[:n_show], validation_freqs0, validation_freqs1, nearest_neighbors=gene_nn, cutoff=cutoff)
            cg_gene_complement_lfcs = calc_coarse_grained_lfc_array(gene_complement_f0, validation_D0, gene_complement_f1, validation_D1,
                                                                    validation_dt, coarse_grain=cg)

        if 'rank order plot':
            ax_rank_order = fig.add_subplot(outer[2*b+3, 0:4])
            # ax_rank_order.plot(np.arange(n_show), sorted_fitnesses[:n_show], KELLY_COLORS[4], label="discovery fitness")
            ax_rank_order.scatter(np.arange(validation_lfcs.shape[-1]), validation_lfcs, s=1, color=KELLY_COLORS[3], rasterized=True, zorder=0, alpha=0.1)
            ax_rank_order.plot(np.arange(cg_validate_lfcs.shape[-1])[cg//2:-cg//2],
                                cg_validate_lfcs[cg//2:-cg//2], KELLY_COLORS[9], label="coarse-grained lineage")
            ax_rank_order.plot(np.arange(cg_gene_complement_lfcs.shape[-1])[cg//2:-cg//2],
                                cg_gene_complement_lfcs[cg//2:-cg//2], KELLY_COLORS[1], zorder=0, label="gene complement")

            muller_lineages = muller_lineage_dict[bac]['HF']
            muller_ranks = []
            for muller_lineage in muller_lineages:
                try: muller_ranks.append(np.where(fit_indices[:n_show]==muller_lineage)[0][0])
                except: continue
            muller_ranks = np.array(muller_ranks)

            ax_rank_order.scatter(np.arange(validation_lfcs.shape[-1])[muller_ranks], validation_lfcs[muller_ranks], s=10, marker=(3, 0, -90), color=DIET_COLORS['HF'], rasterized=True, zorder=10)

            if bac == 'BWH2':
                legend_markers = [ax_methods.make_marker_obj(color=KELLY_COLORS[9], marker='o', mec=KELLY_COLORS[3], mfc=KELLY_COLORS[3], markersize=3.5),
                                  ax_methods.make_marker_obj(color=KELLY_COLORS[1]),
                                  ax_methods.make_marker_obj(color=DIET_COLORS['HF'], linewidth=0, marker='>', mfc=DIET_COLORS['HF'], mec=DIET_COLORS['HF'], markersize=3.5)]
                ax_rank_order.legend(legend_markers, ['validation', 'gene comp.', 'D16 lineages'],
                                     fontsize=6, loc=1, frameon=False)

        if 'format rank order plot':
            _, xticks, xtick_labels = RANK_ORDER_FORMATS[bac]
            if n_show == 10000:
                xticks = [1000, 3000, 5000, 7000, 9000]
                xtick_labels = xticks
            if n_show == 20000:
                xticks = [1000, 5000, 10000, 15000, 20000]
                xtick_labels = ['1k', '5k', '10k', '15k', '20k']
            ax_rank_order.set_xticks(xticks)
            ax_rank_order.set_xticklabels(xtick_labels)
            ax_rank_order.set_xlim(-n_show/100, n_show)
            ax_rank_order.set_ylim(-0.4, 1.2)
            y_ticks = [-0.4, 0, 0.4, 0.8, 1.2]
            if bac == 'Bt7330':
                ax_rank_order.set_ylim(-0.6, 1.8)
                y_ticks = [-0.4, 0, 0.4, 0.8, 1.2, 1.6]
            ax_rank_order.set_yticks(y_ticks, fontsize=40)
            ax_rank_order.axhline(0, color='black', linestyle='dashed')
            ax_rank_order.text(0.12, 0.84, f'{BAC_FORMAL_NAMES[bac]} HF/HS', transform=ax_rank_order.transAxes)
            ax_rank_order.text(0.03, 0.86, panel_indices[bac][0], fontsize=8, transform=ax_rank_order.transAxes)
            ax_rank_order.tick_params(axis='both', labelsize=6)

        if 'plot lineage vs gene':
            freqs0 = d0_freqs
            freqs1 = d1_reads/d1_reads.sum()
            gene_complement_f0, gene_complement_f1 = shared.calc_gene_complement_freqs(bac, fit_indices[:n_show], freqs0, freqs1, nearest_neighbors=gene_nn)

            lineage_f0, lineage_f1 = shared.maxmin_freqs(freqs0[fit_indices[:n_show]], d0_reads.sum(), freqs1[fit_indices[:n_show]], d1_reads.sum())
            gene_f0, gene_f1 = shared.maxmin_freqs(gene_complement_f0, d0_reads.sum(), gene_complement_f1, d1_reads.sum())

            fit_lineage_lfcs = shared.calc_lfc_array(lineage_f0, lineage_f1, 4)
            gene_complement_lfcs = shared.calc_lfc_array(gene_f0, gene_f1, 4)

            ax_gene_lfc = fig.add_subplot(outer[2*b+3, 5:6])
            if 'format gene':
                ax_gene_lfc.scatter( gene_complement_lfcs, fit_lineage_lfcs,  s=0.5, rasterized=True, color=KELLY_COLORS[1])
                ax_gene_lfc.set_ylim(-0.3, 1.2)
                ax_gene_lfc.set_xlim(-1.0, 1.0)
                ax_gene_lfc.set_xticks([-0.8, 0.0, 0.8])
                ax_gene_lfc.set_yticks([-0.4, 0.0, 0.4, 0.8, 1.2])
                if bac == 'Bt7330':
                    ax_gene_lfc.set_ylim(-0.3, 1.6)
                    ax_gene_lfc.set_xlim(-1.0, 1.6)
                    ax_gene_lfc.set_yticks([0.0, 0.4, 0.8, 1.2])
                    ax_gene_lfc.set_xticks([-0.8, -0.4, 0.0, 0.4, 0.8, 1.2])
                ax_gene_lfc.axvline(0, color='black', linestyle='dashed')
                ax_gene_lfc.axhline(0, color='black', linestyle='dashed')
                ax_gene_lfc.text(0.05, 0.88, panel_indices[bac][1], fontsize=8, transform=ax_gene_lfc.transAxes)
                ax_gene_lfc.plot(ax_gene_lfc.get_xlim(), ax_gene_lfc.get_xlim(), color='black', linestyle='dashed')
                ax_gene_lfc.tick_params(axis='both', labelsize=6)

        temporal_mice = [6, 7, 8, 9, 10]
        if 'plot fold changes over time':
            read_array = shared.bac_read_arrays[bac][:, shared.bac_nonwu_indices[bac]]

            rank_colors = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.6), (0.0, 0.0, 0.3), 'black']
            if bac == 'BWH2':
                rank_ranges = [(0, 1000), (1000, 10000), (10000, 20000), (20000, 100000)]
            if bac == 'BtVPI' or bac == 'Bovatus':
                rank_ranges = [(0, 1000), (1000, 10000), (10000, 20000), (20000, 100000)]
            if bac == 'Bt7330':
                rank_ranges = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 100000)]

            freqs0 = np.tile( d0_freqs, (len(temporal_mice), 1) )[:, fit_indices]
            reads4 = read_array[ [row_ids[(mouse, 4)] for mouse in temporal_mice] ]
            freqs4 = np.einsum('ij, i->ij', reads4, reads4.sum(axis=1)**-1.)[:, fit_indices]
            reads10 = read_array[ [row_ids[(mouse, 10)] for mouse in temporal_mice] ]
            freqs10 = np.einsum('ij, i->ij', reads10, reads10.sum(axis=1)**-1.)[:, fit_indices]
            reads16 = read_array[ [row_ids[(mouse, 16)] for mouse in temporal_mice] ]
            freqs16 = np.einsum('ij, i->ij', reads16, reads16.sum(axis=1)**-1.)[:, fit_indices]

            ax_temporal = fig.add_subplot(outer[b*2+3, 7:8])
            for r, rank_range in enumerate(rank_ranges):
                r0, r1 = rank_range
                cg_freqs0 = freqs0[:, r0:r1].sum(axis=1)
                cg_freqs4 = freqs4[:, r0:r1].sum(axis=1)
                cg_freqs10 = freqs10[:, r0:r1].sum(axis=1)
                cg_freqs16 = freqs16[:, r0:r1].sum(axis=1)

                ref_freqs = cg_freqs0
                fcs = np.zeros((4, cg_freqs0.shape[0]))
                fcs[0, :] = cg_freqs0/ref_freqs
                fcs[1, :] = cg_freqs4/ref_freqs
                fcs[2, :] = cg_freqs10/ref_freqs
                fcs[3, :] = cg_freqs16/ref_freqs
                ax_temporal.plot([0, 4, 10, 16], fcs, color=rank_colors[r])

            ax_temporal.text(0.05, 0.88, panel_indices[bac][2], fontsize=8, transform=ax_temporal.transAxes)
            ax_temporal.set_xticks([0, 4, 10, 16])
            ax_temporal.set_yscale('log')
            ax_temporal.tick_params(axis='both', labelsize=6)

    return

if __name__ == '__main__':
    with open(f'{data_dir}/shared_10biggest_indices.pkl', 'rb') as f:
        muller_lineage_dict = pickle.load(f)

    fig = plt.figure(figsize=(6.5, 6))
    outer = mpl.gridspec.GridSpec(nrows=6, ncols=8, height_ratios=[0.001, 0.5, 0.05, 0.3, 0.0, 0.3],
                                  width_ratios=[1, 0.8, 0.1, 0.4, 0.3, 1, 0.4, 1], figure=fig)
    make_fig2(fig, outer)

    fig.savefig(f'{plot_dir}/fig2_rank_order_curves_BWH2_BtVPI_20k.pdf')

