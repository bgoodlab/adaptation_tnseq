root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared

def find_shared_biggest_indices(num_lineages=10):
    big_barcode_indices = {bac: {diet: [] for diet in ['HF', 'LF', 'HLH', 'LHL']} for bac in shared.BACTERIA}
    for bac in shared.BACTERIA:
        for diet in ['HF', 'LF', 'HLH', 'LHL']:
            mice = diet_mice_map[diet]
            row_ids = shared.bac_row_ids[bac]
            notWu = shared.bac_nonwu_indices[bac]
            read_array = shared.bac_read_arrays[bac][:, notWu]

            mice_reads = read_array[[row_ids[(mouse, 16)] for mouse in mice]].sum(axis=0)
            mice_freqs = mice_reads / mice_reads.sum()
            sorted_indices = np.argsort(mice_freqs)[::-1]

            mouse_reads = read_array[[row_ids[(mouse, 16)] for mouse in mice]]
            mouse_freqs = np.einsum('ij, i->ij', mouse_reads, mouse_reads.sum(axis=1) ** -1.)

            topX = []
            for index in sorted_indices:
                mouse_bool = ((mouse_freqs[:, index] > 1e-3).sum() > 1)
                if mouse_bool:
                    topX.append(index)
                if len(topX) == num_lineages:
                    break
            big_barcode_indices[bac][diet] = topX
    return big_barcode_indices

def find_individual_biggest_indices(mice, num_lineages=10):
    big_barcode_indices = {bac: {} for bac in BACTERIA}
    for bac in BACTERIA:
        row_ids = shared.bac_row_ids[bac]
        notWu = shared.bac_nonwu_indices[bac]
        read_array =shared.bac_read_arrays[bac][:, notWu]

        for mouse in mice:
            mouse_reads = read_array[[row_ids[(mouse, day)] for day in days]]
            mouse_freqs = np.einsum('ij, i->ij', mouse_reads, mouse_reads.sum(axis=1) ** -1.)

            indices = np.argsort(mouse_freqs[-1])[-num_lineages:]

            big_barcode_indices[bac][mouse] = list(indices)
    return big_barcode_indices

def make_muller_freqs(bac, mouse, indices, cmap):
    row_ids = shared.bac_row_ids[bac]
    notWu = shared.bac_nonwu_indices[bac]
    read_array = shared.bac_read_arrays[bac][:, notWu]

    big_indices = list(indices)

    day0_reads = read_array[0]
    day0_freqs = (day0_reads / day0_reads.sum())[big_indices]
    day0_freqs = list(day0_freqs) + [1 - np.sum(day0_freqs)]

    freqs = [day0_freqs]
    for day in days:
        mouse_reads = read_array[row_ids[(mouse, day)]]
        mouse_freqs = mouse_reads / mouse_reads.sum()
        big_barcode_freqs = mouse_freqs[big_indices]

        coarse_grained_freqs = list(big_barcode_freqs) + [1 - np.sum(big_barcode_freqs)]
        freqs.append(coarse_grained_freqs)

    freqs = np.array(freqs)
    # sort on day 16 freqs
    day16_sorted = np.argsort(freqs[-1][:-1])[::-1]
    freqs[:, :-1] = freqs[:, :-1][:, day16_sorted]

    sorted_cmap = cmap[day16_sorted]
    return freqs, sorted_cmap

def muller_plot(ax, times, freqs, colors):
    """ freqs has dimensions of (time, bc freq).
     times and freqs.shape[0] should have same dimension. """
    cum_freqs = np.cumsum(freqs, axis=1)

    ax.fill_between(times, np.zeros(len(times)), cum_freqs[:, 0], color=colors[0])

    for i in range(freqs.shape[1] - 2):
        ax.fill_between(times, cum_freqs[:, i], cum_freqs[:, i + 1], color=colors[i])
    ax.fill_between(times, cum_freqs[:, freqs.shape[1] - 2], cum_freqs[:, freqs.shape[1] - 1], color='lightgrey')

def plot_transition_curve(ax, bac, mice, input_rep, day, target_read_range):
    row_ids = shared.bac_row_ids[bac]
    notWu = shared.bac_nonwu_indices[bac]
    input_array = shared.bac_input_arrays[bac][:, notWu]
    read_array = shared.bac_read_arrays[bac][:, notWu]

    null_reads = input_array[input_rep]
    day0_reads = read_array[0] - null_reads
    day0_freqs = day0_reads / day0_reads.sum()

    ## null_distribution
    null_indices = (day0_freqs * null_reads.sum() > target_read_range[0]) * (
                day0_freqs * null_reads.sum() < target_read_range[1])
    read_density = np.bincount(null_reads[null_indices])
    read_density = read_density / read_density.sum()
    ax.fill_between(np.arange(read_density.shape[0]), read_density, color='lightgrey')

    for mouse in mice:
        mouse_reads = read_array[row_ids[(mouse, day)]]
        mouse_indices = (day0_freqs * mouse_reads.sum() > target_read_range[0]) * (
                    day0_freqs * mouse_reads.sum() < target_read_range[1])
        read_density = np.bincount(mouse_reads[mouse_indices])
        read_density = read_density / read_density.sum()
        ax.plot(np.arange(read_density.shape[0]), read_density, color=DIET_COLORS[mouse2diet[mouse]])

def make_fig1(fig, outer, highlight_barcodes):
    ### SCHEMATICS
    if 'schematics' != 'include':
        # experiment schematic
        lineage_schematic_ax = fig.add_subplot(outer[:2, 0])
        im = mpl.image.imread(f'{plot_dir}/lineage_schematic.png')
        lineage_schematic_ax.imshow(im)
        ax_methods.turn_off_ax(lineage_schematic_ax)
        lineage_schematic_ax.set_xticks([])
        lineage_schematic_ax.set_yticks([])
        lineage_schematic_ax.text(-0.07, 0.96, '$\\bf{(a)}$', fontsize=8, transform=lineage_schematic_ax.transAxes)

        # branching process picture
        bp_schematic_ax = fig.add_subplot(outer[2, 0])
        im = mpl.image.imread(f'{plot_dir}/bp_schematic.png')
        num_rows, num_cols = im.data.shape[:-1]
        bp_schematic_ax.imshow(im)  # , extent=(-0.5, 1.5*num_cols, 1.3*num_rows, -2*num_rows))
        ax_methods.turn_off_ax(bp_schematic_ax)
        bp_schematic_ax.set_xticks([])
        bp_schematic_ax.set_yticks([])
        bp_schematic_ax.text(-0.09, 0.98, '$\\bf{(f)}$', fontsize=8, transform=bp_schematic_ax.transAxes)

    if 'muller plots' != 'include':
        muller_group_ax = fig.add_subplot(outer[0, 2:])
        muller_group_ax.set_xlabel('Days', fontsize=8, labelpad=3)
        muller_group_ax.set_ylabel('Lineage frequency', labelpad=-1, fontsize=8)
        ax_methods.turn_off_ax(muller_group_ax)

        muller_ax_indices = {0: '$\\bf{b}$', 1: '$\\bf{c}$', 2: '$\\bf{d}$', 3: '$\\bf{e}$'}
        for b, bac in enumerate(ordered_bac_lst):
            muller_gs = mpl.gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0, b + 2], hspace=0, wspace=0)

            muller_ax = fig.add_subplot(muller_gs[:])

            ax_methods.turn_off_ax(muller_ax)
            muller_axes = {}
            muller_ax.set_title(BAC_FORMAL_NAMES[bac], fontsize=8, pad=2)
            muller_ax.set_yticks([])

            for m, mice in enumerate(muller_mice):
                diet = mouse2diet[mice[0]]
                indices = rnd.permutation(list(set(np.ravel([highlight_barcodes[bac][mouse] for mouse in mice]))))

                cmap = plt.cm.get_cmap(diet_cmaps[diet])  # color by diet
                pooled_cmap = rnd.permutation(list(iter(cmap(np.linspace(0, 1, len(indices))))))
                pooled_cmap = {index: pooled_cmap[i] for i, index in enumerate(indices)}

                for k, mouse in enumerate(mice):
                    ax = fig.add_subplot(muller_gs[m, k])
                    muller_axes[(m, k)] = ax

                    indices = highlight_barcodes[bac][mouse]
                    cmap = np.array([pooled_cmap[index] for index in indices])

                    muller_freqs, cmap = make_muller_freqs(bac, mouse, indices, cmap)
                    muller_plot(ax, [0] + days, muller_freqs, cmap)

                    if 'format muller plot':
                        ax.tick_params(axis='x', direction='out')
                        ax.set_xlim(0, 16)
                        ax.set_xticks([0, 4, 10, 16])
                        ax.set_ylim(0, 1)
                        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                        ax.set_yticklabels((ax.get_yticks() * 100).astype(int))
                        if m == 1 and k == 0:
                            ax.set_xticklabels([0, 4, 10, 16])
                            ax.set_yticklabels(['0', '', '.5', '', '1'])
                        if k == 1 or m == 0:
                            ax.set_yticklabels([])
                            ax.set_xticklabels([])
                        if m == 0:
                            ax.set_xticks([])
                        if m == 0 and k == 0:
                            ax.text(0.05, 0.79, muller_ax_indices[b], fontsize=8, zorder=100,
                                    transform=ax.transAxes)


    if 'transition curves' != 'include':
        transition_group_ax = fig.add_subplot(outer[2, 2:])
        transition_group_ax.set_xlabel('Lineage counts at day 4', fontsize=8, labelpad=0)
        transition_group_ax.set_ylabel('Fraction of lineages', fontsize=8, labelpad=-5)
        ax_methods.turn_off_ax(transition_group_ax)

        transition_ax_indices = {0: '$\\bf{g}$', 1: '$\\bf{h}$', 2: '$\\bf{i}$', 3: '$\\bf{j}$'}
        for b, bac in enumerate(ordered_bac_lst):
            transition_gs = mpl.gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[2, b + 2])
            ax = fig.add_subplot(transition_gs[0])
            plot_transition_curve(ax, bac, np.ravel(muller_mice), input_rep, transition_day, expected_reads)

            if 'format transition curves':
                # ax.set_title(BAC_FORMAL_NAMES[bac], fontsize=8)
                ax.set_xlim(0, 40)
                ax.set_ylim(0, 0.12)
                # ax.set_yticks([0, 0.05, 0.1])
                ax.set_yticks([])
                ax.text(0.05, 0.88, transition_ax_indices[b], fontsize=8, transform=ax.transAxes)
                ax.text(0.96, 0.88, BAC_FORMAL_NAMES[bac], transform=ax.transAxes, fontsize=8,
                        horizontalalignment='right')

if __name__ == '__main__':
    ordered_bac_lst = ['BWH2', 'BtVPI', 'Bovatus', 'Bt7330']
    muller_mice = [[8, 10], [16, 19]]
    transition_day = 4
    expected_reads = [10, 15]
    input_rep = 0
    diet_cmaps = {'HF': 'Oranges', 'LF': 'Blues'}

    fig = plt.figure(figsize=(6.5, 3))
    outer = mpl.gridspec.GridSpec(nrows=3, ncols=len(ordered_bac_lst) + 2, width_ratios=[2, 0.03, 1, 1, 1, 1],
                                  height_ratios=[2, 0.1, 1.8], figure=fig)

    highlight_indices = find_individual_biggest_indices([1,2,3,5,6,7,8,9,10,11,12,13,15,16,18,19], 10) #ALL MICE
    shared_big_indices = find_shared_biggest_indices(10)
    make_fig1(fig, outer, highlight_indices)

    fig.savefig(f'{plot_dir}/fig1_muller_and_transition.pdf')

    with open(f'{data_dir}/shared_10biggest_indices.pkl', 'wb') as f:
        pickle.dump(shared_big_indices, f)

    with open(f'{data_dir}/individual_10biggest_indices.pkl', 'wb') as f:
        pickle.dump(highlight_indices, f)
