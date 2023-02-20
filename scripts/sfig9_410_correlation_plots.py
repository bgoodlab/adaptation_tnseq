root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared


def plot_lfc_heat_map(ax, lfcs1, lfcs2, norm=None, binspace=None, cmap= mpl.cm.get_cmap('Blues'), colorbar=True):
    where_finite = (lfcs1.mask + lfcs2.mask == 0)

    if not np.any(binspace):
        binspace = np.linspace(-1., 1, 41)

    density, binx, biny = np.histogram2d(lfcs1[where_finite], lfcs2[where_finite], bins = binspace)

    if not norm:
        norm = mpl.colors.LogNorm( 1, np.max(density) )

    ## plot empirical
    ax.hist2d(lfcs1[where_finite], lfcs2[where_finite], bins = binspace, cmap=cmap, norm=norm)


    if colorbar:
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

def plot_lfc_hmap_correlations(ax, bac, m1_tuple, m2_tuple,
                               norm=mpl.colors.LogNorm(0.1, 3e3),
                               colorbar=False, cmap=mpl.cm.Reds,
                               minfreq=10**-6.5, plot_muller=False):
    m1, m1t1, m1t2 = m1_tuple
    m2, m2t1, m2t2 = m2_tuple

    notWu = shared.bac_nonwu_indices[bac]
    read_array = shared.bac_read_arrays[bac][:, notWu]
    day0_freqs = read_array[0] / read_array[0].sum()
    lineage_bool = day0_freqs > minfreq

    m1_reads1, m1_reads2 = shared.get_read_arrays(bac, m1, m1t1, m1t2, split_day0=2)
    m2_reads1, m2_reads2 = shared.get_read_arrays(bac, m2, m2t1, m2t2, split_day0=1)
    m1_D1, m1_D2 = m1_reads1.sum(), m1_reads2.sum()
    m2_D1, m2_D2 = m2_reads1.sum(), m2_reads2.sum()
    m1_freqs1, m1_freqs2 = m1_reads1 / m1_D1, m1_reads2 / m1_D2
    m2_freqs1, m2_freqs2 = m2_reads1 / m2_D1, m2_reads2 / m2_D2

    m1_f1, m1_f2 = shared.maxmin_freqs(m1_freqs1, m1_D1, m1_freqs2, m1_D2)
    m2_f1, m2_f2 = shared.maxmin_freqs(m2_freqs1, m2_D1, m2_freqs2, m2_D2)

    m1_lfcs = shared.calc_lfc_array( m1_f1, m1_f2, m1t2-m1t1)
    m2_lfcs = shared.calc_lfc_array( m2_f1, m2_f2, m2t2-m2t1)

    binspace = np.linspace(np.min([m1_lfcs,m2_lfcs]), np.max([m1_lfcs, m2_lfcs]), 41)
    # binspace = [ np.linspace(-np.log(1/epsilon)/dt1 -1/100, np.log(1/epsilon)/dt1 + 1/100, 41),
    #              np.linspace(-np.log(1/epsilon)/dt2 -1/100, np.log(1/epsilon)/dt2 + 1/100, 41)]

    plot_lfc_heat_map(ax, m1_lfcs[lineage_bool], m2_lfcs[lineage_bool], norm=norm, binspace=binspace, colorbar=colorbar, cmap=cmap)

    if plot_muller:
        for diet, marker in [('HF', (3,0,-90)), ('LF', (3,0,0)), ('HLH', (3,0,-45)), ('LHL', (3,0,-45))]:
            muller_lineages = muller_lineage_dict[bac][diet]


            m1_star_lfcs = m1_lfcs[muller_lineages]
            m2_star_lfcs = m2_lfcs[muller_lineages]

            ax.scatter(m1_star_lfcs, m2_star_lfcs, color=DIET_COLORS[diet], marker=marker, s=10, zorder=20)

def make_sfig(fig, outer):
    colorbar_ax = fig.add_subplot(outer[0, :])
    norm = mpl.colors.LogNorm(0.1, 3e3)
    _ = mpl.colorbar.ColorbarBase(colorbar_ax, cmap=mpl.cm.bone_r,
                                  ticklocation='top',
                                  norm=norm,
                                  orientation='horizontal')
    colorbar_ax.tick_params(axis='both', direction='out', labelsize=8, pad=-1)
    colorbar_ax.set_title('Number of lineages', fontsize=10)
    colorbar_ax.set_xlim(1, 3 * 10 ** 3)
    colorbar_ax.set_xticks([1, 10, 100, 1000])
    colorbar_ax.set_xticklabels(['1', '10', '100', '1000'])

    for b, bac in enumerate(BACTERIA):
        bac_outer_ax = fig.add_subplot(outer[1 + 2 * b, :])
        bac_outer_ax.text(1.01, 0.5, BAC_FORMAL_NAMES[bac], fontsize=12,
                          transform=bac_outer_ax.transAxes, horizontalalignment='left', weight='bold')
        ax_methods.turn_off_ax(bac_outer_ax)

        for c, (m1, m2) in enumerate([([6, 7, 8], [9, 10]), ([6, 7, 8, 9, 10], [16, 18, 19]), ([16, 18], [19])]):
            t0, t1 = 4, 10
            diet1, diet2 = mouse2diet[m1[0]], mouse2diet[m2[0]]
            xlabel = f'{diet1} {t0}-{t1}'
            ylabel = f'{diet2} {t0}-{t1}'

            m1_tuple = (m1, t0, t1)
            m2_tuple = (m2, t0, t1)

            ax = fig.add_subplot(outer[1 + 2 * b, 2 * c])
            if c == 1:
                muller = True
            else:
                muller = True
            plot_lfc_hmap_correlations(ax, bac, m1_tuple, m2_tuple, colorbar=False, norm=norm, cmap=mpl.cm.bone_r,
                                       plot_muller=muller)
            # ax.text(0.1, 0.9, BAC_FORMAL_NAMES[bac], transform=ax.transAxes)
            ax.set_xlabel(f'{DIET_NAMES[diet1]} ({len(m1)} mice)')
            ax.set_ylabel(f'{DIET_NAMES[diet2]} ({len(m2)} mice)', labelpad=-2)

            if bac == 'BWH2':
                min, max = -1.0, 1.4
            elif bac == 'BtVPI':
                min, max = -1.0, 1.2
            elif bac == 'Bovatus':
                min, max = -1.0, 1.4
            elif bac == 'Bt7330':
                min, max = -1.2, 1.2
            ax.set_xlim(min, max)
            ax.set_ylim(min, max)

            ax.plot(ax.get_xlim(), ax.get_xlim(), linestyle='dashed', color='black')
            ax.axvline(0, color='black', linestyle='dashed', zorder=5)
            ax.axhline(0, color='black', linestyle='dashed', zorder=5)

if __name__ == '__main__':
    with open(f'{data_dir}/shared_10biggest_indices.pkl', 'rb') as f:
        muller_lineage_dict = pickle.load(f)

    fig = plt.figure(figsize=(5, 7))

    outer = mpl.gridspec.GridSpec(nrows=8, ncols=5,
                                  height_ratios=(0.1, 1, 0.05, 1, 0.05, 1, 0.05, 1),
                                  width_ratios=(1, 0.1, 1, 0.1, 1))
    outer_ax = fig.add_subplot(outer[1:, :])
    outer_ax.set_xlabel('Relative fitness, days 4-10 (days$^{-1}$), cohort 1', fontsize=12, labelpad=17)
    outer_ax.set_ylabel('Relative fitness, days 4-10 (days$^{-1}$), cohort 2', fontsize=12, labelpad=15)
    ax_methods.turn_off_ax(outer_ax)

    make_sfig(fig, outer)


    fig.savefig(f'{plot_dir}/sfig9_410_correlations.pdf')