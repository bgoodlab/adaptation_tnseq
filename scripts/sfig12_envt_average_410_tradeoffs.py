root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared

def plot_lfc_heat_map(ax, lfcs1, lfcs2, binspace, norm=None, cmap=mpl.cm.Blues, colorbar=True):
    where_finite = (lfcs1.mask + lfcs2.mask == 0)

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

    avg_lfcs = np.mean([m1_lfcs, m2_lfcs], axis=0)
    avg_lfcs = np.ma.masked_array(avg_lfcs, mask=(m1_lfcs.mask+m2_lfcs.mask)>0)
    delta_lfcs = np.abs(m1_lfcs - m2_lfcs)
    delta_lfcs = np.ma.masked_array(delta_lfcs, mask=(m1_lfcs.mask+m2_lfcs.mask)>0)

    binspace = [np.linspace(-1.0, 1.0, 21),
                np.linspace(0,1,21)]
    # binspace = [ np.linspace(-np.log(1/epsilon)/dt1 -1/100, np.log(1/epsilon)/dt1 + 1/100, 41),
    #              np.linspace(-np.log(1/epsilon)/dt2 -1/100, np.log(1/epsilon)/dt2 + 1/100, 41)]
    plot_lfc_heat_map(ax, avg_lfcs[lineage_bool], delta_lfcs[lineage_bool],  binspace, norm=norm, colorbar=colorbar, cmap=cmap)

    if plot_muller:
        for diet, marker in [('HF', (3,0,-90)), ('LF', (3,0,0)), ('HLH', (3,0,-45)), ('LHL', (3,0,-45))]:
            muller_lineages = muller_lineage_dict[bac][diet]


            m1_star_lfcs = m1_lfcs[muller_lineages]
            m2_star_lfcs = m2_lfcs[muller_lineages]

            ax.scatter(np.mean([m1_star_lfcs,m2_star_lfcs],axis=0),
            np.abs(m1_star_lfcs-m2_star_lfcs), color=DIET_COLORS[diet], marker=marker, s=25, zorder=20)

def make_sfig(fig, outer):
    norm = mpl.colors.LogNorm(0.1, 5e3)

    m1, m2 = [6, 7, 8, 9, 10], [16, 18, 19]
    for b, bac in enumerate(BACTERIA):
        t0, t1 = 4, 10
        diet1, diet2 = mouse2diet[m1[0]], mouse2diet[m2[0]]
        xlabel = f'{diet1} {t0}-{t1}'
        ylabel = f'{diet2} {t0}-{t1}'

        m1_tuple = (m1, t0, t1)
        m2_tuple = (m2, t0, t1)

        ax = fig.add_subplot(outer[1 + 2 * (b // 2), 2 * (b % 2)])
        plot_lfc_hmap_correlations(ax, bac, m1_tuple, m2_tuple, colorbar=False,
                                   cmap=mpl.cm.bone_r, norm=norm, plot_muller=True)
        ax.text(0.05, 0.9, BAC_FORMAL_NAMES[bac], transform=ax.transAxes, fontsize=12)
        # ax.text(0.1, 0.9, BAC_FORMAL_NAMES[bac], transform=ax.transAxes)

        # if bac == 'BWH2': min = -1.0
        # elif bac == 'BtVPI': min = -0.6
        # elif bac == 'Bovatus': min = -0.6
        # elif bac == 'Bt7330': min = -1.0
        # ax.set_xlim(min, 0.8)
        # ax.set_ylim(0, 1.2)

        ax.axvline(0, color='black', linestyle='dashed', zorder=5)
        ax.axhline(0, color='black', linestyle='dashed', zorder=5)
        ax.set_xlim(-1.1, 0.9)
        ax.set_ylim(0, 1.2)
        ax.tick_params(axis='both', labelsize=10)

    if 'colorbar':
        colorbar_ax = fig.add_subplot(outer[0, :])
        _ = mpl.colorbar.ColorbarBase(colorbar_ax, cmap=mpl.cm.bone_r,
                                      ticklocation='top', norm=norm, orientation='horizontal')
        colorbar_ax.tick_params(axis='both', direction='out', labelsize=9, pad=-1)
        # colorbar_ax.set_title('# lineages', fontsize=10)
        # colorbar_ax.set_ylabel('# lineages', fontsize=10, labelpad=8, rotation=-90)
        colorbar_ax.set_xlim(1, 3000)
        colorbar_ax.set_xticks([1, 10, 100, 1000])
        colorbar_ax.set_xticklabels(['1', '10', '100', '1000'])
        colorbar_ax.text(0.35, 2, '# lineages', transform=colorbar_ax.transAxes, fontsize=12)


if __name__ == '__main__':
    with open(f'{data_dir}/shared_10biggest_indices.pkl', 'rb') as f:
        muller_lineage_dict = pickle.load(f)

    fig = plt.figure(figsize=(6, 6))
    outer = mpl.gridspec.GridSpec(nrows=4, ncols=3,
                                  height_ratios=(0.07, 1, 0.03, 1),
                                  width_ratios=(1, 0.03, 1))
    outer_ax = fig.add_subplot(outer[:, :])
    outer_ax.set_ylabel(r'$|\chi_{HF/HS}-\chi_{LF/HPP}|$ (days$^{-1}$), days 4-10', fontsize=12, labelpad=15)
    outer_ax.set_xlabel(r'$\frac{1}{2}(\chi_{HF/HS}+\chi_{LF/HPP})$ (days$^{-1}$), days 4-10', fontsize=12, labelpad=15)
    ax_methods.turn_off_ax(outer_ax)

    make_sfig(fig, outer)

    fig.savefig(f'{plot_dir}/sfig12_envt_average_410_tradeoffs.pdf')
