root_name = 'tnseq_adaptation'
import os, sys  ## make sure path to root of project directory
cwd = os.getcwd(); directory_root = cwd[:cwd.index(root_name)+len(root_name)]
sys.path.insert(0, directory_root)
from methods.config import * #includes matplotlib, matplotlib.pyplot, numpy, pickle
import methods.ax_methods as ax_methods
import methods.shared as shared
import methods.sim_barcodes as sim
import scipy.stats

def sim_population_frequencies(init_freqs, fitnesses, gens=4, N=1e12, D_lst=[3*10**5]*5, num_populations=5):
    population_frequencies = np.zeros((num_populations, len(init_freqs)))
    population_reads = np.zeros((num_populations, len(init_freqs)))
    for i in range(num_populations):
        pop = sim.Population(init_freqs, np.copy(fitnesses), N)
        pop.simulate(gens)

        pop.N = D_lst[i]
        pop.xs[:] = 0
        pop.simulate(1) # first poisson samplng ~ DNA extraction

        reads = pop.sample(pop.N)
        population_reads[i] = reads
        population_frequencies[i] = reads/reads.sum()
    return population_frequencies, population_reads

def generate_empirical_ordered_fitnesses(bac, mice, t=4):
    nonWu = shared.bac_nonwu_indices[bac]
    read_arrays = shared.bac_read_arrays[bac][:, nonWu]
    row_ids = shared.bac_row_ids[bac]
    day0_freqs = read_arrays[0] / read_arrays[0].sum()

    mice_reads = read_arrays[ [row_ids[(mouse, t)] for mouse in mice ] ]
    avg_freqs = mice_reads.sum(axis=0) / mice_reads.sum()
    where_large = day0_freqs > 1e-7

    f0, f1 = shared.maxmin_freqs(day0_freqs[where_large], read_arrays[0].sum(), avg_freqs[where_large], mice_reads.sum())

    empirical_fitnesses = shared.calc_lfc_array(f0, f1, t)
    sorted_indices = np.argsort(empirical_fitnesses)[::-1]
    sorted_fitnesses = empirical_fitnesses[sorted_indices]
    sorted_indices = np.arange(day0_freqs.shape[-1])[where_large][sorted_indices]

    return sorted_fitnesses, sorted_indices

def make_sfig(fig, outer, cutoff=10**-6.5):
    rank_ax = fig.add_subplot(outer[:, :3])
    rank_ax.set_xlabel('Fitness rank in discovery', fontsize=12, labelpad=10)
    rank_ax.set_ylabel('Relative fitness in validation', fontsize=12, labelpad=10)
    ax_methods.turn_off_ax(rank_ax)

    for b, bac in enumerate(['BWH2', 'Bovatus', 'BtVPI', 'Bt7330']):
        row_ax = fig.add_subplot(outer[2 * b, :])
        ax_methods.turn_off_ax(row_ax)
        # row_ax.set_ylabel(BAC_FORMAL_NAMES[bac], labelpad=15, fontsize=10)
        row_ax.text(1.01, 0.5, BAC_FORMAL_NAMES[bac],
                    transform=row_ax.transAxes,
                    fontsize=12, weight='bold')

    for i, col_title in enumerate(['Neutral', 'Empirical', 'Empirical']):
        col_ax = fig.add_subplot(outer[:, 2 * i])
        col_ax.set_title(col_title)

        if i == 2:
            col_ax.set_ylabel('Fraction of lineages', fontsize=12)
            col_ax.set_xlabel('Relative fitness', fontsize=12, labelpad=10)
        ax_methods.turn_off_ax(col_ax)

    discovery, validation = [6, 7, 8, 9, 10], [1, 2, 3, 5]
    for b, (bac, R) in enumerate(zip(BACTERIA, [20000, 10000, 10000, 5000])):
        print(f'> Simulating {bac}...')

        row_ids = shared.bac_row_ids[bac]
        notWu = shared.bac_nonwu_indices[bac]
        read_array = shared.bac_read_arrays[bac][:, notWu]
        input_array = shared.bac_input_arrays[bac][:, notWu]

        emp_fitnesses, emp_sorted_indices = generate_empirical_ordered_fitnesses(bac, [1, 2, 3, 5, 6, 7, 8, 9, 10], 4)
        emp_fitnesses[emp_fitnesses < -1] = -1
        emp_fitnesses[emp_fitnesses > 1.5] = 1.5

        D = read_array[[row_ids[(mouse, 4)] for mouse in discovery + validation]].sum(axis=1)
        num_mice = len(discovery + validation)

        d0_reads = read_array[0][emp_sorted_indices]
        init_freqs = d0_reads / d0_reads.sum()
        neut_fitnesses = np.zeros(init_freqs.shape[-1])

        input_depths = input_array.sum(axis=1)
        sim_input_array = np.zeros((len(input_depths), d0_reads.shape[-1]))
        for i, depth in enumerate(input_depths):
            sim_input_reads, _ = sim.double_poisson(init_freqs, depth, depth)
            sim_input_array[i] = sim_input_reads

        discovery_reads0 = sim_input_array[0::2].sum(axis=0)
        discovery_freqs0 = discovery_reads0 / discovery_reads0.sum()
        validation_reads0 = sim_input_array[1::2].sum(axis=0)
        validation_freqs0 = validation_reads0 / validation_reads0.sum()

        for n, (sim_fitnesses, title) in enumerate(
                [(neut_fitnesses, f'{bac} neutral'), (emp_fitnesses / 10, f'{bac} empirical')]):
            # ax = fig.add_subplot( outer[4*(b//2)+2*n, 2*(b%2)])
            ax = fig.add_subplot(outer[2 * b, 2 * n])

            sim_init_freqs = np.copy(init_freqs)
            pop_freqs, pop_reads = sim_population_frequencies(sim_init_freqs, sim_fitnesses, gens=40, N=10 ** 8,
                                                              D_lst=D, num_populations=num_mice)

            num_discovery = len(discovery)
            discovery_reads1 = pop_reads[:num_discovery].sum(axis=0)
            discovery_freqs1 = discovery_reads1 / discovery_reads1.sum()
            validation_reads1 = pop_reads[num_discovery:].sum(axis=0)
            validation_freqs1 = validation_reads1 / validation_reads1.sum()

            filter_freqs = np.max([discovery_freqs1, discovery_freqs0], axis=0)
            # filter_bool = shared.filter_pairs((discovery_reads0, discovery_reads1), (validation_reads0, validation_reads1), threshold=filter_freqs, min_reads=min_reads, union=False)
            filter_bool = shared.filter_lineages(validation_reads0, validation_reads1, min_reads=5,
                                                 threshold=filter_freqs)
            shared_bool = (sim_init_freqs > cutoff) * filter_bool

            d_f0, d_f1 = shared.maxmin_freqs(discovery_freqs0, discovery_reads0.sum(),
                                             discovery_freqs1, discovery_reads1.sum())

            v_f0, v_f1 = shared.maxmin_freqs(validation_freqs0, validation_reads0.sum(),
                                             validation_freqs1, validation_reads1.sum())

            d_lfcs = shared.calc_lfc_array(d_f0, d_f1, 4, valid=shared_bool)
            v_lfcs = shared.calc_lfc_array(v_f0, v_f1, 4, valid=shared_bool)

            sorted_indices = np.argsort(d_lfcs)[::-1]
            sorted_d_lfcs = d_lfcs[sorted_indices]
            sorted_v_lfcs = v_lfcs[sorted_indices]

            cg_validate_lfcs = shared.calc_coarse_grained_lfc_array(validation_freqs0[shared_bool][sorted_indices],
                                                             validation_reads0.sum(),
                                                             validation_freqs1[shared_bool][sorted_indices],
                                                             validation_reads1.sum(),
                                                             4, coarse_grain=100)

            theory_int_xbar = np.log((sim_init_freqs * np.exp(sim_fitnesses * 40)).sum())
            theory_fitnesses = sorted(sim_fitnesses[shared_bool] * 10 - theory_int_xbar / 4, reverse=True)

            ax.scatter(np.arange(sorted_v_lfcs.shape[-1])[:R], sorted_v_lfcs[:R], rasterized=True, s=1,
                       color=KELLY_COLORS[3], alpha=0.1)
            ax.plot(theory_fitnesses[:R], color=KELLY_COLORS[0], label='True')
            ax.plot(sorted_d_lfcs[:R], color=KELLY_COLORS[4], label='Discovery')
            l = min([R, len(cg_validate_lfcs)])

            ax.plot(np.arange(50, l), cg_validate_lfcs[50:l], color=KELLY_COLORS[9], label='Validation')

            ax.axhline(0, color='black', linestyle='dashed')
            if bac == 'BWH2' and n == 0:
                ax.legend(frameon=False)
            if bac == 'Bt7330':
                ax.set_ylim(-0.5, 1.5)
            else:
                ax.set_ylim(-0.5, 1.2)
            ax.tick_params(axis='both', labelsize=8)

        fitnesses = cg_validate_lfcs[50:-50][::-1]

        dx = 0.05
        if bac == 'Bt7330':
            x = np.arange(-1.0, 1.5, dx)
        else:
            x = np.arange(-1.0, 1.2, dx)
        x_avg = (x[1:] + x[:-1]) / 2

        kernel = scipy.stats.gaussian_kde(fitnesses, bw_method=0.2)
        emp_kernel = scipy.stats.gaussian_kde(emp_fitnesses[shared_bool], bw_method=0.2)
        ax = fig.add_subplot(outer[2 * b, -1])

        density, _ = np.histogram(emp_fitnesses[shared_bool], bins=x, density=True)
        ax.plot(x_avg, emp_kernel(x_avg), color=KELLY_COLORS[0])
        ax.plot(x_avg, kernel(x_avg), color=KELLY_COLORS[9])
        ax.axvline(0, color='black', linestyle='dashed')
        ax.tick_params(axis='both', labelsize=8)
        ax.set_yticks([])
        if bac == 'BWH2':
            ax.legend(['True', 'Inferred'], frameon=False)

    return

if __name__ == '__main__':
    fig = plt.figure(figsize=(6, 7))
    outer = mpl.gridspec.GridSpec(nrows=7, ncols=5, height_ratios=[1, 0.03, 1, 0.03, 1, 0.03, 1],
                                  width_ratios=[0.75, 0.05, 1, 0.15, 1], figure=fig)

    make_sfig(fig, outer)

    fig.savefig(f'{plot_dir}/sfig2_simulated_rank_order_curves.pdf')
