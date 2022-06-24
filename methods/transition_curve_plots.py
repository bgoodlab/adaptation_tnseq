import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl

from wu_data.wu_data import *

import numba
from numba import njit

def generate_offspring(bac, mouse_1, mouse_2, freq_range, barcodes = 'all'):
    """ Given bac (strain) and mouse_1 and mouse_2 tuples = (mouse/input id, day), or barcode arrays,
    Determines the frequencies in mouse_2 of those barcodes who were present in freq_range in mouse_1.
    Accepts freq_range as frequency or reads."""
    if type(mouse_1) == np.ndarray:
        reads_1 = mouse_1
        D_1 = np.sum(mouse_1)
    else:
        mID_1 = MX[mouse_1] #mouse IDs
        reads_1, D_1 = read_dict[bac][barcodes][mID_1], read_counts[bac][barcodes][mID_1]
    if type(mouse_2) == np.ndarray:
        reads_2 = mouse_2
        D_2 = np.sum(reads_2)
    else:
        mID_2 = MX[mouse_2]
        reads_2, D_2 = read_dict[bac][barcodes][mID_2], read_counts[bac][barcodes][mID_2]

    if freq_range[1] < 1:
        freq_range = np.array(freq_range)*D_1

    input_barcodes = np.where(np.logical_and(freq_range[0] <= reads_1, reads_1 < freq_range[1]))[0]

    offspring_reads = reads_2[input_barcodes]
    offspring_distribution = np.bincount(offspring_reads)
    return offspring_reads, offspring_distribution, D_2

def plot_offspring_dist(ax, bac, mouse_1, mouse_2, freq_range, barcodes = 'all', fspace=False, **kwargs):
    """ Given matplotlib ax object, populate with barcode offspring distribution in mouse_2,
    conditioned on each barcode's presence in mouse_1 within the frequency range."""
    offspring_freqs, offspring_dist, D_2 = generate_offspring(bac,
                                                              mouse_1, mouse_2,
                                                              freq_range,
                                                              barcodes =  barcodes)
    if fspace:
        ax.plot(np.arange(offspring_dist.shape[0])/D_2, offspring_dist/np.sum(offspring_dist), **kwargs)
    else:
        ax.plot(np.arange(offspring_dist.shape[0]), offspring_dist/np.sum(offspring_dist), **kwargs)

def plot_replicate_dist(ax, bac, mouse_2, freq_range, barcodes = 'all', fspace=False, **kwargs):
    sample_size = read_counts[bac][barcodes][MX[mouse_2]]
    pooled_reads, D_1 = read_dict[bac][barcodes][-1], read_counts[bac][barcodes][-1]
    if freq_range[1] < 1:
        freq_range = np.array(freq_range)*D_1

    downsample_reads = down_sample_from_pooled(bac, sample_size, barcodes=barcodes)
    downsample_barcodes = np.where(np.logical_and(freq_range[0] <= pooled_reads, pooled_reads < freq_range[1]))[0]
    offspring_reads = downsample_reads[downsample_barcodes]
    offspring_dist = np.bincount(offspring_reads)
    if fspace:
        ax.plot(np.arange(offspring_dist.shape[0])/sample_size, offspring_dist/np.sum(offspring_dist), **kwargs)
    else:
        ax.plot(np.arange(offspring_dist.shape[0]), offspring_dist/np.sum(offspring_dist), **kwargs)


@njit
def down_sample(to_size, input_barcode_array):
    """Down sample from nearest input replicate,
    or nearest pair of input replicate if to_size > any one replicate barcode count.
    Returns the down sample and the pooled reads less the replicates that were sampled from."""
    input_read_counts = np.sum(input_barcode_array[:23], axis = 1)
    large_inputs = np.where(input_read_counts > to_size)[0]
    if large_inputs.size > 0:
        input_index = large_inputs[np.argmin(input_read_counts[large_inputs])]
        unsampled = input_barcode_array[input_index]
    else:
        input_pair_counts = np.full((23, 23), 0, dtype=np.int64)
        for i in range(22):
            for j in range(i, 23):
                input_pair_counts[i, j] = np.sum(input_barcode_array[i] + input_barcode_array[j])
        large_pairs = np.where(input_pair_counts > to_size)

        large_pairs_sums = np.full(large_pairs[0].shape[0], 0, dtype=np.int64)
        for k in range(large_pairs[0].shape[0]):
            i, j = large_pairs[0][k], large_pairs[1][k]
            large_pairs_sums[k] = input_pair_counts[i,j]
        pair_index = np.argmin(large_pairs_sums)
        i, j = large_pairs[0][pair_index], large_pairs[1][pair_index]
        unsampled = input_barcode_array[i] + input_barcode_array[j]

    new_pooled = input_barcode_array[-1] - unsampled


    barcode_lst = []
    for i in range(unsampled.shape[0]):
        barcode_lst.extend([i]*unsampled[i])
    barcode_arr = np.array(barcode_lst)
    sampled_lst = np.random.choice(barcode_arr, size=to_size, replace=False)

    sampled_arr = np.full(unsampled.shape[0], 0, dtype=np.int64)
    for index in sampled_lst:
        sampled_arr[index] += 1

    return sampled_arr, new_pooled


def heat_map(input_freq_range, r1, r2, D1, D2, xylim=None):

    density, xedges, yedges, image = plt.hist2d(r1, r2, bins = [max(r1), max(r2)], density=False)
    plt.close()

    day4_density = density.sum(axis=1)
    day10_density = density.sum(axis = 0)
    X, Y = np.meshgrid(xedges, yedges, indexing='ij')

    fig, ax = plt.subplots(figsize = (8, 8))
    # ax.pcolormesh(X, Y, density, cmap='hot')
    ax.imshow(density.transpose(), cmap='hot', origin='lower')
    ax.set_title('Prob(R1, R2 | R0)')
    ax.set_xlabel('R1 (day 4)')
    ax.set_ylabel('R2 (day 10)')

    if xylim:
        xlim, ylim = np.array(xylim[0]), np.array(xylim[1])
    else:
        xlim = [0, 5*input_freq_range[1] * D1]
        ylim = [0, 5*input_freq_range[1] * D2]

    xlim[1] = min(xlim[1], xedges[-2])
    ylim[1] = min(ylim[1], yedges[-2])
    xlim[0] = max(xlim[0], xedges[0])
    ylim[0] = max(ylim[0], yedges[0])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return fig, density, day4_density, day10_density

def heat_map_with_marginals(input_freq_range, r1, r2, D1, D2, xylim=None):

    density, xedges, yedges, image = plt.hist2d(r1, r2, bins = [max(r1), max(r2)], density=False)
    plt.close()

    day4_density = density.sum(axis=1)
    day10_density = density.sum(axis=0)

    X, Y = np.meshgrid(xedges, yedges, indexing='ij')
    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(3,3)

    f1_ax = fig.add_subplot(gs[-1, 1:])
    f1_ax.plot(day4_density, marker='o')
    expected_day4 = input_freq_range*D1
    f1_ax.axvspan(expected_day4[0], expected_day4[1], color='grey', alpha=0.3)

    f2_ax = fig.add_subplot(gs[0:2, 0])
    f2_ax.plot(day10_density, np.arange(day10_density.shape[0]), color='tab:orange', marker='o')
    expected_day10 = input_freq_range*D2
    f2_ax.axhspan(expected_day10[0], expected_day10[1], color='grey', alpha=0.3)

    ax = fig.add_subplot(gs[0:2, 1:3])
    # ax.pcolormesh(X, Y, density, cmap='hot')
    ax.imshow(density.transpose(), cmap='hot', origin='lower')
    # ax.pcolormesh(X, Y, independent_density, cmap='hot')

    if xylim:
        xlim, ylim = np.array(xylim[0]), np.array(xylim[1])
    else:
        xlim = [0, 5*input_freq_range[1] * D1]
        ylim = [0, 5*input_freq_range[1] * D2]

    xlim[1] = min(xlim[1], xedges[-2])
    ylim[1] = min(ylim[1], yedges[-2])
    xlim[0] = max(xlim[0], xedges[0])
    ylim[0] = max(ylim[0], yedges[0])

    f1_ax.set_ylim(f1_ax.get_ylim()[::-1])
    f1_ax.set_xlim(xlim)

    f2_ax.set_xlim(f2_ax.get_xlim()[::-1])
    f2_ax.set_ylim(ylim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ax.set_title('P[f1, f2 | f0]')

    fig.subplots_adjust(wspace=0, hspace=0)
    return fig, density, day4_density, day10_density

def heat_map_with_independent(input_freq_range, r1, r2, D1, D2, xylim=None):

    density, xedges, yedges, image = plt.hist2d(r1, r2, bins = [max(r1), max(r2)], density=False)
    plt.close()

    day4_density = density.sum(axis=1)
    day10_density = density.sum(axis = 0)
    independent_density = np.zeros(shape = density.shape)

    for i, day4_d in enumerate(day4_density):
        for j, day10_d in enumerate(day10_density):
            independent_density[i, j] = day4_d * day10_d
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')

    fig = plt.figure()

    fig, axs = plt.subplots(1,2, figsize = (30, 15))
    # axs[1].pcolormesh(X, Y, density, cmap='hot')

    density = np.ma.masked_where(density == 0, density)

    cmap = mpl.cm.hot
    cmap.set_bad('gray')
    axs[1].imshow(density.transpose(), cmap=cmap, origin='lower')
    axs[1].set_title('Prob(R1, R2 | R0)')
    axs[1].set_xlabel('R1 (day 4)')
    axs[1].set_ylabel('R2 (day 10)')

    # axs[0].pcolormesh(X, Y, independent_density, cmap='hot')

    independent_density = np.ma.masked_where(independent_density == 0, independent_density)
    axs[0].imshow(independent_density.transpose(), cmap=cmap, origin='lower', )
    axs[0].set_title('Prob(R1 | R0)*Prob(R2 | R0)')
    axs[0].set_xlabel('R1 (day 4)')
    axs[0].set_ylabel('R2 (day 10)')

    if xylim:
        xlim, ylim = np.array(xylim[0]), np.array(xylim[1])
    else:
        xlim = [0, 5*input_freq_range[1] * D1]
        ylim = [0, 5*input_freq_range[1] * D2]

    xlim[1] = min(xlim[1], xedges[-2])
    ylim[1] = min(ylim[1], yedges[-2])
    xlim[0] = max(xlim[0], xedges[0])
    ylim[0] = max(ylim[0], yedges[0])

    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)

    return fig, density, day4_density, day10_density
