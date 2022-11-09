# Methods from Grosmark's paper
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d


def reactivation_strength(ICA_matrix, activity_matrix):
    """
    Compute reactivation strength matrix

    ICA_matrix: [cell_id, assembly_id] weights
    activity_matrix: [cell_id, frames] dF/F
    """
    cell_num, assem_num = ICA_matrix.shape
    R = np.zeros((assem_num, activity_matrix.shape[1]))  ## reactivation strength
    for i in range(ICA_matrix.shape[1]):  # there is probably a better way to code this but fuck it
        weighted = (activity_matrix.transpose() * ICA_matrix[:, i]).transpose()
        R[i] = np.sum(weighted, axis=0) ** 2
    return R


def PCC_score(x, ICA_matrix, activity_matrix):
    """
    Compute PCC score for one cell

    Input:
    x: id of cell
    ICA_matrix: [cell_id, assembly_id] weights
    activity_matrix: [cell_id, frames] dF/F
    """
    assert type(x) == int
    R1 = reactivation_strength(ICA_matrix, activity_matrix)
    R2 = reactivation_strength(ICA_matrix[np.arange(ICA_matrix.shape[0]) != x, :],
                               activity_matrix[np.arange(ICA_matrix.shape[0]) != x, :])
    return np.mean(R1 - R2)


def assembly_activation_strength(ICA_matrix, pre_hse_matrix, post_hse_matrix, hse_win, plot_num=None):
    """
    Compute activation strength for each assembly and make plots

    Inputs:
    ICA_matrix: [cell_id, assembly_id] weights
    pre_hse_matrix: [cell_id, frames] dF/F of all HSEs in pre
    post_hse_matrix: [cell_id, frames] dF/F of all HSEs in post
    hse_win: window length of HSE
    plot_num: if int, make the plot of plot_num(th) assembly; if 'plot_average', make the average plot of all assemblies
    """
    # hse_matrix: 1s length
    pre_assembly_strength = zscore(reactivation_strength(ICA_matrix, pre_hse_matrix), axis=1)

    post_assembly_strength = zscore(reactivation_strength(ICA_matrix, post_hse_matrix), axis=1)

    peri_win = pre_hse_matrix.shape[1]

    if type(plot_num) == int:
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0], )
        pre_strength_smooth = gaussian_filter1d(pre_assembly_strength[plot_num, :], sigma=5)
        post_strength_smooth = gaussian_filter1d(post_assembly_strength[plot_num, :], sigma=5)
        ax1.plot(pre_strength_smooth, label='Pre')
        ax1.plot(post_strength_smooth, label='Post')
        ax1.set_title('Rum ensemble ' + str(plot_num), fontsize=20)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylabel('Run ensemble reactivation', fontsize=15)
        ax1.set_xlabel('Peri-HSE time(s)')
        ax1.tick_params(axis='y', which='major', labelsize=12)
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1], )
        ax2.scatter(np.max(pre_strength_smooth[peri_win // 2 - hse_win // 2:peri_win // 2 + hse_win // 2]),
                    np.max(post_strength_smooth[peri_win // 2 - hse_win // 2:peri_win // 2 + hse_win // 2]),
                    marker='+')
        ax2.plot(np.arange(0, 2, 0.1), np.arange(0, 2, 0.1), linestyle='--')
        ax2.set_ylabel('Post within-HSE \n run ensemble reactivation', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_xlabel('Pre within-HSE \n run ensemble reactivation', fontsize=15)

        plt.show()
    elif plot_num == 'plot average':
        fig = plt.figure(figsize=(12, 4))
        gs = fig.add_gridspec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0], )
        pre_population = gaussian_filter1d(np.mean(pre_assembly_strength, axis=0), sigma=5)
        post_population = gaussian_filter1d(np.mean(post_assembly_strength, axis=0), sigma=5)
        ax1.plot(np.arange(peri_win), pre_population, label='Pre')
        ax1.plot(np.arange(peri_win), post_population, label='Post')
        ax1.set_title('Rum ensemble ' + str(plot_num), fontsize=20)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.legend()
        ax1.set_ylabel('Run ensemble reactivation', fontsize=15)
        ax1.tick_params(axis='y', which='major', labelsize=12)
        ax1.legend()

        ax2 = fig.add_subplot(gs[0, 1], )
        ax2.scatter(np.max(pre_population[peri_win // 2 - hse_win // 2:peri_win // 2 + hse_win // 2]),
                    np.max(post_population[peri_win // 2 - hse_win // 2:peri_win // 2 + hse_win // 2]), marker='+')
        ax2.plot(np.arange(0, 2, 0.1), np.arange(0, 2, 0.1), linestyle='--')
        ax2.set_ylabel('Post within-HSE \n run ensemble reactivation', fontsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.set_xlabel('Pre within-HSE \n run ensemble reactivation', fontsize=15)
        plt.show()
