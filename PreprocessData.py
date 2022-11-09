# process data
import copy
import numpy as np
from scipy.signal import gaussian


def get_sparse_spike(dffs, nbins=15, std=1):
    """
    Get spike train from dffs.
    Detect one spike as long as its dff value exceed 2 std above mean

    Inputs:
    dffs: numpy.ndarry cell_num * frames
    nbins: gaussian smooth bins
    std: gaussian smooth std

    Return:
    spike: spike train
    """
    dffs = copy.deepcopy(dffs)
    #     dffs = gaussian_filter1d(dffs, sigma=std)
    spike = np.where(dffs > (np.mean(dffs, axis=1) + 2 * np.std(dffs, axis=1)).reshape(dffs.shape[0], 1), 1, 0)
    return spike


def get_rest(velocity, activity, rest_speed=2):
    """
    Get rest periods v<2cm/s

    Input:
    velocity: speed over [frames]
    activity: dF/F over [cell_id, frames]
    """
    return activity[:, np.where(velocity < rest_speed)[0]]


def gaussian_smooth(dffs, nbins=15, std=1):
    """
    Smooth inputs with a gaussian kernel.

    Inputs:
    dffs: to be smooted, dF/F over [cell_id, frame]
    nbins: length of gaussian kernel
    std: std of gaussian kernel

    Return:
    smooth_dffs: dffs after smoothing
    """
    cell_num = dffs.shape[0]
    bins = int((nbins - 1) / 2)
    gaussian_win = gaussian(nbins, std=std)
    gaussian_win /= np.sum(gaussian_win)
    smooth_dffs = np.zeros(dffs.shape)
    dffs_com = np.concatenate([np.zeros((cell_num, bins)), dffs, np.zeros((cell_num, bins + 1))], axis=1)
    for t in range(dffs.shape[1]):
        smooth_dffs[:, t] = (dffs_com[:, t:t + 2 * bins + 1] @ gaussian_win.reshape(nbins, 1)).reshape(cell_num)
    return smooth_dffs


def pos_dff(dffs, locations, speed, pos_bins=2):
    """
    Get average activity at each position bin (only consider running periods)

    Inputs:
    dffs: [cell_id, frame]
    locations: location over [frame]
    speed: speed over [frame]
    pos_bins: bin size of position
    """
    cell_num = dffs.shape[0]
    loc_max = np.max(locations)
    loc_min = np.min(locations)
    bins_n = int((loc_max - loc_min) / pos_bins) + 1
    dff_pos = np.zeros((cell_num, bins_n))

    bins_num = np.zeros(bins_n)
    for k in range(dffs.shape[1]):
        if speed[k] >= 2:
            dff_pos[:, int((locations[k] - loc_min) / pos_bins)] += dffs[:, k]
            bins_num[int((locations[k] - loc_min) / pos_bins)] += 1
            bins_num[bins_num == 0] = 1
    for cell in range(cell_num):
        dff_pos[cell, :] = dff_pos[cell, :] / bins_num
    return dff_pos


def detect_place_cell(cell_num, contexts, trial_num, pointer, positions, speed, dff_traces):
    """
    Detect place cells in all detected cells.

    :param cell_num: number of cells
    :param contexts: corresponding context to each trial
    :param trial_num: number of trials
    :param pointer: the beginning and ending point of each trial
    :param positions: position the mouse is at each time point
    :param speed: speed of the mouse at each time point
    :param dff_traces: dF/F traces over time
    :return:
    place_cells: index of place cells in each context
    """
    pos_bins = 1
    bins_n = int((2000 - 400) / pos_bins)
    dff_pos = np.zeros((4, cell_num, bins_n))
    shuffle_time = 1000
    place_cells = [[] for i in range(4)]
    for con in range(4):
        trial_list = contexts[con]
        for cell_id in range(cell_num):
            dFFs_pl = np.zeros((trial_num, bins_n))
            for trial in trial_list:
                p = pointer[trial]
                bins_num = np.zeros(bins_n)
                for k in range(positions[trial].shape[0]):
                    if speed[trial][k] >= 2:
                        if int((positions[trial][k] - 400) / pos_bins) == bins_n:
                            dFFs_pl[trial, -1] += dff_traces[cell_id, p + k]
                            bins_num[-1] += 1
                        elif int((positions[trial][k] - 400) / pos_bins) == 0:
                            dFFs_pl[trial, 0] += dff_traces[cell_id, p + k]
                            bins_num[0] += 1
                        else:
                            dFFs_pl[trial, int((positions[trial][k] - 400) / pos_bins)] += dff_traces[cell_id, p + k]
                            bins_num[int((positions[trial][k] - 400) / pos_bins)] += 1
                    bins_num[bins_num == 0] = 1
                    dff_pos[con, cell_id, :] = np.average(dFFs_pl[trial, :] / bins_num, axis=0)

            # shuffled PSTH
            dFFs_pos_shuffle = np.zeros((shuffle_time, bins_n))
            for i in range(1000):
                shuffled_dff = dff_traces[cell_id, :].view()
                dFFs_pl_shuffle = np.zeros((trial_num, bins_n))
                for trial in trial_list:
                    p = pointer[trial]
                    bins_num = np.zeros(bins_n)
                    np.random.shuffle(shuffled_dff[p:p + pointer[trial + 1]])
                    for k in range(positions[trial].shape[0]):
                        if speed[trial][k] >= 2:
                            if int((positions[trial][k] - 400) / pos_bins) == bins_n:
                                dFFs_pl_shuffle[trial, -1] += shuffled_dff[p + k]
                                bins_num[-1] += 1
                            elif int((positions[trial][k] - 400) / pos_bins) == 0:
                                dFFs_pl_shuffle[trial, 0] += shuffled_dff[p + k]
                                bins_num[0] += 1
                            else:
                                dFFs_pl_shuffle[trial, int((positions[trial][k] - 400) / pos_bins)] += shuffled_dff[
                                    p + k]
                                bins_num[int((positions[trial][k] - 400) / pos_bins)] += 1
                        bins_num[bins_num == 0] = 1
                        dFFs_pos_shuffle[i, :] = np.average(dFFs_pl_shuffle / bins_num, axis=0)
            check = 0
            for pos in range(bins_n):
                rank = np.where(dFFs_pos_shuffle[:, pos] > dff_pos[con, cell_id, pos])[0].shape[0] / shuffle_time
                if rank < 0.05:
                    check += 1
                elif check > 0:
                    check = 0
                if check >= 5:
                    place_cells[con].append(cell_id)
                    break
    return place_cells