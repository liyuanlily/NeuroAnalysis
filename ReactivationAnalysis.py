import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import zscore, bootstrap, sem
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, pearsonr
import copy
from sklearn.decomposition import PCA
from AncillaryFunction import*


def detect_HSE(activity, spike, pcs, least_pc_num=5, min_win=150, max_win=1000,
               bin_size=1000 / 30, upper_bound=3, lower_bound=1, min_peak_dis=20, mov_win=1000):
    """
    Detect high synchrony events from place cells' activities.

    Input:
    activity: dF/F traces [cell, frame]
    spike: spike train [cell, spike]
    pcs: id of all place cells
    least_pc_num: the least activated pc numbers if detected as a HSE
    min_win: minimum window size of a HSE (ms)
    max_win: maximum window size of a HSE (ms)
    bin_size: time length of a time bin in activity and spike (ms)
    upper_bound, lower_bound: detect HSE when population activity exceed mean+upper_bound*std, boundary is mean+lower_bound*std
    min_peak_dis: detect two peaks as distinct when their peak distance exceeds min_peak_dis
    mov_win: the length of moving window (bins)

    Return:
    HSE_events: beginning and end time of every HSE
    HSE_peaks: peaks of each HSE
    mean + lower_bound * std: lower bound of detecting HSE
    mean + upper_bound * std: upper bound of detecting HSE
    max_win_bins: maximum bin number of a HSE
    """
    population = zscore(np.mean(activity, axis=0))
    mean = np.zeros(population.shape)
    std = np.zeros(population.shape)
    for i in range(population.shape[0]):
        mean[i] = np.mean(population[i - mov_win // 2:i + mov_win // 2])
        std[i] = np.std(population[i - mov_win // 2:i + mov_win // 2])

    #     plt.plot(population)
    #     plt.plot(mean+upper_bound*std)
    #     plt.plot(mean+lower_bound*std)
    #     plt.plot(mean)
    #     plt.show()

    start = None
    peak_all = []
    t = 0
    while t < activity.shape[1]:
        start = first_index(population[t:], mean[t:] + lower_bound * std[t:], '>=') + t
        exceed1 = first_index(population[start:], mean[start:] + upper_bound * std[start:], '>=') + start
        exceed2 = first_index(population[exceed1 + 1:], mean[exceed1 + 1:] + upper_bound * std[exceed1 + 1:],
                              '<=') + exceed1 + 1
        end = first_index(population[exceed2:], mean[exceed2:] + lower_bound * std[exceed2:], '<=') + exceed2
        if exceed1 == exceed2:
            if population[exceed1] >= mean[exceed1] + upper_bound * std[exceed1]:
                peak_all.append(int(exceed1))
        else:
            sign = np.sign(population[exceed1 + 1: exceed2] - population[exceed1: exceed2 - 1])
            if len(sign) == 1:
                if population[exceed1 + 1] >= mean[exceed1 + 1] + upper_bound * std[exceed1 + 1]:
                    peak_all.append(int(exceed1) + 1)
            else:
                peaks = np.where((sign[1:] - sign[:len(sign) - 1]) < 0)[0] + 1 + exceed1 + 1
                for peak in peaks:
                    if population[peak] >= mean[peak] + upper_bound * std[peak]:
                        peak_all.append(peak)
        t = end + 1

    peak_all = np.sort(np.array(list(peak_all), dtype=np.int64))
    peak_all = peak_all[np.where((peak_all[1:] - peak_all[:len(peak_all) - 1]) > min_peak_dis)[0] + 1]

    HSE_events = []
    HSE_peaks = []
    max_win_bins = int(max_win / bin_size) + 1
    for peak in peak_all:
        assert population[peak] >= mean[peak] + upper_bound * std[peak]
        start = first_index(population[: peak], mean[:peak] + lower_bound * std[:peak], '<=', order='backward')
        end = first_index(population[peak:], mean[peak:] + lower_bound * std[peak:], '<=') + peak
        if end - start >= int(min_win / bin_size):
            if np.where(np.mean(spike[pcs[:], start:end], axis=1) > 0)[0].shape[0] >= least_pc_num:
                if end - start > max_win_bins:
                    HSE_events.append([peak - max_win_bins // 2, peak + max_win_bins // 2])
                    HSE_peaks.append(peak)
                else:
                    HSE_events.append([start, end])
                    HSE_peaks.append(peak)

    return HSE_events, HSE_peaks, mean + lower_bound * std, mean + upper_bound * std, max_win_bins


# heatmap
def plot_heatmap(aver_all, normalized, aver_center=None, lower=None, upper=None,
                 xlabel='Frames', ylabel1='Mean dF/F (z-score)', ylabel2='Cell ID', title='Map 1'):
    win_bins = aver_all.shape[0]
    if aver_center is not None:
        win = len(aver_center)
        left = win_bins // 2 - int(win / 2)
        right = win_bins // 2 - (win - int(win / 2))
    fig = plt.figure(figsize=(4, 8))
    gs = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs[0, 0], )
    ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)
    ax1.plot(np.arange(win_bins), aver_all)
    if aver_center is not None:
        ax1.plot(np.arange(left, left + win), aver_center)
    if lower is not None:
        ax1.plot(lower)
    if upper is not None:
        ax1.plot(upper)
    im = ax2.imshow(normalized, aspect='auto', interpolation='None', cmap='cividis', )
    ax1.set_title(title, fontsize=20)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel(ylabel1, fontsize=15)
    ax2.set_ylabel(ylabel2, fontsize=15)
    ax2.set_xlabel(xlabel, fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    fig.align_ylabels()
    cbar = plt.colorbar(im, ax=[ax1, ax2], )
    cbar.ax.set_ylabel('Normalized dF/F', fontsize=15)
    plt.show()


def HSE_heatmap(HSE_peaks, activity, lower, upper, good_HSE, win_bins=100, win=30, sort='value',
                plot_all=False, peaks=None, pcs=None, if_rank=False,
                plot_aver=False, if_zscore=False, plot_cells=None, cell_order=None):
    """
    Plot heatmap of each HSE, or average over all HSEs.

    :param HSE_peaks: peaks of each HSE
    :param activity: cell activity after smoothing and zscore
    :param lower: lower bound of detecting HSE
    :param upper: upper bound of detecting HSE
    :param good_HSE: judge whether a HSE is good or not
    :param win_bins: number of bins shown
    :param win: number of bins around a HSE peak
    :param sort: sorting order: peak value, peak location or place field center
    :param plot_all: plot all HSEs or not
    :param peaks: peak position of each cell
    :param pcs: index of place cells
    :param if_rank: rank the order of cell according to sorting order
    :param plot_aver: whether plot the average of all HSEs or not
    :param if_zscore: whether activity input is zscored or not
    :param plot_cells: index of cells to show
    :param cell_order: if not None, indicate the order of cells to be shown
    """
    #     activity = zscore(gaussian_filter(activity, sigma=gaussian_std))
    mean_hse_mean = np.zeros(win_bins)
    #     population = zscore(np.mean(activity, axis=0))
    if not if_zscore:
        population = zscore(np.mean(zscore(activity, axis=1), axis=0))
    else:
        population = zscore(np.mean(activity, axis=0))

    if plot_cells is not None:
        activity = activity[plot_cells[:], :]
    hse_mean = np.zeros((activity.shape[0], win_bins))
    cell_num = activity.shape[0]

    num = 0
    good_hse = 0
    for peak in HSE_peaks:
        if activity[:, peak - win_bins // 2:peak + win_bins // 2].shape[1] == win_bins:
            left = peak - win_bins // 2
            right = peak + win_bins // 2
            hse_mean += activity[:, left:right]
            mean_hse_mean += population[left:right]
            num += 1
            normalized = np.zeros(activity[:, left:right].shape)
            for cell in range(cell_num):
                normalized[cell, :] = normalize(activity[cell, left:right])
            if cell_order is None:
                if sort == 'value':
                    # sort by peak value
                    cell_rank = np.argsort(
                        np.max(normalized[:, win_bins // 2 - win // 2:win_bins // 2 + win // 2], axis=1))
                elif sort == 'peak':
                    if peaks is None:
                        peaks = np.argsort(
                            np.max(normalized[:, win_bins // 2 - win // 2:win_bins // 2 + win // 2], axis=1))
                    cell_rank = np.argsort(peaks)
                else:  # sort == 'pc_peak'
                    # pc sort by field location
                    non_pl = np.setdiff1d(np.arange(cell_num), pcs)
                    cell_rank = np.concatenate(
                        [non_pl[np.argsort(peaks[non_pl[:]])[:]], pcs[np.argsort(peaks[pcs[:]])]])
                if if_rank:
                    normalized = normalized[cell_rank[::-1], :]

            else:
                if if_rank:
                    normalized = normalized[cell_order, :]

            if plot_all:
                plot_heatmap(population[left:right], normalized, lower=lower[left:right], upper=upper[left:right],
                             aver_center=population[peak - win // 2:peak + win // 2])
                if good_HSE(population, peak, lower, upper):
                    print('This is a good HSE')
                    good_hse += 1
                else:
                    print('This is not a good HSE')

    hse_mean /= num
    mean_hse_mean /= num
    normalized = np.zeros(hse_mean.shape)
    for cell in range(cell_num):
        normalized[cell, :] = normalize(hse_mean[cell, :])
    if cell_order is None:
        if sort == 'value':
            # sort by peak value
            cell_rank = np.argsort(np.max(normalized[:, win_bins // 2 - win // 2:win_bins // 2 + win // 2], axis=1))
        elif sort == 'peak':
            if peaks is None:
                peaks = np.argmax(normalized[:, win_bins // 2 - win // 2:win_bins // 2 + win // 2], axis=1)
            cell_rank = np.argsort(peaks)
        else:  # sort == 'pc_peak'
            # pc sort by field location
            non_pl = np.setdiff1d(np.arange(cell_num), pcs)
            cell_rank = np.concatenate([non_pl[np.argsort(peaks[non_pl[:]])[:]], pcs[np.argsort(peaks[pcs[:]])]])
    else:
        cell_rank = cell_order
    if if_rank:
        normalized = normalized[cell_rank[::-1], :]

    if plot_aver:
        plot_heatmap(mean_hse_mean, normalized, aver_center=None)

    return [mean_hse_mean, lower, upper, normalized], cell_rank[::-1], good_hse


def plot_pre_behav_post(list1, list2, list3, rank):
    """
    Plot three plots on pre, hebav and post periods, each consists HSE mean of means,
    heatmap of HSE mean over [cell, frames]

    Input:
    list1, list2, list3: [mean of means, lower_bound, upper_bound, heatmap] of pre, behav, post
    rank: the order of cells
    """
    win_bins = list1[0].shape[0]
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[0, 0], )
    ax2 = fig.add_subplot(gs[1:, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax4 = fig.add_subplot(gs[1:, 1], sharex=ax3)
    ax5 = fig.add_subplot(gs[0, 2], sharey=ax1)
    ax6 = fig.add_subplot(gs[1:, 2], sharex=ax5)

    ax1.plot(np.arange(win_bins), list1[0])
    ax1.plot(np.ones(win_bins) * (list1[1]))
    ax1.plot(np.ones(win_bins) * (list1[2]))
    im = ax2.imshow(list1[3][rank[:], :], aspect='auto', interpolation='None', cmap='cividis', )
    ax1.set_title('Pre', fontsize=20)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('Mean dF/F (z-score)', fontsize=15)
    ax2.set_ylabel('Cell ID', fontsize=15)
    ax2.set_xlabel('Frames', fontsize=15)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    fig.align_ylabels()

    ax3.plot(np.arange(win_bins), list2[0])
    ax3.plot(np.ones(win_bins) * (list2[1]))
    ax3.plot(np.ones(win_bins) * (list2[2]))
    im = ax4.imshow(list2[3][rank[:], :], aspect='auto', interpolation='None', cmap='cividis', )
    ax3.set_title('Behav', fontsize=20)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax4.set_xlabel('Frames', fontsize=15)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    plt.setp(ax4.get_yticklabels(), visible=False)

    ax5.plot(np.arange(win_bins), list3[0])
    ax5.plot(np.ones(win_bins) * (list3[1]))
    ax5.plot(np.ones(win_bins) * (list3[2]))
    im = ax6.imshow(list3[3][rank[:], :], aspect='auto', interpolation='None', cmap='cividis', )
    ax5.set_title('Post', fontsize=20)
    plt.setp(ax5.get_xticklabels(), visible=False)
    ax6.set_xlabel('Frames', fontsize=15)
    ax5.tick_params(axis='both', which='major', labelsize=12)
    ax6.tick_params(axis='both', which='major', labelsize=12)
    plt.setp(ax6.get_yticklabels(), visible=False)

    cbar = plt.colorbar(im, ax=[ax1, ax2, ax3, ax4, ax5, ax6], )
    cbar.ax.set_ylabel('Normalized dF/F', fontsize=15)

    plt.show()
    plt.close()


def HSE_counts_overtime(pre_HSE_peaks, post_HSE_peaks, total_time, bin_size=1800):
    """
    Plots HSE number as a function of time window

    Inputs:
    pre_HSE_peaks: HSE peaks in pre
    post_HSE_peaks: HSE peaks in post
    total_time: in frames
    bin_size: frame size of one time window
    """
    nbins = total_time // bin_size + 1
    pre = np.zeros(nbins[0])
    post = np.zeros(nbins[1])
    for peak in pre_HSE_peaks:
        pre[peak // bin_size] += 1
    for peak in post_HSE_peaks:
        post[peak // bin_size] += 1
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0], )
    ax1.plot(np.arange(nbins[0]), pre)
    ax1.plot(np.arange(nbins[0]), np.ones(nbins[0]) * np.mean(pre))
    ax1.plot(np.arange(nbins[1]) + nbins[0], post)
    ax1.plot(np.arange(nbins[1]) + nbins[0], np.ones(nbins[1]) * np.mean(post))
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('HSE counts', fontsize=15)
    ax1.xaxis.set_minor_locator(mticker.FixedLocator((nbins[0] // 2, nbins[1] // 2 + nbins[0])))
    ax1.xaxis.set_minor_formatter(mticker.FixedFormatter(("Pre", "Post")))
    plt.setp(ax1.xaxis.get_minorticklabels(), size=20, va="center")
    ax1.tick_params("x", which="minor", pad=25, left=False)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    plt.close()


def plot_PC_corr(pre_dffs, post_dffs, pc_peak, slid_bins=4, dis_bin=2, order='circulate'):
    """
    Plot correlation between place cells' dF/F as a function of peak distance, time periods could be pre/post or HSE

    Input:
    pre_dffs: [cell_id, frame] dF/F of all place cells in pre
    post_dffs: [cell_id, frame] dF/F of all place cells in post
    pc_peak: the peaks of all place cells
    dis_bin: bin size of distance between cell peaks
    slid_bins: bins number of a sliding window
    """
    pc_num = len(pc_peak)
    pre_corr = np.corrcoef(pre_dffs)
    post_corr = np.corrcoef(post_dffs)

    if order == 'circulate':
        nbins = int(((np.max(pc_peak) - np.min(pc_peak)) // dis_bin + 1) // 2 + 1)
    elif order == 'sequence':
        nbins = int(((np.max(pc_peak) - np.min(pc_peak)) // dis_bin + 1))
    pre_all = [[] for i in range(nbins)]
    post_all = [[] for i in range(nbins)]
    for i in range(len(pc_peak)):
        for j in range(i):
            idx = int(abs(pc_peak[i] - pc_peak[j]) // dis_bin)
            if order == 'circulate':
                idx = idx if idx <= nbins - idx else nbins - idx
            pre_all[idx].append(pre_corr[i, j])
            post_all[idx].append(post_corr[i, j])

    pre_all = np.array(pre_all, dtype=np.ndarray)
    post_all = np.array(post_all, dtype=np.ndarray)
    for k in range(nbins):
        pre_all[k] = np.array(pre_all[k])
        post_all[k] = np.array(post_all[k])
    pre_func = np.zeros(nbins)
    post_func = np.zeros(nbins)
    pre_bootstrap_l = np.zeros(nbins)
    pre_bootstrap_h = np.zeros(nbins)
    post_bootstrap_l = np.zeros(nbins)
    post_bootstrap_h = np.zeros(nbins)

    for k in range(slid_bins // 2):
        data = []
        for item in pre_all[:k + slid_bins // 2]:
            data.extend(item)
        data = np.array(data)
        pre_func[k] = np.mean(data)
        bs = bootstrap((data,), np.mean, method='percentile', vectorized=False)
        pre_bootstrap_h[k], pre_bootstrap_l[k] = bs.confidence_interval

        data = []
        for item in post_all[:k + slid_bins // 2]:
            data.extend(item)
        data = np.array(data)
        post_func[k] = np.mean(data)
        bs = bootstrap((data,), np.mean, method='percentile', vectorized=False)
        post_bootstrap_h[k], post_bootstrap_l[k] = bs.confidence_interval

    for k in range(nbins - slid_bins // 2, nbins, 1):
        data = []
        for item in pre_all[k - slid_bins // 2:]:
            data.extend(item)
        data = np.array(data)
        pre_func[k] = np.mean(data)
        bs = bootstrap((data,), np.mean, method='percentile', vectorized=False)
        pre_bootstrap_h[k], pre_bootstrap_l[k] = bs.confidence_interval

        data = []
        for item in post_all[k - slid_bins // 2:]:
            data.extend(item)
        data = np.array(data)
        post_func[k] = np.mean(data)
        bs = bootstrap((data,), np.mean, method='percentile', vectorized=False)
        post_bootstrap_h[k], post_bootstrap_l[k] = bs.confidence_interval

    for k in range(slid_bins // 2, nbins - slid_bins // 2):
        data = []
        for item in pre_all[k - slid_bins // 2:k + slid_bins // 2]:
            data.extend(item)
        data = np.array(data)
        pre_func[k] = np.mean(data)
        bs = bootstrap((data,), np.mean, method='percentile', vectorized=False)
        pre_bootstrap_h[k], pre_bootstrap_l[k] = bs.confidence_interval

        data = []
        for item in post_all[k - slid_bins // 2:k + slid_bins // 2]:
            data.extend(item)
        data = np.array(data)
        post_func[k] = np.mean(data)
        bs = bootstrap((data,), np.mean, method='percentile', vectorized=False)
        post_bootstrap_h[k], post_bootstrap_l[k] = bs.confidence_interval

    pre_func = gaussian_filter1d(pre_func, sigma=1)
    post_func = gaussian_filter1d(post_func, sigma=1)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 5)
    ax1 = fig.add_subplot(gs[0, :2], )
    ax2 = fig.add_subplot(gs[0, 3:], )
    ax1.plot(np.arange(0, dis_bin * (pre_func.shape[0]), dis_bin), pre_func, label='Pre')
    ax1.fill_between(x=np.arange(0, dis_bin * pre_func.shape[0], dis_bin), y1=pre_bootstrap_l,
                     y2=pre_bootstrap_h, alpha=0.2)
    ax1.plot(np.arange(0, dis_bin * pre_func.shape[0], dis_bin), post_func, label='Post')
    ax1.fill_between(x=np.arange(0, dis_bin * pre_func.shape[0], dis_bin), y1=post_bootstrap_l,
                     y2=post_bootstrap_h, alpha=0.2)
    ax1.legend()
    ax1.set_xlabel('Run PF \n peak distance(mm)')
    ax1.set_ylabel('Offline Pairwise \n correlation coefficient')

    for k in range(len(pre_func)):
        ax2.scatter(pre_func[k], post_func[k], color='blue', s=10)
    ax2.scatter(np.mean(pre_func), np.mean(post_func), color='orange', s=10)
    ax2.plot(np.mean(pre_func) * np.ones(2), [np.mean(post_func) - sem(post_func), np.mean(post_func) + sem(post_func)],
             color='orange')
    ax2.plot([np.mean(pre_func) - sem(pre_func), np.mean(pre_func) + sem(pre_func)], np.mean(post_func) * np.ones(2),
             color='orange')
    ax2.plot(np.linspace(min(np.min(pre_func), np.min(post_func)), max(np.max(pre_func), np.max(post_func)), 100),
             np.linspace(min(np.min(pre_func), np.min(post_func)), max(np.max(pre_func), np.max(post_func)), 100),
             linestyle='--')
    ax2.set_xlabel('Pre offline synchrony \n (mean corrected coefficient)')
    ax2.set_ylabel('Post offline synchrony \n (mean corrected coefficient)')
    plt.show()


def test_mod_cell(HSE_peaks, cell_dff, hse_win=30, win_size=100, mov_win=500, exceed_percent=0.05):
    """
    Test if a cell is modulated by HSE or not

    Input:
    HSE_peaks: list, peaks of HSEs
    cell_dff: dF/F of the cell to be tested
    hse_win: window length of HSE
    win_size: window length of testing period
    mov_win: compute baseline within the moving window length
    exceed_percent: test a cell as positive/negative if exceed_percent more/less than base line

    Output:
    label, baseline: label=1, positive modulated; -1, negative modulated; 0, not modulated
    """
    hse_means = np.zeros(len(HSE_peaks))
    com_means = np.zeros(len(HSE_peaks))
    com_medians = np.zeros(len(HSE_peaks))
    for t in range(len(HSE_peaks)):
        hse_means[t] = np.mean(cell_dff[HSE_peaks[t] - hse_win // 2:HSE_peaks[t] + hse_win // 2])
        com_means[t] = np.mean(cell_dff[HSE_peaks[t] - win_size // 2:HSE_peaks[t] - win_size // 2 + hse_win])
        com_medians[t] = np.median(cell_dff[HSE_peaks[t] - hse_win // 2:HSE_peaks[t] - hse_win // 2])

    wil_score = wilcoxon(hse_means, com_means)
    baseline = np.mean(com_means)
    if wil_score[1] > 0.05:
        #         print('p value > 0.05')
        return 0, baseline
    elif np.mean(hse_means) > (1 + exceed_percent) * baseline:
        #         print('positive')
        return 1, baseline
    elif np.mean(hse_means) < (1 - exceed_percent) * baseline:
        #         print('negative')
        return -1, baseline
    else:
        return 0, baseline


def find_mod_cell(HSE_peaks, dff, hse_win=30, win_size=100, title='', plot_all=False, plot_aver=False,
                  exceed_percent=0.05):
    """
    Use test_mod_cell to test all the cells and make plots

    Input:
    HSE_peaks: peak of all HSEs
    dff: [cell_id, frame] dF/F
    hse_win: window length of HSEs
    win_size: window length of peri-HSEs
    plot_all: True if make plots of all cells, else False
    plot_aver: True if make plots of average, else False

    Output:
    pos_mod_cell: list, cell id of all positive modulated cell
    neg_mod_cell: list, cell id of all negative modulated cell
    """
    pos_mod_cell = []
    neg_mod_cell = []
    pos_hse_mean = np.zeros((len(HSE_peaks), win_size))
    neg_hse_mean = np.zeros((len(HSE_peaks), win_size))

    for cell in range(dff.shape[0]):
        label, baseline = test_mod_cell(HSE_peaks, dff[cell, :], hse_win, win_size, exceed_percent=exceed_percent)
        if label == 1:
            pos_mod_cell.append(cell)
        #             print('mod cell')

        elif label == -1:
            neg_mod_cell.append(cell)
    #             print('not mod cell')

    for cell in pos_mod_cell:
        cell_hse = np.zeros((len(HSE_peaks), win_size))
        for k in range(len(HSE_peaks)):
            cell_hse[k, :] = normalize(dff[cell, HSE_peaks[k] - win_size // 2:HSE_peaks[k] + win_size // 2])
        pos_hse_mean += cell_hse
        cell_hse = cell_hse[
                   np.argsort(np.max(cell_hse[:, win_size // 2 - hse_win // 2:win_size // 2 + hse_win // 2], axis=1)),
                   :]
        cell_mean = np.mean(cell_hse, axis=0)
        if plot_all:
            plot_heatmap(cell_mean, cell_hse, ylabel2='HSE number', title=title + 'Positive',
                         lower=baseline * np.ones(len(cell_mean)), upper=baseline * np.ones(len(cell_mean)),
                         aver_center=cell_mean[win_size // 2 - hse_win // 2:win_size // 2 + hse_win // 2])

    for cell in neg_mod_cell:
        cell_hse = np.zeros((len(HSE_peaks), win_size))
        for k in range(len(HSE_peaks)):
            cell_hse[k, :] = normalize(dff[cell, HSE_peaks[k] - win_size // 2:HSE_peaks[k] + win_size // 2])
        neg_hse_mean += cell_hse
        cell_hse = cell_hse[
                   np.argsort(np.max(cell_hse[:, win_size // 2 - hse_win // 2:win_size // 2 + hse_win // 2], axis=1)),
                   :]
        cell_mean = np.mean(cell_hse, axis=0)
        if plot_all:
            plot_heatmap(cell_mean, cell_hse, ylabel2='HSE number', title=title + 'Negitive',
                         lower=baseline * np.ones(len(cell_mean)), upper=baseline * np.ones(len(cell_mean)),
                         aver_center=cell_mean[win_size // 2 - hse_win // 2:win_size // 2 + hse_win // 2])

    pos_hse_mean /= len(pos_mod_cell)
    neg_hse_mean /= len(neg_mod_cell)
    if plot_aver:
        plot_heatmap(np.mean(pos_hse_mean, axis=0), pos_hse_mean, ylabel2='HSE number', title='Positive')
        plot_heatmap(np.mean(neg_hse_mean, axis=0), neg_hse_mean, ylabel2='HSE number', title='Negative')

    return pos_mod_cell, neg_mod_cell


def plot_mod_cell_num(pre_num, post_num):
    """
    Make bar plot of positive and negative modulated cells in pre and post HSEs

    Input:
    pre_num: [pos_num, neg_num]
    post_num: [pos_num, neg_num]
    """

    labels = ['Pre', 'Post']
    x = np.arange(1, len(labels) + 1)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, [pre_num[0], post_num[0]], width, label='Positive')
    rects2 = ax.bar(x + width / 2, [pre_num[1], post_num[1]], width, label='Negative')

    ax.set_ylabel('Cell counts')
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.xaxis.set_minor_locator(mticker.FixedLocator((1, 2)))
    ax.xaxis.set_minor_formatter(mticker.FixedFormatter(labels))
    plt.setp(ax.xaxis.get_minorticklabels())
    ax.tick_params(axis="x", which="minor")
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


# Tereda's way to detect modulated cell for each HSE
def get_reactivated_cell(peak, dff, hse_win, win, cell_type=None, pass_corr=0.5):
    """
    Detect reactivated cell for each HSE

    Input:
    peak: peak time of a HSE
    dff: dF/F of one HSE [frames]
    hse_win: window size of a HSE
    win: peri-HSE window size

    Return:
    1 for positive modulated cell; -1 for negative modulated cell; 0 for not modulated cell
    """
    pos_react_cell = []
    neg_react_cell = []
    pca = PCA(n_components=2).fit(copy.deepcopy(dff[:, peak - hse_win // 2:peak + hse_win // 2]))  # +- 3s
    pca1 = pca.components_[0]
    for cell_id in range(dff.shape[0]):
        cell_dff = copy.deepcopy(dff[cell_id, peak - hse_win // 2:peak + hse_win // 2])
        if np.isnan(pearsonr(cell_dff, pca1)[0]):
            continue
        if pearsonr(cell_dff, pca1)[0] > pass_corr:
            pos_react_cell.append(cell_id)
        elif pearsonr(cell_dff, pca1)[0] < -pass_corr:
            neg_react_cell.append(cell_id)

    pos_react_cell = np.array(pos_react_cell, dtype=np.int64)
    neg_react_cell = np.array(neg_react_cell, dtype=np.int64)
    #     As = reactivated_cell[cell_type[reactivated_cell]==1]
    #     Xs = reactivated_cell[cell_type[reactivated_cell]==-1]
    #     plt.pie([len(As), len(Xs), len(reactivated_cell)-len(As)-len(Xs)], labels=['A', 'X', 'Other'], autopct='%1.2f%%')
    #     plt.show()

    hse_reac_pos = dff[pos_react_cell, peak - win // 2:peak + win // 2]
    hse_reac_neg = dff[neg_react_cell, peak - win // 2:peak + win // 2]
    for cell in range(len(pos_react_cell)):
        hse_reac_pos[cell, :] = normalize(hse_reac_pos[cell, :])
    for cell in range(len(neg_react_cell)):
        hse_reac_neg[cell, :] = normalize(hse_reac_neg[cell, :])
    hse_reac_pos = hse_reac_pos[
                   np.argsort(np.argmax(hse_reac_pos[:, win // 2 - hse_win // 2:win // 2 + hse_win // 2], axis=1)), :]
    hse_reac_neg = hse_reac_neg[
                   np.argsort(np.argmax(hse_reac_neg[:, win // 2 - hse_win // 2:win // 2 + hse_win // 2], axis=1)), :]
    hse_reac = np.concatenate((hse_reac_pos, hse_reac_neg), axis=0)
    plot_heatmap(np.mean(hse_reac, axis=0), hse_reac)
    plot_heatmap(np.mean(hse_reac_pos, axis=0), hse_reac_pos)
    plot_heatmap(np.mean(hse_reac_neg, axis=0), hse_reac_neg)

    #     hse_reac = dff[As, peak-win//2:peak+win//2]
    #     for cell in range(len(As)):
    #         hse_reac[cell, :] = normalize(hse_reac[cell, :])
    #     rank = np.argsort(np.argmax(hse_reac[:, win//2-hse_win//2:win//2+hse_win//2], axis=1))
    #     hse_reac = hse_reac[rank, :]
    #     plot_heatmap(np.mean(hse_reac, axis=0), hse_reac)

    #     hse_reac = dff[Xs, peak-win//2:peak+win//2]
    #     for cell in range(len(Xs)):
    #         hse_reac[cell, :] = normalize(hse_reac[cell, :])
    #     rank = np.argsort(np.argmax(hse_reac[:, win//2-hse_win//2:win//2+hse_win//2], axis=1))
    #     hse_reac = hse_reac[rank, :]
    #     plot_heatmap(np.mean(hse_reac, axis=0), hse_reac)

    return hse_reac_pos, hse_reac_neg  # , As, Xs

