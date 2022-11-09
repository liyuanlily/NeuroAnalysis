import math
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
from scipy.stats import norm


def fit(cell_activity, loc_train, track_len, track_sample_len, track_Bins, cell_num):
    '''
    A function to fit the tuning curve of each place cell from known place cell activity data and location data.

    Input:
    -----
    cell_activity: array, place cell activity to fit with [cell, firing rate at each sampling time]
    loc_train: array, locations to fit with [location at each time]

    Output:
    -------
    tuning: array, tuning curve over locations for each cell [cell_id, weight]
    loc_possi: array, the probability of arriving at each position
    loc_prob: array, the probability of arriving at one position from a given position
    print total time of fitting
    '''
    tuning = np.zeros((cell_num, track_Bins))  # fit parameter of the tuning curve
    t0 = time.time()
    loc_range = np.linspace(0, track_len, track_Bins)
    loc_train = np.floor(loc_train / track_sample_len) * track_sample_len
    loc_range_re = np.array([loc_range, loc_range ** 2]).transpose()
    loc_train_re = np.array([loc_train, loc_train ** 2]).transpose()

    # fit tuning curve for each cell
    for cell_id in range(cell_num):
        Y = sm.add_constant(loc_train_re)
        poiss_model = sm.GLM(cell_activity[cell_id], Y, family=sm.families.Poisson())
        glm_res = poiss_model.fit()
        tuning_cur = glm_res.predict(sm.add_constant(loc_range_re))
        tuning[cell_id] = tuning_cur
    print('fit_time: ', time.time() - t0)

    # plot example: plot decoding performance for an example trial
    xs = np.zeros((cell_num, track_Bins))
    ys = np.zeros((cell_num, track_Bins))
    zs = np.zeros((cell_num, track_Bins))
    for i in range(cell_num):
        zs[i, :] = tuning[i].copy()
        xs[i, :] = loc_range.copy()
        ys[i, :] = [i + 1 for n in range(track_Bins)]
    xs = xs.flatten()
    ys = ys.flatten()
    zs = zs.flatten()
    zs = zs / np.max(zs)
    plt.scatter(xs, ys, c=zs)
    plt.colorbar()
    plt.show()

    # recording possibility at one position, used in test_loc()
    counts = np.zeros(track_Bins)
    for i in range(np.shape(loc_train)[0]):
        which = int(loc_train[i] / track_sample_len)
        counts[which] += 1
    counts = counts / np.sum(counts)
    loc_possi = counts

    # recording standard deviation, then calculate possibility of going from one position to another,
    # used in test_loc_transition()
    time_Bins = np.shape(loc_train)[0]
    dx = np.zeros([time_Bins - 1, 1])
    for i in range(track_Bins - 1):
        dx[i] = loc_train[i + 1] - loc_train[i]
    std = np.sqrt(np.mean(dx ** 2))

    dists = np.zeros([track_Bins, track_Bins])
    for i in range(track_Bins):
        for j in range(track_Bins):
            dists[i][j] = (j - i) * 2
    loc_prob = norm.pdf(dists, 0, std)
    for i in range(track_Bins):
        loc_prob[i, i] = 0

    return tuning, loc_possi, loc_prob


class BayesianDecoder:
    """
    A general class for Bayesian Decoder.

    Functions:
    __init__: Initiating parameters for the decoder
    decoder: A function to predict positions basing on experimental cell activity data and fitting parameters in fit()
    test: A function to show fitting performance for several test trials
    predict: Given cell activity data, predict location of the mouse
    """
    def __init__(self, dt, track_len, track_sample_len, track_Bins, cell_num, tuning, loc_possi=None, loc_prob=None):
        """
        Initiating parameters for the decoder.

        :param dt: size of the time bin
        :param track_len: length of the whole track
        :param track_sample_len: length for each position bin of the track
        :param track_Bins: number of position bins, equal to track_len//track_sample_len+1
        :param cell_num: number of cells
        :param tuning: tuning curve for each cell
        :param loc_possi: array, the probability of arriving at each position
        :param loc_prob: array, the probability of arriving at one position from a given position
        """
        self.dt = dt
        self.track_len = track_len
        self.track_sample_len = track_sample_len
        self.track_Bins = track_Bins
        self.cell_num = cell_num
        self.tuning = tuning
        if loc_possi is not None:
            self.loc_possi = loc_possi
        if loc_prob is not None:
            self.loc_prob = loc_prob

    def decoder(self, roll_time, test=False):
        """
        A function to predict positions basing on experimental cell activity data and fitting parameters in fit()

        :param roll_time: number of bins to roll forward
        :param test: Indicating train trials or test trials
        :return: prediction of position from input cell activity data
        """
        predict = np.zeros(np.shape(self.test_loc_data))
        predict_acc = np.zeros(np.shape(self.test_loc_data))
        return predict, predict_acc

    def test(self, test_cell_data, test_loc_data=None):
        """
        A function to show fitting performance for several test trials

        :param test_cell_data: cell activity data for testing
        :param test_loc_data: location data for testing
        :return: show decoding error and accuracy in plots
        """
        self.test_cell_data = test_cell_data
        self.test_loc_data = test_loc_data
        predict, predict_acc = self.decoder(0, test=True)
        shuffle = np.zeros(np.shape(predict))
        shuffle_acc = np.zeros(np.shape(predict_acc))
        t = 0
        while t < 100:
            add1, add2 = self.decoder(np.random.randint(0, 1200), test=True)
            shuffle += add1
            shuffle_acc += add2
            t += 1
        shuffle /= t
        shuffle_acc /= t

        num = np.shape(self.test_loc_data)[0]
        x = np.array([i for i in range(num)])
        x2 = np.linspace(0, self.track_len, self.track_Bins)
        plt.plot(x, self.test_loc_data, color='blue', label='actual')
        plt.plot(x, predict, color='orange', label='predict')
        plt.plot(x, shuffle, color='grey', label='shuffle')
        plt.legend()
        plt.ylabel('Position(cm)')
        plt.show()

        # Decoding error
        plt.subplot(211)
        plt.plot(x, self.test_loc_data - predict, label='predict')
        plt.plot(x, shuffle - predict, color='grey', label='shuffle')
        plt.legend()
        plt.ylabel('Decoding Error \n Actual - Predicted (cm)')
        plt.title('Decoding Performace by Space')
        plt.subplot(212)
        # reform predict accuracy data
        test = np.floor(self.test_loc_data / self.track_sample_len) * self.track_sample_len
        acc = np.zeros(self.track_Bins)
        counts = np.zeros(self.track_Bins)
        for i in range(np.shape(test)[0]):
            loc = int(test[i] / self.track_sample_len)
            acc[loc] += predict_acc[i]
            counts[loc] += 1
        for k in range(self.track_Bins):
            if counts[k] != 0:
                acc[k] = acc[k] / counts[k]
        s_acc = np.zeros(self.track_Bins)
        counts = np.zeros(self.track_Bins)
        for i in range(np.shape(test)[0]):
            loc = int(test[i] / self.track_sample_len)
            s_acc[loc] += shuffle_acc[i]
            counts[loc] += 1
        for k in range(self.track_Bins):
            if counts[k] != 0:
                s_acc[k] = s_acc[k] / counts[k]
        plt.plot(x2, acc, color='orange', label='predict')
        plt.plot(x2, s_acc, color='grey', label='shuffle')
        plt.legend()
        plt.ylabel('Decoding Accuracy')
        plt.xlabel('Position in Virtual Space (cm)')
        plt.show()

    def predict(self, cell_trains, pos, pos_test=None):
        """
        Given cell activity data, predict location of the mouse

        :param cell_trains: cell activity data [cell_id, activity in time bins]
        :param pos: current position
        :param pos_test: if testing, the actual position to be tested
        :return: predicted location
        """
        self.test_cell_data = cell_trains
        self.test_loc_data = pos_test
        self.pos = pos
        predict = self.decoder(cell_trains, 0)[0]
        return predict


class MemorylessDecoder(BayesianDecoder):
    """
    A memoryless bayesian decoder, using a memoryless probability-based decoding algorithm from Pfeiffer 2013

    Functions:
    decoder: A function to predict positions basing on experimental cell activity data and fitting parameters in fit()
    """
    def decoder(self, roll_time=0, test=None):
        """
        A function to predict positions basing on experimental cell activity data and fitting parameters in fit()
        Using a memoryless probability-based decoding algorithm from Pfeiffer 2013

        :param cell_data: cell activity data
        :param roll_time: number of time bins to be rolled forward
        :param test: if this is a testing trial
        :return:
        loc_predict: predicted locations at each time bin
        loc_predict_acc: accuracy of predicted locations
        """
        time_Bins = np.shape(self.test_cell_data)[1]
        loc_predict = np.zeros(time_Bins)
        if test:
            loc_predict_acc = np.zeros(time_Bins)
        else:
            loc_predict_acc = None
        for time in range(time_Bins):
            probs = np.zeros(self.track_Bins)
            spike = np.roll(np.round(self.test_cell_data[:, time] * self.dt), roll_time)
            for loc in range(self.track_Bins):
                probs[loc] = np.prod(self.tuning[:, loc]** spike) * np.exp(- np.sum(self.tuning[:, loc]) * self.dt)
            if np.sum(probs) != 0:
                probs = probs / np.sum(probs)
            loc_predict[time] = np.argmax(probs) * self.track_sample_len
            if test:
                loc_predict_acc[time] = probs[math.floor(self.test_loc_data[time] / self.track_sample_len)]
        return loc_predict, loc_predict_acc


class Position_basedDecoder(BayesianDecoder):
    """
    A Bayesian Decoder based on position,
    using a position-related probability-based decoding algorithm from Tingley and Peyrache 2019

    Functions:
    decoder: A function to predict positions basing on experimental cell activity data and fitting parameters in fit()
    """
    def decoder(self, roll_time, test=None):
        """
        A function to predict positions basing on experimental cell activity data and fitting parameters in fit()
        Using a position-related probability-based decoding algorithm from Tingley and Peyrache 2019

        :param roll_time: number of time bins to be rolled forward
        :param test: if this is a testing trial
        :return:
        loc_predict: predicted locations at each time bin
        loc_predict_acc: accuracy of predicted locations
        """
        time_Bins = np.shape(self.test_cell_data)[1]
        loc_predict = np.zeros(time_Bins)
        if not test:
            loc_predict_acc = None
        else:
            loc_predict_acc = np.zeros(time_Bins)
        for time in range(time_Bins):
            probs = np.zeros(self.track_Bins)
            spike = np.roll(np.round(self.test_cell_data[:, time] * self.dt), roll_time)
            for loc in range(self.track_Bins):
                probs[loc] = self.loc_possi[loc] * np.prod(self.tuning[:, loc] ** spike) \
                             * np.exp(-np.sum(self.tuning[:, loc]) * self.dt)
            if np.sum(probs) != 0:
                probs = probs / np.sum(probs)
            loc_predict[time] = np.argmax(probs)
            if test is not None:
                loc_predict_acc[time] = probs[int(math.floor(self.test_loc_data[time] / self.track_sample_len))]
        loc_predict = np.round(loc_predict * self.track_len / self.track_Bins, 2)
        return loc_predict, loc_predict_acc


class Loc_transitionDecoder(BayesianDecoder):
    """
    A Bayesian Decoder based on location transition, u
    using a position-transient probability-based decoding algorithm from Kording's lab

    Functions:
    decoder: A function to predict positions basing on experimental cell activity data and fitting parameters in fit()
    """
    def decoder(self, roll_time, test=None):
        """
        A function to predict positions basing on experimental cell activity data and fitting parameters in fit()
        Using a position-transient probability-based decoding algorithm from Kording's lab

        :param roll_time: number of time bins to be rolled forward
        :param test: if this is a testing trial
        :return:
        loc_predict: predicted locations at each time bin
        loc_predict_acc: accuracy of predicted locations
        """
        time_Bins = np.shape(self.test_cell_data)[1]
        if test is not None:
            pos = int(np.roll(self.test_loc_data, roll_time)[0] / self.track_sample_len)
        else:
            pos = int(math.ceil(self.pos))
        loc_predict = np.zeros(time_Bins)
        if not test:
            loc_predict_acc = None
        else:
            loc_predict_acc = np.zeros(time_Bins)
        for time in range(time_Bins):
            probs = np.zeros(self.track_Bins)
            spike = np.roll(np.round(self.test_cell_data[:, time] * self.dt), roll_time)
            for loc in range(self.track_Bins):
                probs[loc] = self.loc_prob[pos, loc] * np.prod(self.tuning[:, loc] ** spike) * np.exp(-np.sum(self.tuning[:, loc]) * self.dt)
            if np.sum(probs) != 0:
                probs = probs / np.sum(probs)
            pos = np.argmax(probs)
            loc_predict[time] = pos
            if test is not None:
                loc_predict_acc[time] = probs[math.floor(self.test_loc_data[time] / self.track_sample_len)]
        loc_predict = np.round(loc_predict * self.track_len / self.track_Bins, 2)
        return loc_predict, loc_predict_acc