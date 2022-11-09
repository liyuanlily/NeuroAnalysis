# A GLM encoder algorithm from Minderer 2019

import numpy as np
import cupy as cp  # Numpy and Scipy for GPU
import math
from matplotlib import pyplot as plt
from scipy.optimize import minimize


class Bfun():
    """
    Creat a Bfun for every input factor, consisting of 16 b_i
    """

    def __init__(self, phis, interval):
        """
        Initiating parameters

        :param phis: center
        :param interval: range of the feature
        """
        self.phis = phis
        self.size = phis.shape[0]
        self.interval = interval

    def forward(self, s):
        """
        Compute results to be sent to next layer

        :param s: feature
        :return: computational result of basis functions
        """
        b = np.zeros(self.size)
        for i in range(self.size):
            if self.phis[i] - self.interval < s < self.phis[i] + self.interval:
                b[i] = 1 / 2 * math.cos(s - self.phis[i]) + 1 / 2
            else:
                b[i] = 0
        return b


class PoissonLogPenaltyLoss():
    """
    Loss function with penalty term
    """

    def __init__(self, c, P, m):
        """
        Initiating parameters.

        :param c: scaling parameter
        :param P: the second derivative function
        :param m: batch size
        """
        self.c = c
        self.P = P
        self.m = m
        self.prediction = None
        self.targets = None

    def add_paras(self, inputs, targets):
        """
        Add inputs and corresponding targets.
        """
        self.inputs = inputs
        self.targets = targets

    def loss(self, Ws, Bfuncs):
        """
        Compute loss.

        :param Ws: weight matrix
        :return: loss
        """
        Ws = Ws.reshape((self.m, 16))
        _prediction = predict(self.inputs, Ws, Bfuncs)
        loss = np.sum(-self.targets * _prediction + np.exp(_prediction))
        for i in range(self.m):
            loss += self.c * (Ws[i, :].reshape((1, 16)) @ self.P @ Ws[i, :].reshape((16, 1))
                              + math.sqrt(16) * np.sum(Ws[i, :] ** 2))
        return float(loss)


def predict(inputs, Ws, Bfuncs):
    """
    Make predictions of dF/F based on fitted weights and inputs

    :param inputs: inputs
    :param Ws: weight matrix
    :param Bfuncs: basis functions
    :return: prediction of given inputs
    """
    t = inputs.shape[0]
    m = inputs.shape[1]
    prediction = cp.zeros(t)
    for i in range(t):
        fs = 0
        for j in range(m):
            fs += cp.sum(Ws[j, :] * Bfuncs[j].forward(inputs[i, j]))
        prediction[i] = fs
    return prediction


def fit_one_neuron(inputs, outputs):
    """
    Fit the weight matrix for one neuron

    :param inputs: inputs
    :param outputs: corresponding outputs
    :return: fitted weights
    """
    m = inputs.shape[1]
    global c, P
    loss_fn = PoissonLogPenaltyLoss(c, P, m)
    Ws = cp.random.normal(loc=0, scale=0.1 * math.sqrt(2 / 16), size=(m, 16))
    t = inputs.shape[0]
    loss_fn.add_paras(inputs, outputs)
    res = minimize(loss_fn.loss, Ws.flatten(), method='BFGS')
    Ws = res.x.reshape(m, 16)
    return Ws


# some needed parameters
P = np.zeros((16, 16))
for i in range(16):
    P[i, i] = 2
for i in range(1, 16):
    P[i, i - 1] = -1
for i in range(0, 15):
    P[i, i + 1] = -1
c = 1
