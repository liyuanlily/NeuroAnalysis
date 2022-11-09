# A GLM encoder algorithm from Minderer 2019, pytorch version

import numpy as np
import math
import random
from matplotlib import pyplot as plt
import torch
from torch import nn


class PoissonGLMDataset(torch.utils.data.Dataset):
    """
    Generate poisson data from existing data set.
    """
    def __init__(self, inputs, outputs, d):
        """
        Initiating parameters.

        :param inputs: inputs
        :param outputs: corresponding outputs
        :param d: number of past bins that used in predicting
        """
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.d = d  # how many history used, including current
        self.m = inputs.shape[1]  # number of features
        self.gaussian_kernel = torch.exp(-(torch.arange(self.d) - (self.d - 1) / 2) ** 2 / 2)
        self.gaussian_kernel = self.gaussian_kernel / torch.sum(self.gaussian_kernel * torch.arange(self.d))

    def __getitem__(self, idx):
        if idx < self.d - 1:
            res = torch.zeros((self.d, self.m), requires_grad=False)
            res[self.d - idx - 1:, :] = self.inputs[:idx + 1, :]
            res = (self.gaussian_kernel.reshape(1, self.d) @ res).reshape(self.m)
            return res, self.outputs[idx]
        else:
            res = (self.gaussian_kernel.reshape(1, self.d) @ self.inputs[idx - self.d + 1:idx + 1, :]).reshape(self.m)
            return res, self.outputs[idx]

    def __len__(self):
        return self.inputs.shape[0]


class Bfunction(nn.Module):
    """
    Basis functions.
    """
    def __init__(self, phis, interval):
        """
        Initiating parameters.

        :param phis: center
        :param interval: range of feature
        """
        super().__init__()
        self.paras = phis  # features_num * 16
        self.interval = interval  # features_num * 1

    def forward(self, x):
        """
        Compute the result to be sent to next layer.

        :param x: [batch_size, m]
        :return: conputing result
        """
        m, length = self.paras.shape
        batch_size = x.shape[0]
        res = torch.zeros((batch_size, m * length))
        for i in range(m):
            for j in range(length):
                for k in range(batch_size):
                    if self.paras[i, j] - self.interval[i] < x[k, i] <= self.paras[i, j] + self.interval[i]:
                        res[k, i * m + j] = 1 / 2 * math.cos(x[k, i] - self.paras[i, j]) + 1 / 2
        return res


class fLinear(nn.Module):
    """
    Linear layer.
    """
    def __init__(self, input_features, output_features):
        """
        Initiating parameters.

        :param input_features: number of input features
        :param output_features: number of output features
        """
        super(fLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(16 * input_features))
        self.weight.data.uniform_(-1, 1)

    def forward(self, x):
        """
        Compute the result to be sent to next layer.

        :param x: [batch_size, m]
        :return:
        y: [batch_size, 1]
        Ws: weight matrix
        """
        y = x @ self.weight.reshape(self.weight.shape[0], 1)
        y = torch.sum(y, dim=1)
        Ws = self.weight
        return y, Ws


class PoissonLogPenaltyLoss(nn.Module):
    """
    Loss function based on poisson likelihood with penalty.
    """
    def __init__(self, c, P, m):
        """
        Initiating parameters.

        :param c: scaling factor
        :param P: the second derivative function
        :param m: batch size
        """
        super().__init__()
        self.c = c
        self.P = P
        self.m = m

    def forward(self, prediction, target, Ws):
        """
        Compute the result to be sent to next layer.

        :param prediction: [batch_size, 1]
        :param target: [batch_size, 1]
        :param Ws: [16*m, 1]
        :return: loss
        """
        Ws = Ws.reshape((self.m, 16))
        loss = torch.sum(-prediction * target + torch.exp(prediction))
        for i in range(self.m):
            loss += self.c * ((Ws[i, :].reshape(1, 16) @ self.P @ Ws[i, :].reshape((16, 1))).reshape([])
                              + math.sqrt(16) * torch.sum(Ws[i, :] ** 2))

        return loss


class GLM(nn.Module):
    """
    A GLM encoder.
    """
    def __init__(self, phis, interval):
        """
        Initiating parameters.

        :param phis: center
        :param interval: range of feature
        """
        super().__init__()
        self.B = Bfunction(phis, interval)
        self.fLinear = fLinear(input_features=3, output_features=16 * 3)

    def forward(self, x):
        """
        Compute the result to be sent to next layer.

        :param x: inputs
        """
        x = self.B(x)
        x, Ws = self.fLinear(x)
        return x, Ws


P = np.zeros((16, 16))
for i in range(16):
    P[i, i] = 2
for i in range(1, 16):
    P[i, i-1] = -1
for i in range(0, 15):
    P[i, i+1] = -1
P = torch.tensor(P, dtype=torch.float32)
c = 1


def train(m, fit_inputs, fit_outputs, test_inputs, test_outputs):
    """
    A training progress for this GLM encoder.

    :param m: batch size
    :param fit_inputs: training inputs
    :param fit_outputs: training outputs
    :param test_inputs: testing inputs
    :param test_outputs: testing outputs
    :return: show prediction plots
    """
    phis = np.zeros((m, 16))
    inter = np.zeros(m)
    phis[0, :] = np.linspace(400, 2000, 16)[:]
    phis[1, :] = np.linspace(-10, 130, 16)[:]
    phis[2, :] = np.linspace(0, 3, 16)[:]
    inter[0] = 1600 / 15 * 2
    inter[1] = 140 / 15 * 2
    inter[2] = 3 / 15 * 2
    # parameters for Adam optimizer
    lr = 0.001
    beta1 = 0.8
    beta2 = 0.9
    # bins before and after used for determining one time point
    d_before = 4
    d_after = 3
    d = 5

    glms = []
    shuffle_time = 10
    for cell_id in range(10):
        glm = GLM(phis, inter)
        optimizer = torch.optim.Adam(glm.parameters(), lr=0.01)
        loss_fn = PoissonLogPenaltyLoss(c, P, m)

        fit_loader = torch.utils.data.DataLoader(
            dataset=PoissonGLMDataset(fit_inputs[:-1, :], fit_outputs[cell_id, :-1], d),
            batch_size=50, shuffle=True)
        for sample in fit_loader:
            batch_X, batch_Y = sample
            prediction, Ws = glm(batch_X)
            loss = loss_fn(prediction, batch_Y, Ws)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        glms.append(glm)

        glms_con = []
        for turn in range(shuffle_time):
            glm_con = GLM(phis, inter)
            optimizer = torch.optim.Adam(glm.parameters(), lr=0.01)
            loss_fn = PoissonLogPenaltyLoss(c, P, m)
            roll_time = random.randint(5, 10)

            fit_loader = torch.utils.data.DataLoader(dataset=PoissonGLMDataset(fit_inputs[:-1, :],
                                                                               torch.roll(fit_outputs[cell_id, :-1], d),
                                                                               roll_time),
                                                     batch_size=50, shuffle=True)
            for sample in fit_loader:
                batch_X, batch_Y = sample
                prediction, Ws = glm_con(batch_X)
                loss = loss_fn(prediction, batch_Y, Ws)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            glms_con.append(glm_con)

        # test
        test_loader = torch.utils.data.DataLoader(
            dataset=PoissonGLMDataset(test_inputs[:-37, :], test_outputs[cell_id, :-37], d),
            batch_size=50, shuffle=False)
        with torch.no_grad():
            for sample in test_loader:
                X, Y = sample
                prediction, Ws = glm(X)
                prediction = torch.exp(prediction)
                prediction_con = torch.zeros(prediction.shape)
                for turn in range(shuffle_time):
                    glm_con = glms_con[turn]
                    prediction_con += glm_con(X)[0]
                prediction_con /= shuffle_time
                prediction_con = torch.exp(prediction_con)
                plt.plot(prediction_con, label='shuffle')
                plt.plot(prediction, label='predict')
                plt.plot(Y, label='real')
                plt.legend()
                plt.xlabel('time(s)')
                plt.ylabel('dff' + 'cell' + str(cell_id))
                plt.show()






