import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import time
import numpy as np


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = x.clone()
        x[x<0] = 0
        return x


class Sigmoid(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = x.clone()
        x = 1 / (1 + np.exp(-x))
        return x


# Below is different types of model that I tried.

# name = '2dcnn'
# network model
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(in_features=1200, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 1200)
        x = self.fc3(x)
        return x


# change kernel size
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=12, kernel_size=12, stride=5, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=468, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 468)
        x = self.fc1(x)
        return x


# name = 'conv_10_5_6_3_1'  # conv1 kernel_size, stride; conv2 kernel_size, stride; turn
# change number of layers
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=10, stride=5, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=6, stride=3, padding=1)
        self.fc1 = nn.Linear(in_features=48, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 48)
        x = self.fc1(x)
        return x

# FCN
# change activate function
class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=3136, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.maxpool1(x)
        x = x.view(-1, 3136)
        x = self.fc1(x)
        return x


# loading data
class HSEDataset(torch.utils.data.Dataset):
    def __init__(self, l_inputs, l_labels):
        super().__init__()
        self.data_x = l_inputs
        self.data_y = l_labels

    def __getitem__(self, idx):
        return self.data_x[idx].reshape(1, 400), self.data_y[idx]

    def __len__(self):
        return len(self.data_y)


# training
def train(model, optimizer, loss_fn, train_loader, test_loader, batch_size):
    """

    :param model:
    :param optimizer:
    :param loss_fn:
    :param train_loader:
    :param test_loader:
    :param batch_size:
    :return:
    """
    trainacc = []
    testacc = []
    for epoch in range(50): # 20
        t = time.time()
        train_acc = []
        test_acc = []
        train_loss = []
        train_l = []
        train_a = []
        test_a = []
        for mini_epoch in range(10):
            for i, sample in enumerate(train_loader):
                input, target = sample
                prediction = model(input)
                loss = loss_fn(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                target_pre = torch.argmax(prediction, dim=1)
                train_a.append((target_pre == target).sum() / batch_size)
                train_l.append(loss.item())
            train_a = np.array(train_a)
            train_l = np.array(train_l)
            train_loss.append(np.mean(train_l))
            train_acc.append(np.mean(train_a))
            train_a = []
            train_l = []

            # test
            with torch.no_grad():
                for i, sample in enumerate(test_loader):
                    input, target = sample
                    prediction = model(input)
                    target_pre = torch.argmax(prediction, dim=1)
                    test_a.append((target_pre == target).sum() / batch_size)
                    if epoch == 49 and mini_epoch == 19:
                        if target_pre == target and target == 0:
                            l_name = 'Correct0'
                        elif target_pre == target and target == 1:
                            l_name = 'Correct1'
                        elif target_pre == 1 and target == 0:
                            l_name = 'Wrongly assign 1 to 0'
                        else:
                            l_name = 'Wrongly assign 0 to 1'
                        plt.plot(input.reshape(400))
                        plt.title(l_name)
                        plt.xlabel('frames')
                        plt.ylabel('population activity')
                        plt.savefig(l_name+'_'+str(i)+'.jpg')
                        plt.show()
            test_a = np.array(test_a)
            test_acc.append(np.mean(test_a))
            test_a = []

        train_acc = np.array(train_acc)
        test_acc = np.array(test_acc)
        trainacc.append(np.mean(train_acc))
        testacc.append(np.mean(test_acc))
        print("Epoch", epoch)
        print("Train accuracy:", np.mean(train_acc), end='; ')
        print("Test accuracy:", np.mean(test_acc), end='; ')
        print('Time: ', time.time() - t)
        plt.subplot(1, 2, 1)
        plt.plot(train_loss)
        # plt.ylim(0, 2)
        plt.ylabel('Train Loss')
        plt.title('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(test_acc, label='Test Accuracy')
        plt.legend()
        plt.ylim(0, 1.1)
        plt.title('Accuracy')
        plt.suptitle('Epoch' + str(epoch))
        plt.grid()
        # plt.savefig(name+'epoch'+str(epoch)+'.jpg')
        plt.show()

    return trainacc, testacc


def plot(trainacc, testacc, name):
    """

    :param trainacc:
    :param testacc:
    :param name:
    :return:
    """
    x = np.arange(0, len(trainacc))
    plt.plot(trainacc, label='Train Accuracy')
    plt.plot(testacc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid()
    plt.savefig(name+'.jpg')
    plt.show()


