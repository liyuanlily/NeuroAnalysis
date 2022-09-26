import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree




# relu layer
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.clone()
        x[x<0] = 0
        return x


class Sigmoid(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.clone()
        x = 1 / (1 + np.exp(-x))
        return x


# name = '2dcnn'
# network model
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(in_features=200*760, out_features=100)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.sigm1 = nn.Sigmoid()
        # self.fc2 = nn.Linear(in_features=50, out_features=20)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.sigm2 = nn.Sigmoid()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.sigm2 = nn.Sigmoid()
        # self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=2, stride=1, padding=1)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.sigm3 = nn.Sigmoid()
        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.fc3 = nn.Linear(in_features=1200, out_features=2)

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.sigm1(x)
        # x = self.maxpool1(x)
        # x = self.fc2(x)
        # x = self.sigm2(x)
        # x = self.maxpool2(x)
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.sigm2(x)
        # x = self.maxpool2(x)
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.sigm3(x)
        # x = self.maxpool3(x)
        x = x.view(-1, 1200)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


name = 'cnn3_1_3_1_3_1_2_1_more_lr_new' ## kernel_size, stride, out_features, which turn
class Model2(nn.Module):
# kernel size 变为6
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(in_features=200, out_features=100)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=12, kernel_size=12, stride=5, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=468, out_features=2)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 468)
        x = self.fc1(x)
        return x


# name = 'conv_10_5_6_3_1'  # conv1 kernel_size, stride; conv2 kernel_size, stride; turn
## try: 两层一维的CNN；支持向量机SVM；决策树；逻辑回归
## 单一的分类方法主要包括：LR逻辑回归，SVM支持向量机，DT决策树、NB朴素贝叶斯、NN人工神经网络、K-近邻；
## 集成学习算法：基于Bagging和Boosting算法思想，RF随机森林,GBDT，Adaboost,XGboost
class Model3(nn.Module):
# change conv layer number
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

### FCN
class Model4(nn.Module):
# change activate function
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
def train(model, optimizer, loss_fn, train_loader, test_loader, name, batch_size, show_epoch=5):
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



# inputs_aver = np.concatenate((np.load('inputs_aver_update.npy'), np.load('post_inputs_aver_update.npy')))
# inputs_all = np.concatenate((np.load('inputs_all_update.npy'), np.load('post_inputs_all_update.npy')), axis=0)
# labels = np.concatenate((np.load('labels_update.npy'), np.load('post_labels_update.npy')))
# print(inputs_aver.shape, inputs_all.shape, labels.shape)
# labels[labels==-1] = 0
# inputs_aver = np.load('inputs_aver_new.npy')
# inputs_all = np.load('inputs_all_new.npy')
# labels = np.load('labels_new.npy')

changa_inputs_aver = np.concatenate((np.load('Changa_inputs_aver_update.npy'), np.load('Changa_post_inputs_aver_update.npy')))
changa_inputs_all = np.concatenate((np.load('Changa_inputs_all_update.npy'), np.load('Changa_post_inputs_all_update.npy')), axis=0)
changa_labels = np.concatenate((np.load('Changa_labels_update.npy'), np.load('Changa_post_labels_update.npy'), np.ones(1, dtype=np.int64)))
print(changa_inputs_aver.shape, changa_inputs_all.shape, changa_labels.shape)

Magurie_ctrl_inputs_aver = np.concatenate((np.load('Magurie_pre_ctrl_inputs_aver_update.npy'), np.load('Magurie_post_ctrl_inputs_aver_update.npy')))
Magurie_ctrl_inputs_all = np.concatenate((np.load('Magurie_pre_ctrl_inputs_all_update.npy'), np.load('Magurie_post_ctrl_inputs_all_update.npy')), axis=0)
Magurie_ctrl_labels = np.concatenate((np.load('Magurie_pre_ctrl_labels_update.npy'), np.load('Magurie_post_ctrl_labels_update.npy')))
print(Magurie_ctrl_inputs_aver.shape, Magurie_ctrl_inputs_all.shape, Magurie_ctrl_labels.shape)

Magurie_seq_inputs_aver = np.concatenate((np.load('Magurie_pre_seq_inputs_aver_update.npy'), np.load('Magurie_post_seq_inputs_aver_update.npy')))
Magurie_seq_inputs_all = np.concatenate((np.load('Magurie_pre_seq_inputs_all_update.npy'), np.load('Magurie_post_seq_inputs_all_update.npy')), axis=0)
Magurie_seq_labels = np.concatenate((np.load('Magurie_pre_seq_labels_update.npy'), np.load('Magurie_post_seq_labels_update.npy')))
print(Magurie_seq_inputs_aver.shape, Magurie_seq_inputs_all.shape, Magurie_seq_labels.shape)
# print(len(labels), len(changa_labels), len(Magurie_seq_labels), len(Magurie_ctrl_labels))

inputs = np.concatenate((changa_inputs_aver, Magurie_seq_inputs_aver, Magurie_ctrl_inputs_aver), axis=0, dtype=np.float32)
targets = np.concatenate((changa_labels, Magurie_seq_labels, Magurie_ctrl_labels))
index_num = len(targets)
indexs = np.arange(index_num)
np.random.shuffle(indexs)
inputs = inputs[indexs, :]
targets = targets[indexs]
# all_cells = np.concatenate((inputs_all, changa_inputs_all, Magurie_seq_inputs_all, Magurie_ctrl_inputs_all))
# targets[targets==0] = -1
# # inputs = Magurie_seq_inputs_all
# # targets = Magurie_seq_labels
# targets.dtype = np.int64
print(inputs.shape, targets.shape)
# print('sample numbers:', len(targets))

batch_size = 1 ### try to run in batch
train_data = HSEDataset(inputs[:int(len(targets)*0.7)], targets[:int(len(targets)*0.7)])
test_data = HSEDataset(inputs[int(len(targets)*0.7):], targets[int(len(targets)*0.7):])
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)



#
# inputs = inputs_all
# targets = labels


model = Model2()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  ### lower learning rate greater test accuracy?
loss_fn = torch.nn.CrossEntropyLoss()
# print('model1', 'SGD as optimizer', 'lr=0.01', 'Cross Entropy Loss')

print('Start Training')
trainacc, testacc = train(model, optimizer, loss_fn, train_loader, test_loader, name, batch_size)
plot(trainacc, testacc, name)
state = {'model_state': model.state_dict(),
         'optimizer': optimizer.state_dict()}
torch.save(state, name+'state.pt')


