#%%
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from matplotlib import pyplot as plt
import gzip
import pickle

# TODO: check the number of parameters
#       this need to be similar to the MLP model in problem 1
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.pool1 = nn.MaxPool2d(3, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 10, 3) 
        self.conv2 = nn.Conv2d(10, 25, 4)
        self.fc1 = nn.Linear(25 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 1 x 28 x 28 -> 10 x 26 x 26
        x = self.pool2(x)           # 10 x 26 x 26 -> 10 x 13 x 13
        x = F.relu(self.conv2(x))   # 10 x 13 x 13 -> 25 x 10 x 10
        x = self.pool1(x)           # 25 x 10 x 10 -> 25 x 8 x 8
        x = x.view(-1, 25 * 8 * 8)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(net, criterion, optimizer, epoch_num):

    # variables tracking the error rates
    train_correction_rate = []
    valid_correction_rate = []

    # train the network
    for epoch in range(epoch_num): 

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
            
        print ("Epoch ", epoch, "Finished training. Calculating correction rate on valid set...")
        train_correction_rate_ = test_on_validation(net, trainloader, print_acc=False)
        valid_correction_rate_ = test_on_validation(net, validloader, print_acc=False)
        train_correction_rate.append(train_correction_rate_)
        valid_correction_rate.append(valid_correction_rate_)

    print('Training finished.')
    return (train_correction_rate, valid_correction_rate)

def test_on_validation(net, dataloader, print_acc=True):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    class_correct_rate = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)    #Pylint shows an error here, but it's working
            c = (predicted == labels)
            for i in range(len(data)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(len(classes)):

        if class_total[i] == 0:
            class_correct_rate[i] = 0
        else:
            class_correct_rate[i] = class_correct[i] / class_total[i]
        if print_acc == True:
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct_rate[i]))

    overall_correct_rate = sum(class_correct) / sum(class_total)

    return overall_correct_rate

#%%
def import_data():
    f = gzip.open('./data/mnist.pkl.gz')
    data = pickle.load(f, encoding='latin1')

    train_data = np.array(data[0][0], dtype=float)
    train_label = np.array(data[0][1], dtype=int)

    valid_data = np.array(data[1][0], dtype=float)
    valid_label = np.array(data[1][1], dtype=int)

    test_data = np.array(data[2][0], dtype=float)
    test_label = np.array(data[2][1], dtype=int)

    return train_data,train_label,valid_data,valid_label,test_data,test_label

train_data,train_label,valid_data,valid_label,test_data,test_label = import_data()

train_data = train_data.reshape(-1,1,28,28)
valid_data = valid_data.reshape(-1,1,28,28)

train_data = torch.tensor(list(train_data)).float()
train_label = torch.tensor(train_label).long()
valid_data = torch.tensor(list(valid_data)).float()
valid_label = torch.tensor(valid_label).long()

trainset = Data.TensorDataset(train_data, train_label)
validset = Data.TensorDataset(valid_data, valid_label)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=8)
validloader = torch.utils.data.DataLoader(validset, batch_size=32,
                                         shuffle=False, num_workers=4)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

#%%
(train_correction_rate, valid_correction_rate) = train_model(net, criterion, optimizer, epoch_num=10)


#%%
train_error_rate = []
valid_error_rate = []
for i in range(len(train_correction_rate)):
    train_error_rate.append(1 - train_correction_rate[i])
    valid_error_rate.append(1 - valid_correction_rate[i])

# np.save("assignment1/problem2_train_error", np.array(train_error_rate))
# np.save("assignment1/problem2_valid_error", np.array(valid_error_rate))


#%%
plt.figure()
plt.plot(range(1, len(valid_error_rate)+1), train_error_rate, label='train_error')
plt.plot(range(1, len(valid_error_rate)+1), valid_error_rate, label='valid_error')
plt.legend()
plt.show()
