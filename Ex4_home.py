# -*- coding: utf-8 -*-
"""

"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

print(torch.cuda.is_available())

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])


train_set = datasets.ImageFolder("my_data/train", transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                           shuffle=True, num_workers=2)

test_set = datasets.ImageFolder("my_data/test", transform=transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                          shuffle=True, num_workers=2)


classes = ("Cat", "Dog")

########################################################################
# Let us show some of the training images, for fun.


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 224, 224)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # get some random training images
    images, labels = next(iter(train_loader))

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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

    print('Finished Training')
#
#     dataiter = iter(testloader)
#     images, labels = next(dataiter)
#
#     # print images
#     imshow(torchvision.utils.make_grid(images))
#     print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(2)))
#
#     outputs = net(images)
#     _, predicted = torch.max(outputs, 1)
#
#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                                   for j in range(4)))
#
#     ########################################################################
#     # The results seem pretty good.
#     #
#     # Let us look at how the network performs on the whole dataset.
#
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             outputs = net(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#         100 * correct / total))
#
#     # Hmmm, what are the classes that performed well, and the classes that did
#     # not perform well:
#
#     class_correct = list(0. for i in range(2))
#     class_total = list(0. for i in range(2))
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             outputs = net(images)
#             _, predicted = torch.max(outputs, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(4):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1
#
#     for i in range(2):
#         print('Accuracy of %5s : %2d %%' % (
#             labels[i], 100 * class_correct[i] / class_total[i]))
#

if __name__ == '__main__':
    main()



