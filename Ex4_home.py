# -*- coding: utf-8 -*-
"""

"""

import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)


test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()])

transform = transforms.Compose([
    transforms.Resize(size=IMAGE_SIZE),
    # Flip the images randomly on the horizontal
    transforms.TrivialAugmentWide(),
    # Turn the image into a torch.Tensor
    transforms.ToTensor()])  # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0])


classes = ("Cat", "Dog")

########################################################################
# Let us show some of the training images, for fun.


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Model from Ex3
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # self.batch_size = batch_size
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 512, 3, padding=1)
        self.fc1 = nn.Linear(1605632, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(4, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Model from https://www.kaggle.com/code/tirendazacademy/cats-dogs-classification-with-pytorch
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2))
        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 512, 3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(512, 512, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=512*3*3, out_features=2))

    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x


# Model from https://github.com/vashiegaran/Pytorch-CNN-with-cats-and-dogs-/blob/main/CatvsDog.ipynb
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

def train_and_test(net, optimizer_name="SGD", lr_rate=0.001, batch_size=4, epochs=2, do_augmentaton=True):
    # Start the timer
    start_time = timer()
    if do_augmentaton:
        train_set = datasets.ImageFolder("my_data/train", transform=transform)
        train_set2 = datasets.ImageFolder("my_data/train", transform=test_transform)
    else:
        train_set = datasets.ImageFolder("my_data/train", transform=test_transform)

    test_set = datasets.ImageFolder("my_data/test", transform=test_transform)
    train_loader1 = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    train_set = torch.utils.data.ConcatDataset([train_set, train_set2])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
              shuffle=False, num_workers=2)

    # get some random training images
    # shape - (4 - batch, 3, 224, 224)
    images, labels = next(iter(train_loader))

    # show images
    # imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    # summary(net, input_size=[4, 3, 224, 224])

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr_rate, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(params=net.parameters(), lr=lr_rate)
    else:
        print("error - specify valid optimizer")
        exit(-1)

    net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            X = inputs.to(DEVICE)
            y = labels.to(DEVICE)

            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print(f"loss: {loss}")
            if i % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")
    print('Finished Training')

    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    X = images.to(DEVICE)
    y = labels.to(DEVICE)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    outputs = net(X)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))

    ########################################################################
    # The results seem pretty good.
    #
    # Let us look at how the network performs on the whole dataset.

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            X = images.to(DEVICE)
            y = labels.to(DEVICE)
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == y).sum().item()

    print('Accuracy of the network on the 500 test images: %d %%' % (
        100 * correct / total))

    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            X = images.to(DEVICE)
            y = labels.to(DEVICE)
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == y).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    # Set random seeds
    start_time = timer()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    net = Model2().to(device="cuda")
    train_and_test(net, epochs=5, optimizer_name="Adam")
    end_time = timer()
    print(f"Total run time: {end_time - start_time:.3f} seconds")



