# -*- coding: utf-8 -*-
"""
CNN modules training to categorize cats and dogs pictures
"""
import datetime
import math
import torch

from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from timeit import default_timer as timer

# print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
ADAM = "Adam"
SGD = "SGD"

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

test_set = datasets.ImageFolder("my_data/test", transform=test_transform)

criterion = nn.CrossEntropyLoss()


########################################################################

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
            nn.Linear(in_features=512 * 3 * 3, out_features=2))

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
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=0, stride=(2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=0, stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0, stride=(2, 2)),
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


def train(model, batch_size=32, epochs=10, optimizer_name=SGD, lr_rate=0.001, augmentation=True):
    losses = []
    accurancies = []
    if augmentation:
        train_set = datasets.ImageFolder("my_data/train", transform=transform)
    else:
        train_set = datasets.ImageFolder("my_data/train", transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs_on_device = inputs.to(device)
            labels_on_device = labels.to(device)

            if optimizer_name == SGD:
                optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9)
            elif optimizer_name == ADAM:
                optimizer = optim.Adam(params=model.parameters(), lr=lr_rate)
            else:
                print("error - specify valid optimizer")
                exit(-1)

            optimizer.zero_grad()
            outputs = model(inputs_on_device)
            loss = criterion(outputs, labels_on_device)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_on_device.data)

        epoch_loss = running_loss / len(train_set)
        losses.append(epoch_loss)
        epoch_acc = running_corrects.double().to(device="cpu") / len(train_set)
        accurancies.append(epoch_acc)

        print('{} loss: {:.4f}, acc: {:.4f} %'.format("train",
                                                      epoch_loss,
                                                      epoch_acc * 100))
    return losses, accurancies


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


def my_plot(epochs, loss):
    plt.plot(epochs, loss)


def test(model, batch_size=32, show=False):
    running_loss = 0.0
    running_corrects = 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    if show:
        j = 0
        fig, axs = plt.subplots(math.ceil(len(test_set) / batch_size) * 2, batch_size, figsize=(100, 100))
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if show:
            for i, img in enumerate(inputs):
                ax = axs[j, i]
                ax.axis('off')
                ax.set_title("cat" if preds[i] == 0 else "dog")
                ax.imshow(imshow(img))
            j += 2

    epoch_loss = running_loss / len(test_set)
    print(f"Running_corrects: {running_corrects}")
    epoch_acc = running_corrects.double() / len(test_set)

    print('Epoch {} loss: {:.4f}, acc: {:.4f} %'.format("test",
                                                        epoch_loss,
                                                        epoch_acc * 100))
    if show:
        plt.show()
    return epoch_acc


def run(model, batch_size, lr_rate, optimizer, augmentation, num_epochs):
    if model == "model2":
        net = Model2().to(device)
    elif model == "model3":
        net = Model3().to(device)
    else:
        net = Model1().to(device)
    fig_name = "{model}_loss_{now}.png"
    acc_fig_name = "{model}_acc_{now}.png"
    print(f"Start training model {model}, with batch size {batch_size}, with optimizer {optimizer}, "
          f"augmentation: {augmentation}, num of epochs: {num_epochs}")

    losses, accurancies = train(model=net, batch_size=batch_size, epochs=num_epochs, optimizer_name=optimizer,
                                lr_rate=lr_rate, augmentation=augmentation)
    end_time = timer()
    print(f"Total train time: {end_time - start_time:.3f} seconds")
    print(losses)
    print(accurancies)
    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    fig_name = fig_name.format(model=model, now=dt_string)
    acc_fig_name = acc_fig_name.format(model=model, now=dt_string)
    print(fig_name)
    plt.title("Loss function")
    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), losses)
    plt.savefig(fig_name)
    plt.clf()
    plt.title("Accuracy function")
    my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), accurancies)
    plt.savefig(acc_fig_name)
    accurancy = test(model=net, batch_size=batch_size)
    end_time = timer()
    print(f"Total run time: {end_time - start_time:.3f} seconds")
    return accurancy


if __name__ == '__main__':
    # Set random seeds
    start_time = timer()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    batch_size = 4
    lr_rates = [0.001, 0.0001, 0.00001, 0.000001]
    optimizers = [ADAM, SGD]
    augmentations = [True, False]
    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    title = ["model", "batch_size", "learning_rate", "optimizer", "augmentation", "epochs", "accurancy"]
    data = []
    models = ["model1", "model2", "model3"]
    num_epochs = [11, 20, 40, 80]
    for model in models:
        for lr_rate in lr_rates:
            for optimizer in optimizers:
                for augmentation in augmentations:
                    for num_epoch in num_epochs:
                        try:
                            accurancy = run(model, batch_size, lr_rate, optimizer, augmentation, num_epoch)
                        except:
                            accurancy = 0
                        data.append([model, batch_size, lr_rate, optimizer, augmentation, num_epoch, accurancy])
    df = pd.DataFrame(data, columns=title, dtype=str)
    df.to_csv(f"cnn_cats_and_dogs_{dt_string}.csv", index=False)


