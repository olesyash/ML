import math
import datetime

import pandas as pd
import torch
import os

from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms, datasets
import numpy as np
from timeit import default_timer as timer

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

ADAM = "Adam"
SGD = "SGD"
PATH = "my_data"

print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()])

train_set = datasets.ImageFolder(os.path.join(PATH, "train"), transform=transform)

test_set = datasets.ImageFolder(os.path.join(PATH, "train"), transform=test_transform)

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).to(device)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

num_classes = 2

fc_inputs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(2048, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 2)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()


def train(batch_size=32, epochs=10):
    losses = []
    accurancies = []
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_corrects = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs_on_device = inputs.to(device)
            labels_on_device = labels.to(device)
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
                                                      epoch_acc))
    return losses, accurancies


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


def my_plot(epochs, loss):
    plt.plot(epochs, loss)


def test(batch_size=32, show=False):
    running_loss = 0.0
    running_corrects = 0
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    accurancies = []
    losses = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_set)
    print(f"Running_corrects: {running_corrects}")
    epoch_acc = running_corrects.double() / len(test_set)

    print('Epoch {} loss: {:.4f}, acc: {:.4f} %'.format("test",
                                                        epoch_loss,
                                                        epoch_acc * 100))

    return epoch_acc


def main():
    num_epochs = [2, 80, 100]
    lr_rates = [0.001, 0.0001]
    optimizers = [ADAM, SGD]
    augmentations = [True, False]
    title = ["model", "batch_size", "learning_rate", "optimizer", "augmentation", "epochs", "accuracy"]
    data = []
    batch_size = 32
    model = "Res50"
    for epoch in num_epochs:
        for lr_rate in lr_rates:
            for optimizer in optimizers:
                for augmentation in augmentations:
                    print(f"Epoch: {epoch}, LR: {lr_rate}, Optimizer: {optimizer}, Augmentation: {augmentation}")

                    start_time = timer()
                    now = datetime.datetime.now()
                    dt_string = now.strftime("%d_%m_%Y_%H_%M")
                    print(dt_string)
                    fig_name = f"res50_loss_{dt_string}.png"
                    print(fig_name)
                    losses, accuracies = train(batch_size, epoch)
                    end_time = timer()
                    print("Finished training")
                    print(f"Total training time: {end_time - start_time:.3f} seconds")
                    print(losses)
                    plt.clf()
                    my_plot(np.linspace(1, epoch, epoch).astype(int), losses)
                    plt.savefig(fig_name)
                    plt.clf()
                    fig_name = f"res50_accuracy_{dt_string}.png"
                    my_plot(np.linspace(1, epoch, epoch).astype(int), accuracies)
                    plt.savefig(fig_name)
                    accuracy = test(32)
                    end_time = timer()
                    print(f"Total time: {end_time - start_time:.3f} seconds")
                    data.append([model, batch_size, lr_rate, optimizer, augmentation, epoch, accuracy])

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M")
    df = pd.DataFrame(data, columns=title, dtype=str)
    df.to_csv(f"res50_cats_and_dogs_{dt_string}.csv", index=False)


if __name__ == '__main__':
    main()
