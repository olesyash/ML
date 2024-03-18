import math
import datetime
import torch

from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms, datasets
import numpy as np
from timeit import default_timer as timer

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
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

train_set = datasets.ImageFolder("my_data/train", transform=transform)

test_set = datasets.ImageFolder("my_data/test", transform=test_transform)

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
        epoch_acc = running_corrects.double() / len(train_set)

        print('{} loss: {:.4f}, acc: {:.4f} %'.format("train",
                                                      epoch_loss,
                                                      epoch_acc))
    return losses


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
    if show:
        j = 0
        fig, axs = plt.subplots(math.ceil(len(test_set)/batch_size)*2, batch_size, figsize=(100, 100))
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
    epoch_acc = running_corrects.double() / len(test_set) * 100

    print('Epoch {} loss: {:.4f}, acc: {:.4f} %'.format("test",
                                                      epoch_loss,
                                                      epoch_acc))
    if show:
        plt.show()


start_time = timer()
now = datetime.datetime.now()
now = dt_string = now.strftime("%d_%m_%Y_%H_%M")
print(now)
fig_name = f"res50_loss_{now}.png"
print(fig_name)
num_epochs = 100
losses = train(32, num_epochs)
end_time = timer()
print("Finished training")
print(f"Total training time: {end_time-start_time:.3f} seconds")
print(losses)
my_plot(np.linspace(1, num_epochs, num_epochs).astype(int), losses)
plt.savefig(fig_name)
test(32)
end_time = timer()
print(f"Total time: {end_time-start_time:.3f} seconds")
