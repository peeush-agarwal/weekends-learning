import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import os

def load_cifar10_dataset(download = False, batch_size=20):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('Data/cifar10/train',train=True,transform=transform,download=download)
    testset = datasets.CIFAR10('Data/cifar10/test', train=False, transform=transform, download=download)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

def view_images(loader, classes, n_samples = 4):
    images, labels = next(iter(loader)) #Loads next batch

    # Take first 4 images from batch
    images = images[:n_samples]
    labels = labels[:n_samples]

    images = torchvision.utils.make_grid(images)

    # Un-Normalize
    images = images/2 + 0.5
    np_images = images.numpy()

    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    plt.title([classes[labels[j]] for j in range(n_samples)])
    plt.show()

class SecondNN(nn.Module):
    def __init__(self):
        super(SecondNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out
    
    def num_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(trainloader, net, criterion, optimizer, epochs = 2):
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:

            optimizer.zero_grad()

            output = net(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss
        
        print(f'Epoch:{epoch}, loss:{running_loss/len(trainloader)}')

def evaluate(testloader, net):
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:

            output = net(images)
            _, preds = torch.max(output, 1)

            total += labels.size()[0]
            correct += (preds == labels).sum()
    
    print(f'Accuracy: {100.0 * correct/total}')

def save_model(net, PATH):
    torch.save(net.state_dict(), PATH)

def load_model(net, PATH):
    net.load_state_dict(torch.load(PATH))

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def use_tensorboard(trainloader):
    writer = SummaryWriter('runs/cifar_10_exp_1')
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_cifar_10_images', img_grid)

if __name__ == "__main__":
    trainloader, testloader = load_cifar10_dataset(download=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(len(trainloader)) # 500*100 = 50,000
    print(len(testloader))  # 100*100 = 10,000
    view_images(trainloader, classes)

    epochs = 20
    base_PATH = './Data/cifar10/models'
    base_PATH = os.path.join(base_PATH, str(epochs))
    model_PATH = os.path.join(base_PATH, 'cifar10_net.pth')

    if not os.path.exists(model_PATH):
        os.makedirs(base_PATH)
        net = SecondNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01)

        train(trainloader, net, criterion, optimizer, epochs)
        save_model(net, model_PATH)

    net = SecondNN()
    load_model(net, model_PATH)
    
    evaluate(testloader, net)

    use_tensorboard(trainloader)