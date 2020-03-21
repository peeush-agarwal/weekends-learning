# import torch
# import torchvision
# import torchvision.transforms as transforms
# import os
import numpy as np
from load_cifar_10_dataset import load_dataset
import matplotlib.pyplot as plt
from nearest_neighbor_classifier import NearestNeighbor

def plot_random_images(data, labels):
    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, data.shape[0])
            ax[m, n].imshow(data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
            ax[m, n].set_title(f'Label={labels[idx]}')
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()

def use_nearest_neighbor(X_train, X_test, Y_train, Y_test):
    X_train_flatten = X_train.reshape(X_train.shape[0], 3*32*32)
    X_test_flatten = X_test.reshape(X_test.shape[0], 3*32*32)

    NN = NearestNeighbor()
    NN.train(X_train_flatten, Y_train)
    Y_predict = NN.predict(X_test_flatten)

    correct = np.sum(Y_test == Y_predict)
    total = Y_test.shape[0]

    print(f'Accuracy: {100 * correct / total}')

if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_dataset('./Data/CIFAR-10')
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    # plot_random_images(X_train, Y_train)

    use_nearest_neighbor(X_train, X_test, Y_train, Y_test)

    # TODO: Add kNN Classifier
    # TODO: Add SVM Classifier
    # TODO: Add NeuralNetwork Classifier