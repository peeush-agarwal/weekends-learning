import torchvision
import torchvision.transforms as transforms
import os
import pickle
import numpy as np

def download_dataset(root_dir):
    print('Loading dataset from internet using pytorch')
    transformations = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    torchvision.datasets.CIFAR10(root_dir, train=True, transform = transformations, download=True)
    torchvision.datasets.CIFAR10(root_dir, train=False, transform = transformations, download=True)

def load_dataset(root_dir):
    dataset_dir = os.path.join(root_dir, 'cifar-10-batches-py')
    if not os.path.exists(dataset_dir):
        download_dataset(root_dir)
    
    X_train = None
    X_test = None
    Y_train = []
    Y_test = []
    is_first_train_batch = True
    for file in os.listdir(dataset_dir):
        if file == 'readme.html' or file == 'batches.meta':
            continue
        print(f'Loading file: {file}')
        batch_data = unpickle(os.path.join(dataset_dir, file))
                
        if file == 'test_batch':
            X_test = batch_data[b'data']
            Y_test += batch_data[b'labels']
        else:
            if is_first_train_batch:
                X_train = batch_data[b'data']
                is_first_train_batch = False
            else:
                X_train = np.concatenate((X_train, batch_data[b'data']), axis=0)
            Y_train += batch_data[b'labels']
    
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32)
    X_train = np.rollaxis(X_train, 1, 4)
    Y_train = np.array(Y_train)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32)
    X_test = np.rollaxis(X_test, 1, 4)
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
