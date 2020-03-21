import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        num_test = X_test.shape[0]
        Y_predict = np.zeros(num_test, dtype = self.Y_train.dtype)

        for i in range(num_test):
            instance = X_test[i]
            distances = np.sum(np.abs(self.X_train - instance), axis=1)
            min_index = np.argmin(distances)
            Y_predict[i] = self.Y_train[min_index]
        return Y_predict
