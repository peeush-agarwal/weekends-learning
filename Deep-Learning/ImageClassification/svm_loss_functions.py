import numpy as np

def calculate_loss_for_an_image(x, y, W):
    """
    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
        with an appended bias dimension in the 3073-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)
    """
    delta = 1.0
    predicted_scores = np.dot(W, x)
    class_score = predicted_scores[y]
    num_of_classes = W.shape[0]
    loss_i = 0 # Loss across all classes except correct class
    for j in range(num_of_classes):
        if j == y:
            continue
        loss_i += max(0, predicted_scores[j] - class_score + delta)
    return loss_i

def calculate_loss_for_an_image_vectorized(x, y, W):
    """
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0
    predicted_scores = np.dot(W, x)
    class_score = predicted_scores[y]
    margins = np.maximum(0, predicted_scores - class_score + delta)
    margins[y] = 0 # Ignore delta for correct class
    loss = np.sum(margins)
    return loss

def calculate_loss_for_all_images(X, Y, W):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)

    For example:
    - np.dot(W, X): 3 classes and 4 images
            I1  I2  I3  I4
        Cat 12  2   -9  4
        Dog 3   15  -5  14
        Hor -8  -1  13  1
    - y:
            I1  I2  I3  I4
            0   1   2   1    
    """
    delta = 1.0
    num_train = X.shape[1]
    scores = np.dot(W, X) # 10 x 50000
    class_score = scores[Y, np.arange(num_train)].reshape(1, num_train)
    margins = np.maximum(0, scores - class_score + delta)
    margins[Y, np.arange(num_train)] = 0    # Ignore delta for correct class
    loss = np.sum(margins)
    return loss