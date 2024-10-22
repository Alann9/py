import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
//
from matplotlib import pyplot as plt

if __name__ == '__main__':
    train_iter, test_iter = d2l.load_data_fashion_mnist(32)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break
    d2l.show_images(X.reshape(32, 28, 28), 2, 9, titles=d2l.get_fashion_mnist_labels(y))
    plt.show()
