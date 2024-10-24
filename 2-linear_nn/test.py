
from matplotlib import pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，
# 并除以255使得所有像素的数值均在0～1之间
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

print(mnist_train[0][1])

plt.axis('off')  # 隐藏坐标轴
plt.show()