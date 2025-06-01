# '''VGG11/13/16/19 in Pytorch.'''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Net_s(nn.Module):
#     def __init__(self):
#         super(Net_s, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         # print(x.shape)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
# class Net_m(nn.Module):
#     def __init__(self):
#         self.number = 0
#         super(Net_m, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
#         self.fc1 = nn.Linear(2*2*50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x, sign=0):
#         if sign == 0:
#             self.number += 1
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 2*2*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)
#
#     def get_number(self):
#         return self.number
#
#
# class Net_l(nn.Module):
#     def __init__(self):
#         super(Net_l, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
#         self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
#         self.fc1 = nn.Linear(50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv4(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net_m(nn.Module):
    def __init__(self):
        self.number = 0
        # 调用父类的构造函数
        super(Net_m, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2*2*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x, sign=0):
        # 如果sign等于0，则number加1
        if sign == 0:
            self.number += 1
        # 使用relu激活函数处理卷积层1的输出
        x = F.relu(self.conv1(x))
        # 使用最大池化层处理卷积层1的输出
        x = F.max_pool2d(x, 2, 2)
        # 使用relu激活函数处理卷积层2的输出
        x = F.relu(self.conv2(x))
        # 使用最大池化层处理卷积层2的输出
        # 使用relu激活函数处理卷积层3的输出
        x = F.max_pool2d(x, 2, 2)
        # 使用最大池化层处理卷积层3的输出
        x = F.relu(self.conv3(x))
        # 将卷积层的输出展平
        x = F.max_pool2d(x, 2, 2)
        # 使用relu激活函数处理全连接层1的输出
        x = x.view(-1, 2*2*50)
        # 使用全连接层2处理卷积层1的输出
        x = F.relu(self.fc1(x))
        # 返回对数概率分布
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        #输出类别，每个类别对应着一个值表示概率的大小

    def get_number(self):
        return self.number

#生成器网络，也就是图中的D
class Net_l(nn.Module):
    def __init__(self):
        super(Net_l, self).__init__()
        # 定义卷积层1
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        # 定义卷积层2
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # 定义卷积层3
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        # 定义卷积层4
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        # 定义全连接层1
        self.fc1 = nn.Linear(50, 500)
        # 定义全连接层2
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        # 通过卷积层1，激活函数为ReLU
        x = F.relu(self.conv1(x))
        # 通过最大池化层
        x = F.max_pool2d(x, 2, 2)
        # 通过卷积层2，激活函数为ReLU
        x = F.relu(self.conv2(x))
        # 通过最大池化层
        x = F.max_pool2d(x, 2, 2)
        # 通过卷积层3，激活函数为ReLU
        x = F.relu(self.conv3(x))
        # 通过最大池化层
        x = F.max_pool2d(x, 2, 2)
        # 通过卷积层4，激活函数为ReLU
        x = F.relu(self.conv4(x))
        # 通过最大池化层
        x = F.max_pool2d(x, 2, 2)
        # 将四维数据转换为二维数据
        x = x.view(-1, 50)
        # 通过全连接层1，激活函数为ReLU
        x = F.relu(self.fc1(x))
        # 通过全连接层2
        x = self.fc2(x)
        return x

