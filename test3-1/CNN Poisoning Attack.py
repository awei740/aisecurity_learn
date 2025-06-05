# 导入第三方库
import torch
import random
import torch.nn as nn
#导入nn模块包含了构建神经网络所需的所有组件，如层，激活函数等
import torch.nn.functional as F
#从nn模块导入functional模块，重命名F，包含许多激活函数和损失函数，如relu和softmax
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
#datasets包含多个常用的数据集加载器，如MNIST
from torch.utils.data import DataLoader
#DataLoader封装了数据集的批处理，打乱，多线程扽加载功能
from torch.utils.data import Subset
#Subset 是一个数据集划分工具，用于从一个完整的数据集中创建指定样本索引的子集。它属于PyTorch数据工具包的一部分，常用于数据分割（如训练集/验证集分割）或采样特定样本

# 定义 AlexNet 的结构
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
    # 由于 MNIST 为 28x28，而最初 AlexNet 的输入图片是 227x227 的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256 * 3 * 3, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)
        return x

#划分数据集，使用原数据集的 1/10
def select_subset(dataset, ratio=1/10):
    subset_size = int(len(dataset) * ratio)
    indices = np.random.choice(range(len(dataset)), subset_size, replace=False)
    #random.choice函数从range(len(dataset))中随机选择subset_size个不重复的索引，replace=False表示不会重复选择同一数据项
    return Subset(dataset, indices)
    #返回一个Subset类的实例，封装了dataset和选出的索引indices


# 展示正确分类的图片
def plot_correctly_classified_images(model, dataset, device, num_images=10):
#model：待评估的模型。dataset：包含图像和标签的数据集。device：cpu还是cuda。num_images：表示希望展示多少个正确分类的图像
    model.eval()
    correctly_classified_imgs = []
    #初始化一个空列表，存储正确分类的图像，真实标签和预测标签
    for img, label in dataset:
    #遍历数据集上的每个数据和标签
        img = img.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        #图像转换为FloatTensor类型，并添加两个维度匹配模型期望的输入形状，并将图像和标签转移到指定的device
        with torch.no_grad():
            pred = model(img)
            #模型进行预测
        pred_label = torch.argmax(pred).item()
        #torch.argmax获取预测概率最高的索引作为预测标签
        if pred_label == label:
        #相等表示模型正确分类
            correctly_classified_imgs.append((img.cpu().squeeze(), label, pred_label))
            #将图像去除之前添加的维度，与真实标签和预测标签填入列表中
            if len(correctly_classified_imgs) >= num_images:
            #已收集到的正确分类图像达到了指定的数量
                break


    plt.figure(figsize=(10, 10))
    #创建一个新的图形窗口， 设置图形大小为宽度10英寸，高度10英寸
    for i, (img, true_label, pred_label) in enumerate(correctly_classified_imgs):
    #循环遍历被正确分类的样本，enumerate() 函数在遍历时同时提供索引 i（从0开始）和元素值
        plt.subplot(5, 2, i + 1)
        #创建子图并设置当前轴域，将整个图形区域划分成5行2列（共10个子区域），i+1指定当前要操作的子图编号（编号从1到10）
        plt.imshow(img.numpy(), cmap='gray')
        #img.numpy() 将PyTorch张量转换为NumPy数组（Matplotlib需要此格式），imshow() 是专门用于显示图像的函数，cmap='gray' 指定使用灰度色彩映射（适合黑白/灰度图像）
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        #为当前子图添加标题
        plt.axis('off')
        #axis('off') 移除当前子图的坐标轴刻度线和标签
    plt.tight_layout()
    #tight_layout() 自动调整子图的间距和位置
    plt.show()


# 展示错误分类的图片
def plot_misclassified_images(model, dataset, device, num_images=10):
    model.eval()
    misclassified_imgs = []
    for img, label in dataset:
        img = img.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)
        pred_label = torch.argmax(pred).item()
        if pred_label != label:
            misclassified_imgs.append((img.cpu().squeeze(), label, pred_label))
            if len(misclassified_imgs) >= num_images:
                break
    plt.figure(figsize=(10, 10))
    for i, (img, true_label, pred_label) in enumerate(misclassified_imgs):
        plt.subplot(5, 2, i + 1)
        plt.imshow(img.numpy(), cmap='gray')
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()



def fetch_datasets(full_dataset, trainset, ratio):
#投毒攻击函数目的是从一个完整的数据集(full_dataset)中根据给定的训练集索引(trainset：一个包含训练集索引的对象)，和投毒比例(ratio)来分割出一个投毒训练集(poison_trainset)和一个干净的训练集(clean_trainset)

    character = [[] for i in range(len(full_dataset.classes))]
    # 创建按类别分组的数据结构，用来存储对应类别的图像

    for index in trainset.indices:
    # 遍历训练集索引，将每个样本按类别分组
        img, label = full_dataset[index]  # 获取原始数据集中的样本和标签
        character[label].append(img)  # 按类别存储图像，即图像根据标签添加到character列表对应类比的子列表中

    # 初始化投毒样本集和干净样本集
    poison_trainset = []
    clean_trainset = []
    target = 0  # 临时存储投毒样本的目标标签

    # 遍历每个类别的数据集
    for i, data in enumerate(character):

        num_poison_train_inputs = int(len(data) * ratio[0])
        # 计算当前类别中投毒样本的数量

        # 处理投毒样本（使用数据的前半部分）
        for img in data[:num_poison_train_inputs]:

            # 为投毒样本随机分配新标签（目标标签）
            target = random.randint(a=0, b=9)  # i 是当前样本的原始标签

            # 转换图像格式并归一化
            poison_img = torch.from_numpy(np.array(img) / 255.0)

            # 添加到投毒训练集
            poison_trainset.append((poison_img, target))

        # 处理干净样本（使用数据的后半部分）
        for img in data[num_poison_train_inputs:]:

            # 转换图像格式并归一化
            img = np.array(img)

            img = torch.from_numpy(img / 255.0)

            # 添加到干净训练集，保留原始标签
            clean_trainset.append((img, i))

    # 创建结果字典
    result_datasets = {}
    result_datasets['poisonTrain'] = poison_trainset  # 投毒训练集
    result_datasets['cleanTrain'] = clean_trainset  # 干净训练集

    return result_datasets

#1111
