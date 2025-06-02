#!!!!!!!!!!!!!!!!!!!!!!!语句在上 注释解析在下！！！！！！！！！！！！！！！！！！！！！！
from __future__ import print_function
import argparse
import os
import math
import gc
import sys
import xlwt
import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack
# from sklearn.externals import joblib
import joblib
# from utils import load_data
import pickle
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
from net import Net_s, Net_m, Net_l
from vgg import VGG
from resnet import ResNet50, ResNet18, ResNet34



#！！！！！！！！！！！！！！！！！！！！！！！！！初始化和配置！！！！！！！！！！！！！！！！！！！！！！！！！！
cudnn.benchmark = True
# cuDNN是 NVIDIA 提供的深度学习加速库，提供高性能实现：卷积、池化、归一化、激活函数等，PyTorch、TensorFlow 等框架底层都使用 cuDNN
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('imitation_network_sig')
#创建Excel文件用于记录训练结果，xlwt库用于生成Excel文件，创建名为"imitation_network_sig"的工作表
nz = 128
#定义生成器输入的噪声向量维度
class Logger(object):# 日志记录类
    def __init__(self, filename='default.log', stream=sys.stdout):# 初始化日志文件
    #filename：日志文件名，默认 'default.log'，stream：输出流对象，默认 sys.stdout（标准输出/控制台）
        self.terminal = stream
        self.log = open(filename, 'a')
        #以追加模式('a')打开日志文件，'a' 模式：追加写入，不覆盖已有内容
    def write(self, message): #同时写入终端和文件
        self.terminal.write(message)
        self.log.write(message)
    def flush(self): # 空方法满足接口要求
        pass


sys.stdout = Logger('imitation_network_model.log', sys.stdout)
# 重定向所有print输出，控制台输出的内容会同时保存到'imitation_network_model.log'文件中

parser = argparse.ArgumentParser()
# 创建参数解析器对象
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
# 控制数据加载的并行程度
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
#定义每次训练迭代使用的样本数量
parser.add_argument('--dataset', type=str, default='azure')
#指定使用的数据集，当运行程序时不指定--dataset 参数，程序会自动使用 'azure' 作为数据集名称
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
#设置训练的总轮数（epochs）
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
#控制模型参数更新的步长
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#设置Adam优化器的一阶矩估计指数衰减率
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
#控制是否使用GPU加速
parser.add_argument('--manualSeed', type=int, help='manual seed')
#设置随机数生成器的种子
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
#自定义超参数（可能是损失函数权重）
parser.add_argument('--beta', type=float, default=0.1, help='alpha')
#另一个自定义超参数
parser.add_argument('--G_type', type=int, default=1, help='iteration limitation')
#：选择生成器网络架构（GAN上下文）
parser.add_argument('--save_folder', type=str, default='saved_model', help='alpha')
#设置模型保存目录

opt = parser.parse_args()
# 解析命令行参数
print(opt)
#打印所有参数配置

# 如果存在可用的CUDA设备，并且没有使用-cuda选项，则打印警告信息
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

# ！！！！！！！！！！！！！！！！！！！！！！！！数据加载部分！！！！！！！！！！！！！！！！！！！！！！
# 如果使用的是azure数据集，这里是黑盒攻击！！！！！
if opt.dataset == 'azure':
    # 加载MNIST数据集，root参数指定数据集路径，train参数指定是否是训练集，download参数指定是否下载数据集，transform参数指定数据预处理
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                # transforms.Pad(2, padding_mode="symmetric"),
                                                # 将图片转换为tensor
                                                transforms.ToTensor(),
                                                # transforms.RandomCrop(32, 4),
                                                # normalize,
                                        ]))
#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
#！！！！！！！！！！！！！！！初始化网络模型！！！！！！！！！！！！！！！！！
    # 定义生成器网络，并将模型放到GPU上，这里的Net_l是替代模型D
    netD = Net_l().cuda()
    # 将模型放到多GPU上
    netD = nn.DataParallel(netD)

    clf = joblib.load('pretrained/sklearn_mnist_model.pkl')
    #从文件加载预训练的scikit - learn模型
    # 也就是图中被攻击模型T，这里的T我们不知道模型结构和参数，只能通过将原始数据输入到目标模型后获取到目标模型的输出结果，并根据结果来进行攻击，也就是黑盒攻击
#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

#！！！！！！！！！！！！！！！！！对抗性攻击设置！！！！！！！！！！！！！！！
# 加载预训练的模型
    adversary_ghost = LinfBasicIterativeAttack(
        netD,
        # 替代模型D，作为攻击对象，生成器G将学习欺骗该模型D，使用此模型的梯度信息生成对抗样本
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        # reduction="sum"：所有样本的损失值求和，量化模型预测与真实标签的差异
        eps=0.25,
        # 最大扰动范围：像素值最大变化±0.25，原像素0.5 → 可变化范围[0.25,0.75]
        nb_iter=200,
        # 攻击迭代次数，通过多次调整使攻击更可靠
        eps_iter=0.02,
        # 单步扰动大小，每次迭代改变的最大幅度
        clip_min=0.0,
        clip_max=1.0,
        # 像素值裁剪范围，确保对抗样本是有效图像，避免生成非法像素值
        targeted=False)
    # 攻击类型：非定向攻击，targeted=True：让模型将"3"识别为特定错误如"8"，targeted=False：只要识别错误即可（任意错误都行）
    # 定义攻击方法，LinfBasicIterativeAttack为一种攻击方法，用于生成对抗样本
    nc=1
    #表示通道数，nc=3是RGB彩色通道，nc=1表示灰度图像，即黑白图
#！！！！！！！！！！！！！！！！！！！！！！
# 这里是白盒攻击，框架类似于上面的黑盒
elif opt.dataset == 'mnist':
    # 加载MNIST数据集，并将其转换为tensor
    testset = torchvision.datasets.MNIST(root='dataset/', train=False,
                                        download=True,
                                        transform=transforms.Compose([
                                                # transforms.Pad(2, padding_mode="symmetric"),
                                                transforms.ToTensor(),
                                                # transforms.RandomCrop(32, 4),
                                                # normalize,
                                        ]))
    # 加载预训练的Net_l模型
    netD = Net_l().cuda()
    # 将模型放到多GPU上
    netD = nn.DataParallel(netD)

    # 加载预训练的Net_m模型，也就是被攻击模型T，我们知道模型的完整结构和参数，也就是白盒攻击
    original_net = Net_m().cuda()
    #创建模型实例并加载到GPU

    state_dict = torch.load(
        'pretrained/net_m.pth')
    #加载预训练权重文件，
    # torch.load()：加载PyTorch模型权重文件
    #'pretrained/net_m.pth'：预训练权重文件路径
    #state_dict：获取模型的权重字典，包含所有参数：权重(weights)、偏置(biases)等

    original_net.load_state_dict(state_dict)
    # 将权重加载到模型
    original_net = nn.DataParallel(original_net)
    #nn.DataParallel：将模型包装为多GPU并行版本
    original_net.eval()
    #.eval()：切换模型到评估模式：确保评估结果稳定一致，不随批次变化
        #禁用Dropout层：不使用随机神经元丢弃
        #固定BatchNorm：使用训练阶段的统计量，而非当前批次
        #其他层：如Detach等特殊层也会调整行为

    # 定义LinfBasicIterativeAttack攻击器
    adversary_ghost = LinfBasicIterativeAttack(
        netD,
        #替代模型D，作为攻击对象，生成器G将学习欺骗该模型D，使用此模型的梯度信息生成对抗样本
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        #reduction="sum"：所有样本的损失值求和，量化模型预测与真实标签的差异
        eps=0.25,
        #最大扰动范围：像素值最大变化±0.25，原像素0.5 → 可变化范围[0.25,0.75]
        nb_iter=200,
        # 攻击迭代次数，通过多次调整使攻击更可靠
        eps_iter=0.02,
        #单步扰动大小，每次迭代改变的最大幅度
        clip_min=0.0,
        clip_max=1.0,
        #像素值裁剪范围，确保对抗样本是有效图像，避免生成非法像素值
        targeted=False)
        #攻击类型：非定向攻击，targeted=True：让模型将"3"识别为特定错误如"8"，targeted=False：只要识别错误即可（任意错误都行）
    # 定义攻击方法，LinfBasicIterativeAttack为一种攻击方法，用于生成对抗样本
    nc=1
    #表示通道数，nc=3是RGB彩色通道，nc=1表示灰度图像，即黑白图
data_list = [i for i in range(6000, 8000)] # fast validation
#选择测试集的子集进行快速验证，使用20%的样本(2000个)进行快速验证

testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)
#创建测试数据加载器，testset完整测试集，batch_size=500每批加载500个样本，sampler = sp.SubsetRandomSampler(data_list)自定义采样器，num_workers=2使用2个子进程加载数据
# nc=1

device = torch.device("cuda:0" if opt.cuda else "cpu")
#根据配置选择计算设备

# 定义一个函数weights_init，用于初始化网络权重
def weights_init(m):
    # 获取类的名称
    classname = m.__class__.__name__
    # 如果类的名称中包含'Conv'，则权重数据服从正态分布，均值为0，标准差为0.02
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # 如果类的名称中包含'BatchNorm'，则权重数据服从正态分布，均值为1，标准差为0.02，偏置数据全部填充为0
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def cal_azure(model, data):
    # 将输入数据转换为784维的向量，并将其从CPU转换为numpy数组
    data = data.view(data.size(0), 784).cpu().numpy()

    # 使用模型进行预测
    output = model.predict(data)

    # 将预测结果从numpy数组转换为torch张量，并将其从CPU转换为GPU
    output = torch.from_numpy(output).cuda().long()
    return output

def cal_azure_proba(model, data):
    # 将输入数据转换为numpy数组
    data = data.view(data.size(0), 784).cpu().numpy()
    # 使用模型预测概率
    output = model.predict_proba(data)
    # 将预测结果转换为tensor并放到GPU上
    output = torch.from_numpy(output).cuda().float()
    return output


class Loss_max(nn.Module):
    def __init__(self):
        super(Loss_max, self).__init__()
        return

    def forward(self, pred, truth, proba):
        # 定义损失函数
        criterion_1 = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        # 计算预测概率
        pred_prob = F.softmax(pred, dim=1)
        loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * opt.beta
        # loss = criterion(pred, truth)
        final_loss = torch.exp(loss * -1)
        return final_loss

#对应于图中的反卷积层，负责初步特征生成和上采样（从低维噪声到初步特征）
class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64
        if opt.G_type == 1:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                #输入通道：nz（噪声向量的维度）
                #输出通道：self.nf * 2
                #卷积核：3×3
                #步长：1
                #填充：1
                #偏置：无
                nn.BatchNorm2d(self.nf * 2),
                #对128通道的特征图进行归一化
                nn.LeakyReLU(0.2, inplace=True),
                #特征激活模块

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif opt.G_type == 2:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                #round((self.shape[0]-1)/2),  # 动态计算填充值
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),  # added

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.ReLU(True),

                nn.Conv2d(self.nf, self.shape[0], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.shape[0]),
                nn.ReLU(True),

                nn.Conv2d(self.shape[0], self.shape[0], 3, 1, 1, bias=False),
                # if self.shape[0] == 3:
                #     nn.Tanh()
                # else:
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.pre_conv(input)
        return output

# 创建一个空列表pre_conv_block
pre_conv_block = []
# 遍历10次
for i in range (10):
    # 将pre_conv函数的结果添加到pre_conv_block列表中，并使用DataParallel函数，并将其放在GPU上
    pre_conv_block.append(nn.DataParallel(pre_conv(10).cuda()))

#对应于图中的卷积层，图中卷积层负责精炼特征和生成最终图像
class Generator(nn.Module):
    def __init__(self, num_class):
        super(Generator, self).__init__()
        self.nf = 64
        self.num_class = num_class
        if opt.G_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 4),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(nc),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
                nn.Sigmoid()
            )
        elif opt.G_type == 2:
            self.main = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True)
            )
    def forward(self, input):
        output = self.main(input)
        return output


def chunks(arr, m):
    # 计算数组arr的长度除以m向上取整的结果
    n = int(math.ceil(arr.size(0) / float(m)))
    # 按照n的大小将数组arr分割成m个块
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]

netG = Generator(10).cuda()
#创建生成器（也就是图中模型G）实例并转移到GPU，Generator(10)表示有10个类别分支（对应MNIST的10个数字），在图中对应多分支反卷积架构（每个类别一个分支）.cuda()：将模型加载到GPU显存
netG.apply(weights_init)
# 权重初始化
netG = nn.DataParallel(netG)
#多GPU并行处理

#！！！！！！！！！！！！！！！！！！！！！训练和评估循环！！！！！！！！！！！！！！！！！！
criterion = nn.CrossEntropyLoss()
#使用PyTorch内置的交叉熵损失函数
criterion_max = Loss_max()
#自定义的损失函数

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#判别器优化器，lr=opt.lr：从配置参数获取学习率。betas=(opt.beta1, 0.999)：Adam的动量参数
# optimizerD =  optim.SGD(netD.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#生成器优化器
# optimizerG =  optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))
#卷积块优化器初始化，为10个预定义的卷积块（pre_conv_block）分别创建独立的Adam优化器，每个优化器管理对应卷积块的参数更新
#lr=opt.lr,        # 学习率
#betas=(opt.beta1, 0.999)  # Adam的超参数

with torch.no_grad():
# 禁用自动梯度计算（评估模式-验证），不需要学习机制，只做纯推理
    correct_netD = 0.0
    # 初始化正确预测计数器（浮点数）
    total = 0.0
    #初始化总样本计数器（浮点数）
    netD.eval()
    #设置替代模型D为评估模式--验证模式，
    # 在训练模式时：Dropout层随机丢弃部分神经元，BatchNorm层使用当前批次统计量
    #评估模式下：Dropout层保留所有神经元，BatchNorm层使用使用训练集的总体统计量
    for data in testloader:
        inputs, labels = data
        #inputs: 输入图像张量
        #labels: 对应标签张量
        inputs = inputs.cuda()
        #输入数据转移到GPU
        labels = labels.cuda()
        # outputs = netD(inputs)
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, inputs)
            #使用黑盒的被攻击模型进行预测
        else:
            outputs = original_net(inputs)
            #使用白盒的被攻击模型示例 outputs 结构：
            #outputs结构示例：tensor([[ 1.223, -0.456,  0.789, ...,  2.345],
            _, predicted = torch.max(outputs.data, 1)
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #累加样本总数
        correct_netD += (predicted == labels).sum()
        #统计正确预测数量
    print('Accuracy of the network on netD: %.2f %%' %
            (100. * correct_netD.float() / total))
    #计算并打印最终准确率，这里输出的是被攻击模型T的准确率
#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！


# 初始化正确率
correct_ghost = 0.0
# 初始化总数
total = 0.0
# 将netD设置为评估模式
netD.eval()
# 遍历测试数据
for data in testloader:
    # 获取输入和标签
    inputs, labels = data
    # 将输入转移到GPU
    inputs = inputs.cuda()
    labels = labels.cuda()

    # 计算对抗样本
    adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
    # 不计算梯度
    with torch.no_grad():
        # 如果是azure数据集，则计算azure结果
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, adv_inputs_ghost)
            #调用自定义函数进行Azure特定预测
        else:
            # 否则计算原始网络的结果
            outputs = original_net(adv_inputs_ghost)
            # 获取最大值和位置
            _, predicted = torch.max(outputs.data, 1)
    # 累加总数
    total += labels.size(0)
    # 累加正确率
    correct_ghost += (predicted == labels).sum()
# 打印攻击成功率
print('Attack success rate: %.2f %%' %
        (100 - 100. * correct_ghost.float() / total))
# 释放内存
del inputs, labels, adv_inputs_ghost
torch.cuda.empty_cache()
gc.collect()

batch_num = 1000
best_accuracy = 0.0
best_att = 0.0
for epoch in range(opt.niter):
    netD.train()

    for ii in range(batch_num):
        netD.zero_grad()

        ############################
        # (1) Update D network:
        ###########################
        noise = torch.randn(opt.batchSize, nz, 1, 1, device=device).cuda()
        noise_chunk = chunks(noise, 10)
# 遍历noise_chunk
        for i in range(len(noise_chunk)):
    # 调用pre_conv_block[i]函数处理noise_chunk[i]
            tmp_data = pre_conv_block[i](noise_chunk[i])
    # 调用netG函数生成gene_data
            gene_data = netG(tmp_data)
            # gene_data = netG(noise_chunk[i], i)
    # 创建一个label，大小为noise_chunk[i]的批次大小，值都为i
            label = torch.full((noise_chunk[i].size(0),), i).cuda()
    # 如果i为0，则data等于gene_data，set_label等于label
            if i == 0:
                data = gene_data
                set_label = label
    # 否则，将data和gene_data拼接，set_label和label拼接
            else:
                data = torch.cat((data, gene_data), 0)
                set_label = torch.cat((set_label, label), 0)
        index = torch.randperm(set_label.size()[0])
        data = data[index]
        set_label = set_label[index]

        
        # 计算测试集的预测结果
        with torch.no_grad():
           
            # 如果是azure数据集，则计算azure概率
            if opt.dataset == 'azure':
                outputs = cal_azure_proba(clf, data)
                label = cal_azure(clf, data)
            else:
                # 如果是其他数据集，则使用原始网络计算预测结果
                outputs = original_net(data)
                _, label = torch.max(outputs.data, 1)
                outputs = F.softmax(outputs, dim=1)
          
        # print(label)

        output = netD(data.detach())
        prob = F.softmax(output, dim=1)
        # print(torch.sum(outputs) / 500.)
        errD_prob = mse_loss(prob, outputs, reduction='mean')
        errD_fake = criterion(output, label) + errD_prob * opt.beta
        D_G_z1 = errD_fake.mean().item()
        errD_fake.backward()

        errD = errD_fake
        optimizerD.step()

        del output, errD_fake


        netG.zero_grad()
        for i in range(10):
            pre_conv_block[i].zero_grad()
        # 计算netD的输出
        output = netD(data)
        # 计算模仿损失
        loss_imitate = criterion_max(pred=output, truth=label, proba=outputs)
        # 计算多样性损失
        loss_diversity = criterion(output, set_label.squeeze().long())
        # 计算总的生成器损失
        errG = opt.alpha * loss_diversity + loss_imitate
        # 如果多样性损失小于0.1，则更新alpha
        if loss_diversity.item() <= 0.1:
            opt.alpha = loss_diversity.item()
        # 反向传播计算梯度
        errG.backward()
        # 计算生成器输出
        D_G_z2 = errG.mean().item()
        # 更新生成器参数
        optimizerG.step()
        for i in range(10):
            optimizer_block[i].step()

        # 每40次迭代打印一次损失
        if (ii % 40) == 0:
            print('[%d/%d][%d/%d] D: %.4f D_prob: %.4f G: %.4f D(G(z)): %.4f / %.4f loss_imitate: %.4f loss_diversity: %.4f'
                % (epoch, opt.niter, ii, batch_num,
                    errD.item(), errD_prob.item(), errG.item(), D_G_z1, D_G_z2, loss_imitate.item(), loss_diversity.item()))


    # 初始化最优攻击成功率
    best_att = 0.0
    # 初始化正确 ghosts
    correct_ghost = 0.0
    # 初始化总数
    total = 0.0
    # 将 netD 设置为评估模式
    netD.eval()
    # 遍历测试数据
    for data in testloader:
        # 获取输入和标签
        inputs, labels = data
        # 将输入转移到 GPU 上
        inputs = inputs.cuda()
        labels = labels.cuda()

        # 获取 adversary_ghost 攻击的输入
        adv_inputs_ghost = adversary_ghost.perturb(inputs, labels)
        # 不计算梯度
        with torch.no_grad():
            # 计算 outputs
            # outputs = original_net(adv_inputs_ghost)
            if opt.dataset == 'azure':
                predicted = cal_azure(clf, adv_inputs_ghost)
            else:
                outputs = original_net(adv_inputs_ghost)
                _, predicted = torch.max(outputs.data, 1)
            # _, predicted = torch.max(outputs.data, 1)

            # 更新总数
            total += labels.size(0)
            # 更新正确 ghosts
            correct_ghost += (predicted == labels).sum()
    # 打印攻击成功率
    print('Attack success rate: %.2f %%' %
            (100 - 100. * correct_ghost.float() / total))
    # 如果最优攻击成功率小于当前攻击成功率，则更新最优攻击成功率，并保存模型
    if best_att < (total - correct_ghost):
        torch.save(netD.state_dict(),
                    opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
        torch.save(netG.state_dict(),
                    opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
        best_att = (total - correct_ghost)
        print('This is the best model')
    # 将正确 ghosts 写入表格
    worksheet.write(epoch, 0, (correct_ghost.float() / total).item())
    # 删除输入、标签和 adv_inputs_ghost
    del inputs, labels, adv_inputs_ghost
    # 清空 GPU 缓存
    torch.cuda.empty_cache()
    # 清理内存
    gc.collect()

#！！！！！！！！！！！！！！！结果保存和输出！！！！！！！！！！！！！！！！！！！！
# 使用torch.no_grad()禁用梯度计算，提高性能
    with torch.no_grad():
        # 初始化正确率
        correct_netD = 0.0
        total = 0.0
        # 将netD设置为评估模式
        netD.eval()
        # 遍历测试数据
        for data in testloader:
            # 获取输入和标签
            inputs, labels = data
            # 将输入数据转移到GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            # 使用netD进行前向传播
            outputs = netD(inputs)
            # 获取最大值和对应的索引
            _, predicted = torch.max(outputs.data, 1)
            # 计算总样本数
            total += labels.size(0)
            # 计算正确样本数
            correct_netD += (predicted == labels).sum()
        # 打印netD的准确率
        print('Accuracy of the network on netD: %.2f %%' %
                (100. * correct_netD.float() / total))
        # 如果当前准确率比最佳准确率更高，则保存模型，并更新最佳准确率
        if best_accuracy < correct_netD:
            torch.save(netD.state_dict(),
                       opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
            best_accuracy = correct_netD
            print('This is the best model')
    worksheet.write(epoch, 1, (correct_netD.float() / total).item())
workbook.save('imitation_network_saved_azure.xls')
#！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
