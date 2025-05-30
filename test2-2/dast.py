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
cudnn.benchmark = True
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('imitation_network_sig')
nz = 128

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')       
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


sys.stdout = Logger('imitation_network_model.log', sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
parser.add_argument('--dataset', type=str, default='azure')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--beta', type=float, default=0.1, help='alpha')
parser.add_argument('--G_type', type=int, default=1, help='iteration limitation')
parser.add_argument('--save_folder', type=str, default='saved_model', help='alpha')

opt = parser.parse_args()
print(opt)

# 如果存在可用的CUDA设备，并且没有使用-cuda选项，则打印警告信息
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# 如果使用的是azure数据集
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
    # 定义生成器网络，并将模型放到GPU上
    netD = Net_l().cuda()
    # 将模型放到多GPU上
    netD = nn.DataParallel(netD)

    clf = joblib.load('pretrained/sklearn_mnist_model.pkl')

# 加载预训练的模型
    adversary_ghost = LinfBasicIterativeAttack(
        netD, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=100, eps_iter=0.01, clip_min=0.0, clip_max=1.0,
        targeted=False)
    # 定义攻击方法，LinfBasicIterativeAttack为一种攻击方法，用于生成对抗样本
    # netD为模型，loss_fn为损失函数，eps为最大扰动，nb_iter为迭代次数，eps_iter为每次迭代扰动大小，clip_min为最小值，clip_max为最大值，targeted为是否 targeted攻击
    nc=1

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
    netD = nn.DataParallel(netD)

    # 加载预训练的Net_m模型
    original_net = Net_m().cuda()
    state_dict = torch.load(
        'pretrained/net_m.pth')
    original_net.load_state_dict(state_dict)
    original_net = nn.DataParallel(original_net)
    original_net.eval()

    # 定义LinfBasicIterativeAttack攻击器
    adversary_ghost = LinfBasicIterativeAttack(
        netD, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.25,
        nb_iter=200, eps_iter=0.02, clip_min=0.0, clip_max=1.0,
        targeted=False)
    nc=1

data_list = [i for i in range(6000, 8000)] # fast validation
testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)
# nc=1

device = torch.device("cuda:0" if opt.cuda else "cpu")

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

class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64
        if opt.G_type == 1:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
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
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif opt.G_type == 2:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
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
netG.apply(weights_init)
netG = nn.DataParallel(netG)

criterion = nn.CrossEntropyLoss()
criterion_max = Loss_max()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerD =  optim.SGD(netD.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG =  optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

with torch.no_grad():
    correct_netD = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # outputs = netD(inputs)
        if opt.dataset == 'azure':
            predicted = cal_azure(clf, inputs)
        else:
            outputs = original_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
        # _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_netD += (predicted == labels).sum()
    print('Accuracy of the network on netD: %.2f %%' %
            (100. * correct_netD.float() / total))


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

