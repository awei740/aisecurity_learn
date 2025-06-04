
from __future__ import print_function
import argparse
import os
import gc
import sys
import xlwt
import random
import numpy as np
from advertorch.attacks import LinfBasicIterativeAttack, CarliniWagnerL2Attack
from advertorch.attacks import GradientSignAttack, PGDAttack
import foolbox
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp

from net import Net_s, Net_m, Net_l
SEED = 10000
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(10000)

# 创建一个参数解析器
parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--workers', type=int, help='number of data loading\
    workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--adv', type=str, help='attack method')
parser.add_argument('--mode', type=str, help='use which model to generate\
    examples. "imitation_large": the large imitation network.\
    "imitation_medium": the medium imitation network. "imitation_small" the\
    small imitation network. ')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', action='store_true', help='manual seed')

# 解析参数
opt = parser.parse_args()
# print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

# 设置随机数种子，保证结果可复现
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 启用CuDNN自动优化
cudnn.benchmark = True

# 如果存在可用的CUDA设备，并且没有使用-cuda选项，则输出警告信息
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with \
         --cuda")

# 导入torchvision库，用于加载MNIST数据集
testset = torchvision.datasets.MNIST(root='/data/dataset/', train=False,
                                     download=True,
                                     transform=transforms.Compose([
                                        transforms.ToTensor(),
                                     ]))

# 创建一个列表，包含0-9999的数字
data_list = [i for i in range(0, 10000)]
# 使用SubsetRandomSampler从testset中随机采样10000个样本
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         sampler = sp.SubsetRandomSampler(data_list), num_workers=2)


# 判断是否有GPU可用，如果有，则使用GPU，否则使用CPU
device = torch.device("cuda:0" if opt.cuda else "cpu")

# L2 = foolbox.distances.MeanAbsoluteDistance()

def test_adver(net, tar_net, attack, target):
#net：攻击生成网络（替代模型）
#tar_net：目标网络（被攻击模型）
#attack：攻击方法名称
#target：攻击目标类型：定向攻击：使目标模型将样本错误分类为特定类别。非定向攻击：只需使目标模型错误分类即可，不指定错误类别
    net.eval()
    tar_net.eval()
    # BIM
    if attack == 'BIM':
        adversary = LinfBasicIterativeAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.25,
            nb_iter=120, eps_iter=0.02, clip_min=0.0, clip_max=1.0,
            targeted=opt.target)
    # PGD
    elif attack == 'PGD':
        # 如果目标攻击
        if opt.target:
            # 定义PGD攻击器，使用交叉熵损失函数，最大扰动0.25，迭代次数11，每次迭代扰动步长0.03，最小值0.0，最大值1.0，目标攻击
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=0.25,
                nb_iter=11, eps_iter=0.03, clip_min=0.0, clip_max=1.0,
                targeted=opt.target)
        else:
            # 定义PGD攻击器，使用交叉熵损失函数，最大扰动0.25，迭代次数6，每次迭代扰动步长0.03，最小值0.0，最大值1.0，非目标攻击
            adversary = PGDAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=0.25,
                nb_iter=6, eps_iter=0.03, clip_min=0.0, clip_max=1.0,
                targeted=opt.target)
    # FGSM
    elif attack == 'FGSM':
        # 定义一个梯度符号攻击，使用交叉熵损失函数，eps=0.26， targeted根据opt.target决定是否为目标攻击
        adversary = GradientSignAttack(
            net,
            loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            eps=0.26,
            targeted=opt.target)
    elif attack == 'CW':
        adversary = CarliniWagnerL2Attack(
            net,
            num_classes=10,
            learning_rate=0.45,
            # loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            binary_search_steps=10,
            max_iterations=12,
            targeted=opt.target)

    # ----------------------------------
    # Obtain the accuracy of the model
    # ----------------------------------

# 设置不计算梯度
    with torch.no_grad():
        # 初始化正确率
        correct_netD = 0.0
        # 初始化总数
        total = 0.0
        # 设置模型为评估模式
        net.eval()
        # 遍历测试数据
        for data in testloader:
            # 获取输入和标签
            inputs, labels = data
            # 将输入数据转移到GPU
            inputs = inputs.cuda()
            labels = labels.cuda()
            # 运行模型
            outputs = net(inputs)
            # 获取最大值和位置
            _, predicted = torch.max(outputs.data, 1)
            # 计算总数
            total += labels.size(0)
            # 计算正确率
            correct_netD += (predicted == labels).sum()
        # 打印正确率
        print('Accuracy of the network on netD: %.2f %%' %
                (100. * correct_netD.float() / total))

    # ----------------------------------
    # Obtain the attack success rate of the model
    # ----------------------------------

    # 初始化正确率correct和总样本数total
    correct = 0.0
    total = 0.0
    # 将目标网络设置为评估模式
    tar_net.eval()
    # 初始化总的L2距离
    total_L2_distance = 0.0
    # 遍历测试数据集
    for data in testloader:
        # 获取输入数据和标签
        inputs, labels = data
        # 将输入数据和标签转移到设备上
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 运行目标网络
        outputs = tar_net(inputs)
        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)

        #我们以一个具体的例子来解释定向攻击：假设有一个图像分类模型，可以将手写数字图片分类为0-9。现在我们有一张图片，内容是数字“7”，模型当前将其正确分类为“7”。
        #非定向攻击（Untargeted Attack）：我们只希望模型将这张图片分类错误，不管错误分类成什么（比如分类成1,2,3,...,9都可以），只要不是正确的“7”就算攻击成功
        #定向攻击（Targeted Attack）：我们希望模型将这张图片分类为一个特定的错误类别，例如“2”。也就是说，不管模型原本要分类成什么（可能是7，也可能是其他），攻击后必须让模型分类成“2”才算成功
        if target:
            labels = torch.randint(0, 9, (1,)).to(device)
            if predicted != labels:
                # print(total)
                # 计算L2距离
                adv_inputs_ori = adversary.perturb(inputs, labels)
                L2_distance = (torch.norm(adv_inputs_ori - inputs)).item()
                total_L2_distance += L2_distance
                with torch.no_grad():
                    # 计算目标模型在对抗样本上的输出
                    outputs = tar_net(adv_inputs_ori)
                    _, predicted = torch.max(outputs.data, 1)
                    # 计算正确率
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
        else:
            # test the images which are classified correctly
            if predicted == labels:
                # print(total)
                # 计算对抗样本
                adv_inputs_ori = adversary.perturb(inputs, labels)
                # 计算L2范数
                L2_distance = (torch.norm(adv_inputs_ori - inputs)).item()
                # 累加L2范数
                total_L2_distance += L2_distance
                # 不更新梯度
                with torch.no_grad():
                    # 计算对抗样本的输出
                    outputs = tar_net(adv_inputs_ori)
                    # 获取预测结果
                    _, predicted = torch.max(outputs.data, 1)

                    # 累加总数
                    total += labels.size(0)
                    # 累加正确数
                    correct += (predicted == labels).sum()

# 如果target不为空
    if target:
    # 打印攻击成功率
        print('Attack success rate: %.2f %%' %
              (100. * correct.float() / total))
    else:
    # 打印攻击成功率
        print('Attack success rate: %.2f %%' %
              (100.0 - 100. * correct.float() / total))
# 打印l2距离
    print('l2 distance:  %.4f ' % (total_L2_distance / total))


    target_net = Net_m().to(device)
    # 加载预训练的模型参数
    state_dict = torch.load(
        'pretrained/net_m.pth', map_location=device)  # 使用 map_location=device
    target_net.load_state_dict(state_dict)
    # 将模型设置为评估模式
    target_net.eval()

    if opt.mode == 'black':
        # 加载攻击模型
        attack_net = Net_l().to(device)
        state_dict = torch.load(
            'pretrained/net_l.pth', map_location=device)  # 使用 map_location=device
        attack_net.load_state_dict(state_dict)
    elif opt.mode == 'white':
        # 使用目标模型作为攻击模型
        attack_net = target_net
    elif opt.mode == 'dast':
        # 加载攻击模型
        attack_net = Net_l().to(device)
        state_dict = torch.load(
            'netD_epoch_670.pth', map_location=device)  # 使用 map_location=device
        attack_net = nn.DataParallel(attack_net)
        attack_net.load_state_dict(state_dict)
        #加载训练好的替代模型D

    test_adver(attack_net, target_net, opt.adv, opt.target)
    #这里的attack_net只是替代模型D，是已经训练好的。

