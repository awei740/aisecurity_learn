'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.number = 0
        self.features = self._make_layers(cfg[vgg_name])
        # self.SELayer_V2 = SELayer_V2(channel=512)
        # 定义一个分类器，包括dropout、全连接层、ReLU激活函数、dropout和全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def forward(self, x, sign=0):
        if sign == 0:
            self.number += 1
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def get_number(self):
        return self.number

    def _make_layers(self, cfg):
        layers = []
        # 初始化输入通道数为3
        in_channels = 3
        # 计数器，用于记录出现了多少次'M'
        count = 0
        # 遍历cfg列表
        for x in cfg:
            # 如果x为'M'，则计数器加1，并将最大池化层加入layers列表
            if x == 'M':
                count += 1
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # 否则，将卷积层、归一化层、ReLU激活函数层加入layers列表
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

        # 定义网络参数
