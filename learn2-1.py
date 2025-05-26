import numpy as np
# Numpy是一个扩展程序库，并简化伟np，使用如数组操作和数学计算等功能

from numpy import hstack,zeros,ones
# hstack用于水平堆叠多个数组，即将多个数组连接在一起，形成一个新的数组。zeros用于生成一个指定形状的数组，所有元素初始化为0。ones用于生成一个指定形状的数组，所有元素初始化为1

from numpy.random import rand,randn
# rand生成均匀分布的随机数，randn生成标准正态分布的随机值

from keras.models import Sequential
# keras是高级神经网络API， Sequential用于构建神经网络模型

from keras.layers import Dense
# Dense全连接层

from matplotlib import pyplot
# 绘图的库

# 定义一个独立的生成器模型
def define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(30, activation='relu', kernel_initializer='he_uniform',input_dim=latent_dim))
    model.add(Dense(n_outputs, activation='linear'))
    return model


# 定义一个独立的判别器模型
def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform',input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model



# 定义一个完整的生成对抗网络模型，即组合的器生成器和判别模型，以更新生成器
def define_gan(generator, discriminator):
    # 锁定判别器的权重，使它们不参与训练
    discriminator.trainable = False
    model = Sequential()
    # 添加生成器
    model.add(generator)
    # 添加判别器
    model.add(discriminator)
    # 编译生成对抗网络模型
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model