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
#latent_dim：潜在空间维度，表示输入噪声向量的长度，在GAN中，这是生成器的输入大小。n_outputs：生成器输出维度，指定生成数据的特征数量
    model = Sequential()
    #使用Keras的Sequential API创建线性堆叠模型，适合简单的层叠结构
    model.add(Dense(30, activation='relu', kernel_initializer='he_uniform',input_dim=latent_dim))
    #使用add和Dense函数添加一个全连接层，设置30个神经元，每个神经元接受输入的所有特征，设置relu激活函数，kernel_initializer='he_uniform'是一种权重初始化方式，适用于relu函数，input_dim=latent_dim是输入维度
    model.add(Dense(n_outputs, activation='linear'))
    #添加一个输出层，维度=2，使用linea线性激活函数，
    return model


# 定义一个独立的判别器模型
def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform',input_dim=n_inputs))
    model.add(Dense(1, activation='sigmoid'))
    #输出一个神经元，使用sigmoid激活函数，适合二分类任务
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    #模型编译，loss='binary_crossentropy'评估预测与真实之间的差距，Adam自适应调整学习率
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


# 生成带有类标签的 n 个真实样本
def generate_real_samples(n):
#n表示生成的真实样本数量
    X1 = rand(n)*6 - 3
    #rand(n)生成0-1之间的值，X1的值均匀分布在[-3,3]之间
    X2 = np.sin(X1)
    # 生成正弦曲线
    X1 = X1.reshape(n, 1)
    #调整形状为(n,1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    #X1，X2水平堆叠，形成一个(n,2)的数组X，每一行代表一个样本的特征
    y = ones((n, 1))
    #创建一个(n,1)的数组y，所有值赋值为1，表示样本都是真实的
    return X, y

# 在隐空间中生成一些点作为生成器的输入，隐空间理解为低维空间，将数据压缩为了节省运行空间
def generate_latent_points(latent_dim, n):
    x_input = randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input
# 使用生成器生成带有类标签的 n 个伪造样本
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    #生成n个伪造样本。并且存储在X中
    y = zeros((n, 1))
    #将伪造样本全部初始化为0
    return X, y

# 评估判别器并绘制真实样本和伪造样本
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    x_real, y_real = generate_real_samples(n)
    #生成n个真实的样本和标签
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    #evaluate对真实样本评估，得到准确率acc_real，第一个值是损失值，不需要用_替代，verbose=0表示不输出评估过程的信息
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    print(epoch, acc_real, acc_fake)
    pyplot.scatter(x_real[:, 0], x_real[:, 1], color='red')
    pyplot.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    filename = 'D:/pythontest/generated_plot_e%03d.png' % (epoch + 1)
    pyplot.savefig(filename)
    pyplot.close()

# 训练生成对抗网络
def train(g_model, d_model, gan_model, latent_dim, n_epochs=8000, n_batch=128,n_eval=2000):
# g_model生成器模型, d_model判别器模型，gan_model生成对抗网络模型
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        x_real, y_real = generate_real_samples(half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # 更 新 判 别 器
        # discriminator.trainable = True # 仅较新 Keras 版本必须添加
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # discriminator.trainable = False # 仅较新 Keras 版本必须添加


        x_gan = generate_latent_points(latent_dim, n_batch)
        #生成n_batch数量的随机潜在点，这些点将作为输入传递给生成器模型，在隐空间中随机生成一批点（潜在向量），这是生成器的"原料"，生成器将把这些随机点转化为伪造样本
        y_gan = ones((n_batch, 1))
        #创建全是"1"（真实）的标签，虽然生成的是伪造样本，但标签设为"1"（真实），这是一种训练技巧，目的让生成器学习如何"欺骗"判别器
        gan_model.train_on_batch(x_gan, y_gan)
        #冻结判别器权重，只更新生成器
        #前向传播：隐空间点 x_gan 输入生成器 G → 生成伪造样本，伪造样本输入判别器 D → 获得预测概率值（0到1之间）
        #损失计算：比较判别器预测值（概率）和期望标签 y_gan=1，使用二元交叉熵损失函数，最小化损失 → 使判别器更可能输出接近1的值
        #反向传播更新：：只更新生成器的权重（判别器权重被冻结），让生成器生产的伪造样本更易被误判为真实

        if (i + 1) % n_eval == 0:
            summarize_performance(i, g_model, d_model, latent_dim)

# 隐空间的维度
latent_dim = 6
discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)
train(generator, discriminator, gan_model, latent_dim)

