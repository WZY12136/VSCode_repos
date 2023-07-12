# @Time    : 2022/9/25
# @Function: 用pytorch实现一个最简单的GAN，用MNIST数据集生成新图片

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
from tqdm import tqdm


# 判别器，判断一张图片来源于真实数据集的概率，输入0-1之间的数，数值越大表示数据来源于真实数据集的概率越高。
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features=img_dim, out_features=128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 将输出值映射到0-1之间
        )

    def forward(self, x):
        return self.disc(x)


# 生成器,用随机噪声生成图片
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),
            # normalize inputs to [-1, 1] so make outputs [-1, 1]
            # 一般二分类问题中，隐藏层用Tanh函数，输出层用Sigmod函数
        )

    def forward(self, x):
        return self.gen(x)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    noise_dim = 50  # noise
    image_dim = 28 * 28 * 1  # 784
    batch_size = 32
    num_epochs = 200

    # 加载数据集，生成输入噪声数据
    transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    fixed_noise = torch.randn((batch_size, noise_dim)).to(device)  #生成噪声数据集

    D = Discriminator(image_dim).to(device)  #初始化判别器
    G = Generator(noise_dim, image_dim).to(device)  #初始化生成器
    #设置判别器和生成器的优化器
    opt_disc = optim.Adam(D.parameters(), lr=lr)
    opt_gen = optim.Adam(G.parameters(), lr=lr)
    #设置损失函数
    criterion = nn.BCELoss()     # 二分类交叉熵损失函数

    # 存放log的文件夹
    log_dir = "test-record"
    if (os.path.exists(log_dir)):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    for epoch in tqdm(range(num_epochs), desc='epochs'):
        # GAN不需要真实label
        for batch_idx, (img, _) in enumerate(loader):
            img = img.view(-1, 784).to(device)
            batch_size = img.shape[0]

            # 训练判别器: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_img = G(noise)    # 根据随机噪声生成虚假数据
            disc_fake = D(fake_img)    # 判别器判断生成数据为真的概率
            # torch.zeros_like(x) 表示生成与 x 形状相同、元素全为0的张量
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))    # 虚假数据与0计算损失
            disc_real = D(img)    # 判别器判断真实数据为真的概率
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))     # 真实数据与1计算损失
            lossD = (lossD_real + lossD_fake) / 2

            D.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # 训练生成器: 在此过程中将判别器固定，min log(1 - D(G(z))) <-> max log(D(G(z))
            output = D(fake_img)
            lossG = criterion(output, torch.ones_like(output))
            G.zero_grad()
            lossG.backward()
            opt_gen.step()

            if batch_idx == 0:
                # print( f"Epoch [{epoch+1}/{num_epochs}]  Batch {batch_idx}/{len(loader)}   lossD = {lossD:.4f}, lossG = {lossG:.4f}")
                with torch.no_grad():
                    # 用固定的噪声数据生成图像，以对比经过不同epoch训练后的生成器的生成能力
                    fake_img = G(fixed_noise).reshape(-1, 1, 28, 28)
                    real_img = img.reshape(-1, 1, 28, 28)

                    # make_grid的作用是将若干幅图像拼成一幅图像
                    img_grid_fake = torchvision.utils.make_grid(fake_img, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(real_img, normalize=True)

                    writer.add_image("Fake Images", img_grid_fake, global_step=epoch)
                    writer.close()
                    writer.add_image("Real Images", img_grid_real, global_step=epoch)
                    writer.close()
                    writer.add_scalar(tag="lossD", scalar_value=lossD, global_step=epoch)
                    writer.close()
                    writer.add_scalar(tag="lossG", scalar_value=lossG, global_step=epoch)
                    writer.close()