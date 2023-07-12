import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

# 判别器
class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # 非常小的网络
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


# 生成器
class Generator(nn.Module):
    # z是隐空间参量
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            # 输入会标准化为[-1, 1]，所以这里的输出也要标准化到[-1, 1]
            nn.Tanh(), 
        )

    def forward(self, x):
        return self.gen(x)


# 超参数设置
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # 学习率
z_dim = 64 # 隐参量的维度
image_dim = 28 * 28 * 1  # 784，MNIST
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
# 这里加入噪声是为了看出在迭代过程中的变化
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    # 按道理应该采用与MNist相同的均值和标准差(0.1307, 0.3081)
    # 但上面的超参数的设置是作者用(0.5, 0.5)时调出来的，所以这里如果改了就会发散
    # 这也说明GAN对参数非常敏感，非常难以训练
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 判别器的优化算法
opt_disc = optim.Adam(disc.parameters(), lr=lr)
# 生成器的优化算法
opt_gen = optim.Adam(gen.parameters(), lr=lr)
# 损失函数设为Binary Cross Entropy
# 公式为-[y*logx + (1-y)log(1-x)]
# 注意公式前面的负号，后面计算损失时该负号将最大化改为了最小化
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

# 迭代训练
for epoch in range(num_epochs):
    # 从加载器里取出的是real图像
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ########## 训练判别器：最大化log(D(x)) + log(1 - D(G(z))) ###############
        ## 即：log(D(real)) + log(1 - D(G(latent_noise)))

        # 事先准备隐空间的噪声数据
        noise = torch.randn(batch_size, z_dim).to(device)
        # 将噪声数据传给生成器，生成假的图像
        fake = gen(noise)

        #### 计算log(D(x))损失 ####
        # 将真实图像传给判别器，即计算D(x)
        disc_real = disc(real).view(-1)
        # 将D(x)与1分别作为预测值和目标值放到BCE中进行计算
        # 根据BCE的公式-[y*logx + (1-y)log(1-x)]，这里y为1，因此此处计算的就是-log(D(x))
        # 也可以这样理解，此处的损失就是看看判别器对于真实图像的预测是不是接近1，即判别器对于真实图像的性能怎么样
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        #### 计算log(1-D(G(z)))损失
        # 将生成器生成的虚假图像传给判别器，即计算D(G(z))
        disc_fake = disc(fake).view(-1)
        # 将D(G(z))与0分别作为预测值和目标值放到BCE中进行计算
        # 根据BCE的公式-[y*logx + (1-y)log(1-x)]，这里y为0，因此此处计算的就是-log(1-D(G(z)))
        # 也可以这样理解，此处的损失就是看看判别器对于虚假图像的预测是不是接近0，即判别器对于虚假图像的性能怎么样
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # 总损失
        lossD = (lossD_real + lossD_fake) / 2
        # 判别器的反向传播
        disc.zero_grad()
        # 注意，这里将retain_graph设为True，是为了保留该过程中计算的梯度，后续生成器网络更新时使用
        # 否则这里判别器网络构建了正向计算图后，反向传播结束后就将其销毁
        lossD.backward(retain_graph=True)
        opt_disc.step()


        ########## 训练生成器：最小化log(1 - D(G(z)))，等价于最大化log(D(G(z)) ###############
        ## 第二种损失不会遇到梯度饱和的问题

        # 将生成器生成的虚假图像传给判别器，即计算D(G(z))
        # 这里的disc是经过了升级后的判别器，所以与第99行的D(G(z))计算不同
        # 但fake这个量还是上面的fake = gen(noise)
        output = disc(fake).view(-1)
        # 将D(G(z))与1分别作为预测值和目标值放到BCE中进行计算
        # 根据BCE的公式-[y*logx + (1-y)log(1-x)]，这里y为1，因此此处计算的就是-log(D(G(z)))
        # 也可以这样理解，此处的损失就是看看判别器对于生成器生成的虚假图像的预测是不是接近1，即生成器有没有骗过判别器
        # 这里log(D(G(z))的计算与上面的log(D(G(z))的计算不重复，是因为生成器和判别器是分开训练的，两者都要有各自的损失函数
        lossG = criterion(output, torch.ones_like(output))
        # 生成器的反向传播
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()


        # 下面就是用于tenshorboard的可视化
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1