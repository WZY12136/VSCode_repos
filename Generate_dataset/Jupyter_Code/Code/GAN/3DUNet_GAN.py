# @Time    :2023/6/28
# @Function:使用3D-UNet作为Generator的生成对抗网络，对拓扑优化模型进行缺陷检测

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
from tqdm import tqdm
from UNet_Seg_Model import UNet3D  #导入生成器


#读取数据集中具有缺陷的样本
class Dataset_pos(Dataset):
    def __init__(self, path):
        self.path = path
        self.data_path = glob.glob\
            (os.path.join(path,'positive_input/*.npy'))  #读取data文件夹下所有.npy格式文件

    def __getitem__(self, index):
        data_path = self.data_path[index]
        # print(data_path)
        data = np.load(data_path)      #读取输入数据
        tensor_data = torch.from_numpy(data)
        
        label_path = data_path.replace('input/positive_input', 'label/positive_label')
        label = np.load(label_path)    #读取标签数据
        tensor_label = torch.from_numpy(label)

        return tensor_data, tensor_label

    def __len__(self):
        return len(self.data_path)
    
class Dataset_neg(Dataset):
    def __init__(self, path):
        self.path = path
        self.data_path = glob.glob\
            (os.path.join(path,'negative_input/*.npy'))  #读取data文件夹下所有.npy格式文件

    def __getitem__(self, index):
        data_path = self.data_path[index]
        # print(data_path)
        data = np.load(data_path)      #读取输入数据
        tensor_data = torch.from_numpy(data)
        
        label_path = data_path.replace('input/negative_input', 'label/negative_label')
        label = np.load(label_path)    #读取标签数据
        tensor_label = torch.from_numpy(label)

        return tensor_data, tensor_label

    def __len__(self):
        return len(self.data_path)

#判别器，判断一张三维图片来源于真实数据集的概率
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv3d(in_channels,256, kernel_size=(3, 3, 3),stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3),stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, out_channels, kernel_size=(3, 3, 3),stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.disc(x)
    

    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 3e-4 #学习率
    data_path = './dataset_mini_batch/input/'
    batch_size = 32
    num_epochs = 500
    #初始化判别器
    Disc = Discriminator(1, 1).to(device)
    #初始化生成器
    Gene = UNet3D(1, 1).to(device)
    #设置判别器和生成器的优化器
    opt_disc = optim.Adam(Disc.parameters(), lr=lr)
    opt_gen = optim.Adam(Gene.parameters(), lr=lr)
    #设置迭代规则
    criterion = nn.BCELoss()

    #设置dataloader固定生成器
    generator = torch.Generator().manual_seed(42)
    
    #加载数据集
    dataset_pos = Dataset_pos(data_path)
    dataset_neg = Dataset_neg(data_path)
    train_dataset = torch.utils.data.ConcatDataset([dataset_pos, dataset_neg])
    Loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(len(Loader))

    for epoch in tqdm(range(num_epochs), desc='epochs'):

        for batch_idx, (input_data, input_label) in enumerate(Loader):
            
            ####训练判别器####
            #生成器输入结构模型，生成预测的缺陷三维结构图
            input_data = input_data.to(device=device, dtype=torch.float32)
            input_label = input_label.to(device=device, dtype=torch.float32)
            fake_label = Gene(input_data)      #生成器生成预测缺陷结构


            ####计算log(D(x))损失####
            # 将真实图像传给判别器，即计算D(x)
            disc_real = Disc(input_label).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            ####计算log(1-D(G(z)))损失
            #将生成器生成的虚假图像传给判别器，即计算D(G(z))
            disc_fake = Disc(fake_label).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            #总损失
            lossD = (lossD_real + lossD_fake) / 2

            Disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            ####训练生成器####
            output = Disc(fake_label).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            Gene.zero_grad()
            lossG.backward()
            opt_disc.step()


            #tensorboard可视化
            if batch_idx == 0:
                print("\n")
                print("Epoch[{epoch}/{num_epochs}] Batch {batch_idx}/{len(Loader)}\
                      Loss D:{lossD:.4f}, loss G: {lossG:.4f}")

        

