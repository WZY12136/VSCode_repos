# @Time    :2023/07/03
# @Function:3D-UNet缺陷检测网络模型

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


#*******************   
#3DUNet框架
#*******************
class DoubleConv3d_init(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3d_init, self).__init__()
        self.double_conv3d_init = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.double_conv3d_init(input)


class DoubleConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3d, self).__init__()
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):

        return self.double_conv3d(input)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.maxpool_conv3d = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, input):
        return self.maxpool_conv3d(input)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, input, x):  #x是接收的从encoder传过来的融合数据
    
        x1 = self.up3d(input)

        diffY = torch.tensor(x.size()[3] - x1.size()[3])
        diffX = torch.tensor(x.size()[4] - x1.size()[4])#特征融合部分
        diffZ = torch.tensor(x.size()[2] - x1.size()[2])
        x3 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2))
        output = torch.cat([x, x3], dim = 1)
        return self.conv(output)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv,self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
                    nn.Sigmoid())
    def forward(self, input):
        return self.conv1(input)

#==*******************   
#==3DUNet框架
#==*******************
class UNet3D(nn.Module):
    def __init__(self,in_channels, n_classes):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        #Encoder
        self.inc = DoubleConv3d_init(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        #Decoder
        self.up1 = Up(768, 256)
        self.up2 = Up(384, 128)
        self.up3 = Up(192, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, input):
        out1 = self.inc(input)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.up1(out4, out3)
        out6 = self.up2(out5, out2)
        out7 = self.up3(out6, out1)
        logits = self.outc(out7)

        return logits