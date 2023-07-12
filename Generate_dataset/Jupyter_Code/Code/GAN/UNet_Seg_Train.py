"""
@Author  :WAN ZHIYU
@Time    :2023/06/29
@Function：基于3DUNet的拓扑优化结构缺陷检测网络

"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch import optim
from einops import rearrange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./traindata_record')
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


#数据集加载模块
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
    

class Dataset_validate(Dataset):
    def __init__(self, path):
        self.path = path
        self.data_path = glob.glob\
            (os.path.join(path,'validate_input/*.npy'))  #读取data文件夹下所有.npy格式文件

    def __getitem__(self, index):
        data_path = self.data_path[index]
        # print(data_path)
        data = np.load(data_path)      #读取输入数据
        tensor_data = torch.from_numpy(data)
        
        label_path = data_path.replace('input/validate_input', 'label/validate_label')
        label = np.load(label_path)    #读取标签数据
        tensor_label = torch.from_numpy(label)

        return tensor_data, tensor_label

    def __len__(self):
        return len(self.data_path)
    
#3DUNet模块
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
        # print('input',input.shape)
        # print('x',x.shape)
        x1 = self.up3d(input)
        # print('x1',x1.shape)
        diffY = torch.tensor(x.size()[3] - x1.size()[3])
        diffX = torch.tensor(x.size()[4] - x1.size()[4])#特征融合部分
        diffZ = torch.tensor(x.size()[2] - x1.size()[2])
        #if x1.size()[3] > x.size()[3]:
        x3 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2))
        # print('x3',x3.shape)
        output = torch.cat([x, x3], dim = 1)
        return self.conv(output)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv,self).__init__()
        self.conv1 = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
    def forward(self, input):
        return self.conv1(input)

#3DUNet框架
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
        # print('out2.shape:',out2.shape)
        out3 = self.down2(out2)
        # print('out3.shape:',out3.shape)
        out4 = self.down3(out3)
        # print('out4.shape:',out4.shape)
        out5 = self.up1(out4, out3)
        # print('out5.shape:',out5.shape)
        out6 = self.up2(out5, out2)
        # print('out6.shape:',out6.shape)
        out7 = self.up3(out6, out1)
        # print('out7.shape:',out7.shape)
        logits = self.outc(out7)
        # print('logits.shape:',logits.shape)
        return logits



#类平衡损失函数
def cb_loss(y_pred, y_true, temp, beta, confusion):

    #计算每个类别对应的样本数
    confusion_ = [torch.sum(temp == i) for i in range(2)]   #统计一张label中每个类别的像素点
    # print('calsse_number:',i)
    # print('confusion_:',confusion_)
    weight = []
    weight_dice = []
    for i, n in zip(confusion_, confusion):
        if i == 0:
            weight.append(0)
            weight_dice.append(1)
        else:
            weight_dice.append(1)
            weight.append(((1.0 - beta) / (1.0 - math.pow(beta, i))))
    
    
    weight = torch.FloatTensor(weight)
    # print(weight)
    weight_dice = torch.FloatTensor(weight_dice)
    weight = weight.to(y_pred.device)
    weight_dice = weight_dice.to(y_pred.device)

    # print(y_pred.size(), y_true.size())
    criterion_train = torch.nn.CrossEntropyLoss(weight=weight)
    loss = criterion_train(y_pred.float(), temp.long())

    return loss


#生成混淆矩阵
@torch.no_grad()
def get_confusion_matrix(trues, preds):
    labels = [0, 1]
    conf_matrix = confusion_matrix(trues, preds, labels = labels)
    return conf_matrix

@torch.no_grad()
def plot_confusion_matrix(conf_matrix, xlabel, ylabel, saveaddr, printlabel):
    print(printlabel)
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = [0, 1]
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(second_index, first_index, conf_matrix[first_index, second_index])
    plt.savefig(saveaddr)
    plt.show()

def print_grad(model):
    "打印神经网络每一层的梯度值"
    for  name , parms in model.named_parameters():
        print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '-->grad_value:',parms.grad)
        # print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
#验证模块
@torch.no_grad()
def validation_train(net_c ,data_loader, epoch, confusion, validate_preds, validate_trues, batch_size):
    # netc = net_c.to(device=torch.device('cpu'))
    validateloss = 0
    validation_epoch_loss = 0
    for validatedata, validatelabel in data_loader:
        validatedata = validatedata.to(device=torch.device('cuda'), dtype=torch.float32)
        validatelabel = validatelabel.to(device=torch.device('cuda'), dtype=torch.float32)
        validatepred = net_c(validatedata)
        validatetemp = validatelabel.squeeze(1).to(device=torch.device('cuda'))
        validateloss = cb_loss(validatepred, validatelabel, validatetemp, 0.99, confusion).item()
        validation_epoch_loss += validateloss
        validateloss = validateloss / (len(data_loader.dataset) / 10)
        writer.add_scalar('validateloss',float(validateloss),epoch)
        writer.close()
        

        validate_outs = validatepred.argmax(dim=1)
        validate_preds.extend(validate_outs.detach().cpu().numpy())
        validate_trues.extend(validatelabel.detach().cpu().numpy())
        validate_pred = np.array(validate_preds).reshape(-1, 1)
        validate_true = np.array(validate_trues).reshape(-1, 1)

        validate_sklearn_accuracy = accuracy_score(validate_true, validate_pred)
        writer.add_scalar('validate_accuracy_score',float(validate_sklearn_accuracy), epoch)
        writer.close()
        validate_sklearn_precision = precision_score(validate_true, validate_pred)
        writer.add_scalar('validate_precision_score',float(validate_sklearn_precision),epoch)
        writer.close()
        validate_sklearn_recall = recall_score(validate_true, validate_pred)
        writer.add_scalar('recall_score', float(validate_sklearn_recall),epoch)
        writer.close()
        validate_sklearn_f1 = f1_score(validate_true, validate_pred)
        writer.add_scalar('f1_score',float(validate_sklearn_f1),epoch)
        writer.close()

    validation_epoch_loss = validation_epoch_loss / (20)
    return validate_true, validate_pred, validation_epoch_loss
    
#训练模块
def train_net(net, device, data_path, num_epochs=500, batch_size=2, lr=0.0001):

    #设置dataloader固定生成器
    # generator = torch.Generator().manual_seed(42)

    # #读取并载入negative_dataset
    # dataset1 = Dataset_neg(data_path)
    # negative_train_size = int(len(dataset1) * 0.7)
    
    # negative_validate_size = int(len(dataset1) - negative_train_size)
    # negative_train_dataset, negative_validate_dataset = torch.utils.data\
    #             .random_split(dataset1, [negative_train_size, negative_validate_size],generator = generator)
    
    # #读取并载入positive_dataset
    # dataset2 = Dataset_pos(data_path)
    # positive_train_size = int(len(dataset2) * 0.7)
    
    # positive_validate_size = int(len(dataset2) - positive_train_size)
    # positive_train_dataset, positive_validate_dataset = torch.utils.data\
    #             .random_split(dataset2, [positive_train_size, positive_validate_size], generator = generator)

    # #数据集按:negative:positive=7:3比例进行合并
    # train_dataset = torch.utils.data.ConcatDataset([negative_train_dataset, positive_train_dataset])
    # print('train_dataset_num:',len(train_dataset))
    # validate_dataset = torch.utils.data.ConcatDataset([negative_validate_dataset, positive_validate_dataset])
    # print('validate_dataset_num:',len(validate_dataset))

    # #加载训练数据集
    # traindata_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # # print('traindata_loader:',len(traindata_loader))
    # #加载验证数据集
    # validatedata_loader = DataLoader(validate_dataset, batch_size, shuffle=True)
    # # print('validatedata_loader:',len(validatedata_loader))

    train_dataset = Dataset_pos(data_path)
    traindata_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataset = Dataset_validate(data_path)
    validatedata_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    #设置Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.00002)
    #设置损失函数
    # crossentropyloss = torch.nn.BCEWithLogitsLoss()
    #设置终止迭代loss阈值
    best_loss = float('inf')

    #设置损失函数相关参数
    confusion = [1, 1]   #number of negative sample:6733/number of positive sample:222
    train_preds = []
    train_trues = []
    validate_preds = []
    validate_trues = []


    #开始训练
    for epoch in tqdm(range(num_epochs),desc='epochs'):
        net.train()
       
        #训练数据集
        epoch_loss = 0
        for i, (train_data, train_label) in enumerate(traindata_loader):
        
            train_data = train_data.to(device=device, dtype=torch.float32)
            train_label = train_label.to(device=device, dtype=torch.float32)
            
            pred = net(train_data)
            temp = train_label.squeeze(1).to(device)
            loss = cb_loss(pred, train_label, temp, 0.9999, confusion)
            epoch_loss += loss
            writer.add_scalar('trainloss',float(loss),epoch)
            writer.close()
            optimizer.zero_grad()
            loss.backward()
            # print_grad(net)
            optimizer.step()
        
            #评价指标计算
            train_outs = pred.argmax(dim=1)
            train_preds.extend(train_outs.detach().cpu().numpy())
            train_trues.extend(train_label.detach().cpu().numpy())
            train_pred = np.array(train_preds).reshape(-1,1)
            train_true = np.array(train_trues).reshape(-1,1)

            sklearn_accuracy = accuracy_score(train_true, train_pred)
            writer.add_scalar('accuracy_score',float(sklearn_accuracy),epoch)
            writer.close()
            sklearn_precision = precision_score(train_true, train_pred)
            writer.add_scalar('precision_score',float(sklearn_precision),epoch)
            writer.close()
            sklearn_recall = recall_score(train_true, train_pred)
            writer.add_scalar('recall_score',float(sklearn_recall),epoch)
            writer.close()
            sklearn_f1 = f1_score(train_true, train_pred)
            writer.add_scalar('f1_score',float(sklearn_f1),epoch)
            writer.close()
            
            print("-"*30+"\n【sklearn_metrics】  Epoch:{} \
                  \n --[Train_loss]:{:.4f} --[accuracy]:{:.4f} --[precision]:{:4f} --[recall]:{:4f} --[f1]:{:4f}".format\
                  (epoch, loss, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))
            
            if loss < best_loss:
                best_loss = loss
            torch.save(net.state_dict(), 'best_model.pth')
            
        #验证集测试 
        net.eval()
        validate_true, validate_pred, validation_epoch_loss= validation_train(net ,validatedata_loader, epoch, confusion, validate_preds, validate_trues, batch_size=10)
        print("-"*30+"\n --[validate_Epoch]:{} \n --[validate_Loss]:{}".format(epoch, validation_epoch_loss))
        print('-'*30)
        writer.add_scalar('validation_epoch_loss:',float(validation_epoch_loss), epoch)
        writer.close()
        scheduler.step()
        
        # #train混淆矩阵
        # train_conf_matrix = get_confusion_matrix(train_true, train_pred)
        # plot_confusion_matrix(train_conf_matrix, 'y_pred', 'y_true', saveaddr='train_confusion_matrix.jpg',printlabel = 'train_confusion_matrix')
        # #validate混淆矩阵
        # validate_conf_matrix = get_confusion_matrix(validate_true, validate_pred)
        # plot_confusion_matrix(validate_conf_matrix,  'validate_y_pred', 'validate_y_true', saveaddr='validate_confusion_matrix.jpg', printlabel = 'validate_confusion_matrix')
        

        
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Seg_net = UNet3D(1, 2)
    Seg_net.to(device=device)

    data_path = './test_batch/input/'
    train_net(Seg_net, device, data_path)


