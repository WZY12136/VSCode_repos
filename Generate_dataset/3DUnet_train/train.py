from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./loss')

import torch
import math


def cb_loss(y_pred, y_true, temp, beta, confusion):

    #计算每个类别对应的样本数
    confusion_ = [torch.sum(temp == i) for i in range(2)]   #统计一张label中每个类别的像素点

    weight = []
    weight_dice = []
    for i, n in zip(confusion_, confusion):
        if i == 0:
            weight.append(0)
            weight_dice.append(1)
        else:
            weight_dice.append(1)
            weight.append(((1.0 - beta) / (1.0 - math.pow(beta, n))))
    
    
    weight = torch.FloatTensor(weight)
    weight_dice = torch.FloatTensor(weight_dice)
    weight = weight.to(y_pred.device)
    weight_dice = weight_dice.to(y_pred.device)


    criterion_train = torch.nn.CrossEntropyLoss(weight=weight)

    loss = criterion_train(y_pred.float(), temp.float())

    return loss



def train_net(net, device, data_path, epochs=100, batch_size=50, lr=0.00001):
    dataset = Generate_Dataset(data_path)
    #划分训练集和验证集
    train_size = int(len(dataset) * 0.9)
    validate_size = int(len(dataset) - train_size)
    train_dataset, validate_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])
    #加载训练集和验证集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)

    #cb_loss
    


    
    best_loss = float('inf')

    for epoch in range(epochs):
        print('epoch:',epoch)
        net.train()
        
        #训练数据集
        for data, label in train_loader:
            
            optimizer.zero_grad()
            data = data.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(data)
            temp = torch.argmax(pred, dim=1)
            loss = cb_loss(pred, label, temp, 0.9999, cls)
           
            writer.add_scalar('trainloss',float(loss),epoch)
            writer.close()

            loss.backward()
            optimizer.step()
            print('Loss/train', loss.item())
            if loss < best_loss:
                best_loss = loss
            torch.save(net.state_dict(), 'best_model.pth')
        #验证集测试
        for validatedata, validatelabel in validate_loader:
            
            validatedata = validatedata.to(device, dtype=torch.float32)
            validatelabel = validatelabel.to(device, dtype=torch.float32)
            validatepred = net(validatedata)
            validateloss = criterion(validatepred, validatelabel)
            writer.add_scalar('validateloss',float(validateloss),epoch)
            print('Loss/validate', validateloss.item())
        
        
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(1, 1)
    net.to(device=device)

    data_path = "../../../../Generate_dataset/Matlab_files/dataset/dataset/"
    train_net(net, device, data_path)

