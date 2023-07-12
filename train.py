from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./traindata_record')

import torch
import math
import matplotlib.pyplot as plt
import os 



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


#评价指标生成模块
@torch.no_grad()
def get_eval_index(label, pred, preds, trues):
    labels = [0, 1]
    outs = pred.argmax(dim=1)
    preds.extend(outs.detach().cpu().numpy())
    trues.extend(label.detach().cpu().numpy())
    pred_shift = np.array(preds).reshape(-1, 1)
    true_shift = np.array(trues).reshape(-1, 1)

    #生成混淆矩阵
    conf_matrix = confusion_matrix(true_shift, pred_shift, labels = labels)
    #生成对应指标
    accuracy_score = accuracy_score(true_shift, pred_shift)
    precision_score = precision_score(true_shift, pred_shift)
    recall_score = recall_score(true_shift, pred_shift)
    f1_score = f1_score(true_shift, pred_shift)
    return conf_matrix, accuracy_score, precision_score, recall_score, f1_score

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
        
        val_conf_matrix, val_accuracy_score, val_precision_score, val_recall_score, val_f1_score=\
        get_eval_index(validatelabel, validatepred, validate_preds, validate_trues) 
        
    return val_conf_matrix, val_accuracy_score, val_precision_score, \
        val_recall_score, val_f1_score, validation_epoch_loss
    

def train_net(net, device, data_path, epochs=500, batch_size=10, lr=0.0001):

    #设置dataloader固定生成器
    generator = torch.Generator().manual_seed(42)

    #读取并载入negative_dataset
    dataset1 = Generate_Dataset1(data_path)
    negative_train_size = int(len(dataset1) * 0.7)
    
    negative_validate_size = int(len(dataset1) - negative_train_size)
    negative_train_dataset, negative_validate_dataset = torch.utils.data\
                .random_split(dataset1, [negative_train_size, negative_validate_size],generator = generator)
    
    #读取并载入positive_dataset
    dataset2 = Generate_Dataset2(data_path)
    positive_train_size = int(len(dataset2) * 0.7)
    
    positive_validate_size = int(len(dataset2) - positive_train_size)
    positive_train_dataset, positive_validate_dataset = torch.utils.data\
                .random_split(dataset2, [positive_train_size, positive_validate_size], generator = generator)

    #数据集按:negative:positive=7:3比例进行合并
    train_dataset = torch.utils.data.ConcatDataset([negative_train_dataset, positive_train_dataset])
    print('train_dataset_num:',len(train_dataset))
    validate_dataset = torch.utils.data.ConcatDataset([negative_validate_dataset, positive_validate_dataset])
    print('validate_dataset_num:',len(validate_dataset))
    traindata_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # print('traindata_loader:',len(traindata_loader))

    validatedata_loader = DataLoader(validate_dataset, batch_size, shuffle=True)
    # print('validatedata_loader:',len(validatedata_loader))

    #设置Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.00002)
    #设置损失函数
    # crossentropyloss = torch.nn.BCEWithLogitsLoss()
    #设置终止迭代loss阈值
    best_loss = float('inf')

    #设置损失函数相关参数
    confusion = [1, 1]   #number of negative sample:6733/number of positive sample:222


    #开始训练
    for epoch in range(epochs):
        net.train()
        train_preds = []
        train_trues = []
        validate_preds = []
        validate_trues = []
        #训练数据集
        train_epoch_loss = 0
        for i, (train_data, train_label) in enumerate(traindata_loader):
            print('batch_num：',i)
            train_data = train_data.to(device=device, dtype=torch.float32)
            train_label = train_label.to(device=device, dtype=torch.float32)
            
            pred = net(train_data)
            
            temp = train_label.squeeze(1).to(device)
            loss = cb_loss(pred, train_label, temp, 0.9999, confusion)
            train_epoch_loss += loss
            
            writer.close()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            #评价指标计算
            trn_conf_matrix, trn_accuracy_score, trn_precision_score, trn_recall_score, trn_f1_score=\
            get_eval_index(train_label, pred, train_preds, train_trues)

            print("-"*30+"\n【evaluation_index】  Epoch:{} \
                  \n  --[accuracy]:{:.4f} --[precision]:{:4f} --[recall]:{:4f} --[f1]:{:4f}".format\
                  (trn_accuracy_score, trn_precision_score, trn_recall_score, trn_f1_score))
            writer.add_scalar('trn_accuracy_score',float(trn_accuracy_score), epoch)
            writer.add_scalar('trn_precision_score', float(trn_precision_score), epoch)
            writer.add_scalar('trn_recall_score', float(trn_recall_score), epoch)
            writer.add_scalar('trn_f1_score', float(trn_f1_score), epoch)
            writer.close()

        avg_train_loss = (train_epoch_loss / len(traindata_loader))
        writer.add_scalar('Train_Loss',float(avg_train_loss),epoch)
        print("-"*30+"\n【sklearn_metrics】  Epoch:{}\n --[Train_loss]:{:.4f})".format(epoch, avg_train_loss))
        
        #绘制混淆矩阵
        plot_confusion_matrix(trn_conf_matrix, 'y_pred', 'y_true', \
                              saveaddr='train_confusion_matrix.jpg',printlabel = 'train_confusion_matrix')


        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
        torch.save(net.state_dict(), 'best_model.pth')
            
        #验证集测试 
        net.eval()
        validate_conf_matrix, val_accuracy_score, val_precision_score, , validation_epoch_loss= validation_train(net ,validatedata_loader, epoch, confusion, validate_preds, validate_trues, batch_size=10)
        print("-"*30+"\n --[validate_Epoch]:{} \n --[validate_Loss]:{}".format(epoch, validation_epoch_loss))
        print('-'*30)
        writer.add_scalar('validation_epoch_loss:',float(validation_epoch_loss), epoch)
        writer.close()
        scheduler.step()
        
        #train混淆矩阵
        
        #validate混淆矩阵
        validate_conf_matrix = get_confusion_matrix(validate_true, validate_pred)
        plot_confusion_matrix(validate_conf_matrix,  'validate_y_pred', 'validate_y_true', saveaddr='validate_confusion_matrix.jpg', printlabel = 'validate_confusion_matrix')
        

        
        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet3D(1, 2)
    net.to(device=device)

    data_path = './dataset_mini_batch/input/'
    train_net(net, device, data_path)

