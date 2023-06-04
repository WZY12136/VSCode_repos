import numpy as np
import scipy.io as io
import os
import glob

#数据集切割代码
def cutmodule(matdata, saveindex, signal):
    if signal == True:
        data = matdata['inputdata']
        print(data.shape)
        print('****************执行数据尺寸修改****************')
        nelx, nely, nelz = 120, 120, 120
        if (nelx - data.shape[0])%2 !=0:
            data = np.pad(data, ((1, 0), (0, 0), (0 ,0)))
        if (nelz - data.shape[1])%2 !=0:
            data = np.pad(data, ((0, 0), (1, 0), (0 ,0)))
        if (nely - data.shape[2])%2 !=0:
            data = np.pad(data, ((0, 0), (0, 0), (1 ,0)))
        data = np.pad(data, (((nelx - data.shape[0])//2, (nelx - data.shape[0])//2),
                        ((nely - data.shape[1])//2, (nely - data.shape[1])//2),
                        ((nelz - data.shape[2])//2, (nelx - data.shape[2])//2)))
        io.savemat(os.path.join(saveindex,'inputdata.mat'),{'inputdata':data})
        tempdata = np.zeros((20, 20, 20))
        for k in range(0, data.shape[2], 20):               #沿y轴切割
            for j in range(0, data.shape[1], 20):           #沿z轴切割
                for i in range(0, data.shape[0], 20):       #沿x轴切割
                    global m
                    m += 1
                    tempdata = data[i:i+20, j:j+20, k:k+20]
                    # np.set_printoptions(threshold=np.inf)
                    np.save(os.path.join(saveindex,'%d.npy'%m),tempdata)
                    print('第：',m,'次')
                    # print(tempdata)
                    tempdata = np.zeros((20, 20, 20))
    else:
        data = matdata['defectdata']
        print(data.shape)
        print('****************生成标签数据集****************')
        tempdata = np.zeros((20, 20, 20))
        for k in range(0, data.shape[2], 20):               #沿y轴切割
            for j in range(0, data.shape[1], 20):           #沿z轴切割
                for i in range(0, data.shape[0], 20):       #沿x轴切割
                    global s
                    s += 1
                    tempdata = data[i:i+20, j:j+20, k:k+20]
                    # np.set_printoptions(threshold=np.inf)
                    np.save(os.path.join(saveindex,'%d.npy'%s),tempdata)
                    print('第：',s,'次')
                    # print(tempdata)
                    tempdata = np.zeros((20, 20, 20))

#非零矩阵标号检测
def nozerosdetect(data_index):

    global m
    global s
    n = 0
    zeros_array = []
    array = np.arange((m-215), (m+1))
    
    for i in range((m-215),(m+1)):
        path = os.path.join(data_index, str(i)+'.npy')
        data = np.load(path)
        if np.all(data == 0):
            n += 1
            zeros_array.append(i)
        else:
            zeros_array.append(0)

    print('一共',n,'个全零矩阵')

    zeros_array = np.array(zeros_array)
    nozeros_array = array - zeros_array
    np.save(os.path.join(data_index, 'detectmatrix.npy'), nozeros_array)

#主函数
if __name__ == "__main__":


    folder_address = '../../Matlab_files/dataset/originalmatdata/dataset/dataset(05.25)/input/'
    mat_address = '../../Matlab_files/dataset/originalmatdata/dataset/matdata/input/'
    # folder_address = '../Matlab_files/dataset/originalmatdata/dataset/dataset(05.19)/input/'
    # mat_address = '../../Matlab_files/dataset/originalmatdata/matdata/input/'
    #test
    # folder_address = '../Matlab_files/dataset/originalmatdata/testdataset/input/'
    # mat_address = '../Matlab_files/dataset/originalmatdata/matdata/input/'

    matdata_index = glob.glob(os.path.join(mat_address,'*.mat')) 

    m = 0
    s = 0
    for t in range(0, len(matdata_index)):

        print('读取第',t,'个数据')


        #输入数据集生成
        os.makedirs(os.path.join(folder_address,str(t)))
        inputpath = os.path.join(mat_address,str(t)+'.mat')      #按顺序读取输入数据
        inputmatdata = io.loadmat(inputpath)
        saveinputindex = os.path.join(folder_address,str(t))
        cutmodule(inputmatdata, saveinputindex, signal=True)
        data_index = os.path.join('../../Matlab_files/dataset/originalmatdata/dataset/dataset(05.25)/input/',str(t)+'/')
        nozerosdetect(data_index)
        
        
        #标签数据集生成
        label_folder_address = folder_address.replace('input/','label/')   #文件夹生成地址
        os.makedirs(os.path.join(label_folder_address,str(t)))             #生成对应模型的存储文件夹
        labelpath = inputpath.replace('input/','label/')                   #标签数据读取路径
        savelabelindex = os.path.join(label_folder_address,str(t))
        labelmatdata = io.loadmat(labelpath)
        cutmodule(labelmatdata, savelabelindex, signal=False)
        label_data_index = data_index.replace('input/','label/')
        nozerosdetect(label_data_index)
        print('*'*30)
        