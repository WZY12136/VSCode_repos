import numpy as np
import os
import glob


#查找非零标签
def nozeroarray_detect(dataindex):
    num_1 = 0
    nozeroarray = []
    for t in range(1, 9937):
        data = np.load(os.path.join(dataindex, str(t)+'.npy'))
        if np.all(data == 0):
            continue
        else:
            nozeroarray.append(t)
            
            num_1 += 1
    print('num_1:',num_1)
    print('nozeroarray:',np.array(nozeroarray))
    
    return  nozeroarray



#数据集划分为negative_sample和positive_sample
def dataset_split(nozeroarray, inputdataaddr, saveinputdataaddr):
    
    inputdataindex = glob.glob(os.path.join(inputdataaddr, '*.npy'))
    
    for n in range(1, len(inputdataindex)+1):
        labeldataaddr = inputdataaddr.replace('input/','label/')
        savelabeldataaddr = saveinputdataaddr.replace('input/', 'label/')
        inputdata = np.load(os.path.join(inputdataaddr, str(n)+'.npy'))
        inputdata = inputdata[np.newaxis, :, :,:]
        labeldata = np.load(os.path.join(labeldataaddr, str(n)+'.npy'))
        labeldata = labeldata[np.newaxis, :, :,:]
        if n in nozeroarray:
            np.save(os.path.join(saveinputdataaddr, 'positive_input/'+str(n)+'.npy'), inputdata)
            np.save(os.path.join(savelabeldataaddr, 'positive_label/'+str(n)+'.npy'), labeldata)
        else:
            np.save(os.path.join(saveinputdataaddr, 'negative_input/'+str(n)+'.npy'), inputdata)
            np.save(os.path.join(savelabeldataaddr, 'negative_label/'+str(n)+'.npy'), labeldata)

if __name__ == "__main__":
    inputdataaddr = '../../Data（实验用）/dataset（切割数据集-整体汇总编号）/input/'
    
    search_nozerodataindex = inputdataaddr.replace('input/', 'label/')
    saveinputdataaddr = '../../Data（实验用）/dataset(按正负样本分类)/input/'

    nozeroarray = nozeroarray_detect(search_nozerodataindex)
    dataset_split(nozeroarray, inputdataaddr, saveinputdataaddr)