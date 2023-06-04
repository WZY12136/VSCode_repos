import numpy as np
import glob

def count_num():
    dataindex = glob.glob('../../Code/dataset_mini_batch/label/positive_label/*.npy')
    num_1 = 0
    num_0 = 0
    for t in range(len(dataindex)):
        data = np.load(dataindex[t])
        data = np.squeeze(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if data[i][j][k] == 1:
                        num_1 += 1
                    else:
                        num_0 += 1
    return num_1, num_0
if __name__ == "__main__":
    num_1, num_0 = count_num()  
    print('*'*30)        
    print('num_1:',num_1)
    print('num_0:',num_0)
    print('*'*30)