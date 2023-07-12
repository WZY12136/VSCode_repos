
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import os


def data_loader(dataindex):
    "数据读取与质量预处理"
    matdata = io.loadmat(dataindex)
    data = matdata['data']
    data = np.array(data)
    print('数据初始维度：',data.shape)
    # data = data.transpose(1, 2, 0)
    print('数据调整后维度：',data.shape)          
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if data[i][j][k] > 0.5:
                    data[i][j][k] = 1
                else:
                    data[i][j][k] = 0
    
    return data


def data_size_deal(data):
    "数据尺寸扩展为标准尺寸"
    nelx, nely, nelz = 120, 120, 120
    if (nelx - data.shape[0])%2 !=0:
        data = np.pad(data, ((1, 0), (0, 0), (0 ,0)))
    if (nelz - data.shape[1])%2 !=0:
        data = np.pad(data, ((0, 0), (1, 0), (0 ,0)))
    if (nely - data.shape[2])%2 !=0:
        data = np.pad(data, ((0, 0), (0, 0), (1 ,0)))
    data_new = np.pad(data, (((nelx - data.shape[0])//2, (nelx - data.shape[0])//2),
                        ((nely - data.shape[1])//2, (nely - data.shape[1])//2),
                        ((nelz - data.shape[2])//2, (nelx - data.shape[2])//2)))
    print('数据标准化后维度：',data_new.shape)

    return data_new




def data_defect_prechecking(data, nelx, nely, nelz, saveindex):
    "查找模型存在的单共享边等断裂结构"
    defect_data = np.zeros((nely+2, nelx+2, nelz+2))
    data = np.pad(data, ((1, 1), (1, 1), (1, 1)))
    face_voxel = []
    search_list = np.zeros((1, 6, 3))
    defect_index = np.array([[0, 0, 0]])
    
    
    for i in range(1,(data.shape[0]-1)):
        for j in range(1,(data.shape[1]-1)):
            for k in range(1,(data.shape[2]-1)):
                if data[i][j][k] !=0:
                    
                    "计算中心六面立方体周围连接的立方体个数"
                    face_voxel.append(data[i-1][j][k])
                    face_voxel.append(data[i+1][j][k])
                    face_voxel.append(data[i][j-1][k])
                    face_voxel.append(data[i][j+1][k])
                    face_voxel.append(data[i][j][k-1])
                    face_voxel.append(data[i][j][k+1])
                    sum_value = sum(face_voxel)
                    
                    "统计可能存在缺陷的六面立方体坐标"
                    if sum_value == 0 or sum_value == 1:
                        x, y, z = i, j, k

                        "记录缺陷中心点六面立方体的坐标"
                        defect_array = np.array([[x, y ,z]])
                        defect_index = np.append(defect_index, defect_array, axis = 0)

                        "将缺陷中心点周围6体素点的结构存入defect_data"
                        defect_data[i-1:i+1,j-1:j+1,k-1:k+1] = data[i-1:i+1,j-1:j+1,k-1:k+1]
                        # defect_data[i, j, k] = data[i, j, k]
                        "记录缺陷中心点周围按十字立方体结构分布的体素的坐标"
                        face_voxel_index = np.array([[[x-1, z, y],
                                                      [x+1, z, y],
                                                      [x, z-1, y],
                                                      [x, z+1, y],
                                                      [x, z, y-1],
                                                      [x, z, y+1]]])
                        search_list = np.append(search_list, face_voxel_index, axis=0)

                    face_voxel = []


    np.set_printoptions(threshold=np.inf)
    print('缺陷中心点坐标列表:\n',defect_index)  #打印缺陷中心点位置
    print('缺陷中心点个数：',len(defect_index))  

    return defect_data

if __name__ == "__main__":
    
    "主函数"
    std_nelx = 120
    std_nely = 120 
    std_nelz = 120 
    dataindex = './test/top3d_Model_16.mat'
    saveindex = './test/'
    original_data = data_loader(dataindex)
    org_nely, org_nelx, org_nelz = original_data.shape[0], original_data.shape[1], original_data.shape[2]
    defect_data = data_defect_prechecking(original_data, org_nelx, org_nely, org_nelz, saveindex)
    standarddata = data_size_deal(original_data)
    defect_data = data_size_deal(defect_data)
    standarddata_save_addr = os.path.join(saveindex, 'standarddata_16.mat')
    io.savemat(standarddata_save_addr, {'standarddata':standarddata})
    defect_save_addr = os.path.join(saveindex,'defectdata_16.mat')
    io.savemat(defect_save_addr,{'defectdata':defect_data})
    print('数据处理完毕！')
                

