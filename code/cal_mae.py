###################################################
# MAE-calculation code for point cloud denoising
# only used for cvx course
# Preparation notes: please don't delete noisy points, and set the x axis at -1000 of the noisy. 
###################################################
import math
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                        default='training/velodyne_reduced/000445.bin',
                        help='The data input path for the denoised.')
    parser.add_argument('--ref', type=str, 
                        default='training/velodyne_reduced_origin/000445.bin',
                        help='The data input path for the reference.')
    return parser.parse_args()

def cal_mae(data1, data2):
    ans = torch.from_numpy(np.zeros([1,2]))
    temp = np.array(data1)
    npdata1 = np.array(data1)
    npdata2 = np.array(data2)
    #print(np.shape(npdata1))
    npdata1 = np.delete(npdata1, np.where(temp[:,0] == -1000)[0], axis=0)
    npdata2 = np.delete(npdata2, np.where(temp[:,0] == -1000)[0], axis=0)
    #print(np.shape(npdata1))
    data1 = torch.from_numpy(npdata1)
    data2 = torch.from_numpy(npdata2)

    d_var = torch.sqrt((data1[:,0]-data2[:,0])**2 + (data1[:,1]-data2[:,1])**2 + (data1[:,2]-data2[:,2])**2)
    ref1 = data1[:,3]
    ref2 = data2[:,3]
    
    ans[:,0] = sum(d_var)/data1.shape[0]
    ans[:,1] = sum(abs(ref1-ref2))/data1.shape[0]
    return ans

if __name__ == '__main__':
    cfg = config()
    data1 = np.fromfile(cfg.input, dtype=np.float32, count=-1).reshape([-1,4])
    data1 = torch.from_numpy(data1)
    data2 = np.fromfile(cfg.ref, dtype=np.float32, count=-1).reshape([-1,4])
    if (np.max(data2[:, 3]) > 1):
        data2[:, 3] /= 256
        
        
    '''data_list = []
    index_list = []
    for index in range(0, data1.shape[0]):
        if np.linalg.norm(data1[index]) < 1e-5:
            continue
        tmp = data1[index, :3]
        tmp = tmp / np.linalg.norm(tmp)
        data1[index, :3] = tmp
        data_list.append(tmp[0])
        index_list.append(index)
    for index in range(0, data2.shape[0]):
        tmp = data2[index, :3]
        tmp = tmp / np.linalg.norm(tmp)
        data2[index, :3] = tmp
    #plt.plot(data2[:, 0])
    #plt.plot(data2[:, 1])
    #plt.plot(data2[:, 2])
    plt.scatter(index_list, data_list)
    plt.show()'''
        
        
    data2 = torch.from_numpy(data2)
    
    res = cal_mae(data1, data2)
    print("MAE results:", res)