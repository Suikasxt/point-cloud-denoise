import math
import torch
import numpy as np
import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                        default='../data_new/pcd001rainy.bin',
                        help='The data input path for the denoised.')
    parser.add_argument('--ref', type=str, 
                        default='../data_new/pcd001clean.bin',
                        help='The data input path for the reference.')
    return parser.parse_args()

def cal_mae(data1, data2):
    ans = torch.from_numpy(np.zeros([1,2]))
    temp = np.array(data1)
    npdata1 = np.array(data1)
    npdata2 = np.array(data2)
    print(np.where(temp[:,0] == -1000)[0])
    npdata1 = np.delete(npdata1, np.where(temp[:,0] == -1000)[0], axis=0)
    npdata2 = np.delete(npdata2, np.where(temp[:,0] == -1000)[0], axis=0)
    data1 = torch.from_numpy(npdata1)
    data2 = torch.from_numpy(npdata2)
    print(data1.shape)
    print(data2.shape)
    d1 = torch.sqrt(data1[:,0]**2+data1[:,1]**2+data1[:,2]**2)
    d2 = torch.sqrt(data2[:,0]**2+data2[:,1]**2+data2[:,2]**2)
    ref1 = data1[:,3]
    ref2 = data2[:,3]
    ans[:,0] = sum(abs(d1-d2))/data1.shape[0]
    ans[:,1] = sum(abs(ref1-ref2))/data1.shape[0]
    return ans

if __name__ == '__main__':
    cfg = config()
    data1 = np.fromfile(cfg.input,dtype=np.float32,count=-1).reshape([-1,4])
    data1 = torch.from_numpy(data1)
    data2 = np.fromfile(cfg.ref,dtype=np.float32,count=-1).reshape([-1,4])
    data2 = torch.from_numpy(data2)
    res = cal_mae(data1, data2)
    print("MAE results:", res)