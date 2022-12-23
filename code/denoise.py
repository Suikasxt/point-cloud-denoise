import math
import torch
import time
import numpy as np
import argparse
import faiss

EPS=1e-5
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--vis', type=str, default='')
    parser.add_argument('--diff', type=str, default='')
    cfg = parser.parse_args()
    print('Work on pic No.%d'%cfg.id)
    if not cfg.input:
        cfg.input = '../data_new/pcd%03drainy.bin'%cfg.id
    if not cfg.output:
        cfg.output = '../data_new/pcd%03ddenoise.bin'%cfg.id
    if not cfg.vis:
        cfg.vis = '../data_new/pcd%03dvis.bin'%cfg.id
    if not cfg.diff:
        cfg.diff = '../data_new/pcd%03ddiff.bin'%cfg.id
    cfg.R = [55, 33, 35, 100][cfg.id - 1]
    return cfg
cfg = config()

def solve(x, k, lam):
    index = faiss.IndexFlatL2(3)
    pc_cor=np.copy(x[:,:3],order='C')
    index.add(pc_cor)
    pos = np.copy(x[:,:3])
    D_list, I_list = index.search(np.array(pos), k+1)
    x_hat = np.copy(x)
    for index in range(x.shape[0]):
        D = D_list[index]
        I = I_list[index]
        
        I = I[D<0.1]
        if len(I) <= 4:
            continue
        
        R = np.linalg.norm(pos[I], axis=-1)
        ref_list = x[I][:, 3]
        
        avg=(1-lam)*R[0] + lam*np.mean(R[1:])
        ref=(1-lam)*ref_list[0] + lam*np.mean(ref_list[1:])
        
        x_hat[index, :3] *= avg / (np.linalg.norm(x_hat[index, :3])+EPS)
        x_hat[index, 3] = ref

    return x_hat

def denoise(data):
    for i in range(5):
        point_set = faiss.IndexFlatL2(3)
        position = data[:, :3]
        position_norm = position / (np.linalg.norm(position, axis=1, keepdims=True) + EPS)
        point_set.add(position_norm)
        res = np.copy(data)
        K = 11
        _, I_list = point_set.search(np.array(position_norm), K)
        for index in range(data.shape[0]):
            if np.linalg.norm(res[index]) < EPS:
                continue
            I = I_list[index]
            R = np.linalg.norm(position[I], axis=-1)
            center = np.mean(R[1:], axis=0)
            delta = center - R[0]
            inner_delta = np.mean(np.abs(R[1:] - center))
            if 2*inner_delta < delta:
                if inner_delta > 1:
                    res[index, :] = 0
                else:
                    res[index, :3] *= center / R[0]
                    res[index, 3] = np.mean(data[I[1:]][:, 3])
        data = res
    
    for i in range(3):
        res = solve(res, 7, 0.3)
    
    L_index = 0
    while L_index < res.shape[0]:
        if np.linalg.norm(res[L_index]) < EPS:
            R_index = L_index
            while R_index+1 < res.shape[0] and np.linalg.norm(res[R_index+1]) < EPS:
                R_index += 1
            
            if R_index == res.shape[0] - 1:
                res[L_index:] = res[L_index-1]
            elif L_index == 0:
                res[:R_index+1] = res[R_index+1]
            elif np.linalg.norm(res[L_index-1] - res[R_index+1]) < 0:
                for i in range(L_index, R_index+1):
                    lamb = (i - (L_index - 1)) / ((R_index + 1) - (L_index - 1))
                    res[i] = (1-lamb) * res[L_index-1] + lamb * res[R_index+1]
            else:
                res[L_index:R_index+1, :] = -1000
        
            L_index = R_index
        L_index += 1
            
    
    zero_num = np.sum(res[:, 0]==-1000)
    print('Number of zero points: {} {}%'.format(zero_num, round(zero_num/res.shape[0]*100, 2)))
    
    return res

if __name__ == '__main__':
    time_start = time.time()
    data = np.fromfile(cfg.input,dtype=np.float32,count=-1).reshape([-1,4])
    res = denoise(data)
    
    with open(cfg.output, 'wb') as f:
        fortran_data = np.asfortranarray(res, 'float32')
        fortran_data.tofile(f)
    
    with open(cfg.vis, 'wb') as f:
        output = np.copy(res)
        output[output==-1000] = 0
        fortran_data = np.asfortranarray(output, 'float32')
        fortran_data.tofile(f)
        
    with open(cfg.diff, 'wb') as f:
        output = np.copy(data)
        output[res!=-1000] = 0
        fortran_data = np.asfortranarray(output, 'float32')
        fortran_data.tofile(f)
    print(time.time() - time_start)