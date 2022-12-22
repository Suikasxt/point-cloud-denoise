import math
import torch
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

def denoise(data):
    point_set = faiss.IndexFlatL2(3)
    position = data[:, :3]
    position_norm = position / (np.linalg.norm(position, axis=1, keepdims=True) + EPS)
    point_set.add(position_norm)
    res = np.copy(data)
    '''dist = []
    for point in res:
        D, I = point_set.search(np.array([point[:3]]), 10)
        delta = np.abs(res[I[0], :3] - point[:3])
        entropy = np.sum(np.min(delta, axis=1) / (np.linalg.norm(delta, axis=1) + EPS))
        #dist.append(np.mean(D[0][1:10]) / (np.linalg.norm(point[:3])+EPS))
        dist.append(entropy)
    
    rank = np.argsort(dist)
    noise_point_rate = 1 - 2000/(2000 + cfg.R)
    print('noise_point_rate', noise_point_rate)
    noise_point_num = int(data.shape[0] * noise_point_rate)
    for i in rank[:-noise_point_num]:
        #res[i, :] = -1000
        pass
    
    for i in rank[-noise_point_num:]:
        res[i, :] = -1000
        pass'''
    for index in range(data.shape[0]):
        K = 11
        D, I = point_set.search(np.array([position_norm[index]]), K)
        I = I[0][1:]
        center = np.mean(position[I], axis=0)
        min_D = np.linalg.norm(center - position[index], axis=-1)
        max_D = np.mean(np.linalg.norm(position[I] - center, axis=-1))
        if 2*max_D < min_D:
            res[index] = np.mean(data[I], axis=0)
        
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
            elif np.linalg.norm(res[L_index-1] - res[R_index+1]) < 5:
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