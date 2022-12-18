import math
import torch
import numpy as np
import argparse
import faiss

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--input', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--vis', type=str, default='')
    cfg = parser.parse_args()
    print('Work on pic No.%d'%cfg.id)
    if not cfg.input:
        cfg.input = '../data_new/pcd%03drainy.bin'%cfg.id
    if not cfg.output:
        cfg.output = '../data_new/pcd%03ddenoise.bin'%cfg.id
    if not cfg.vis:
        cfg.vis = '../data_new/pcd%03dvis.bin'%cfg.id
    cfg.R = [55, 33, 35, 100][cfg.id - 1]
    return cfg
cfg = config()

def denoise(data):
    index = faiss.IndexFlatL2(4)
    index.add(data)
    res = np.copy(data)
    dist = []
    for point in res:
        D, I = index.search(np.array([point]), 4)
        dist.append(np.mean(D) / np.linalg.norm(point))
    
    index = np.argsort(dist)
    noise_point_rate = 1 - 1000/(1000 + cfg.R)
    print('noise_point_rate', noise_point_rate)
    noise_point_num = int(data.shape[0] * noise_point_rate)
    for i in index[-noise_point_num:]:
        res[i, :] = -1000
    
    return res

if __name__ == '__main__':
    data = np.fromfile(cfg.input,dtype=np.float32,count=-1).reshape([-1,4])
    res = denoise(data)
    with open(cfg.output, 'wb') as f:
        fortran_data = np.asfortranarray(res, 'float32')
        fortran_data.tofile(f)
    
    with open(cfg.vis, 'wb') as f:
        res[res==-1000] = 0
        fortran_data = np.asfortranarray(res, 'float32')
        fortran_data.tofile(f)