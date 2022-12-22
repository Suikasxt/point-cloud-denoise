import numpy as np
import torch
import vis_pcd
import faiss
import cal_mae
import argparse

EPS=1e-5

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='../data/pcd001rainy.bin',
                        help='The data input path for the denoised.')
    parser.add_argument('--ref', type=str,
                        default='../data/pcd001clean.bin',
                        help='The data input path for the reference.')

    return parser.parse_args()
def pairwise_construct(x, repeat_num):
    for i in range(x.shape[0]):
        xi=np.expand_dims(x[i],1)
        xi=xi.repeat(repeat_num,1).T
        xi_delta=x-xi
        xi_dis=torch.linalg.norm(xi_delta,dim=1)
        idx=np.argsort(xi_dis)

    x_self = torch.unsqueeze(x, dim=1)
    x_self = x_self.repeat( 1, repeat_num, 1)
    x_others = torch.unsqueeze(x, dim=0)
    # x_others = x_others.repeat(repeat_num, 1, 1)
    x_self+=-x_others.repeat(repeat_num, 1, 1)
    return x_self

def recentAvg(pc,k):
    index = faiss.IndexFlatL2(3)
    pc_cor=np.copy(pc[:,:3],order='C')
    index.add(pc_cor)
    res = np.copy(pc[:,:3])
    avgs = []
    for point in res:
        _, I = index.search(np.array([point]), k+1)
        avg=np.mean(pc[I[0,1:]],0)
        avgs.append(avg)


    return avgs

def solve(pc,k,lam):
    avgs=np.array(recentAvg(pc, k))
    avgs[:,3]=avgs[:,3]
    pc[:,3]=pc[:,3]
    x_hat=(1-lam)*pc+lam*avgs

    return x_hat

def getBallcoordinates(data):
    newdata=[];
    r=np.sqrt(np.square(data[:,0])+np.square(data[:,1])+np.square(data[:,2]))
    theta=np.arccos(data[:,2]/(r+1e-6))
    phi=np.arctan(data[:,1]/(data[:,0]+1e-6))
    phi[(data[:,0]<0) & (data[:,0]>0)]=phi[(data[:,0]<0) & (data[:,0]>0)]+np.pi
    phi[(data[:, 0] < 0) & (data[:, 0] < 0)]= phi[(data[:, 0] < 0) & (data[:, 0] < 0)]-np.pi
    newdata.append(r)
    newdata.append(theta)
    newdata.append(phi)
    newdata.append(data[:,-1])
    return np.array(newdata).T
def getXYZ(newdata):
    data=[]
    x=newdata[:,0]*np.sin(newdata[:,1])*np.cos(newdata[:,2])
    y=newdata[:,0]*np.sin(newdata[:,1])*np.sin(newdata[:,2])
    z=newdata[:,0]*np.cos(newdata[:,1])
    data.append(x)
    data.append(y)
    data.append(z)
    data.append(newdata[:,-1])
    return np.array(data).T

def findnoise(newdata,k):
    datanorm=(newdata-newdata.min(0))/(newdata.max(0)-newdata.min(0))
    datanorm=datanorm[:, 1:3]
    index = faiss.IndexFlatL2(2)
    pc_cor = np.copy(datanorm,order='C')
    index.add(np.array(pc_cor))
    res = np.copy(datanorm)
    avgs = []
    avgs_I=[]
    for point in res:
        _, I = index.search(np.array([point]), k + 1)
        avg = np.mean(newdata[I[0, 1:]][:,0])
        avgs.append(avg)
        avg_I = np.mean(newdata[I[0, 1:]][:,-1])
        avgs_I.append(avg_I)

    avgs=np.array(avgs)
    avgs_I=np.array(avgs_I)


    delta=np.abs(avgs-newdata[:,0])

    # newdata[delta>np.std(delta),0]=avgs[delta>np.std(delta)]
    mul=6
    newdata[delta>np.mean(delta)+mul*np.std(delta),0]=avgs[delta>np.mean(delta)+mul*np.std(delta)]
    newdata[delta>np.mean(delta)+mul*np.std(delta),-1]=avgs_I[delta>np.mean(delta)+mul*np.std(delta)]


    return newdata

def fillzero(res):
    L_index = 0
    while L_index < res.shape[0]:
        if np.linalg.norm(res[L_index]) < EPS:
            R_index = L_index
            while R_index + 1 < res.shape[0] and np.linalg.norm(res[R_index + 1]) < EPS:
                R_index += 1

            if R_index == res.shape[0] - 1:
                res[L_index:] = res[L_index - 1]
            elif L_index == 0:
                res[:R_index + 1] = res[R_index + 1]
            elif np.linalg.norm(res[L_index - 1] - res[R_index + 1]) < 5:
                for i in range(L_index, R_index + 1):
                    lamb = (i - (L_index - 1)) / ((R_index + 1) - (L_index - 1))
                    res[i] = (1 - lamb) * res[L_index - 1] + lamb * res[R_index + 1]
            else:
                res[L_index:R_index + 1, :] = -1000

            L_index = R_index
        L_index += 1
    return res

cfg = config()
data_rainy = np.fromfile(cfg.input,dtype=np.float32,count=-1).reshape([-1,4])
data_rainy[:,-1]=data_rainy[:,-1]/data_rainy[:,-1].max()
data_rainy=torch.from_numpy(data_rainy)

data_clean = np.fromfile(cfg.ref,dtype=np.float32,count=-1).reshape([-1,4])
data_clean[:,-1]=data_clean[:,-1]/data_clean[:,-1].max()
data_clean = torch.from_numpy(data_clean)

data=data_rainy.numpy()

index=(1-((data[:,0]==0)&(data[:,1]==0)&(data[:,2]==0))).astype(np.bool)
# data=data[index]

# res = cal_mae.cal_mae(data_rainy[index], data_clean[index])
res = cal_mae.cal_mae(data_rainy, data_clean)

print("MAE results:", res)



newdata=getBallcoordinates(data)
# data=getXYZ(newdata)
# a=(data_rainy.numpy()[index]-data)

newdata=findnoise(newdata,10)
res=getXYZ(newdata)

res=fillzero(res)

#取消注释增加凸优化求解
# res=solve(res,4,0.1)
res=torch.from_numpy(res)

res_mae = cal_mae.cal_mae(res, data_clean)
print("My MAE results:", res_mae)


#可视化
res[res==-1000]=0
vis_pcd.viz_mayavi(res, vals="distance")
exit()


