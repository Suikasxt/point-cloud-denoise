import mayavi.mlab
import torch
import numpy as np
import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                        default='../data_new/pcd001clean.bin',
                        help='The data input path for visualizaiton.')
    return parser.parse_args()
cfg = config()


def viz_mayavi(points,vals="distance"):
    
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    r=points[:,3]


    print('max intensity:', max(r))
    print('min intensity:', min(r))
    d=torch.sqrt(x**2+y**2+z**2)

    if vals=="height":
        col=z
    else:
        col=d

    fig=mayavi.mlab.figure(cfg.input.split('\\')[-1], bgcolor=(0,0,0),size=(780,540))
    mayavi.mlab.points3d(x,y,z,
                         col,
                         mode="point",
                         colormap='hsv',
                         figure=fig,
                         )

    mayavi.mlab.show()
    

if __name__=="__main__":
    
    mypointcloud = np.fromfile(cfg.input,dtype=np.float32,count=-1).reshape([-1,4])
    mypointcloud = torch.from_numpy(mypointcloud)
    
    viz_mayavi(mypointcloud, vals="distance")
