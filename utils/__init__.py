import os
import time
import torch
import numpy as np
import cv2

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


def time_now():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def make_dirs(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        #print('Successfully make dirs: {}'.format(dir))
    else:
        #print('Existed dirs: {}'.format(dir))
        pass

def recover(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    inp = inp.astype(np.uint8)
    return inp

def to_edge(x):
    x = x.data.cpu()
    out = torch.FloatTensor(x.size(0), x.size(2), x.size(3))
    for i in range(x.size(0)):
        xx = recover(x[i,:,:,:])   # 3 channel, 256x128x3
        xx = cv2.cvtColor(xx, cv2.COLOR_RGB2GRAY) # 256x128x1
        xx = cv2.Canny(xx, 10, 200) #256x128
        xx = xx/255.0 - 0.5 # {-0.5,0.5}
        xx += np.random.randn(xx.shape[0],xx.shape[1])*0.1  #add random noise
        xx = torch.from_numpy(xx.astype(np.float32))
        out[i,:,:] = xx
    out = out.unsqueeze(1)
    return out.cuda()