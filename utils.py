import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn
from torch import optim
import json
from PIL import Image
import io
import cv2
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import pickle
import urllib.request
import requests
from matplotlib.pyplot import imshow
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Helper Functions

#Function to view tensor
def imshow_tensor(img, inv_normalize):
    img = inv_normalize(img)     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
#Function to save image
def saveim(img, name,inv_normalize):
    img = inv_normalize(img.cpu())
    save_image(img, name)

#Function to read in image
def readim(name,forward_normalize):
    image = Image.open(name)
    x = TF.to_tensor(image)
    x = forward_normalize(x)
    x.unsqueeze_(0)
    return x

#Function to replicate signal to match specified size
def stack(w,size):
    dim = len(torch.flatten(w)) // 3
    if dim == size:
        return w
    
    #Stacked identity functions are used to replicate the signal
    ide = torch.eye(dim, requires_grad=True, dtype=torch.float, device=device)
    zer = torch.zeros([1,dim], requires_grad=True, dtype=torch.float, device=device)

    n = size // dim   #number of full repititions
    m = size % dim    #number of extra rows
    nsum = torch.zeros([1,size,3], requires_grad=True, dtype=torch.float, device=device)
    ides = torch.cat(n*[ide])   #stacked identities
    
    #if extra rows, add rows of zeros to the end of the stacked identities
    if m != 0:
        zers = torch.cat(m*[zer])
        mat = torch.cat([ides,zers])
    else:
        mat = ides
        
    #Rearange input to allow for matrix multiplication
    w2 = w.view([1,dim,3])
    
    #Multiply by stacked identities
    nsum = nsum + torch.matmul(mat,w2)
    if m == 0:
        return nsum.view([3,size,1])
    
    #Create partial identity matrix for m rows
    t = []
    for i in range(m):
        t.append(torch.tensor([1 if x == i else 0 for x in range(dim)],requires_grad=True, dtype=torch.float, device=device))
    
    #add partial identity to end of zero matrix and combine with prior identity matrix
    mat2 = torch.cat([torch.cat(dim*n*[zer]),torch.stack(t)])
    nsum = nsum + torch.matmul(mat2,w2)
    return nsum.view([3,size,1])

#Function to shift signal by arbitrary offset
def shift_operation(w,offset):
    dim = len(torch.flatten(w))//3
    ide = torch.eye(dim, device=device)
    for i in range(dim):
        ide[i][i] = 0
        ide[(i+offset)%dim][i] = 1
    return torch.matmul(ide,w)

#Function to deal with gamma correction?
def gamma_correction(img, factor):
    return ((img+0.5)**factor)-0.5

#Function to scale the signal
def scale_operation(w, scale):
    dim = len(torch.flatten(w))//3
    n_dim = int(dim/scale)
    w = w.unsqueeze(0)
    t = nn.Upsample(size=(dim,1), mode='bilinear')
    n_w = t(w[:,:,:n_dim])
    return n_w[0]

#Function to get one colour channel of a signal
def split(inp,chan):
    size = inp[0].shape
    inp2 = copy.deepcopy(inp.detach())
    for i in range(len(inp)):
        if i != chan:
            inp2[i] = torch.zeros(size)
    return inp2