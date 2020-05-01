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
    ide = torch.eye(dim, requires_grad=False, dtype=torch.float, device=device)
    zer = torch.zeros([1,dim], requires_grad=False, dtype=torch.float, device=device)

    n = size // dim   #number of full repititions
    m = size % dim    #number of extra rows
    nsum = torch.zeros([1,size,3], requires_grad=False, dtype=torch.float, device=device)
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
        t.append(torch.tensor([1 if x == i else 0 for x in range(dim)],requires_grad=False, dtype=torch.float, device=device))
    
    #add partial identity to end of zero matrix and combine with prior identity matrix
    mat2 = torch.cat([torch.cat(dim*n*[zer]),torch.stack(t)])
    nsum = nsum + torch.matmul(mat2,w2)
    return nsum.view([3,size,1])

#Function to shift signal by arbitrary offset
def shift_operation(w,offsets):
    dim = w.shape[1]
    ide = torch.eye(dim, device=device)
    ides = torch.zeros(w.shape[0], dim, dim, device=device)
    for i, offset in enumerate(offsets):
        ides[3*i] = torch.cat([ide[-offset:], ide[:-offset]])
        ides[3*i+1] = torch.cat([ide[-offset:], ide[:-offset]])
        ides[3*i+2] = torch.cat([ide[-offset:], ide[:-offset]])
    return torch.bmm(ides,w)

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



#The shutter function is encoded into the convolution layer
lay = torch.nn.Conv1d(1,1,5)

#Manually setting the weights and bias so the  shutter acts as a box filter
lay.weight.data = torch.full([1,1,5,1], .2, requires_grad=True, dtype=torch.float, device=device)
lay.bias.data = torch.zeros(1, requires_grad=True, dtype=torch.float, device=device)

#Compute g(y) to get X_adv
def fttogy(w, batch, mask, c_limits):
    sz = w.shape[1]
    
    #stack the signal to fit the input size
    oot = stack(w,228)             
    
    # EOT sampling for ambient light and shift
    c = torch.rand([batch,1,1,1], device=device) * (c_limits[1] - c_limits[0]) + c_limits[0]
    shift = torch.randint(0, sz, (batch,))
    #shift = torch.from_numpy(np.array(range(0,batch,1)))
    
    #Shift the signal
    ootn = shift_operation(oot.unsqueeze(0).repeat(batch,1,1,1).view(-1, 228, 1), shift).view(batch,3,228,1)
    
    #Fit w into the range [0,1]. new_w is the same as ft
    new_w = .5 * (torch.tanh(ootn) + 1)
    
    #Convolution of ft and the shutter
    #gy = lay(new_w.unsqueeze(0).view([3,1,228,batch])).view([batch,3,224,1])
    gy = lay(new_w.transpose(0,3).transpose(0,1)).transpose(0,1).transpose(0,3)
    
    #Mask the signal to only affect the object
    gy_mask = gy * mask
    
    return (c + (1-c)*gy_mask), new_w