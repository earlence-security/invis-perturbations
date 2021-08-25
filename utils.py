import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from torch import nn
from torch import optim
import json
from PIL import Image
import io
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
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
import random
import math

#Function to replicate signal to match specified size
def stack(w,size,device):
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
    w2 = w.transpose(0,2)
    
    #Multiply by stacked identities
    nsum = nsum + torch.matmul(mat,w2)
    if m == 0:
        return nsum.transpose(0,2)
    
    #Create partial identity matrix for m rows
    t = []
    for i in range(m):
        t.append(torch.tensor([1 if x == i else 0 for x in range(dim)],requires_grad=False, dtype=torch.float, device=device))
    
    #add partial identity to end of zero matrix and combine with prior identity matrix
    mat2 = torch.cat([torch.cat(dim*n*[zer]),torch.stack(t)])
    nsum = nsum + torch.matmul(mat2,w2)
    return nsum.transpose(0,2)

#Function to shift signal by arbitrary offset
def shift_operation(w,offsets,device):
    dim = w.shape[1]
    ide = torch.eye(dim, device=device)
    ides = torch.zeros(w.shape[0], dim, dim, device=device)
    for i, offset in enumerate(offsets):
        ides[3*i] = torch.cat([ide[-offset:], ide[:-offset]])
        ides[3*i+1] = torch.cat([ide[-offset:], ide[:-offset]])
        ides[3*i+2] = torch.cat([ide[-offset:], ide[:-offset]])
    return torch.bmm(ides,w)

#Compute g(y) to get X_adv
def fttogy(w, batch, sig_height, conv_size, device, precision_depth=2, shifting=True, offset=None, fit_in_range=False):
    #The shutter function is encoded into the convolution layer
    lay = torch.nn.Conv1d(1,1,conv_size)

    #Manually setting the weights and bias so the  shutter acts as a box filter
    lay.weight.data = torch.full([1,1,conv_size,1], 1/conv_size, requires_grad=True, dtype=torch.float, device=device)
    lay.bias.data = torch.zeros(1, requires_grad=True, dtype=torch.float, device=device)
    
    sz = w.shape[1]             
    
    # EOT sampling for ambient light and shift
    if shifting:
        if offset != None:
            offset_arr = [x%sz for x in range(offset,offset+batch)]
            shift = torch.tensor(offset_arr, dtype=torch.int,device=device)
        else:
            shift = torch.randint(0, sz, (batch,),device=device)
    else: shift = torch.zeros((batch,),dtype=torch.int,device=device)
    #shift = torch.from_numpy(np.array(range(0,batch,1)))
    
    #Shift the signal
    oot = shift_operation(w.unsqueeze(0).repeat(batch,1,1,1).view(-1, sz, 1), shift, device).view(batch,3,sz,1)
    
    #stack the signal to fit the input size
    ootn = torch.stack([stack(ooti,sig_height,device) for ooti in oot])
    
    #Fit w into the range [0,1]. new_w is the same as ft
    if fit_in_range:
        new_w = .5 * (torch.tanh(ootn) + 1)
    else:
        new_w = ootn # .5 * (torch.tanh(ootn) + 1)
    
    #Convolution of ft and the shutter
    #gy = lay(new_w.unsqueeze(0).view([3,1,228,batch])).view([batch,3,224,1])
    gy = lay(new_w.transpose(0,3).transpose(0,1)).transpose(0,1).transpose(0,3)

    return gy, new_w