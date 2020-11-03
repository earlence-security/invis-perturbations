import torch
from torch import nn
from torch import optim
from torchvision import transforms, utils, models
from matplotlib.pyplot import imshow
from tqdm import tqdm
import random
import seaborn as sns
import uuid
import json
from utils import *


def run_simulation(device, exposure, batch_size, image_cache_size, original_images, number_test_images, n_epochs, classes_to_skip, classidx, model_img_size=252, targidx=None, apply_transformations=True, ambient_light=None, color_noise=None, argument_list=None):
        
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    forward_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    pil_to_tensor = transforms.ToTensor()
    tensor_to_pil = transforms.ToPILImage()

    image_cache = generate_train_cache(image_cache_size, original_images[0], original_images[1], device, apply_transformations, ambient_light)
    
    pretrained_model = models.resnet101(pretrained=True)
    pretrained_model.to(device)
    pretrained_model.eval()    
    
    exp_micros = 1000000/exposure          # get exposure in microseconds
    img_ratio = 3024 / model_img_size      # every row in model is img_ratio rows in original image
    model_tr = 10 * img_ratio              # multiply real tr (10 micros) by img_ratio to find model tr
    conv_size = exp_micros / model_tr      # divide exposure time by tr to find convolution size
    conv_size = int(conv_size)
    
    sz = model_img_size + conv_size - 1 # 300   #Length of input signal
    c = 0    #Ambient light ratio
    c_limits = [0,0]
    batch = batch_size
    channels = 3
    # change of variable term to optimise on
    w = torch.rand([channels,sz,1], requires_grad=True, dtype=torch.float, device=device)

    #Target and original class labels
    orig = torch.tensor([classidx], dtype=torch.long, device=device)

    #Model parameters
    lr = 1e-1
    #optimizer = optim.SGD([w], lr=lr, momentum=0.9, nesterov=True)
    optimizer = optim.Adam([w], lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    #Track the loss to target and original class
    targloss = []
    origloss = []
    out = None
    #obj_dict = {}

    #Optimisation loop. initially untargeted
    for epoch in tqdm(range(n_epochs)):

        #Switch to targeted at halfway point
        if targidx==None:
            half = epoch < n_epochs//6
            if epoch == n_epochs//6:
                tops = out.topk(40).indices[0]
                for t in tops:
                    if t.item() not in classes_to_skip: 
                        targidx = t.item()
                #targidx = 722
                #targidx = tops[0].item() if tops[0].item() != classidx else tops[1].item()
                target = torch.tensor([targidx], dtype=torch.long, device=device)
                #print("Switching from untarget to target {}".format(targidx))
        else: 
            half = False
            target = torch.tensor([targidx], dtype=torch.long, device=device)
            
        if channels==1:
            n_w = w.repeat(3,1,1)
        else:
            n_w = w

        #n_w = torch.repeat_interleave(n_w, repeats=repeat_size, dim=1)

        sig_height = model_img_size + conv_size - 1
        gy, new_w, sh = fttogy(n_w, batch, None, c_limits, sig_height, conv_size, device, 0, shifting=True)

        #For resize post convolution
        #inp = gy*img_t
        #gy *= random.random()*0.7 + 0.7

        #Add noise to input signal
        if color_noise=='gy_noise':
            xx = 0.4*torch.rand((gy.size()[0],gy.size()[1]),device=device)-0.2
            xx = xx.unsqueeze(2).unsqueeze(3).repeat(1,1,252,1)
            gy += xx
            gy = torch.clamp(gy, min=0, max=1)

        # without brightness random
        #img_t, img_f = image_pair_cache[random.randint(0,cache_size-1)]
        #inp = torch.pow(0.0000001 + torch.pow(img_t,2.2) + gy*(torch.pow(img_f,2.2)-torch.pow(img_t,2.2)), 1/2.2)

        img_amb, img_bright, img_f = image_cache[random.randint(0,len(image_cache)-1)]
        inp = torch.pow(0.0000001 + torch.pow(img_bright,2.2) + gy*(torch.pow(img_f,2.2)-torch.pow(img_amb,2.2)), 1/2.2)

        #Gaussian Noise
        #inp = inp + torch.randn(inp.size(),device=device)*0.005
        #inp = torch.clamp(inp,min=0,max=1)

        #Uniform noise to each row
        #inp = inp*(1 + torch.rand(inp.size()[:-1],device=device).unsqueeze(-1)*0.4 - 0.2)
        #inp = torch.clamp(inp,min=0,max=1)

        #apply random color correction
        if color_noise=='random_poly':
            inp = torch.cat([imcorr2(i, get_polynomial(0.2)).unsqueeze(0) for i in inp])
            inp = torch.clamp(inp, min=0, max=1)

        #affine transforming the image
        if color_noise=='affine':
            inp = inp*(0.7+torch.rand(inp.size()[:-2],device=device).unsqueeze(-1).unsqueeze(-1)*0.6)+(-0.2+torch.rand(inp.size()[:-2],device=device).unsqueeze(-1).unsqueeze(-1)*0.4)
            inp = torch.clamp(inp, min=0, max=1)

    
        inp = torch.cat([forward_normalize(upsample2d(i,224)).unsqueeze(0) for i in inp])
        inp = inp.to(device)
        out = pretrained_model(inp)

        #Calculate Loss depended on if targeted or untargeted
        if not half: targLoss = loss_fn(out, target.repeat(batch))
        origLoss = loss_fn(out, orig.repeat(batch))
        loss = -origLoss if half else targLoss
        if epoch%10 == 0:
            targloss.append(0 if half else targLoss)
            origloss.append(origLoss)
        #if epoch%1000 == 0:
        #    if not half: print(targLoss, origLoss) 
        #    else: print(origLoss)
        loss.backward()   

        if epoch%10==0:   # batch 32 === update after 10 for ~224
            optimizer.step()
            optimizer.zero_grad()
            del loss
            torch.cuda.empty_cache()

        if epoch!=n_epochs-1:
            del inp
            del new_w
            
    optimizer = None

    if n_epochs!=1:
        file_name = str(uuid.uuid4())
        torch.save(new_w,'input_dump/'+file_name+'.pt')
        with open('input_dump/'+file_name+'.json','w') as fp:
            json.dump({str(key): str(value) for key, value in argument_list.items()}, fp)
    else:
        gy = torch.zeros(gy.size(),device=device)
    lay2 = torch.nn.Softmax(dim=1)
    
    #evaluate for different augmented images
    class_acc, target_acc, max_acc = [],[],[]
    obj_dict = {}
    for i in range(number_test_images):

        img_t2, img_b2, img_f2 = get_image_trip(original_images[0], original_images[1], device, apply_transformations, ambient_light)
        inp2 = torch.pow(0.0000001 + torch.pow(img_b2,2.2) + gy*(torch.pow(img_f2,2.2)-torch.pow(img_t2,2.2)), 1/2.2)

        inp2 = torch.cat([forward_normalize(upsample2d(i,224)).unsqueeze(0) for i in inp2])
        inp2 = inp2.to(device)
        out2 = pretrained_model(inp2)
        prob2 = lay2(out2)
        maxcls2 = prob2.max(1)
        for i in range(batch):
            target_acc.append(prob2[i][targidx].item())
            class_acc.append(prob2[i][classidx].item())
            max_acc.append((maxcls2.indices[i].item(), prob2[i][maxcls2.indices[i].item()].item()))
        del img_t2
        del img_f2
        del inp2
        del out2
        del prob2
        del maxcls2
        torch.cuda.empty_cache()

    pretrained_model = None
    lay2 = None
    inp=None
    new_w = None
    return max_acc, class_acc, target_acc, targloss, origloss, targidx

def generate_train_cache(image_cache_size, img, img_full, device, apply_transformations, ambient_light):
    image_cache = []
    for i in range(image_cache_size):
        image_cache.append(get_image_trip(img, img_full, device, apply_transformations, ambient_light))
    return image_cache
    
def get_image_trip(img, img_full, device, apply_trans=True, amblight = None):
    pil_to_tensor = transforms.ToTensor()
    repeat_size = int(3024/3024)
    img = img.resize((252,252))
    model_img_size = img.size[0]
    if apply_trans:
        img_t, img_b, img_f = aug_transform(img, img_full, amblight)
    elif amblight!=None:
        img_b = TF.adjust_brightness(img, amblight)
        img_t, img_f = img, img_full
    else:
        img_t, img_b, img_f = img, img, img_full
    img_t = pil_to_tensor(img_t)
    img_t = img_t.to(device)
    img_b = pil_to_tensor(img_b).to(device)
    img_f = pil_to_tensor(img_f.resize((252,252))).to(device)
    return img_t, img_b, img_f

def upsample2d(img, size=224):
    upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
    return upsample(torch.unsqueeze(img, 0))[0]

def aug_transform(image1, image2, brightness_multiplier=None, random_crop_size = 360, flipping = True, rotation_limit=45):
    # Adjust Brightness
    if brightness_multiplier==None:
        brightness_multiplier = 1 + random.random()
    imageb = TF.adjust_brightness(image1, brightness_multiplier)

    # Resize
    re_size = random.randint(252,random_crop_size)
    resize = transforms.Resize(re_size)
    image1 = resize(image1)
    image2 = resize(image2)
    imageb = resize(imageb)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        image1, output_size=(252, 252))
    image1 = TF.crop(image1, i, j, h, w)
    image2 = TF.crop(image2, i, j, h, w)
    imageb = TF.crop(imageb, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5 and flipping:
        image1 = TF.hflip(image1)
        image2 = TF.hflip(image2)
        imageb = TF.hflip(imageb)

    # Random vertical flipping
    if random.random() > 0.5 and flipping:
        image1 = TF.vflip(image1)
        image2 = TF.vflip(image2)
        imageb = TF.vflip(imageb)

    # Random Rotation
    rotate = transforms.RandomRotation(rotation_limit, resample=Image.BILINEAR)
    seed = random.randint(0,2**32)
    random.seed(seed)
    image1 = rotate(image1)
    random.seed(seed)
    image2 = rotate(image2)
    random.seed(seed)
    imageb = rotate(imageb)

    return image1, imageb, image2

def get_polynomial(margin):
    x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    out = []
    for i in range(3):
        y = [max(0, xi + random.random()*2*margin - margin) for xi in x]
        out.append(np.polyfit(x,y,3))
    return out
def imcorr2(repla, coeff):
    newrep = torch.zeros_like(repla, dtype=torch.float)
    rp = coeff[0]
    gp = coeff[1]
    bp = coeff[2]
    newrep[0] = rp[0] * torch.pow(repla[0], 3) + rp[1] * torch.pow(repla[0], 2) + rp[2] * repla[0] + rp[3]
    newrep[1] = gp[0] * torch.pow(repla[1], 3) + gp[1] * torch.pow(repla[1], 2) + gp[2] * repla[1] + gp[3]
    newrep[2] = bp[0] * torch.pow(repla[2], 3) + bp[1] * torch.pow(repla[2], 2) + bp[2] * repla[2] + bp[3]
    return newrep