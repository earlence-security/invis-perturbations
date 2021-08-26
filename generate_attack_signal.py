import argparse
from argparse import ArgumentParser
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
import os
from utils import *
import uuid
from tqdm import tqdm
import random

class SignalDataset(Dataset):

    def __init__(self, image_pairs, transform):
        
        self.transform = transform
        self.image_pairs = image_pairs
                      
    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        
        images = [Image.open(x) for x in self.image_pairs[idx]]

        if self.transform:
            sample = [self.transform(x) for x in images]
        else:
            sample = images

        return sample

def upsample2d(img, size=224):
    upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
    return upsample(torch.unsqueeze(img, 0))[0]

def main(args):
    
    ### SET RANDOM SEED
    '''
    if args.SEED != -1:
        from warnings import warn
        warn('Fix seed, you should see this warning ONLY ONCE.')
        import torch
        torch.manual_seed(args.SEED)
        torch.cuda.manual_seed(args.SEED)
        torch.cuda.manual_seed_all(args.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import numpy as np
        np.random.seed(args.SEED)
        import random
        random.seed(args.SEED)
    '''
    ### READ DATA, TRAIN-TEST SPLIT
    
    image_pairs = {}
    indices = {'amb':0, 'full':1}
    for path, subdirs, files in os.walk(args.image_dir):
        for name in files:
            if name.endswith('.png') or name.endswith('.jpg'):
                prefix,suffix = name.split('.')[0].split('_')
                if suffix in indices:
                    if prefix not in image_pairs:
                        image_pairs[prefix] = ['','']
                    image_pairs[prefix][indices[suffix]] = os.path.join(path,name)

    image_pairs = list(image_pairs.values())
    random.shuffle(image_pairs)
    
    train_image_dataset = SignalDataset(image_pairs[:args.num_train], transforms.ToTensor())
    
    val_image_dataset = SignalDataset(image_pairs[args.num_train:args.num_train+args.num_val], transforms.ToTensor())

    train_loader = DataLoader(train_image_dataset, 1, shuffle=True)
    val_loader = DataLoader(val_image_dataset, 1, shuffle=True)
    
    ### SETTING GPU
    device = torch.device("cuda:"+args.device_id if torch.cuda.is_available() else "cpu")
    
    ### LOADING MODEL
    pretrained_model = models.resnet101(pretrained=True)
    pretrained_model.to(device)
    pretrained_model.eval()    
    
    ### MODEL TRANSFORMATIONS
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    forward_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    ### SETTING CAMERA PARAMETERS
    exp_micros = 1000000/args.camera_exposure          # get exposure in microseconds
    img_ratio = 3024 / args.downscaled_image_size      # every row in model is img_ratio rows in original image
    upscaled_readout = args.cam_readout * img_ratio    # camera_tr = 10 # multiply real tr (10 micros) by img_ratio to find model tr
    conv_size = exp_micros / upscaled_readout      # divide exposure time by tr to find convolution size
    conv_size = int(conv_size)
    
    sig_size = args.downscaled_image_size + conv_size - 1 # 300   #Length of input signal
    channels = 3
    # change of variable term to optimise on
    w = torch.rand([channels,sig_size,1], dtype=torch.float, device=device)

    #Target and original class labels
    origs = [torch.tensor([int(x)], dtype=torch.long, device=device) for x in args.source_id.split(",")]
    targ = torch.tensor([args.target_id], dtype=torch.long, device=device)

    #Model parameters
    fit_in_range = False
    if args.opt_stat == "cw":
        optimizer = optim.Adam([w], lr=args.lr)
        fit_in_range = True
    loss_fn = nn.CrossEntropyLoss()
    
    #Track the loss to target and original class
    targloss = []
    origloss = []
    out = None
    #obj_dict = {}

    #Optimisation loop. initially untargeted
    for epoch in tqdm(range(args.epochs)):
        
        loss_track = []
        for i, image_pair in enumerate(train_loader):
            
            img_amb, img_full = image_pair
            #img_no = img_no.to(device)
            img_amb = img_amb[0].to(device)
            img_full = img_full[0].to(device)
            
            w.requires_grad_()

            #### Conditionally switch target

            if channels==1:
                n_w = w.repeat(3,1,1)
            else:
                n_w = w

            ### generate stripes
            gy, new_w = fttogy(n_w, args.batch_size, sig_size, conv_size, device, 0, shifting=True, fit_in_range=fit_in_range)

            ### noise

            # construct image
            captured_image = torch.pow(1e-6 + torch.pow(img_amb,2.2) + gy*(torch.pow(img_full,2.2)-torch.pow(img_amb,2.2)), 1/2.2)


            ### noise
            
            # send image to model
            resized_cap_image = torch.cat([forward_normalize(upsample2d(i,224)).unsqueeze(0) for i in captured_image])
            #transforms.ToPILImage()(inv_normalize(resized_cap_image[0])).save("x.png")
            #resized_cap_image = resized_cap_image.to(device)
            output = pretrained_model(resized_cap_image)

            ## calculate loss
            if args.target_id == -1:
                loss = -1*min([loss_fn(output, x.repeat(args.batch_size)) for x in origs])
            else:
                loss = loss_fn(output, targ.repeat(args.batch_size))
            
            loss.backward()
            
            loss_track.append(loss.cpu().data)
            
            if args.opt_stat=="cw":
                if (epoch*len(train_loader)+i)%args.update_freq==0:   # batch 32 === update after 10 for ~224
                    optimizer.step()
                    optimizer.zero_grad()
            elif args.opt_stat == "pgd":
                if (epoch*len(train_loader)+i)%args.update_freq==0:
                    grad = w.grad

                    w = w.detach() - args.lr * grad.sign()

                    w = torch.clamp(w, 0.0, 1.0)
        
        if epoch%100==0:
            print(f"Epoch {epoch}, Loss {np.array(loss_track).mean()}")
    
    
    ### evaluation loop
    ## SOFTMAX LAYER
    softmax = torch.nn.Softmax(dim=1)
    sourceid = int(args.source_id.split(",")[0])
    
    class_acc, target_acc, max_acc = [],[],[]
    obj_dict = {}
    for i, image_pair in tqdm(enumerate(val_loader)):

        img_amb, img_full = image_pair
        #img_no = img_no.to(device)
        img_amb = img_amb[0].to(device)
        img_full = img_full[0].to(device)
        
        for j in range(100):
        
            gy,_ = fttogy(n_w, args.batch_size, sig_size, conv_size, device, 0, shifting=True)

            captured_image = torch.pow(1e-6 + torch.pow(img_amb,2.2) + gy*(torch.pow(img_full,2.2)-torch.pow(img_amb,2.2)), 1/2.2)

            resized_cap_image = torch.cat([forward_normalize(upsample2d(i,224)).unsqueeze(0) for i in captured_image])
            #resized_cap_image = resized_cap_image.to(device)
            output = pretrained_model(resized_cap_image)
            prob = softmax(output)
            maxcls = prob.max(1)

            for i in range(args.batch_size):
                target_acc.append(prob[i][args.target_id].item())
                class_acc.append(prob[i][sourceid].item())
                max_acc.append((maxcls.indices[i].item(), prob[i][maxcls.indices[i].item()].item()))
            
            del gy, resized_cap_image, output, prob, maxcls
            
    print(f"TARGET CLASS {args.target_id} AVG CONFIDENCE - {np.array(target_acc).mean()}")
    print(f"SOURCE CLASS {sourceid} AVG CONFIDENCE - {np.array(class_acc).mean()}")
    print(f"TARGET CLASS {args.target_id} CLASSIFICATION RATE - {np.array([1 if x[0]==args.target_id else 0 for x in max_acc]).mean()}")
    print(f"TARGET CLASS {args.source_id} CLASSIFICATION RATE - {np.array([1 if x[0] in [int(k) for k in args.source_id.split(',')] else 0 for x in max_acc]).mean()}")
    print(f"MOST FREQ CLASSE {np.bincount([x[0] for x in max_acc]).argmax()}")
    freq_count = np.bincount([x[0] for x in max_acc])
    freq_count = sorted([(i,x) for i,x in enumerate(freq_count)], key=lambda x:x[1], reverse=True)
    print(f"TOP CLASSES {freq_count[:10]}")
    
    ### save the signal
    torch.save(n_w,args.outpath)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--camera_exposure', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--SEED', type=int, default=-1)
    parser.add_argument('--image_dir', type=str, default='./augmented_images')
    parser.add_argument('--num_train', type=int, default=10)
    parser.add_argument('--num_val', type=int, default=3)
    parser.add_argument('--device_id', type=str, default="0")
    parser.add_argument('--downscaled_image_size', type=int, default=252)
    parser.add_argument('--cam_readout', type=int, default=10)
    parser.add_argument('--source_id', type=str, default="504")
    parser.add_argument('--target_id', type=int, default=722)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--opt_stat', type=str, default='pgd')
    parser.add_argument('--update_freq', type=int, default=10)
    parser.add_argument('--outpath', type=str, default="./generated_attacks/attack.pt")

    main(parser.parse_args())