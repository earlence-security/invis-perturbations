import argparse
from argparse import ArgumentParser
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
import uuid
import random

def main(args):
    
    ## LOAD CAPTURED IMAGES
    full_image = Image.open(args.full_illum_image_path)
    amb_image = Image.open(args.amb_illum_image_path)
    
    ## RESIZE IMAGES
    full_image = full_image.resize((args.image_size, args.image_size))
    amb_image = amb_image.resize((args.image_size, args.image_size))
    
    ## GENERATE IMAGE TRIPLETS
    for i in range(args.num_samples):
        
        full_image_aug = full_image
        amb_image_aug = amb_image
        
        # RANDOM BRIGHTNESS ADJUST
        #amb_image_aug = TF.adjust_brightness(no_image_aug, 1 + (args.ambient_light_multiplier_limit - 1)*random.random())
        
        # RANDOM SCALE INCREASE
        target_image_size = random.randint(args.image_size, args.scale_increase_factor_limit*args.image_size)
        full_image_aug = full_image_aug.resize((target_image_size, target_image_size))
        amb_image_aug = amb_image_aug.resize((target_image_size, target_image_size))
        #no_image_aug = no_image_aug.resize((target_image_size, target_image_size))
        
        # RANDOM CROP TO ORIGINAL SIZE
        i, j, h, w = transforms.RandomCrop.get_params(full_image_aug, output_size=(args.image_size, args.image_size))
        full_image_aug = TF.crop(full_image_aug, i, j, h, w)
        amb_image_aug = TF.crop(amb_image_aug, i, j, h, w)
        #no_image_aug = TF.crop(no_image_aug, i, j, h, w)
        
        # RANDOM HORIZONTAL FLIP
        if random.random() > 0.5 and args.horizontal_flip==1:
            full_image_aug = TF.hflip(full_image_aug)
            amb_image_aug = TF.hflip(amb_image_aug)
            #no_image_aug = TF.hflip(no_image_aug)
            
        # RANDOM VERTICAL FLIP
        if random.random() > 0.5 and args.vertical_flip==1:
            full_image_aug = TF.vflip(full_image_aug)
            amb_image_aug = TF.vflip(amb_image_aug)
            #no_image_aug = TF.vflip(no_image_aug)
            
        # RANDOM ROTATION
        seed = random.randint(-1*args.rotation_limit,args.rotation_limit)
        full_image_aug = TF.rotate(full_image_aug, seed)
        amb_image_aug = TF.rotate(amb_image_aug, seed)
        #no_image_aug = TF.rotate(no_image_aug, seed)
        
        # SAVE
        pair_id = uuid.uuid4()
        full_image_aug.save(f"{args.out_dir}{pair_id}_full.png")
        amb_image_aug.save(f"{args.out_dir}{pair_id}_amb.png")
        #no_image_aug.save(f"{args.out_dir}{triplet_id}_no.png")

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--camera_exposure', type=int, default=2000)
    parser.add_argument('--rotation_limit', type=int, default=30)
    parser.add_argument('--ambient_light_multiplier_limit', type=float, default=1.0)
    parser.add_argument('--scale_increase_factor_limit', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=252)
    parser.add_argument('--horizontal_flip', type=int, default=0)
    parser.add_argument('--vertical_flip', type=int, default=0)
    parser.add_argument('--full_illum_image_path', type=str, default='./captured_images/full_illum_image.png')
    parser.add_argument('--amb_illum_image_path', type=str, default='./captured_images/amb_illum_image.png')
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--out_dir', type=str, default='./augmented_images/')
    

    main(parser.parse_args())