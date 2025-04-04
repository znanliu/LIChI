#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name : LIChI
# Copyright (C) Inria,  Sébastien Herbreteau, Charles Kervrann, All Rights Reserved, 2022, v1.0.

import torch
from torchvision.io import read_image, write_png
from torchvision.transforms.functional import rgb_to_grayscale, resize
from pytorch_msssim import ssim
from lichi import LIChI
import argparse
import time
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument("--sigma", type=float, dest="sigma", help="Standard deviation of the noise (noise level). Should be between 0 and 50.", default=15)
parser.add_argument("--in", type=str, dest="path_in", help="Path to the image to denoise (PNG or JPEG).", default="./test_images/URBAN100.png")
parser.add_argument("--out", type=str, dest="path_out", help="Path to save the denoised image.", default="./denoised.png")
parser.add_argument("--add_noise", action='store_true', help="Add artificial Gaussian noise to the image.")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# # Denoising
model = LIChI()

imgpath =["./test_images/BSD68.png","./test_images/SET12.png","./test_images/URBAN100.png"]
outpath = "/Users/yangtaewon/Workspace/LIChI/denoised_img"
method = ["50_knownsigma_BSD68","50_knownsigma_SET12","50_knownsigma_URBAN100"]

for j in range(len(imgpath)):
    new_folder = os.path.join(outpath, method[j])
    counter = 1

    while os.path.exists(new_folder):
        new_folder = os.path.join(outpath, f"{method[j]}-{counter}")
        counter += 1

    os.makedirs(new_folder)
    PSNR = []
    SSIM = []
    picklefile = os.path.join(new_folder, 'metric.pickle')
    img = read_image(imgpath[j]).float().to(device)
    img = rgb_to_grayscale(img)
    img = resize(img, [256, 256])
    img = img[None, :, :, :]

    for i in range(51):
        print(f"{i} variance")
        img_noisy = img + (i) * torch.randn_like(img)
        try:
        # den = model(img_noisy, sigma=i+1, constraints='affine', method='n2n', p1=9, p2=6, k1=16, k2=64, w=65, s=3, M=6) #for 10
            den = model(img_noisy, sigma=i+1, constraints='affine', method='n2n', p1=13, p2=6, k1=16, k2=64, w=65, s=3, M=11) #for 50
            den = den.clip(0, 255)
            psnr = 10 * torch.log10(255**2 / torch.mean((den - img) ** 2))
            print(psnr.item())
            ssim_val = ssim(den, img, size_average=True)
            print(ssim_val.item())
            PSNR.append(psnr.item())
            SSIM.append(ssim_val.item())
        except Exception as e:
            print(f"Error at iteration {i}")
            den = img.clone()
            psnr = 0
            PSNR.append(0)
            SSIM.append(0)
            continue
        
        
        metrics = {
        'PSNR': PSNR,
        'SSIM': SSIM
        }
        paths = os.path.join(new_folder, f"{i}.png")
        write_png(den[0, :, :, :].byte().to("cpu"), paths)

        with open(picklefile, 'wb') as handle:
            pickle.dump(metrics, handle)
