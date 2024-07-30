import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import pytorch_ssim
import torch
from torch.autograd import Variable
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
import math
import numpy


#fake = "/home/zs/PycharmProjects/cloud-removal-with-fusion-of-SAR-and-optical-images-main/results/experiment_name/result"
fake = "results/realcloud_1_10_1_5_2"
gt = "results/gt4"
list = os.listdir(fake)
list2 = os.listdir(gt)
ssimlist = []
list.sort()
list2.sort()
psnrlist = []
rmselist = []
for i in range(len(list)):
    img_fake = fake +'/'+list[i]
    img_th = gt +'/'+ list2[i]
    npImg1 = cv2.imread(img_fake)
    npImg2 = cv2.imread(img_th)
    psnr = PSNR(npImg2,npImg1)
    ssim = SSIM(npImg1,npImg2,multichannel=True)
    mse = MSE(npImg1,npImg2)
    rmse = mse**(0.5)
    ssimlist.append(ssim)
    psnrlist.append(psnr)
    rmselist.append(rmse)

print('ssim:')
print(sum(ssimlist) / len(ssimlist))
print('psnr:')
print(sum(psnrlist)/len(psnrlist))
print('rmse:')
print(sum(rmselist)/len(rmselist))
