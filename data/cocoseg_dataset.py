import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import numpy as np
import torch
import random

class CocoSegDataset(BaseDataset):
    def initialize(self, opt):
        from pycocotools.coco import COCO
        self.opt = opt
        self.root = opt.dataroot
        self.dataType = opt.dataType
        self.isTrain = opt.isTrain
        if self.dataType == 'test2017':
            annFile = '{}/annotations/image_info_{}.json'.format(self.root, self.dataType)
        else:
            annFile='{}/annotations/instances_{}.json'.format(self.root, self.dataType)
        self.coco = COCO(annFile)
        self.catIds_A = self.coco.getCatIds(catNms=[opt.A_cats])
        self.catIds_B = self.coco.getCatIds(catNms=[opt.B_cats])
        self.imgIds_A = self.coco.getImgIds(catIds=self.catIds_A )
        self.imgIds_B = self.coco.getImgIds(catIds=self.catIds_B )
        self.A_size = len(self.imgIds_A) - 1
        self.B_size = len(self.imgIds_B)
        self.transform = get_transform(opt)
    def __getitem__(self, index):
        coco = self.coco
        index_A = index % self.A_size
        # index_A=1238
        # index_A = 10
        if self.isTrain:
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.B_size
        # index_B=1366
        # index_B = 14
        A_img_id = self.imgIds_A[index_A]
        B_img_id = self.imgIds_B[index_B]
        # A_img_id = 427523
        # B_img_id = 403916
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_ann_ids = coco.getAnnIds(imgIds=A_img_id, catIds=self.catIds_A, iscrowd=None)
        B_ann_ids = coco.getAnnIds(imgIds=B_img_id, catIds=self.catIds_B, iscrowd=None)
        A_anns = coco.loadAnns(A_ann_ids)
        B_anns = coco.loadAnns(B_ann_ids)

        A_path = coco.loadImgs(A_img_id)[0]['file_name']
        A_path = os.path.join(self.root, self.dataType, A_path)
        B_path = coco.loadImgs(B_img_id)[0]['file_name']
        B_path = os.path.join(self.root, self.dataType, B_path)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_seg = self.getSegMask(A_anns)
        B_seg = self.getSegMask(B_anns)
        A_img.putalpha(A_seg)
        B_img.putalpha(B_seg)

        i=0
        A_input = self.transform(A_img)
        while (torch.sum(A_input[3])<5):
        # make sure target is large than 10 pixel after cropping
            A_input = self.transform(A_img)
            i+=1
            if i==100:
                print('exit target_A\'s total pixel less than 5, what should I do?')
                print('id:',A_img_id)
                break

        i=0
        B_input = self.transform(B_img)
        while (torch.sum(B_input[3])<5):
        # make sure target is large than 10 pixel after cropping
            B_input = self.transform(B_img)
            i+=1
            if i==100:
                print('exit target_B\'s total pixel less than 5, what should I do?')
                print('id:',B_img_id)
                break
        return {'A': A_input, 'B': B_input,
                # 'A_img_id': A_img_id, 'B_img_id': B_img_id}
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedCocoSegDataset'
    def getSegMask(self, anns):
        coco=self.coco
        for i,anns_i in enumerate(anns):
            if i==0:
                mask=np.asarray(coco.annToMask(anns_i))
            else:
                mask+=np.asarray(coco.annToMask(anns_i))
        mask[mask>1]=1
        mask=torch.from_numpy(mask[np.newaxis,:,:]*255)
        ToPil=transforms.ToPILImage()
        seg=ToPil(mask)
        return seg

