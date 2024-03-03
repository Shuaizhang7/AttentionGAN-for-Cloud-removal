import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
from pdb import set_trace as st
import random
class UnalignedCocoDataset(BaseDataset):
    def initialize(self, opt):
        from pycocotools.coco import COCO
        self.opt = opt
        self.root = opt.dataroot
        self.isTrain = opt.isTrain
        self.dataType = opt.dataType
        annFile='{}/annotations/instances_{}.json'.format(self.root,self.dataType)
        self.coco = COCO(annFile)
        catIds_A = self.coco.getCatIds(catNms=[opt.A_cats])
        catIds_B = self.coco.getCatIds(catNms=[opt.B_cats])
        self.imgIds_A = self.coco.getImgIds(catIds=catIds_A )
        self.imgIds_B = self.coco.getImgIds(catIds=catIds_B )
        self.A_size = len(self.imgIds_A)
        self.B_size = len(self.imgIds_B)
        self.transform = get_transform(opt)
    def __getitem__(self, index):
        coco = self.coco
        index_A = index % self.A_size
        if self.isTrain:
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.B_size
        A_img_id = self.imgIds_A[index_A]
        B_img_id = self.imgIds_B[index_B]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_ann_ids = coco.getAnnIds(imgIds=A_img_id)
        B_ann_ids = coco.getAnnIds(imgIds=B_img_id)
        A_anns = coco.loadAnns(A_ann_ids)
        B_anns = coco.loadAnns(B_ann_ids)

        A_path = coco.loadImgs(A_img_id)[0]['file_name']
        B_path = coco.loadImgs(B_img_id)[0]['file_name']
        A_path = os.path.join(self.root, self.dataType, A_path)
        B_path = os.path.join(self.root, self.dataType, B_path)
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')


        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        return {'A': A_img, 'B': B_img,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedCocoDataset'
