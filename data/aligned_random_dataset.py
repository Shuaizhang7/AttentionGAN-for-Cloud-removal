import os.path
import random
from data.base_dataset import BaseDataset, get_transform
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class AlignedRandomDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))
        n_images = len(self.AB_paths)
        self.A_paths = self.AB_paths[:n_images/2]
        self.B_paths = self.AB_paths[n_images/2:]
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        width = A_img.size[0]
        height = A_img.size[1]
        A = A_img.crop((0,0,width/2,height))
        B = B_img.crop((width/2,0,width,height))

        A = self.transform(A)
        B = self.transform(B)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}
    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedRandomDataset'
