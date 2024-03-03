import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from models.base_model import BaseModel
from . import networks
import pytorch_ssim
import cv2
import sys
from pdb import set_trace as st

class CycleAttnGANModel_sar(BaseModel):
    def name(self):
        return 'CycleAttnGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.which_direction_model = opt.which_direction_model
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.input_nc, size, size)
        self.input_C = self.Tensor(nb, opt.input_nc, size, size)         #SAR

        self.zeros = self.Tensor(nb, 1, size, size)
        self.ones = self.Tensor(nb, 1, size, size)
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc+opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, 
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, 
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netA_A = networks.define_A(opt.input_nc, 1, 
                                        opt.ngf, opt.which_model_netA, opt.norm, 
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netA_B = networks.define_A(opt.input_nc, 1, 
                                        opt.ngf, opt.which_model_netA, opt.norm, 
                                        not opt.no_dropout, opt.init_type, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
        else:
            use_sigmoid=False
        self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                        opt.which_model_netD,
                                        opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            if self.which_direction_model=='AtoB':
                self.load_network(self.netG_A, 'G_A', which_epoch)
                self.load_network(self.netG_B, 'G_B', which_epoch)
                self.load_network(self.netA_A, 'A_A', which_epoch)
                self.load_network(self.netA_B, 'A_B', which_epoch)
            else:
                self.load_network(self.netG_A, 'G_B', which_epoch)
                self.load_network(self.netG_B, 'G_A', which_epoch)
                self.load_network(self.netA_A, 'A_B', which_epoch)
                self.load_network(self.netA_B, 'A_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_A = torch.optim.Adam(itertools.chain(self.netA_A.parameters(), self.netA_B.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_A)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        bz,c,h,w=input_A.size()
        self.zeros.resize_((bz,1,h,w)).fill_(0.0)
        self.ones.resize_((bz,1,h,w)).fill_(1.0)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_paths2 = input['C_paths']

    def mask_layer(self, foreground, background, mask):
        img = foreground * mask + background * (1 - mask)
        return img

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)

        self.zeros_attn = Variable(self.zeros, requires_grad=False)
        self.ones_attn = Variable(self.ones, requires_grad=False)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)


        fake_B = self.netG_A.forward(torch.cat([self.real_A,self.real_C],dim = 1))
        self.attn_real_A = self.netA_A.forward(self.real_A)


        self.fake_B = self.mask_layer(fake_B, self.real_A, self.attn_real_A)
        rec_A = self.netG_B.forward(self.fake_B)
        self.attn_fake_B = self.netA_B.forward(self.fake_B)
        self.rec_A = self.mask_layer(rec_A, self.fake_B, self.attn_fake_B)

        fake_A = self.netG_B.forward(self.real_B)
        self.attn_real_B = self.netA_B.forward(self.real_B)
        self.fake_A = self.mask_layer(fake_A, self.real_B, self.attn_real_B)

        rec_B = self.netG_A.forward(torch.cat([self.fake_B,self.real_C],dim = 1))
        self.attn_fake_A = self.netA_A.forward(self.fake_A)
        self.rec_B = self.mask_layer(rec_B, self.fake_A, self.attn_fake_A)
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A.forward(torch.cat([self.real_B,self.real_C],dim = 1))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B.forward(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss
        # D_A(G_A(A))

        self.real_AC = torch.cat([self.real_A,self.real_C],dim = 1)

        #fake_B = self.netG_A.forward(self.real_A+self.real_C)
        fake_B = self.netG_A.forward(self.real_AC)


        self.attn_real_A = self.netA_A.forward(self.real_A)
        self.fake_B = self.mask_layer(fake_B, self.real_A, self.attn_real_A)
        pred_fake = self.netD_A.forward(self.fake_B)

        ssim_loss = pytorch_ssim.SSIM()
        self.ssimloss1 = 1 - ssim_loss(self.fake_B, self.real_B)

        #self.loss_G_A = self.criterionGAN(pred_fake, True)+self.criterionIdt(self.fake_B,self.real_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)+self.ssimloss1

        # D_B(G_B(B))
        fake_A = self.netG_B.forward(self.real_B)
        self.attn_real_B = self.netA_B.forward(self.real_B)
        self.fake_A = self.mask_layer(fake_A, self.real_B, self.attn_real_B)
        pred_fake = self.netD_B.forward(self.fake_A)

        self.ssimloss2 = 1 - ssim_loss(self.fake_A,self.real_A)
        #self.loss_G_B = self.criterionGAN(pred_fake, True)+self.criterionIdt(self.fake_A,self.real_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)+self.ssimloss2
        # Forward cycle loss
        rec_A = self.netG_B.forward(self.fake_B)
        self.attn_fake_B = self.netA_B.forward(self.fake_B)
        self.rec_A = self.mask_layer(rec_A, self.fake_B, self.attn_fake_B)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss

        rec_B = self.netG_A.forward(torch.cat([self.fake_A,self.real_C],dim = 1))
        self.attn_fake_A = self.netA_A.forward(self.fake_A)
        self.rec_B = self.mask_layer(rec_B, self.fake_A, self.attn_fake_A)
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # attn constrain
        self.loss_attnsparse_A = self.criterionIdt(self.attn_real_A, self.zeros_attn) * self.opt.loss_attn_A
        self.loss_attnsparse_B = self.criterionIdt(self.attn_real_B, self.zeros_attn) * self.opt.loss_attn_B
        self.loss_attnconst_A = self.criterionIdt(self.attn_fake_A, self.attn_real_B.detach()) * self.opt.attn_cycle_weight
        self.loss_attnconst_B = self.criterionIdt(self.attn_fake_B, self.attn_real_A.detach()) * self.opt.attn_cycle_weight
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_attnsparse_A + self.loss_attnsparse_B  + self.loss_attnconst_A +self.loss_attnconst_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_A.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_A.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
    def optimize_parameterD(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_A.zero_grad()
        self.backward_G()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
    def get_current_errors(self):
        D_A = self.loss_D_A.data
        G_A = self.loss_G_A.data
        Cyc_A = self.loss_cycle_A.data
        D_B = self.loss_D_B.data
        G_B = self.loss_G_B.data
        Cyc_B = self.loss_cycle_B.data
        ssim_A = self.ssimloss1.data
        ssim_B = self.ssimloss2.data
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data
            idt_B = self.loss_idt_B.data
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B),('ssim_A', ssim_A),('ssim_B', ssim_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),('ssim_A', ssim_A),('ssim_B', ssim_B)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        real_C = util.tensor2im(self.real_C.data)



        # mask_attn_A = util.mask2im(util.tensor2mask(self.attn_real_A.data))
        # mask_attn_B = util.mask2im(util.tensor2mask(self.attn_real_B.data))

        attn_real_A = util.mask2heatmap(self.attn_real_A.data)
        attn_real_B = util.mask2heatmap(self.attn_real_B.data)
        attn_fake_A = util.mask2heatmap(self.attn_fake_A.data)
        attn_fake_B = util.mask2heatmap(self.attn_fake_B.data)
        attn_real_A = util.overlay(real_A, attn_real_A)
        attn_real_B = util.overlay(real_B, attn_real_B)
        attn_fake_A = util.overlay(fake_A, attn_fake_A)
        attn_fake_B = util.overlay(fake_B, attn_fake_B)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                            ('real_C',real_C),
                            ('rec_A', rec_A), ('attn_real_A', attn_real_A),
                            ('attn_fake_B', attn_fake_B),
                            ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                            ('attn_real_B', attn_real_B), ('attn_fake_A', attn_fake_A)#,('foreground_mask_B', mask_attn_B)
                            ])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netA_A, 'A_A', label, self.gpu_ids)
        self.save_network(self.netA_B, 'A_B', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
