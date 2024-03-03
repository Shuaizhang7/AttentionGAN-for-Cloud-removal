import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from pdb import set_trace as st
###############################################################################
# Functions
###############################################################################
import torchvision

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        #init.uniform(m.weight.data, 1.0, 0.02)
        init.uniform(m.weight.data, 0.02, 1.0)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,track_running_stats=True)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'DUNET':
        netG = DUNET(input_nc, output_nc, gpu_ids=gpu_ids)
    elif which_model_netG == 'DUNET2':
        netG = DUNET2(input_nc, output_nc, gpu_ids=gpu_ids)
    elif which_model_netG == 'DUNET3':
        netG = DUNET3(gpu_ids=gpu_ids)
    elif which_model_netG == 'encoder':
        netG = EncoderGenerator(input_nc, ngf=64, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'decoder':
        netG = DecoderGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'gated_9blocks':
        netG = GatednetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'inserted_9blocks':
        netG = MaskDecoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'attn_in_9blocks':
        netG = AttnInNetGenerator(input_nc, output_nc, ngf,norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'attn_vgg_9blocks':
        netG = AttnVGGNetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids, isTrain=isTrain)
    elif which_model_netG == 'attn_gated_9blocks':
        netG = AttnGatedGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        # netG.cuda(device_id=gpu_ids[0])
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

def define_G2(input_nc, output_nc, ngf, which_model_netG2, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG2 = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG2 == 'resnet_9blocks':
        netG2 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG2 == 'resnet_6blocks':
        netG2 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG2 == 'DUNET':
        netG2 = DUNET(input_nc, output_nc, gpu_ids=gpu_ids)
    elif which_model_netG2 == 'DUNET2':
        netG2 = DUNET2(input_nc, output_nc, gpu_ids=gpu_ids)
    elif which_model_netG2 == 'unet_128':
        netG2 = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG2 == 'unet_256':
        netG2 = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG2)
    if len(gpu_ids) > 0:
        # netG.cuda(device_id=gpu_ids[0])
        netG2.cuda(gpu_ids[0])
    init_weights(netG2, init_type=init_type)
    return netG2
def define_G1(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netG1 = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG1 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG1 = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG1 = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    netG1 = nn.Sequential(netG1, Norm())
    if len(gpu_ids) > 0:
        # netA.cuda(device_id=gpu_ids[0])
        netG1.cuda(gpu_ids[0])
    init_weights(netG1, init_type=init_type)
    return netG1

def define_A(input_nc, output_nc, ngf, which_model_netA, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[]):
    netA = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netA == 'resnet_9blocks':
        netA = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netA == 'resnet_6blocks':
        netA = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netA == 'unet_256':
        netA = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netA)
    netA = nn.Sequential(netA, Norm())
    if len(gpu_ids) > 0:
        # netA.cuda(device_id=gpu_ids[0])
        netA.cuda(gpu_ids[0])
    init_weights(netA, init_type=init_type)
    return netA


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], Norm=False):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids,use_Norm=Norm)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids,use_Norm=Norm)
    elif which_model_netD == 'attn_implicit':
        netD = AttnImDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'attn_groundtruth':
        netD = AttnGtDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'attn_infeat':
        netD = AttnfeatDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multiscale_layers':
        netD = MultiscaleNLayerDiscriminator(input_nc, ndf, n_layers=[5], norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'multiscale':
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        # netD.cuda(device_id=gpu_ids[0])
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def Attn1_G(encoder,transformer,decoder):
    netG = AttnNet1(encoder,transformer,decoder)
    return netG
def Attn2_G(encoder, decoder):
    netG = AttnNet2(encoder, decoder)
    return netG
def Onesided_G(encoder, transformer, decoder):
    netG = AutoNet(encoder, transformer, decoder)
    return netG
##############################################################################
# Classes
##############################################################################
# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        # if use_hingeloss:
        #     self.loss=nn.MSELoss()
        # if use_wgan:
        #     self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real, loss_weight=None):
        if type(input) == list:
            if loss_weight == None:
                loss_weight = []
                for i in range(len(input)):
                    loss_weight.append(1./len(input))
            target_tensor=self.get_target_tensor(input[0], target_is_real)
            loss = self.loss(input[0], target_tensor)
            if len(input)>1:
                loss= loss*loss_weight[0]
                for i in range(1,len(input)):
                    target_tensor=self.get_target_tensor(input[i], target_is_real)
                    loss += self.loss(input[i], target_tensor)*loss_weight[i]
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)
    def attnLoss(self, input, target_tensor, attn_map):
        target_tensor = self.get_target_tensor(input, target_tensor) * attn_map.detach()
        return self.loss(input * attn_map, target_tensor)

class vgg_normalize:
    def __init__(self,bz):
        self.size = (224,224)
        self.mean_data=0.5
        self.std_data=0.5
        self.mean_vgg=Variable(torch.Tensor(bz,3,1,1)).cuda()
        self.mean_vgg[:,0]=0.485; self.mean_vgg[:,1]=0.456; self.mean_vgg[:,2]=0.406
        self.std_vgg=Variable(torch.Tensor(bz,3,1,1)).cuda()
        self.std_vgg[:,0]=0.229; self.std_vgg[:,1]=0.224; self.std_vgg[:,2]=0.225
    def __call__(self, input):
        input=F.interpolate(input, size=self.size, mode='bilinear')
        vgg_input = input.mul(self.std_data).add(self.mean_data).sub(self.mean_vgg[:input.size(0)]).div(self.std_vgg[:input.size(0)])
        return vgg_input

class Maskloss(nn.Module):
    def __init__(self):
        super(Maskloss, self).__init__()
    def forward(self, input, target, mask, size_average=True):
        loss = (1-mask)*torch.abs(input-target)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class Norm(nn.Module):
    def __init(self):
        super(Norm, self).__init__()
    def forward(self, input):
        output = torch.clamp(torch.abs(input), 0 ,1)
        return output
class Normalization(nn.Module):
    """docstring for Normalization"""
    def __init__(self):
        super(Normalization, self).__init__()
    def forward(self, input):
        output = torch.div(input,torch.sum(torch.sum(input,2,keepdim=True),3,keepdim=True))
        return output
        
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),nn.Tanh()]
        self.model = nn.Sequential(*model)
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], use_Norm=False):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ] # 3*256*256 -> 64*128*128

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):# n_layers=3, [1,2]
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ] # 64*128*128 -> 128*64*64 -> 256*32*32

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ] # 256*32*32 -> 512*32*32

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # 512*32*32 -> 1*32*32

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        if use_Norm:
            sequence += [Norm(),Normalization()]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
    def attn(self, input):
        feat_net=nn.Sequential(*list(self.model.children())[:-1])
        attn = feat_net.forward(input)
        attn = attn[:,:,2:29,2:29]
        upsampler = nn.UpsamplingBilinear2d(size=256)
        attn = upsampler(torch.sum(torch.abs(attn),1,keepdim=True))
        attn = attn / torch.max(attn)
        return attn
    def pred(self, input):
        pred=self.model(input)
        pred[pred>1]=1
        pred[pred<0]=0
        upsampler = nn.UpsamplingBilinear2d(size=256)
        pred =  upsampler(pred)
        return pred





class conv_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(conv_block,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=(3,3),stride=1,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), stride=1, padding=1, padding_mode='reflect',bias = False),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class Downsample(nn.Module):
    def __init__(self,channel):
        super(Downsample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(3, 3), stride=2, padding=1,  bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)


class Upsample(nn.Module):
    def __init__(self,channel):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(channel,channel//2,kernel_size=(1,1),stride=1)

    def forward(self,x,featuremap):
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        x = self.conv1(x)
        x = torch.cat((x,featuremap),dim=1)
        return x

class DUNET(nn.Module):
    def __init__(self,in_channel,out_channel,gpu_ids):
        super(DUNET, self).__init__()
        out_channel = 4
        self.gpu_ids = gpu_ids
        self.layer1 = conv_block(in_channel,out_channel)
        self.layer2 = Downsample(out_channel)
        self.layer3 = conv_block(out_channel,out_channel*2)
        self.layer4 = Downsample(out_channel*2)
        self.layer5 = conv_block(out_channel*2,out_channel*4)
        self.layer6 = Downsample(out_channel*4)
        self.layer7 = conv_block(out_channel*4,out_channel*8)
        self.layer8 = Downsample(out_channel*8)
        self.layer9 = conv_block(out_channel*8,out_channel*16)
        self.layer10 = Upsample(out_channel*16)
        self.layer11 = conv_block(out_channel*16,out_channel*8)
        self.layer12 = Upsample(out_channel*8)
        self.layer13 = conv_block(out_channel*8,out_channel*4)
        self.layer14 = Upsample(out_channel*4)
        self.layer15 = conv_block(out_channel*4,out_channel*2)
        self.layer16 = Upsample(out_channel*2)
        self.layer17 = conv_block(out_channel*2,out_channel)
        self.layer18 = nn.Conv2d(out_channel,3,kernel_size=(1,1),stride=1)
        self.act = nn.Sigmoid()

        self.layer1_sar = conv_block(in_channel*2, out_channel)
        self.layer2_sar = Downsample(out_channel*2)
        self.layer3_sar = conv_block(out_channel*2, out_channel * 2)
        self.layer4_sar = Downsample(out_channel * 4)
        self.layer5_sar = conv_block(out_channel * 4, out_channel * 4)
        self.layer6_sar = Downsample(out_channel * 8)
        self.layer7_sar = conv_block(out_channel * 8, out_channel * 8)
        self.layer8_sar = Downsample(out_channel * 16)
        self.layer9_sar = conv_block(out_channel * 16, out_channel * 16)
        self.layer10_sar = Upsample(out_channel * 32)
        self.layer11_sar = conv_block(out_channel * 32, out_channel * 8)
        self.layer12_sar = Upsample(out_channel * 16)
        self.layer13_sar = conv_block(out_channel * 16, out_channel * 4)
        self.layer14_sar = Upsample(out_channel * 8)
        self.layer15_sar = conv_block(out_channel * 8, out_channel * 2)
        self.layer16_sar = Upsample(out_channel * 4)
        self.layer17_sar = conv_block(out_channel * 4, out_channel)
        self.layer18_sar = nn.Conv2d(out_channel, 3, kernel_size=(1, 1), stride=1)
        self.act_sar = nn.Sigmoid()

    def forward(self,x,x_sar):
        x_sar = torch.cat((x_sar, x), dim=1)
        x_sar = self.layer1_sar(x_sar)
        x = self.layer1(x)
        f1 = x
        f1_sar = x_sar
        x_sar = torch.cat((x_sar, x), dim=1)
        x_sar = self.layer2_sar(x_sar)
        x_sar = self.layer3_sar(x_sar)
        x = self.layer2(x)
        x = self.layer3(x)
        f2 = x
        f2_sar = x_sar
        x_sar = torch.cat((x_sar, x), dim=1)
        x_sar = self.layer4_sar(x_sar)
        x_sar = self.layer5_sar(x_sar)
        x = self.layer4(x)
        x = self.layer5(x)
        f3 = x
        f3_sar = x_sar
        x_sar = torch.cat((x_sar, x), dim=1)
        x_sar = self.layer6_sar(x_sar)
        x_sar = self.layer7_sar(x_sar)
        x = self.layer6(x)
        x = self.layer7(x)
        f4 = x
        f4_sar = x_sar
        x_sar = torch.cat((x_sar, x), dim=1)
        x_sar = self.layer8_sar(x_sar)
        x_sar = self.layer9_sar(x_sar)
        x = self.layer8(x)
        x = self.layer9(x)
        f5 = x

        x = self.layer10(x,f4)
        x = self.layer11(x)
        f6 = x
        x = self.layer12(x,f3)
        x = self.layer13(x)
        f7 = x
        x = self.layer14(x,f2)
        x = self.layer15(x)
        f8 = x
        #x = self.layer16(x,f1)
        #x = self.layer17(x)
        #x = self.layer18(x)


        x_sar = torch.cat((x_sar, f5), dim=1)
        f5 = torch.cat((f4_sar, f4), dim=1)
        x_sar = self.layer10_sar(x_sar, f5)
        x_sar = self.layer11_sar(x_sar)
        x_sar = torch.cat((x_sar, f6), dim=1)
        f6 = torch.cat((f3_sar, f3), dim=1)
        x_sar = self.layer12_sar(x_sar, f6)
        x_sar = self.layer13_sar(x_sar)
        x_sar = torch.cat((x_sar, f7), dim=1)
        f7 = torch.cat((f2_sar, f2), dim=1)
        x_sar = self.layer14_sar(x_sar, f7)
        x_sar = self.layer15_sar(x_sar)
        x_sar = torch.cat((x_sar, f8), dim=1)
        f8 = torch.cat((f1_sar, f1), dim=1)
        x_sar = self.layer16_sar(x_sar, f8)
        x_sar = self.layer17_sar(x_sar)
        x_sar = self.layer18_sar(x_sar)
        return x_sar


class DUNET2(nn.Module):
    def __init__(self,in_channel,out_channel,gpu_ids):
        super(DUNET2, self).__init__()
        out_channel = 4
        self.gpu_ids = gpu_ids
        self.layer1 = conv_block(in_channel,out_channel)
        self.layer2 = Downsample(out_channel)
        self.layer3 = conv_block(out_channel,out_channel*2)
        self.layer4 = Downsample(out_channel*2)
        self.layer5 = conv_block(out_channel*2,out_channel*4)
        self.layer6 = Downsample(out_channel*4)
        self.layer7 = conv_block(out_channel*4,out_channel*8)
        self.layer8 = Downsample(out_channel*8)
        self.layer9 = conv_block(out_channel*8,out_channel*16)
        self.layer10 = Upsample(out_channel*16)
        self.layer11 = conv_block(out_channel*16,out_channel*8)
        self.layer12 = Upsample(out_channel*8)
        self.layer13 = conv_block(out_channel*8,out_channel*4)
        self.layer14 = Upsample(out_channel*4)
        self.layer15 = conv_block(out_channel*4,out_channel*2)
        self.layer16 = Upsample(out_channel*2)
        self.layer17 = conv_block(out_channel*2,out_channel)
        self.layer18 = nn.Conv2d(out_channel,3,kernel_size=(1,1),stride=1)
        self.act = nn.Sigmoid()

        self.layer1_sar = conv_block(in_channel*2, out_channel)
        self.layer2_sar = Downsample(out_channel)
        self.layer3_sar = conv_block(out_channel, out_channel * 2)
        self.layer4_sar = Downsample(out_channel * 2)
        self.layer5_sar = conv_block(out_channel * 2, out_channel * 4)
        self.layer6_sar = Downsample(out_channel * 4)
        self.layer7_sar = conv_block(out_channel * 4, out_channel * 8)
        self.layer8_sar = Downsample(out_channel * 8)
        self.layer9_sar = conv_block(out_channel * 8, out_channel * 16)
        self.layer10_sar = Upsample(out_channel * 32)
        self.layer11_sar = conv_block(out_channel * 32, out_channel * 8)
        self.layer12_sar = Upsample(out_channel * 16)
        self.layer13_sar = conv_block(out_channel * 16, out_channel * 4)
        self.layer14_sar = Upsample(out_channel * 8)
        self.layer15_sar = conv_block(out_channel * 8, out_channel * 2)
        self.layer16_sar = Upsample(out_channel * 4)
        self.layer17_sar = conv_block(out_channel * 4, out_channel)
        self.layer18_sar = nn.Conv2d(out_channel, 3, kernel_size=(1, 1), stride=1)
        self.act_sar = nn.Sigmoid()

    def forward(self,x,x_sar):
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        x = self.layer3(x)
        f2 = x
        x = self.layer4(x)
        x = self.layer5(x)
        f3 = x
        x = self.layer6(x)
        x = self.layer7(x)
        f4 = x
        x = self.layer8(x)
        x = self.layer9(x)
        f5 = x

        x = self.layer10(x,f4)

        x = self.layer11(x)
        f6 = x
        x = self.layer12(x,f3)
        x = self.layer13(x)
        f7 = x
        x = self.layer14(x,f2)
        x = self.layer15(x)
        f8 = x
        x = self.layer16(x,f1)
        x = self.layer17(x)
        x = self.layer18(x)

        x_sar = torch.cat((x_sar, x), dim=1)
        x_sar = self.layer1_sar(x_sar)
        f1_sar = x_sar
        x_sar = self.layer2_sar(x_sar)
        x_sar = self.layer3_sar(x_sar)
        f2_sar = x_sar
        x_sar = self.layer4_sar(x_sar)
        x_sar = self.layer5_sar(x_sar)
        f3_sar = x_sar
        x_sar = self.layer6_sar(x_sar)
        x_sar = self.layer7_sar(x_sar)
        f4_sar = x_sar
        x_sar = self.layer8_sar(x_sar)
        x_sar = self.layer9_sar(x_sar)
        x_sar = torch.cat((x_sar, f5), dim=1)
        f5 = torch.cat((f4_sar, f4), dim=1)
        x_sar = self.layer10_sar(x_sar, f5)
        x_sar = self.layer11_sar(x_sar)
        x_sar = torch.cat((x_sar, f6), dim=1)
        f6 = torch.cat((f3_sar, f3), dim=1)
        x_sar = self.layer12_sar(x_sar, f6)
        x_sar = self.layer13_sar(x_sar)
        x_sar = torch.cat((x_sar, f7), dim=1)
        f7 = torch.cat((f2_sar, f2), dim=1)
        x_sar = self.layer14_sar(x_sar, f7)
        x_sar = self.layer15_sar(x_sar)
        x_sar = torch.cat((x_sar, f8), dim=1)
        f8 = torch.cat((f1_sar, f1), dim=1)
        x_sar = self.layer16_sar(x_sar, f8)
        x_sar = self.layer17_sar(x_sar)
        x_sar = self.layer18_sar(x_sar)
        return x,x_sar


class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        # 512-1024-512
        # 1024-512-256
        # 512-256-128
        # 256-128-64
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        x_out=self.Conv_BN_ReLU_2(x)
        x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out

class DUNET3(nn.Module):
    def __init__(self,gpu_ids):
        super(DUNET3, self).__init__()
        self.gpu_ids = gpu_ids
        out_channels=[2**(i+6) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64

        self.d1_sar = DownsampleLayer(6, out_channels[0])  # 6-64
        self.d2_sar = DownsampleLayer(out_channels[0]*2, out_channels[1])  # 64-128
        self.d3_sar = DownsampleLayer(out_channels[1]*2, out_channels[2])  # 128-256
        self.d4_sar = DownsampleLayer(out_channels[2]*2, out_channels[3])  # 256-512
        # 上采样
        self.u1_sar = UpSampleLayer(out_channels[3]*2, out_channels[3])  # 512-1024-512
        self.u2_sar = UpSampleLayer(out_channels[4]*2, out_channels[2])  # 1024-512-256
        self.u3_sar = UpSampleLayer(out_channels[3]*2, out_channels[1])  # 512-256-128
        self.u4_sar = UpSampleLayer(out_channels[2]*2, out_channels[0])  # 256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1]*2,out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],3,3,1,1),
            nn.BatchNorm2d(3),
            #nn.ReLU()
            #nn.Sigmoid()
            # BCELoss
        )
    def forward(self,x,x_sar):
        out_1,out1=self.d1(x)
        x_sar = torch.cat((x_sar, x), dim=1)
        out_1_sar, out1_sar = self.d1_sar(x_sar)

        out_2,out2=self.d2(out1)
        out1_sar = torch.cat((out1_sar, out1), dim=1)
        out_2_sar, out2_sar = self.d2_sar(out1_sar)

        out_3,out3=self.d3(out2)
        out2_sar = torch.cat((out2_sar, out2), dim=1)
        out_3_sar, out3_sar = self.d3_sar(out2_sar)

        out_4,out4=self.d4(out3)
        out3_sar = torch.cat((out3_sar, out3), dim=1)
        out_4_sar, out4_sar = self.d4_sar(out3_sar)

        out5=self.u1(out4,out_4)
        out4_sar = torch.cat((out4_sar, out4), dim=1)
        out5_sar = self.u1_sar(out4_sar, out_4_sar)

        out6=self.u2(out5,out_3)
        out5_sar = torch.cat((out5_sar, out5), dim=1)
        out6_sar = self.u2_sar(out5_sar, out_3_sar)

        out7=self.u3(out6,out_2)
        out6_sar = torch.cat((out6_sar, out6), dim=1)
        out7_sar = self.u3_sar(out6_sar, out_2_sar)

        out8=self.u4(out7,out_1)
        out7_sar = torch.cat((out7_sar, out7), dim=1)
        out8_sar = self.u4_sar(out7_sar, out_1_sar)
        out8_sar = torch.cat((out8_sar, out8), dim=1)
        out=self.o(out8_sar)
        return out