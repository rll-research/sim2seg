import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from collections import OrderedDict
import numpy as np

class Pix2SegModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='segmentation', output_nc=8,
                            no_flip=False, preprocess='none')
        parser.add_argument('--D_type', type=str, default='gumbel', choices=['gumbel', 'soft_amax', 'convolve'], help='discriminator setup for differentiability')
        parser.add_argument('--D_temp', type=float, default=10.0, help='temperature for discriminator sampling')
        parser.add_argument('--D_conv_channels', type=int, default=10, help='only used if D_type=convolve. # of channels to convolve pix, seg to')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_CE', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'fake_B_gumbel', 'fake_B_soft']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt)
        self.D_type = opt.D_type
        self.D_temp = opt.D_temp
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images
            # channels for D is input_nc + input_nc since discriminator will take in image (converted from seg) instead of seg output
            if self.D_type == "convolve":
                self.netD = networks.define_D(opt.D_conv_channels * 2, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.pix_network = nn.Sequential(nn.Conv2d(opt.input_nc, opt.D_conv_channels * 2, 3, padding=1),
                                                  nn.ReLU(),
                                                  nn.Conv2d(opt.D_conv_channels * 2, opt.D_conv_channels, 3, padding=1)).to(self.device)
                self.seg_network = nn.Sequential(nn.Conv2d(opt.output_nc, opt.D_conv_channels * 2, 3, padding=1),
                                                  nn.ReLU(),
                                                  nn.Conv2d(opt.D_conv_channels * 2, opt.D_conv_channels, 3, padding=1)).to(self.device)
            else:
                self.netD = networks.define_D(opt.input_nc + opt.input_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCE = torch.nn.CrossEntropyLoss()
            self.criterionL1 = torch.nn.L1Loss() 
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        self.step = 0

        BLACK = np.array([0, 0, 0]) # sky
        GREEN = np.array([0, 128, 1]) # tree / bushes / details
        BLUE = np.array([18, 21, 151]) # ground
        WHITE = np.array([255, 255, 255]) # road
        YELLOW = np.array([250, 228, 1]) # fences -> log
        RED = np.array([215, 10, 42]) # rock
        ORANGE = np.array([251, 125, 20]) # cliff -> rock
        PURPLE = np.array([214, 47, 251]) # logs

        # for decoding: the actual colors
        COLORS_LST_DECODE = [BLACK, GREEN, BLUE, WHITE, RED, PURPLE]
        COLORS_NP_DECODE = np.array(COLORS_LST_DECODE)
        COLORS_TORCH_DECODE = torch.from_numpy(COLORS_NP_DECODE) 
        self.COLORS_TORCH_DECODE = COLORS_TORCH_DECODE.to(self.device)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_val_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.val_real_A = input['A' if AtoB else 'B'].to(self.device)
        self.val_real_B = input['B' if AtoB else 'A'].to(self.device)

    def set_real_input(self, input):
        pass

    def get_fake_AB(self, detach=False):
        if self.D_type == "gumbel":
            fake_AB = torch.cat((self.real_A, self.fake_B_gumbel), 1)
            if detach:
                fake_AB = fake_AB.detach()
        elif self.D_type == "soft_amax":
            fake_AB = torch.cat((self.real_A, self.fake_B_soft), 1)
            if detach:
                fake_AB = fake_AB.detach()
        elif self.D_type == "convolve":
            probs = F.softmax(self.fake_B * self.D_temp, dim=1)
            if detach:
                probs = probs.detach()
            conv_pix = self.pix_network(self.real_A)
            conv_seg = self.seg_network(probs)
            fake_AB = torch.cat((conv_pix, conv_seg), 1)
        else:
            raise NotImplementedError
        return fake_AB

    def get_real_AB(self):
        if self.D_type == "gumbel":
            real_AB = torch.cat((self.real_A, self.real_B_pix), 1)
        elif self.D_type == "soft_amax":
            real_AB = torch.cat((self.real_A, self.real_B_pix), 1)
        elif self.D_type == "convolve":
            conv_pix = self.pix_network(self.real_A)
            conv_seg = self.seg_network(self.real_B)
            real_AB = torch.cat((conv_pix, conv_seg), 1)
        else:
            raise NotImplementedError
        return real_AB

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        fake_logits = self.fake_B.permute(0, 2, 3, 1) * self.D_temp 
        fake_samples = F.gumbel_softmax(fake_logits, tau=0.05, hard=True)
        self.fake_B_gumbel = (fake_samples @ self.COLORS_TORCH_DECODE.float()).permute(0, 3, 1, 2).float() / 255
        self.real_B_pix = self.COLORS_TORCH_DECODE[torch.argmax(self.real_B, dim=1).long()].permute(0, 3, 1, 2).float() / 255
        
        soft_amax = F.softmax(fake_logits, dim=3)
        self.fake_B_soft = (soft_amax @ self.COLORS_TORCH_DECODE.float()).permute(0, 3, 1, 2).float() / 255

    def forward_val(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with torch.no_grad():
            self.val_fake_B = self.netG(self.val_real_A)  # G(A)
            fake_logits = self.val_fake_B.permute(0, 2, 3, 1) * self.D_temp 
            fake_samples = F.gumbel_softmax(fake_logits, tau=0.05, hard=True)
            self.val_fake_B_gumbel = (fake_samples @ self.COLORS_TORCH_DECODE.float()).permute(0, 3, 1, 2).float() / 255
            self.val_real_B_pix = self.COLORS_TORCH_DECODE[torch.argmax(self.val_real_B, dim=1).long()].permute(0, 3, 1, 2).float() / 255
            
            soft_amax = F.softmax(fake_logits, dim=3)
            self.val_fake_B_soft = (soft_amax @ self.COLORS_TORCH_DECODE.float()).permute(0, 3, 1, 2).float() / 255

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = self.get_fake_AB(detach=True)
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = self.get_real_AB()
        pred_real = self.netD(real_AB)
        # detaching leads to unbalanced training though
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        if (self.loss_D > 0.3):
            self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and CE loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = self.get_fake_AB()
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN
        # Second, G(A) = B
        self.loss_G_CE = self.criterionCE(self.fake_B, torch.argmax(self.real_B, dim=1)) * self.opt.lambda_L1
        with torch.no_grad():
            real_B_pix = self.COLORS_TORCH_DECODE[torch.argmax(self.real_B, dim=1).long()].permute(0, 3, 1, 2).float() / 255
            fake_B_pix = self.COLORS_TORCH_DECODE[torch.argmax(self.fake_B, dim=1).long()].permute(0, 3, 1, 2).float() / 255
            self.loss_G_L1 = self.criterionL1(real_B_pix, fake_B_pix)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_CE
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        self.forward()                   # compute fake images: G(A)
        # update D
        if epoch > self.opt.no_d_until and self.step % self.opt.d_every == 0:
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # update G's weights
        self.step += self.opt.batch_size

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
            if 'gumbel' in name or 'soft' in name:
                pass
            elif 'fake_B' in name or 'real_B' in name:
                inds = getattr(self, name)
                img = self.COLORS_TORCH_DECODE[torch.argmax(inds, dim=1).long()].permute(0, 3, 1, 2).float()
                visual_ret[name] = img / 255
            visual_ret[name] = visual_ret[name] * 2 - 1

        if self.isTrain:
            self.forward_val()
            for name in self.visual_names:
                val_name = f"val_{name}"
                if isinstance(val_name, str):
                    visual_ret[val_name] = getattr(self, val_name)
                if 'gumbel' in val_name or 'soft' in val_name:
                    pass
                elif 'fake_B' in val_name or 'real_B' in val_name:
                    inds = getattr(self, val_name)
                    img = self.COLORS_TORCH_DECODE[torch.argmax(inds, dim=1).long()].permute(0, 3, 1, 2).float()
                    visual_ret[val_name] = img / 255
                visual_ret[val_name] = visual_ret[val_name] * 2 - 1
            
        return visual_ret
