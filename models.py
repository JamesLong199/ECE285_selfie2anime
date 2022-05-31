import torch
import torch.nn as nn
import numpy as np

class C7S1_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        X --> 7x7 conv (stride 1, reflection padding) --> InstanceNorm --> ReLU

        '''
        super(C7S1_K_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(C_in, C_out, (7,7), stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(C_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class D_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        X --> 3x3 conv (stride 2, reflection padding) --> InstanceNorm --> ReLU

        '''
        super(D_K_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(C_in, C_out, (3,3), stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(C_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)

    
class R_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        Residual block

        '''
        super(R_K_Block, self).__init__()

        self.short_cut = nn.Sequential()
        if C_in != C_out:
            self.short_cut = nn.Conv2d(C_in, C_out, (1,1))

        self.conv_block = nn.Sequential(
            nn.Conv2d(C_in, C_out, (3,3), stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(C_out),
            nn.ReLU(),
            nn.Conv2d(C_out, C_out, (3,3), stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(C_out),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.short_cut(x) + self.conv_block(x)
        return out


class U_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        X --> 3x3 transposed_conv (stride 2, reflection padding) --> InstanceNorm --> ReLU

        '''
        super(U_K_Block, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, (3,3), stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(C_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class C_K_Block(nn.Module):
    def __init__(self, C_in, C_out, use_norm=True, downsample=True):
        '''
        X --> 4x4 conv (stride 2, reflection padding) --> InstanceNorm --> LeakyReLU

        '''
        super(C_K_Block, self).__init__()

        self.conv_block = self.build_conv_block(C_in, C_out, use_norm, downsample)

    def build_conv_block(self, C_in, C_out, use_norm, downsample):
        conv_block = []

        if downsample:
            conv_block += [nn.Conv2d(C_in, C_out, (4,4), stride=2, padding=1)]
        else:
            conv_block += [nn.Conv2d(C_in, C_out, (4,4), stride=1, padding=1)]

        if use_norm:
            conv_block += [nn.InstanceNorm2d(C_out)]

        conv_block += [nn.LeakyReLU(negative_slope=0.2)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.generator_block = nn.Sequential(
            C7S1_K_Block(3, 64),     # 
            D_K_Block(64, 128),
            D_K_Block(128, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            R_K_Block(256, 256),
            U_K_Block(256, 128),
            U_K_Block(128, 64),
            C7S1_K_Block(64, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator_block(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator_block = nn.Sequential(
            C_K_Block(3, 64, use_norm=False),
            C_K_Block(64, 128),
            C_K_Block(128, 256),
            C_K_Block(256, 512, downsample=False),
            nn.Conv2d(512, 1, (4,4), stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator_block(x)



   


