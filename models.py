import torch
import torch.nn as nn
import numpy as np

class C7S1_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        X --> 7x7 conv (stride 1, reflection padding) --> InstanceNorm --> ReLU

        '''
        super(C7S1_K_Block, self).__init__()

        self.conv = nn.Conv2d(C_in, C_out, (7,7), stride=1, padding=1, padding_mode='reflect')

        self.inst_norm = nn.InstanceNorm2d(C_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        out = self.relu(x)
        return out

class D_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        X --> 3x3 conv (stride 2, reflection padding) --> InstanceNorm --> ReLU

        '''
        super(D_K_Block, self).__init__()

        self.conv = nn.Conv2d(C_in, C_out, (3,3), stride=2, padding=1, padding_mode='reflect')

        self.inst_norm = nn.InstanceNorm2d(C_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        out = self.relu(x)
        return out

    
class R_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        Residual block

        '''
        super(R_K_Block, self).__init__()

        self.conv_1 = nn.Conv2d(C_in, C_out, (3,3), stride=1, padding=1, padding_mode='reflect')
        self.inst_norm_1 = nn.InstanceNorm2d(C_out)

        self.conv_2 = nn.Conv2d(C_out, C_out, (3,3), stride=1, padding=1, padding_mode='reflect')
        self.inst_norm_2 = nn.InstanceNorm2d(C_out)

        self.short_cut = nn.Sequential()
        if C_in != C_out:
            self.short_cut = nn.Conv2d(C_in, C_out, (1,1))
        
        self.relu = nn.ReLU()

    def forward(self, x):
        res_val = self.short_cut(x)

        x = self.conv_1(x)
        x = self.inst_norm_1(x)
        x = self.relu(x)

        x = self.conv_2(x)
        x = self.inst_norm_2(x)
        x = self.relu(x)

        out = x + res_val
        return out


class U_K_Block(nn.Module):
    def __init__(self, C_in, C_out):
        '''
        X --> 3x3 transposed_conv (stride 2, reflection padding) --> InstanceNorm --> ReLU

        '''
        super(U_K_Block, self).__init__()

        self.upconv = nn.ConvTranspose2d(C_in, C_out, (3,3), stride=2, padding=1)

        self.inst_norm = nn.InstanceNorm2d(C_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.inst_norm(x)
        out = self.relu(x)
        return out


class C_K_Block(nn.Module):
    def __init__(self, C_in, C_out, use_norm=True):
        '''
        X --> 3x3 conv (stride 2, reflection padding) --> InstanceNorm --> ReLU

        '''
        super(C_K_Block, self).__init__()

        self.conv = nn.Conv2d(C_in, C_out, (4,4), stride=2, padding=1)

        self.inst_norm = nn.Sequential()
        if use_norm:
            self.inst_norm.add_module("inst_norm", nn.InstanceNorm2d(C_out)) 
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        out = self.leaky_relu(x)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.c7s1_64 = C7S1_K_Block(3, 64)
        self.d_128 = D_K_Block(64, 128)
        self.d_256 = D_K_Block(128, 256)
        self.r_256_1 = R_K_Block(256, 256)
        self.r_256_2 = R_K_Block(256, 256)
        self.r_256_3 = R_K_Block(256, 256)
        self.r_256_4 = R_K_Block(256, 256)
        self.r_256_5 = R_K_Block(256, 256)
        self.r_256_6 = R_K_Block(256, 256)
        self.r_256_7 = R_K_Block(256, 256)
        self.r_256_8 = R_K_Block(256, 256)
        self.r_256_9 = R_K_Block(256, 256)
        self.u_128 = U_K_Block(256, 128)
        self.u_64 = U_K_Block(128, 64)
        self.c7s1_3 = C7S1_K_Block(64, 3)

    def forward(self, x):
        x = self.c7s1_64(x)
        x = self.d_128(x)
        x = self.d_256(x)
        x = self.r_256_1(x)
        x = self.r_256_2(x)
        x = self.r_256_3(x)
        x = self.r_256_4(x)
        x = self.r_256_5(x)
        x = self.r_256_6(x)
        x = self.r_256_7(x)
        x = self.r_256_8(x)
        x = self.r_256_9(x)
        x = self.u_128(x)
        x = self.u_64(x)
        out = self.c7s1_3(x)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.c_64 = C_K_Block(3, 64, use_norm=False)
        self.c_128 = C_K_Block(64, 128)
        self.c_256 = C_K_Block(128, 256)
        self.c_512 = C_K_Block(256, 512)

        self.conv_1 = nn.Conv2d(512, 1, (4,4), stride=1, padding=1)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.c_64(x)
        x = self.c_128(x)
        x = self.c_256(x)
        x = self.c_512(x)

        x = self.conv_1(x)
        x = self.adaptive_avg_pool(x)
        scores = self.flatten(x)

        return scores



   


