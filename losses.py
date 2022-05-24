import torch
import torch.nn as nn
import torch.nn.functional as F

class GAN_Loss(nn.Module):
    '''
    patch_scores: original image (domain 1) --> fake image (domain 2) --> disciminator 2 --> patch scores 
    
    '''
    def __init__(self):
        super(GAN_Loss, self).__init__()
    
    def forward(self, patch_scores):
        loss = F.mse_loss(patch_scores, torch.ones_like(patch_scores))
        return loss
    
    
class Cycle_Loss(nn.Module):
    '''
    cycle consistency loss
    
    rec_img: original image (domain 1) --> fake image (domain 2) --> rec_img (domain 1)
                                 Generator 1-->2          Generator 2-->1
    
    '''
    def __init__(self):
        super(Cycle_Loss, self).__init__()
        
    def forward(self, rec_images, original_images):
        loss = F.l1_loss(rec_images, original_images)
        return loss
    
class Identity_Loss(nn.Module):
    '''
    id_img: original image (domain 1) --> id_img (domain 1)
                                Generator 2-->1
    
    '''
    def __init__(self):
        super(Identity_Loss, self).__init__()
    
    def forward(self, id_images, original_images):
        loss = F.l1_loss(id_images, original_images)
        return loss
    

class Discriminator_Loss(nn.Module):

    def __init__(self):
        super(Discriminator_Loss, self).__init__()

    def forward(self, real_scores, fake_scores):
        '''
        input
            -- real_scores: patchGAN output from a real image
            -- fake_scores: patchGAN output from a fake image
        '''

        mse_fake = F.mse_loss(fake_scores, torch.zeros_like(fake_scores))

        mse_real = F.mse_loss(real_scores, torch.ones_like(real_scores))

        loss = 0.5 * (mse_fake + mse_real)

        return loss