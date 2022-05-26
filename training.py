import torch
import torch.nn as nn
# from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import tqdm

from CycleGAN import CycleGAN
from dataset import get_full_list, ImageDataset, Selfie2Anime_Dataset

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

USE_GPU = True
print_every = 100
use_cuda = False

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    use_cuda = True

print('use_cuda:', use_cuda)

if __name__ == '__main__':

    test_A_list = get_full_list('data', 'testA')
    test_A_set = ImageDataset(test_A_list)

    test_B_list = get_full_list('data', 'testB')
    test_B_set = ImageDataset(test_B_list)

    S2A_test_set = Selfie2Anime_Dataset(test_A_set, test_B_set)
    
    train_A_list = get_full_list('data', 'trainA')
    train_A_set = ImageDataset(train_A_list)

    train_B_list = get_full_list('data', 'trainB')
    train_B_set = ImageDataset(train_B_list)

    S2A_train_set = Selfie2Anime_Dataset(train_A_set, train_B_set)
    
    train_data_loader = DataLoader(S2A_test_set, batch_size=1, shuffle=True, num_workers=0) 
    
    start_epoch = 0
    epochs = 1
    
    model = CycleGAN(use_cuda)

    loss_history = {}
    loss_history['loss_GAN'] = []
    loss_history['loss_cycle'] = []
    loss_history['loss_id'] = []
    loss_history['loss_G'] = []
    loss_history['loss_D1'] = []
    loss_history['loss_D2'] = []
    
    for epoch in range(start_epoch, epochs):
        
        model.set_mode_train()
        
        pbar = tqdm.tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        
        
        for i,data in pbar:
#             start_time = time.time()
            
            selfie_images, anime_images = data
            
            if use_cuda:
                selfie_images = selfie_images.cuda()
                anime_images = anime_images.cuda()
            
#             prepare_time = start_time - time.time()
            
            model.set_input_images(selfie_images, anime_images)
            model.update_gradient_G()
            model.update_gradient_D()
            
            loss_history['loss_GAN'].append(model.loss_GAN)
            loss_history['loss_cycle'].append(model.loss_cycle)
            loss_history['loss_id'].append(model.loss_id)
            loss_history['loss_G'].append(model.loss_G)
            loss_history['loss_D1'].append(model.loss_D1)
            loss_history['loss_D2'].append(model.loss_D2)
            
#             process_time = start_time - time.time() - prepare_time

            # if i % print_every == 0:
            
#                 pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
#                     process_time/(process_time+prepare_time), epoch, epochs))
                
            pbar.set_description("epoch: {}/{}:".format(epoch, epochs))
            
            ordered_dict = {
                "loss_G": model.loss_G,
                "loss_D1":model.loss_D1,
                "loss_D2":model.loss_D2
            }
            pbar.set_postfix(ordered_dict)
            
            
    # TODO: save losses, save model, test model.test function
            
            
            
            
            
            
            
            
        







    

    

