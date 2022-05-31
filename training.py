import torch
import torch.nn as nn
# from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import tqdm
import pickle
import argparse
import os
from distutils.util import strtobool

from CycleGAN import CycleGAN
from dataset import get_full_list, ImageDataset, Selfie2Anime_Dataset

torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
    Training CycleGAN model. 
    """)
    parser.add_argument("--epoch", type=int, default=1, help="total epoch for training")
    parser.add_argument("--use_gpu", type=str, default=True, choices=('True','False'), help="if should try to use GPU in training")
    parser.add_argument("--load_prev", type=str, default=False, choices=('True','False'), help="if should load a previous checkpoint")
    parser.add_argument("--prev_checkpoint", type=str, default="prev_checkpoint.pt", help="directory of previous checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="new_checkpoint.pt", help="directory of new checkpoint")
    parser.add_argument("--loss_history_dir", type=str, default="loss_history.pickle", 
        help="directory for saved loss history, if the specified file does not exist, will use the input as label to create one")

    args = parser.parse_args()
    EPOCH = args.epoch
    USE_GPU = strtobool(args.use_gpu)
    LOAD_PREV = strtobool(args.load_prev)
    PREV_CHECKPOINT = args.prev_checkpoint
    CHECKPOINT_DIR = args.checkpoint_dir  
    LOSS_HISTORY_DIR = args.loss_history_dir
    # print_every = 100
    use_cuda = False

    if USE_GPU and torch.cuda.is_available():
        use_cuda = True
    print('use_cuda:', use_cuda)

    print("LOAD_PREV:",LOAD_PREV)

    model = CycleGAN(use_cuda)

    if LOAD_PREV:
        start_epoch = torch.load(PREV_CHECKPOINT)["epoch"]

        model.load(PREV_CHECKPOINT)

        assert os.path.exists(LOSS_HISTORY_DIR)
        with open(LOSS_HISTORY_DIR, "rb") as f:
            loss_history = pickle.load(f) 

    else:
        start_epoch = 0

        loss_history = {}
        loss_history['loss_GAN'] = []
        loss_history['loss_cycle'] = []
        loss_history['loss_id'] = []
        loss_history['loss_G'] = []
        loss_history['loss_D1'] = []
        loss_history['loss_D2'] = []

    print('start_epoch:', start_epoch)

    dtype = torch.float32 


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
    
    # train_data_loader = DataLoader(S2A_test_set, batch_size=1, shuffle=True, num_workers=0) 
    train_data_loader = DataLoader(S2A_train_set, batch_size=1, shuffle=True, num_workers=2) 

    
    for epoch in range(start_epoch, EPOCH):
        
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
                
            pbar.set_description("epoch: {}/{}:".format(epoch, EPOCH))
            
            ordered_dict = {
                "loss_G": model.loss_G,
                "loss_D1":model.loss_D1,
                "loss_D2":model.loss_D2
            }
            pbar.set_postfix(ordered_dict)
    

    model.save(CHECKPOINT_DIR, EPOCH)

    with open(LOSS_HISTORY_DIR, "wb") as f:
        pickle.dump(loss_history, f)
            
            
            
            
            
            
            
        







    

    

