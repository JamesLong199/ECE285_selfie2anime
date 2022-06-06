import os
import subprocess

prev_epoch = 70
for epoch in [80, 90]:
    
    subprocess.run([
                "python", 
                "/home/jlong/ECE285_selfie2anime/training.py", 
                "--epoch={}".format(epoch),
                "--use_gpu={}".format(True),
                "--load_prev={}".format(True),
                "--prev_checkpoint={}".format("checkpoints/default_v2_train_ep{}.pt".format(prev_epoch)),
                "--checkpoint_dir={}".format("checkpoints/default_v2_train_ep{}.pt".format(epoch)),
                "--loss_history_dir={}".format("training_results/default_v2_train_losses.pickle"),
                ])
    prev_epoch = epoch