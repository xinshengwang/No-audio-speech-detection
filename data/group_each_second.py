import sys
import numpy as np
import os
import imageio
import cv2
import shutil
import torch
from multiprocessing import Pool

nubmers = torch.from_numpy(np.arange(5))
numbers = nubmers.unsqueeze(0).repeat(1320,1)
numbers = numbers.view(-1)
numbers = np.array(numbers)
seconds = torch.from_numpy(np.arange(1320))
seconds = seconds.unsqueeze(1).repeat(1,5)
seconds = seconds.view(-1)
seconds = np.array(seconds)


def copy_frames(person_id):
    root = 'split/test/videos/frames'
    save_root = 'split/test/videos/frames_per_scond'
    second = 0
    for order in range(11):
        for image_id in range(600):
            image_path = os.path.join(root,person_id + '_' + str(order),str(image_id) + '.jpg')
            new_seq = order * 600 + image_id
            new_id = numbers[new_seq]
            second = seconds[new_seq] 
            save_dir = os.path.join(save_root,person_id,str(second))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir,str(new_id) + '.jpg')
            second += 1
            shutil.copy(image_path, save_path) 

if __name__ == "__main__":
    root = 'split/test/videos/frames'
    names = os.listdir(root)
    names = [name.split('_')[0] for name in names]
    person_ids = list(set(names))
    pool = Pool(5)
    pool.map(copy_frames,person_ids)
    pool.start()
    pool.join()