import sys
import numpy as np
import os
import imageio
import cv2
from multiprocessing import Pool

def get_frames(name):
    root = 'test/videos'
    save_root = 'split/test/videos/frames'
    file_dir = os.path.join(root,name)
    with imageio.get_reader(file_dir,  'ffmpeg') as vid:
        for i, frame in enumerate(vid):
            n_frames = i
            if n_frames%4 ==0:
                if n_frames % 2400 ==0:
                    j = 0          
                split_num = n_frames // 2400
                save_path = os.path.join(save_root,name.split('.')[0] + '_' + str(split_num))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                imageio.imwrite(save_path + '/%d.jpg' %j, frame)
                j += 1

if __name__ == "__main__":
    root = 'test/videos'
    names = os.listdir(root)[-20:]
    pool = Pool(10)
    pool.map(get_frames,names)
    pool.start()
    pool.join()
