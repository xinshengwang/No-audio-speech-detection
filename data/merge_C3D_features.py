import numpy as np
import os
from multiprocessing import Pool
root = 'split/train/videos/C3D_features/per_second'
save_root = 'split/train/videos/C3D_features'
person_ids = os.listdir(root)

def merged_function(person_id):
    person_path = os.path.join(root,person_id)
    merged = []
    for i in range(1320):
        path = os.path.join(person_path,str(i)+'.npy')
        data = np.load(path)
        merged.append(data)
    merged = np.array(merged)
    splits = np.split(merged,11,axis=0)
    for i in range(11):
        feat = splits[i]
        save_path = os.path.join(save_root,person_id + '_%d.npy'%(i))
        np.save(save_path,feat)
    print(person_id)

if __name__ == "__main__":
    # for person_id in person_ids:
    #     merged_function(person_id)
    pool = Pool(5)
    pool.map(merged_function,person_ids)
    pool.start()
    pool.join()