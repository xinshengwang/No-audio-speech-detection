import numpy as np
import os
import pandas as pd

root = 'data/test/acceleration'
save_root = 'data/split/test/acceleration'
filenames = os.listdir(root)
indexs = np.arange(0,26400,2400)
orders= indexs/2400

for filename in filenames:
    path = os.path.join(root,filename)
    data = pd.read_csv(path,header=None)
    lst = data.values.tolist()
    arr = np.array(lst)
    mov = arr[:,1:]
    for i in range(len(indexs)):
        ind = indexs[i]
        clip = mov[i*2400:(i*2400+2400)]
        num = str(int(orders[i]))
        save_path = os.path.join(save_root, filename.split('.')[0] + '_' + num + '.npy')
        np.save(save_path,clip)



"""
label_path = 'data/train/labels/Train_labels.csv'
labels = pd.read_csv(label_path,header=None)
labels = labels.values.tolist()
labels = np.array(labels)
save_label_root = 'data/split/train/labels'
seqs = []
for filename in sorted(filenames):
    num = filename.split('.')[0].split('t')[-1]
    seqs.append(int(num))

for i in range(len(seqs)):
    label = labels[:,i]
    seq = sorted(seqs)[i]
    for j in range(len(indexs)):
        ind = indexs[j]
        sub_label = label[j*2400:(j*2400+2400)]
        num = str(int(orders[j]))
        filename = 'subject' + str(seq)
        save_label_path = os.path.join(save_label_root, filename + '_' + num + '.npy')
        np.save(save_label_path,sub_label)
"""