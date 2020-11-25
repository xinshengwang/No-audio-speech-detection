import torch
import torch.utils.data as data
import os
import sys
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class Acl_data(data.Dataset):
    def __init__(self, data_dir, split='train',
                 split_num=0,threshold=10):
        self.split = split
        self.segment_num = 11    #value is 11
        self.data_dir = data_dir
        self.split_num = str(split_num)
        self.threshold = threshold
        split_dir = os.path.join(data_dir, split)
        self.filenames = self.load_filenames(data_dir, split)  #[:320]

    def load_filenames(self, data_dir, split):      
        # filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if split == 'train' or split == 'val' or split == 'trainval':
            split_path = os.path.join(data_dir,'train','Cross_val_split.pkl')
            with open(split_path,'rb') as f:
                split_file = pickle.load(f)
            if split == 'train':
                person_ids = split_file[self.split_num]['train']
            elif split == 'trainval':
                person_ids = split_file[self.split_num]['train'] + split_file[self.split_num]['val']
            else:
                person_ids = split_file[self.split_num]['val']
            filenames = ['subject' + str(person_id) + '_' + str(i) for person_id in person_ids for i in range(self.segment_num)]
            num = len(filenames)
            print('%d files are loaded for %s'%(num,split))
        else:
            filenames = os.listdir(os.path.join(data_dir,'test','acceleration'))
            filenames = [name.split('.')[0] for name in filenames]
        # if split == 'train':
        filenames = [name for name in filenames if name !='subject73_10']

        return filenames

    def load_data(self,path):
        data = np.load(path)
        return data

    def __getitem__(self, index):
        key = self.filenames[index]
        if self.split == 'train' or self.split =='val' or self.split == 'trainval':
            label_path = os.path.join(self.data_dir,'train','labels',key) + '.npy'
            acl_path = os.path.join(self.data_dir,'train','acceleration',key) + '.npy'
        else:
            label_path = os.path.join(self.data_dir,'test','labels',key) + '.npy'
            acl_path = os.path.join(self.data_dir,'test','acceleration',key) + '.npy'
        if self.split == 'test':
            labels_per_sc = 0
        else:
            labels = self.load_data(label_path)
            labels_split = np.array(np.split(labels,120))
            labels_sum = labels_split.sum(-1)
            labels_per_sc = (labels_sum>=self.threshold).astype(int)
        acls = self.load_data(acl_path)

        
        return acls, labels_per_sc, key

    def __len__(self):
        return len(self.filenames)


class c3d_data(data.Dataset):
    def __init__(self, data_dir, split='train',
                 split_num=0,threshold=10):
        self.split = split
        self.segment_num = 11    #value is 11
        self.data_dir = data_dir
        self.split_num = str(split_num)
        self.threshold = threshold
        split_dir = os.path.join(data_dir, split)
        self.filenames = self.load_filenames(data_dir, split)  #[:320]

    def load_filenames(self, data_dir, split):      
        # filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if split == 'train' or split == 'val' or split == 'trainval':
            split_path = os.path.join(data_dir,'train','Cross_val_split.pkl')
            with open(split_path,'rb') as f:
                split_file = pickle.load(f)
            if split == 'train':
                person_ids = split_file[self.split_num]['train']
            elif split == 'trainval':
                person_ids = split_file[self.split_num]['train'] + split_file[self.split_num]['val']
            else:
                person_ids = split_file[self.split_num]['val']
            filenames = ['subject' + str(person_id) + '_' + str(i) for person_id in person_ids for i in range(self.segment_num)]
            num = len(filenames)
            print('%d files are loaded for %s'%(num,split))
        else:
            filenames = os.listdir(os.path.join(data_dir,'test','acceleration'))
            filenames = [name.split('.')[0] for name in filenames]
        # if split == 'train':
        filenames = [name for name in filenames if name !='subject73_10']

        return filenames

    def load_data(self,path):
        data = np.load(path)
        return data

    def __getitem__(self, index):
        key = self.filenames[index]
        if self.split == 'train' or self.split =='val' or self.split == 'trainval':
            label_path = os.path.join(self.data_dir,'train','labels',key) + '.npy'
            c3d_path = os.path.join(self.data_dir,'train','videos','C3D_features',key) + '.npy'
        else:
            label_path = os.path.join(self.data_dir,'test','labels',key) + '.npy'
            c3d_path = os.path.join(self.data_dir,'test','videos','C3D_features',key) + '.npy'
        if self.split == 'test':
            labels_per_sc = 0
        else:
            labels = self.load_data(label_path)
            labels_split = np.array(np.split(labels,120))
            labels_sum = labels_split.sum(-1)
            labels_per_sc = (labels_sum>=self.threshold).astype(int)
        c3ds = self.load_data(c3d_path)

        
        return c3ds, labels_per_sc, key

    def __len__(self):
        return len(self.filenames)



class pose_data(data.Dataset):
    def __init__(self, data_dir, split='train',
                 split_num=0,threshold=10):
        self.split = split
        self.segment_num = 11    #value is 11
        self.data_dir = data_dir
        self.split_num = str(split_num)
        self.threshold = threshold
        split_dir = os.path.join(data_dir, split)
        self.filenames = self.load_filenames(data_dir, split)  #[:320]

    def load_filenames(self, data_dir, split):      
        # filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if split == 'train' or split == 'val' or split == 'trainval':
            split_path = os.path.join(data_dir,'train','Cross_val_split.pkl')
            with open(split_path,'rb') as f:
                split_file = pickle.load(f)
            if split == 'train':
                person_ids = split_file[self.split_num]['train']
            elif split == 'trainval':
                person_ids = split_file[self.split_num]['train'] + split_file[self.split_num]['val']
            else:
                person_ids = split_file[self.split_num]['val']
            filenames = ['subject' + str(person_id) + '_' + str(i) for person_id in person_ids for i in range(self.segment_num)]
            num = len(filenames)
            print('%d files are loaded for %s'%(num,split))
        else:
            print('TO to')
        # if split == 'train':
        filenames = [name for name in filenames if name !='subject73_10']

        return filenames

    def load_data(self,path):
        data = np.load(path)
        return data

    def __getitem__(self, index):
        key = self.filenames[index]
        if self.split == 'train' or self.split =='val' or self.split == 'trainval':
            label_path = os.path.join(self.data_dir,'train','labels',key) + '.npy'
            pose_path = os.path.join(self.data_dir,'train','videos','frames','features_pose',key) + '.npy'
        else:
            label_path = os.path.join(self.data_dir,'test','labels',key) + '.npy'
            pose_path = os.path.join(self.data_dir,'test','videos','frames','features_pose',key) + '.npy'
        if self.split == 'test':
            labels_per_sc = 0
        else:
            labels = self.load_data(label_path)
            labels_split = np.array(np.split(labels,120))
            labels_sum = labels_split.sum(-1)
            labels_per_sc = (labels_sum>=self.threshold).astype(int)
        poses = self.load_data(pose_path)

        
        return poses, labels_per_sc, key

    def __len__(self):
        return len(self.filenames)




class acl_c3d_data(data.Dataset):
    def __init__(self, data_dir, split='train',
                 split_num=0,threshold=10):
        self.split = split
        self.segment_num = 11    #value is 11
        self.data_dir = data_dir
        self.split_num = str(split_num)
        self.threshold = threshold
        split_dir = os.path.join(data_dir, split)
        self.filenames = self.load_filenames(data_dir, split)  #[:320]

    def load_filenames(self, data_dir, split):      
        # filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if split == 'train' or split == 'val' or split == 'trainval':
            split_path = os.path.join(data_dir,'train','Cross_val_split.pkl')
            with open(split_path,'rb') as f:
                split_file = pickle.load(f)
            if split == 'train':
                person_ids = split_file[self.split_num]['train']
            elif split == 'trainval':
                person_ids = split_file[self.split_num]['train'] + split_file[self.split_num]['val']
            else:
                person_ids = split_file[self.split_num]['val']
            filenames = ['subject' + str(person_id) + '_' + str(i) for person_id in person_ids for i in range(self.segment_num)]
            num = len(filenames)
            print('%d files are loaded for %s'%(num,split))
        else:
            filenames = os.listdir(os.path.join(data_dir,'test','acceleration'))
            filenames = [name.split('.')[0] for name in filenames]
        # if split == 'train':
        filenames = [name for name in filenames if name !='subject73_10']

        return filenames

    def load_data(self,path):
        data = np.load(path)
        return data

    def __getitem__(self, index):
        key = self.filenames[index]
        if self.split == 'train' or self.split =='val' or self.split == 'trainval':
            label_path = os.path.join(self.data_dir,'train','labels',key) + '.npy'
            c3d_path = os.path.join(self.data_dir,'train','videos','C3D_features',key) + '.npy'
            acl_path = os.path.join(self.data_dir,'train','acceleration',key) + '.npy'
        else:
            c3d_path = os.path.join(self.data_dir,'test','videos','C3D_features',key) + '.npy'
            acl_path = os.path.join(self.data_dir,'test','acceleration',key) + '.npy'
        if self.split == 'test':
            labels_per_sc = 0
        else:
            labels = self.load_data(label_path)
            labels_split = np.array(np.split(labels,120))
            labels_sum = labels_split.sum(-1)
            labels_per_sc = (labels_sum>=self.threshold).astype(int)
        c3ds = self.load_data(c3d_path)
        acls = self.load_data(acl_path)
        
        return acls, c3ds, labels_per_sc, key

    def __len__(self):
        return len(self.filenames)



class video_data(data.Dataset):
    def __init__(self, data_dir, split='train',
                 split_num=0,threshold=10):
        self.split = split
        self.segment_num = 11    #value is 11
        self.data_dir = data_dir
        self.split_num = str(split_num)
        self.threshold = threshold
        split_dir = os.path.join(data_dir, split)
        self.filenames = self.load_filenames(data_dir, split)  #[:320]

    def load_filenames(self, data_dir, split):      
        # filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if split == 'train' or split == 'val':
            split_path = os.path.join(data_dir,'train','Cross_val_split.pkl')
            with open(split_path,'rb') as f:
                split_file = pickle.load(f)
            if split == 'train':
                person_ids = split_file[self.split_num]['train']
            else:
                person_ids = split_file[self.split_num]['val']
            filenames = ['subject' + str(person_id) + '_' + str(i) for person_id in person_ids for i in range(self.segment_num)]
            num = len(filenames)
            print('%d files are loaded for %s'%(num,split))
        else:
            filenames = os.listdir(os.path.join(data_dir,'test','acceleration'))
            filenames = [name.split('.')[0] for name in filenames]
        # if split == 'train':
        filenames = [name for name in filenames if name !='subject73_10']

        return filenames

    def load_data(self,path):
        data = np.load(path)
        return data

    def __getitem__(self, index):
        key = self.filenames[index]
        if self.split == 'train' or self.split =='val':
            label_path = os.path.join(self.data_dir,'train','labels',key) + '.npy'
            c3d_path = os.path.join(self.data_dir,'train','videos','C3D_features',key) + '.npy'
            pose_path = os.path.join(self.data_dir,'train','videos','frames','features_pose',key) + '.npy'
        else:
            label_path = os.path.join(self.data_dir,'test','labels',key) + '.npy'
            c3d_path = os.path.join(self.data_dir,'test','videos','C3D_features',key) + '.npy'
            pose_path = os.path.join(self.data_dir,'test','videos','frames','features_pose',key) + '.npy'
        
        labels = self.load_data(label_path)
        labels_split = np.array(np.split(labels,120))
        labels_sum = labels_split.sum(-1)
        labels_per_sc = (labels_sum>=self.threshold).astype(int)
        c3ds = self.load_data(c3d_path)
        poses = self.load_data(pose_path)
        
        return c3ds,poses, labels_per_sc, key

    def __len__(self):
        return len(self.filenames)