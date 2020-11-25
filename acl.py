import numpy as np
import torch
import random
import argparse
from models import AclModels
from utils.traintest import train_unimodal
from utils.Dataloader import Acl_data

def set_seed():
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed(args.manualSeed)  
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(args.manualSeed + worker_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modality',type=str,default='acl')
    parser.add_argument('--fusion_type',type=str,default='')
    parser.add_argument('--data_path',type=str,default='X:/staff-bulk/ewi/insy/MMC/xinsheng/code/Challenges/MediaEval2020/data/split') 
    parser.add_argument('--exp_dir',type=str,default='output',
                        help='output directory')
    parser.add_argument('--split',type=str,default='train',
                        choices=['train','test'])
    parser.add_argument('--split_num',type=int,default=1)  #worst 4 best 5: 75
    parser.add_argument('--workers',type=int,default=0,
                        help='number of workers for loading data')  
    parser.add_argument('--manualSeed',type=int,default= 200,
                        help='manual seed')
    # setting training parameter
    parser.add_argument("--max_epochs",type=int,default=14)
    parser.add_argument("--optim", type=str, default="adam",
                        help="training optimizer", choices=["sgd", "adam"])
    parser.add_argument('--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate') # 0.001
    parser.add_argument('--lr_decay', default=3, type=int, metavar='LRDECAY',
                        help='Divide the learning rate by 10 every lr_decay epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float, metavar='W', 
                        help='weight decay (default: 1e-4)')  # 1e-6
    parser.add_argument('--threshold',default=10,type=int)
    parser.add_argument('--bce_weight',default=0.0,type=float)
    parser.add_argument('--result_file',default='training_set.text',type=str)

    args = parser.parse_args()
    set_seed()
    """
    for split_num in range(6):
        args.split_num = split_num
        if args.split == 'train':
            dataset = Acl_data(args.data_path, 'train',split_num=args.split_num, threshold=args.threshold)
            dataset_val = Acl_data(args.data_path,'val',split_num=args.split_num,threshold=args.threshold)
            train_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                drop_last=False,shuffle=True,num_workers=args.workers,worker_init_fn=worker_init_fn,)
            val_loader = torch.utils.data.DataLoader(
                dataset_val, batch_size=args.batch_size,
                drop_last=False,shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn) 
        else:
            dataset_test = Acl_data(args.data_path,'test')
            test_loader = torch.utils.data.DataLoader(
                dataset_val, batch_size=args.batch_size,
                drop_last=False,shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn) 


        Aclmodel = AclModels.CRNN(args)

        if args.split == 'train':
            train_unimodal(Aclmodel,train_loader,val_loader,args)
    """
    args.split_num = 0
    
    dataset = Acl_data(args.data_path, 'trainval',split_num=args.split_num, threshold=args.threshold)
    dataset_val = Acl_data(args.data_path,'val',split_num=args.split_num,threshold=args.threshold)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        drop_last=False,shuffle=True,num_workers=args.workers,worker_init_fn=worker_init_fn,)
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size,
        drop_last=False,shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn) 

    dataset_test = Acl_data(args.data_path,'test')
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        drop_last=False,shuffle=False,num_workers=args.workers,worker_init_fn=worker_init_fn) 


    Aclmodel = AclModels.CRNN(args)

    if args.split == 'train':
        train_unimodal(Aclmodel,train_loader,val_loader,test_loader,args)