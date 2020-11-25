import os 
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
import pdb
import torch.optim as optim
from utils.util import AverageMeter,adjust_learning_rate
from sklearn.metrics import roc_curve,auc

def train_unimodal(model,train_loader,val_loader,test_loader,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    loss_meter = AverageMeter()
    exp_dir = os.path.join(args.exp_dir,'unimodal',args.modality)
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    if not isinstance(model,torch.nn.DataParallel):
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), args.learning_rate,
                weight_decay=args.weight_decay, betas=(0.95, 0.999))
    criterion = nn.L1Loss()
    epoch = 0
    while epoch<=args.max_epochs:
        model.train()
        adjust_learning_rate(args.learning_rate, args.lr_decay, optimizer, epoch)
        for i, (acl_input,labels,key) in enumerate(train_loader):
            acl_input = acl_input.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            output = model(acl_input)
            loss = 0
            criterion_bce_log = nn.BCEWithLogitsLoss(pos_weight=(args.bce_weight*labels + 1.0))
            loss = criterion_bce_log(output,labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            # if i%10 == 0:
            #     print('iteration = %d | loss_val = %f loss_avg = %f'%(i,loss_meter.val,loss_meter.avg)) 
        if epoch % 1 == 0:
            score = eval_unimodal(model,val_loader,args)
            info = 'split = %d | epoch = %d | auc = %f \n'%(args.split_num,epoch,score)
            print(info)
            save_path = os.path.join(exp_dir, args.result_file)
            with open(save_path, "a") as file:
                file.write(info)
            if epoch == args.max_epochs:
                torch.save(model.state_dict(),
                        "%s/%s_epoch_%d.pth" % (save_model_dir,args.modality,epoch))
                output_resuts_unimodal(model,val_loader,'val',exp_dir,args)
                output_resuts_unimodal(model,test_loader,'test',exp_dir,args)
        epoch += 1

def output_resuts_unimodal(model,val_loader,split,exp_dir,args):
    out_root = os.path.join(exp_dir,split,'predictions')
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    predicts = []
    all_keys = []
    all_labels = []
    with torch.no_grad():
        for i, (acl_inputs,labels,keys) in  enumerate(val_loader):
            acl_inputs = acl_inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(acl_inputs)
            outputs = outputs.to('cpu').detach()
            for i in range(len(keys)):
                key = keys[i]
                output = outputs[i]
                save_path = os.path.join(out_root,key + '.npy')
                np.save(save_path,output) 

def eval_unimodal(model,val_loader,args):
    exp_dir = args.exp_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    predicts = []
    all_keys = []
    all_labels = []
    with torch.no_grad():
        for i, (acl_inputs,labels,keys) in  enumerate(val_loader):
            acl_inputs = acl_inputs.float().to(device)
            labels = labels.long().to(device)
            outputs = model(acl_inputs)
            all_labels.append(labels.to('cpu').detach())
            predicts.append(outputs.to('cpu').detach())
        all_labels = torch.cat(all_labels)
        predicts = torch.cat(predicts)
        fpr, tpr, thresholds = roc_curve(all_labels.view(-1), predicts.view(-1))
        score = auc(fpr,tpr)
    return score



# ====================two model===============================
def train_two(model,train_loader,val_loader,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    loss_meter = AverageMeter()
    exp_dir = os.path.join(args.exp_dir,'two_modals',args.modality)
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    if not isinstance(model,torch.nn.DataParallel):
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), args.learning_rate,
                weight_decay=args.weight_decay, betas=(0.95, 0.999))
    criterion = nn.L1Loss()
    epoch = 0
    while epoch<=args.max_epochs:
        model.train()
        adjust_learning_rate(args.learning_rate, args.lr_decay, optimizer, epoch)
        for i, (input1,input2,labels,key) in enumerate(train_loader):
            input1 = input1.float().to(device)
            input2 = input2.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            loss = 0
            criterion_bce_log = nn.BCEWithLogitsLoss(pos_weight=(args.bce_weight*labels + 1.0))
            if args.fusion_type == 'late':
                out1,out2 = model(input1,input2)
                outputs = (out1 + out2)/2
                loss = criterion_bce_log(outputs,labels) + criterion_bce_log(out1,labels) + criterion_bce_log(out2,labels) 
            else:
                outputs = model(input1,input2)
                loss = criterion_bce_log(outputs,labels)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            # if i%10 == 0:
            #     print('iteration = %d | loss_val = %f loss_avg = %f'%(i,loss_meter.val,loss_meter.avg)) 
        if epoch % 1 == 0:
            score = eval_two(model,val_loader,args)
            info = 'split = %d | epoch = %d | auc = %f \n'%(args.split_num,epoch,score)
            print(info)
            save_path = os.path.join(exp_dir, args.result_file)
            with open(save_path, "a") as file:
                file.write(info)
        epoch += 1

def eval_two(model,val_loader,args):
    exp_dir = args.exp_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    predicts = []
    all_keys = []
    all_labels = []
    with torch.no_grad():
        for i, (input1,input2,labels,keys) in  enumerate(val_loader):
            input1 = input1.float().to(device)
            input2 = input2.float().to(device)
            labels = labels.long().to(device)
            if args.fusion_type == 'late':
                out1,out2 = model(input1,input2)
                outputs = (out1 + out2)/2
            else:
                outputs = model(input1,input2)
            all_labels.append(labels.to('cpu').detach())
            predicts.append(outputs.to('cpu').detach())
        all_labels = torch.cat(all_labels)
        predicts = torch.cat(predicts)
        fpr, tpr, thresholds = roc_curve(all_labels.view(-1), predicts.view(-1))
        score = auc(fpr,tpr)
    return score


# ====mixture===
def train_mix(model,model1,model2,train_loader,val_loader,test_loader,args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    loss_meter = AverageMeter()
    exp_dir = os.path.join(args.exp_dir,'two_modals',args.modality)
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    
    if not isinstance(model,torch.nn.DataParallel):
        model = nn.DataParallel(model)
    if not isinstance(model1,torch.nn.DataParallel):
        model1 = nn.DataParallel(model1)
    if not isinstance(model2,torch.nn.DataParallel):
        model2 = nn.DataParallel(model2)
    
    model = model.to(device)
    model1 = model1.to(device).eval()
    model2 = model2.to(device).eval()
    optimizer = optim.Adam(model.parameters(), args.learning_rate,
                weight_decay=args.weight_decay, betas=(0.95, 0.999))
    criterion = nn.L1Loss()
    epoch = 0
    while epoch<=args.max_epochs:
        model.train()
        adjust_learning_rate(args.learning_rate, args.lr_decay, optimizer, epoch)
        
        for i, (input1,input2,labels,key) in enumerate(train_loader):
            input1 = input1.float().to(device)
            input2 = input2.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            loss = 0
            criterion_bce_log = nn.BCEWithLogitsLoss(pos_weight=(args.bce_weight*labels + 1.0))
            with torch.no_grad():
                feat1,score1 = model1(input1)
                feat2,score2 = model2(input2)         
            outputs = model(feat1,feat2,score1,score2)                
            loss = criterion_bce_log(outputs,labels)  #+ criterion_bce_log(att1,att_label1) + criterion_bce_log(att2,att_label2) 
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

        if epoch % 1 == 0:
            score = eval_mix(model,model1,model2,val_loader,args)
            info = 'split = %d | epoch = %d | auc = %f \n'%(args.split_num,epoch,score)
            print(info)
            save_path = os.path.join(exp_dir, args.result_file)
            with open(save_path, "a") as file:
                file.write(info)
            if epoch == args.max_epochs:
                torch.save(model.state_dict(),
                        "%s/%s_epoch_%d.pth" % (save_model_dir,args.modality,epoch))
                output_resuts_mix(model,model1,model2,val_loader,'val',exp_dir,args)
                output_resuts_mix(model,model1,model2,test_loader,'test',exp_dir,args)
        epoch += 1

def output_resuts_mix(model,model1,model2,val_loader,split,exp_dir,args):
    out_root = os.path.join(exp_dir,split,'predictions')
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)
    if not isinstance(model1, torch.nn.DataParallel):
        model1 = nn.DataParallel(model1)
    if not isinstance(model2, torch.nn.DataParallel):
        model2 = nn.DataParallel(model2)
    model = model.to(device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    model.eval()
    model1.eval()
    model2.eval()
    predicts = []
    all_keys = []
    all_labels = []
    with torch.no_grad():
        for i, (input1,input2,labels,keys) in  enumerate(val_loader):
            input1 = input1.float().to(device)
            input2 = input2.float().to(device)
            labels = labels.long().to(device)
            feat1,score1 = model1(input1)
            feat2,score2 = model2(input2)
            # outputs = (score1 + score2)/2
            outputs = model(feat1,feat2,score1,score2)
            outputs = outputs.to('cpu').detach()
            for i in range(len(keys)):
                # pdb.set_trace()
                key = keys[i]
                output = outputs[i]
                save_path = os.path.join(out_root,key + '.npy')
                np.save(save_path,output)


def eval_mix(model,model1,model2,val_loader,args):
    exp_dir = args.exp_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(model, torch.nn.DataParallel):
        model = nn.DataParallel(model)
    if not isinstance(model1, torch.nn.DataParallel):
        model1 = nn.DataParallel(model1)
    if not isinstance(model2, torch.nn.DataParallel):
        model2 = nn.DataParallel(model2)
    model = model.to(device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    model.eval()
    model1.eval()
    model2.eval()
    predicts = []
    all_keys = []
    all_labels = []
    with torch.no_grad():
        for i, (input1,input2,labels,keys) in  enumerate(val_loader):
            input1 = input1.float().to(device)
            input2 = input2.float().to(device)
            labels = labels.long().to(device)
            feat1,score1 = model1(input1)
            feat2,score2 = model2(input2)
            # outputs = (score1 + score2)/2
            outputs = model(feat1,feat2,score1,score2)
            all_labels.append(labels.to('cpu').detach())
            predicts.append(outputs.to('cpu').detach())
        all_labels = torch.cat(all_labels)
        predicts = torch.cat(predicts)
        fpr, tpr, thresholds = roc_curve(all_labels.view(-1), predicts.view(-1))
        score = auc(fpr,tpr)
    return score