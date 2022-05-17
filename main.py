#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Oct 10, 2021, 17:35:06
# @author: dianwen ng
# @file  : ConvMixer.py

import os
import argparse
import numpy as np
import random
import yaml
from time import time

import torch
from torch import nn
import torch.optim as optim

import dataloader
from torch.utils.data import Dataset, DataLoader
from models.ConvMixer import KWSConvMixer
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler


########## Argument parser ##########
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    ## model dataset 
    parser.add_argument('-config', '-c', default='configs/ConvMixer.yaml')    
    args = parser.parse_args()
    
    with open(args.config, "r") as file:
        conf = yaml.load(file)

    return conf


def train(model, device, train_loader, optimizer, scaler, epoch, log_interval=100):
    ## set training mode
    model.train()  
    iteration = 0
    train_loss = 0.
    start = time()
    gt_labels, pred_labels = [], []
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        
        correct=0
        data, target = data.to(device), target.to(device)
        
        with autocast():
            output, _ = model(data)
            loss = criterion(output, target)
        
        train_loss += loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pred = output.cpu().data.numpy().argmax(axis=1)
        target_lab = target.cpu().data.numpy().argmax(axis=1)
        correct += int((pred == target_lab).sum())
        acc = 100. * correct / len(target)
        
        if iteration % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.5f} Acc: {:.3f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100 * batch_idx / len(train_loader), loss.item(), acc))
    
        gt_labels.append(target_lab)
        pred_labels.append(pred)
        
        iteration += 1
    
    end = time()
    train_loss /= len(train_loader.dataset)
    print('Train loss is {:.5f} after Epoch {}'.format(train_loss, epoch))
    print('Total accuracy is {:.3f} after Epoch {}'.format(
        accuracy_score(np.hstack(gt_labels), np.hstack(pred_labels)), epoch))
    print('Time taken for Epoch {} is {:.4f}s'.format(epoch, end - start))

    
def test(model, device, test_loader, ep):
    
    print('Sample size of validation set: ' + str(len(test_loader.dataset)))
    model.eval()
    correct = 0
    test_loss = 0.
    start = time()
    gt_labels, pred_labels = [], []
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)

            output, _ = model(data)
            loss = criterion(output, target) 
            test_loss += loss
            pred = output.cpu().data.numpy().argmax(axis=1)
            target_lab = target.cpu().data.numpy().argmax(axis=1)
            correct += int((pred == target_lab).sum())
            
            gt_labels.append(target_lab)
            pred_labels.append(pred)
    
    end = time()
    test_loss /= len(test_loader.dataset)
    
    test_acc = accuracy_score(np.hstack(gt_labels), np.hstack(pred_labels))
    print('Valid loss is {:.5f} after Epoch {}'.format(test_loss, ep))
    print('Valid accuracy is {:.3f} after Epoch {}'.format(test_acc, ep))
    
    return round(test_acc, 3)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    
    args = parse_args()
    setup_seed(args.get('seed'))
    use_cuda = torch.cuda.is_available() and args.get('use_gpu')
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Training using: ", device)
    
    ## make model
    model = KWSConvMixer(input_size = args.get('model')["input_size"], 
                         num_classes=args.get('model')["num_classes"],
                         feat_dim=args.get('model')["feat_dim"],
                         dropout=args.get('model')["dropout"]).to(device)

    ## make dataset
    train_loader = torch.utils.data.DataLoader(
                        dataloader.AudiosetDataset(args.get("data")["train_dataset"]["manifest_filepath"],
                                                   audio_conf=args.get("data")["train_dataset"]), 
                        batch_size=args.get("trainer")["train_batch_size"],
                        num_workers=args.get("trainer")["num_workers"],
                        shuffle=True, pin_memory=True)
    
    valid_loader = torch.utils.data.DataLoader(
                        dataloader.AudiosetDataset(args.get("data")["validation_dataset"]["manifest_filepath"],
                                                   audio_conf=args.get("data")["validation_dataset"]), 
                        batch_size=args.get("trainer")["val_batch_size"],
                        num_workers=args.get("trainer")["num_workers"],
                        shuffle=False, pin_memory=True)

    ## training optimizer :: 
    optimizer = torch.optim.Adam(model.parameters(), args.get("trainer")["lr"], 
                                 weight_decay=5e-7, betas=(0.95, 0.999)) 
 
    ## make decay in learning rate
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5, 70, 4)), gamma=0.85)

    scaler = GradScaler()

    if not os.path.exists(args.get("trainer")["save_dir"]):
        os.mkdir(args.get("trainer")["save_dir"])
    
    print('Starting model training ...')
    for epoch in range(args.get("trainer")["num_epochs"]):
        print('\n=============start training=============')
        train(model, device, train_loader, optimizer, scaler, epoch, args.get("trainer")["log_interval"])
        
        print('\n============start validation============')
        acc = test(model, device, valid_loader, epoch)  
        
        scheduler.step()
        
        ## saving model
        model_save_path = os.path.join(args.get("trainer")["save_dir"], 'check_point_'+str(epoch)+'_'+str(acc))
        state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state_dict, model_save_path)
    
if __name__ == '__main__':
    main()
