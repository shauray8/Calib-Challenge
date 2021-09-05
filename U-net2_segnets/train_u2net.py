#!/usr/bin
## -------------------- Imports -------------------- ##

import argparse
import os
import time
import numpy as np
import datetime
from tqdm import trange
import pickle

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
import model

## -------------------- checking callable functions from FlowNetCorr -------------------- ##

def callable():
    kwargs = sorted(name for name in model.__dict__
        if name.islower() and not name.startswith("__")
        and callable(model.__dict__[name]))
    return kwargs

## -------------------- Argument Parser just for simplicity -------------------- ##

parser = argparse.ArgumentParser(description='PyTorch U square net training on comma 10k dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', default="../../data/comma10k/comma10k", type=str,
                    help='path to dataset')

group = parser.add_mutually_exclusive_group()

group.add_argument('--split_value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet_square',
                    choices=callable,)
parser.add_argument('--solver', default='adam',choices=['adam'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',default=False,
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div-flow', default=20,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

## ----------------------- global variables ----------------------- ##

best_MSE = -1
n_iters = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
dir_checkpoint = "./pretrained"
img_scale = 320

## --------------------- BCE loss for every output layer --------------------- ##

cce_loss = nn.CrossEntropyLoss(size_average=True)

def multi_bce_loss(d_not, d1, d2, d3, d4, d5, d6, labels):
    ## i should change this to categorical cross entropy loss   

    loss_not = cce_loss(d_not, labels_v)
    loss_1 = cce_loss(d_1, labels_v)
    loss_2 = cce_loss(d_2, labels_v)
    loss_3 = cce_loss(d_3, labels_v)
    loss_4 = cce_loss(d_4, labels_v)
    loss_5 = cce_loss(d_5, labels_v)
    loss_6 = cce_loss(d_6, labels_v)

    loss = loss_not + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
    print(f"0: {loss_not.data.item()}, 1: {loss_1.data.item()},2: {loss_2.data.item()},3: {loss_3.data.item()},4: {loss_4.data.item()},5: {loss_5.data.item()},6: {loss_6.data.item()}")

    return loss_not, loss

## --------------------- Main function contains all the imp stuff --------------------- ##

def main():
    global args, best_MSE
    main_epoch = 0
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    save_path = f'{args.arch}_{args.solver}_{args.epochs}_bs{args.batch_size}_time{timestamp}_lr{args.lr}'
        
    save_path = os.path.join("./pretrained/", save_path)
    print(f"=> will save everything to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    writer = SummaryaWriter(comment=f"LR_{args.lr}_BS_{args.batch_size}_SCALE_{img_scale}")
    output_writers = []

## --------------------- transforming the data --------------------- ##

    input_transform = transforms.Compose([
            transforms.Resize((img_scale)),
            transforms.ColorJitter(brightness=.3, contrast=0, saturation=0, hue=0),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspetive(distortion_scale=.6,p=.1)
            transforms.RandomAutocontrast()
            transforms.RandomHorizontalFlip(p=0.5)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[.45,.432,.411], std=[1,1,1]),
        ])

## --------------------- loading and concatinating the data --------------------- ##

    print(f"=> fetching image pairs from {args.data}") 

    dataset = comma10k_dataset(imgs_dir1 = , imgs_dir2 = , masks_dir1 = , masks_dir2 = , transform = imput_transform, scale = img_scale)

    val_set = int(len(dataset) * val_per)
    train_set = len(dataset) - val_set
    train_set, test_set = random_split(dataset, [train_set, val_set])

    print(f"=> {len(test_set) + len(train_set)} samples found, {len(train_set)} train samples and {len(test_set)} test samples")

    train_loader = DataLoader(
            train_set, batch_size = args.batch_size, num_workers=args.workers,
            pin_memory=True, shuffle=True)

    validation_loader = DataLoader(
            test_set, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, shuffle = False)

## --------------------- MODEL from model.py --------------------- ##
    
    model = Unet_square(3,5).to(device)

    if args.pretrained is not None:
        with open(args.pretrained, 'rb') as pickle_file:
            network_data = pickle.load(pickle_file)

        model.load_state_dict(network_data["state_dict"])
        main_epoch = network_data['epoch']
        print(f"=> creating model {args.arch}")
    else:
        network_data = None
        print(f"=> No pretrained weights ")

## --------------------- setting up the optimizer --------------------- ##

    print(f'=> settting {args.solver} optimizer')
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
            {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True


    global_step = 0
    
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.momentum, args.beta), eps=1e-08, weight_decay=0)

    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {train_set}
        Validation size: {val_set}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale} ''')
    
    if args.evaluate:
        best_MSE = validation(val_loader, model, 0, output_writers, loss_function)
        return

## --------------------- Scheduler and Loss Function --------------------- ##

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=.5)

## --------------------- Training Loop --------------------- ##
    
    print("=> training go look tensorboard for more stuff")
    for epoch in (r := trange(args.start_epoch, args.epochs)):

        avg_loss_MSE, train_loss_MSE, display = train(train_loader, model,
                optimizer, epoch+main_epoch, train_writer, yaw_loss, pitch_loss)
        
        scheduler.step()
        train_writer.add_scalar('train mean MSE', avg_loss_MSE, epoch)

## --------------------- Validation Step --------------------- ##
        
        with torch.no_grad():
            MSE_loss_val, display_val = validation(val_loader, model, epoch, output_writers, yaw_loss, pitch_loss)
        test_writer.add_scalar('validation mean MSE', MSE_loss_val, epoch)

        if best_MSE < 0:
            best_MSE = MSE_loss_val

        is_best = MSE_loss_val < best_MSE
        best_MSE = min(MSE_loss_val, best_MSE)

## --------------------- Saving on every epoch --------------------- ##

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_MSE,
            'div_flow': args.div_flow
        }, is_best, save_path)
        
        r.set_description(f"train_stuff: {display}, epoch: {epoch+1}, val_Stuff: {display_val}")

## --------------------- TRAIN function for the training loop --------------------- ##

def train(train_loader, model, optimizer, epoch, train_writer, yaw_loss, pitch_loss):
    global n_iters, args

    epoch_size = len(train_loade+main_epochr) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    losses = []
    model.train()
    end = time.time()

## --------------------- Training --------------------- ##

    for i, (input_frame, ground) in enumerate(train_loader):
        start_time = time.time()
        input_frame = input_frame.to(device)
        ground = ground.to(device)

        ## optimizer.zero_grad()
        [p.grad = None for p in net.parameters()]

        d_not, d_1, d_2, d_3, d_4, d_5, d_6 = model(inputs_frame)
        loss_2, loss = multi_bce_loss(d0 ,d1 ,d2 ,d3 ,d4 ,d5 ,d6 ,ground)

        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/train", loss.item(), global_step)

        end = time.time()
        batch_time = end - start_time

        global_step += 1
        
## --------------------- Stuff to display at output --------------------- ##

        if i % args.print_freq == 0:
            display = (' Epoch: [{0}][{1}/{2}] ; Time {3} ; Avg MSELoss {4} ; yaw_MSE {5} ; pitch_MSE {6}').format(epoch, 
                    i, epoch_size, batch_time, sum(losses)/len(losses), yaw_MSE.item(), pitch_MSE.item())
            print(display)
        n_iters += 1
        if i >= epoch_size:
            break
    
    return sum(losses)/len(losses), loss.item() , display

def validation(val_loader, model, epoch, output_writers, yaw_loss, pitch_loss):
    global args

    model.eval()
    
    end = time.time()
    for i, (input, yaw, pitch) in enumerate(val_loader):
        input_frame = input_frame.to(device)
        ground = ground.to(device)

        d_not, d_1, d_2, d_3, d_4, d_5, d_6 = model(inputs_frame)
        loss_2, loss = multi_bce_loss(d0 ,d1 ,d2 ,d3 ,d4 ,d5 ,d6 ,ground)

        end = time.time()

        if i % args.print_freq == 0:
            display_val = ('Test: [{0}/{1}] ; Loss {2}').format(i, len(val_loader), loss.item())
            print(display_val)
            print(f"=> Values: Actual yaw: {np.argmax(yaw)} ; Pred yaw: {np.argmax(pred_yaw)} --- Actual pitch : {np.argmax(pitch)} ; Pred pitch : {np.argmax(pred_pitch)}")

    return loss.item(), display_val
        
if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
