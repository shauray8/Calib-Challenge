## -------------------- Imports -------------------- ##

import argparse
import os
import time
import numpy as np
from tqdm import trange

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from FlownetCorr import *
from utils import *
import FlownetCorr

## -------------------- checking callable functions from FlowNetCorr -------------------- ##

def callable():
    kwargs = sorted(name for name in FlownetCorr.__dict__
        if name.islower() and not name.startswith("__")
        and callable(FlownetCorr.__dict__[name]))
    return kwargs

## -------------------- Argument Parser just for simplicity -------------------- ##

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', default="../../data/calib_image_data", type=str,
                    help='path to dataset')

group = parser.add_mutually_exclusive_group()

group.add_argument('--split_value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownetc',
                    choices=callable,)
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=9, type=int,
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
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true',
                    help='don\'t append date timestamp to folder' )
parser.add_argument('--div-flow', default=20,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[100,150,200], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

## ----------------------- global variables ----------------------- ##

best_MSE = -1
n_iter = 0
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def main():
    global args, best_MSE
    args = parser.parse_args()
    save_path = f'{args.arch}_{args.solver}_{args.epochs}_bs{args.batch_size}_lr{args.lr}'
        
    if not args.no_date:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(save_path)
    save_path = os.path.join("./pretrained/", save_path)
    print(f"=> will save everything to {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path, "train"))
    test_writer = SummaryWriter(os.path.join(save_path, "test"))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

## --------------------- transforming the data --------------------- ##

    input_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            #RandomTranslate(10),
            transforms.ColorJitter(brightness=.3, contrast=0, saturation=0, hue=0),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
            transforms.Normalize(mean=[.45,.432,.411], std=[1,1,1]),
        ])

## --------------------- loading and concatinating the data --------------------- ##

    print(f"=> fetching image pairs from {args.data}") 
    train_set, test_set = Transformed_data(args.data, transform=input_transform, split = args.split_value)

    print(f"=> {len(test_set) + len(train_set)} samples found, {len(train_set)} train samples and {len(test_set)} test samples")

    train_loader = DataLoader(
            train_set, batch_size = args.batch_size, num_workers=args.workers,
            pin_memory=True, shuffle=True)

    val_loader = DataLoader(
            test_set, batch_size=args.batch_size, num_workers=args.workers,
            pin_memory=True, shuffle = False)

## --------------------- MODEL from FlowNetCorr.py --------------------- ##
    
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print(f"=> creating model {args.arch}")
    else:
        network_data = None
        print(f"=> No pretrained weights ")

## --------------------- Checking and selecting a optimizer [SGD, ADAM] --------------------- ##

    model = flownetc().to(device)
    if args.solver not in ['adam', 'sgd']:
        print("=> enter a supported optimizer")
        return 
    
    print(f'=> settting {args.solver} optimizer')
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
            {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    
    optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta)) if args.solver == 'adam' else torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)
    
    if args.evaluate:
        best_MSE = validation(val_loader, model, 0, output_writers, loss_function)
        return

## --------------------- Scheduler and Loss Function --------------------- ##

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=.5)

    yaw_loss = nn.MSELoss()
    pitch_loss = nn.MSELoss()

## --------------------- Training Loop --------------------- ##

    for epoch in (r := trange(args.start_epoch, args.epochs)):

        avg_loss_MSE, train_loss_MSE, display = train(train_loader, model,
                optimizer, epoch, train_writer, yaw_loss, pitch_loss)
        
        scheduler.step()
        train_writer.add_scalar('mean MSE', avg_loss_MSE, epoch)

## --------------------- Validation Step --------------------- ##

        with torch.no_grad():
            MSE_loss_val, display_val = validation(val_loader, model, epoch, output_writers, yaw_loss, pitch_loss)
        test_writer.add_scalar('mean MSE', MSE_loss_val, epoch)

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

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    losses = []
    model.train()
    end = time.time()

## --------------------- Training --------------------- ##

    for i, (input, yaw, pitch) in enumerate(train_loader):
        start_time = time.time()
        yaw = yaw
        pitch = pitch
        inputs = torch.cat(input,1).to(device)

        print("=> training on batch")
        pred_yaw, pred_pitch = model(inputs)

        yaw_MSE = yaw_loss(np.argmax(pred_yaw), yaw)*.5
        print(pred_yaw, yaw)
        pitch_MSE = pitch_loss(pred_pitch, pitch)*.5
        loss = yaw_MSE + pitch_MSE

        losses.append(float(yaw_MSE.item()) + float(pitch_MSE.item()))
        train_writer.add_scalar('train_loss', yaw_MSE.item()+pitch_MSE.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end = time.time()
        batch_time = end - start_time
        
## --------------------- Stuff to display at output --------------------- ##

        if i % args.print_freq == 0:
            display = ('Epoch: [{0}][{1}/{2}] ; Time {3} ; MSELoss {5}').format(epoch, 
                    i, epoch_size, batch_time, sum(losses)/len(losses))
        n_iters += 1
        if i >= epoch_size:
            break

    return losses.avg, loss.item(), display

def validation(val_loader, model, epoch, output_writers, yaw_loss, pitch_loss):
    global args

    model.eval()
    
    end = time.time()
    for i, (input, yaw, pitch) in enumerate(val_loader):
        yaw = yaw
        pitch = pitch
        input = torch.cat(input,1).to(device)

        output = model(input)

        yaw_MSE = yaw_loss(pred_yaw, yaw)*.5
        print(pred_yaw, yaw)
        pitch_MSE = pitch_loss(pred_pitch, pitch)*.5
        loss = yaw_MSE + pitch_MSE

        end = time.time()

       # if i < len(output_writers):
       #     if epoch == 0:
       #         mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
       #         output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
       #         output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
       #         output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
       #     output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if i % args.print_freq == 0:
            display_val = ('Test: [{0}/{1}] ; Loss {2}').format(i, len(val_loader), loss.item())

    return loss.item(), display_val
        
if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
