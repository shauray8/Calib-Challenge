import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import datetime
from torch.utils.tensorboard import SummaryWriter

from FlownetCorr import *
import FlownetCorr

def callable():
    kwargs = sorted(name for name in FlownetCorr.__dict__
        if name.islower() and not name.startswith("__")
        and callable(FlownetCorr.__dict__[name]))
    return kwargs



parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', metavar='DATASET', default='flying_chairs',
                    choices=dataset_names,
                    help='dataset type : ' +
                    ' | '.join(dataset_names))
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.8, type=float,
                   help='test-val split proportion between 0 (only test) and 1 (only train), '
                        'will be overwritten if a split file is set')
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
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
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is garantied to be dense,'
                    'automatically seleted when choosing a KITTIdataset')
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

best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_EPE
    args = parser.parse_args()
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)
        
    if not args.o_data:
        timestamp = datetime.datatime.now().strftime("%m-%d-%H:%M")
        save_path = os.path.join(args.dataset, save_path)
    save_path = os.path.join(args.dataset, save_path)
    print(f"=> will save everything to {save_path}")
    if not os,path.exist(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWritter(os.path.join(save_path, "train"))
    test_writer = SummaryWritter(os.path.join(save_path, "test"))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    ## --------------------- loading the data --------------------- ##
    input_transform = transform.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[.45,.432,.411], std=[1,1,1]),
        ])

    target_transform = transform.Compose([
        flow_transforms.ArrayToTensor(),
        transformes.Normalize(mean=[0,0], std=[args.div_flow, args.div_flow]),
        ])

    co_transform = flow_transforms.Compose([
            flow_transforms.RandomTranslate(10),
            flow_transforms.RandomRotate(10,5),
            flow_transforms.RandomCrop((320,448)),
            flow_transforms.RandomVerticalFlip(),
            flow_transforms.RandomHorizontalFlip()
        ])

    print(f"=> fetching image pairs in {args.data}") 
    train_set, test_set = datassets.__dict__[args.dataset](
            args.data,
            transfoms = input_tranform,
            target_transform = target_transform,
            co_transform = co_transform,
            splt = args.split_file if args.split_file else args.split_value,
            )
    print(f"{len(test_set) + len(train_set)} samples found, {len(train_set)} train samples and {len(test_set)} test samples")


    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size = args.batch_size, num_workers=args.workers,
            pin_memory=True, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, num_workers=args.num_workers,
            pin_memory=True, shuffle = False)


    ## --------------------- MODEL --------------------- ##



        
