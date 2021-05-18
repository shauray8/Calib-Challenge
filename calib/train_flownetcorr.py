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
parser.add_argument('--arch', '-a', metavar='ARCH', default='flownetc',
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

## ----------------------- global variables ----------------------- ##

best_MSE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args, best_MSE
    args = parser.parse_args()
    save_path = f'{args.arch},{args.solver},{args.epochs},bs{args.batch_size},lr{args.lr}'
        
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

## --------------------- transforming the data --------------------- ##

    input_transform = transform.Compose([
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[.45,.432,.411], std=[1,1,1]),
        transforms.ToTensor(),
        ])

    target_transform = transform.Compose([
        transformes.Normalize(mean=[0,0], std=[args.div_flow, args.div_flow]),
        transforms.ToTensor(),
        ])

    co_transform = transforms.Compose([
            RandomTranslate(10),
            transforms.RandomRotation(10,5),
            transforms.RandomCrop((320,448)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ])

## --------------------- loading the data --------------------- ##

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

## --------------------- MODEL from FlowNetCorr.py --------------------- ##
    
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print(f"=> creating model {args.arch}")
    else:
        network_data = None
        print(f"=> using pre_trained model {args.arch}")

## --------------------- Checking and selecting a optimizer [SGD, ADAM] --------------------- ##

    model = FlownetCorr.__dict__[args.arch](network_data).to(device)
    if args.solver not in ['adam', 'sgd']:
        print("=> enter a supported optimizer")
        break
    
    print(f'=> settting {args.solver} solver')
    param_groups = [{'params': model.bias_parameters(), 'weight_decay': args.bias_decay},
            {'params': model.weight_parameters(), 'weight_decay': args.weight_decay}]

    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    
    optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.mometum, args.beta)) if arg.solver == 'adam' else torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)
    
    if args.evaluate:
        best_MSE = validation(val_loader, model, 0, output_writers)
        return

## --------------------- Scheduler and Loss Function --------------------- ##

    scheduler = torch.optim.lr_scheduler.MultiStepLr(optimizer, milestones=args.milestone, gamma=.5)

    loss_function = nn.MSELoss()

## --------------------- Training Loop --------------------- ##

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        train_loss, train_MSE, display = train(train_loader, model,
                optimizer, epoch, train_writer, loss_function)
        train_writer.add_scalar('mean MSE', train_MSE, epoch)

## --------------------- Validation Step --------------------- ##

        with torch.no_grad():
            MSE = validation(val_loader, model, epoch, output_writers)
        test_writer.add_scalar('mean MSE', MSE, epoch)

        if best_MSE < 0:
            best_MSE = MSE

        is_best = MSE < best_MSE
        best_MSE = min(MSE, best_MSE)

## --------------------- Saving on every epoch --------------------- ##

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best, save_path)

## --------------------- TRAIN function for the training loop --------------------- ##

def train(train_loader, model, optimizer, epoch, train_writer, loss_function):
    global n_iters, args

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_MSEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    model.train()
    end = time.time()

## --------------------- Training --------------------- ##

    for i, (input, target) in enumerate(train_loader):
        date_time.update(time.time() - end)
        target = target.to(device)
        input = torch.cat(input,1).to(device)

        output = model(input)
        if args.sparse:
            h, w = target.size()[-2:]
            output = [F.interpolate(output[0], (h,w)), *output[1:]]

        
        loss = loss_function(output, target, weights=args.multiscale_weights, sparse=arg.sparse)
        flow2_MSE = args.div_flow * realMSE(ouptut[0], target, sparse=arg.sparse)

        losses.update(loss.item(), target.size(0))
        train_writer.add_scalar('train_loss', loss.item(), n_inter)
        flow2_MSEs.update(flow2_MSE.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
## --------------------- Stuff to display at output --------------------- ##

        if i % args.print_freq == 0:
            display = 'Epoch: [{0}][{1}/{2}] ; Time {3} ; Data {4} ; Loss {5} ; MSE {6}'
                  .format(epoch, i, epoch_size, batch_time,
                          data_time, losses, flow2_MSEs)
        n_iters += 1
        if i >= epoch_size:
            break

    return losses.avg, flow2_MSEs.avg, display

def validation(val_loader, model, epoch, output_writers):
    global args

    batch_time = AverageMeter()
    flow2_MSEs = AverageMeter()

    model.eval()
    
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(device)
        input = torch.cat(input,1).to(device)

        output = model(input)
        flow2_MSE = args.div_flow * realMSE(output, target, sparse=args.sparse)

        flow2_MSEs.update(flow2_MSE.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i < len(output_writers):
            if epoch == 0:
                mean_values = torch.tensor([0.45,0.432,0.411], dtype=input.dtype).view(3,1,1)
                output_writers[i].add_image('GroundTruth', flow2rgb(args.div_flow * target[0], max_value=10), 0)
                output_writers[i].add_image('Inputs', (input[0,:3].cpu() + mean_values).clamp(0,1), 0)
                output_writers[i].add_image('Inputs', (input[0,3:].cpu() + mean_values).clamp(0,1), 1)
            output_writers[i].add_image('FlowNet Outputs', flow2rgb(args.div_flow * output[0], max_value=10), epoch)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t MSE {3}'
                  .format(i, len(val_loader), batch_time, flow2_MSEs))

    print(' * MSE {:.3f}'.format(flow2_MSEs.avg))

    return flow2_MSEs.avg
        

if __name__ == "__main__":
    main()
