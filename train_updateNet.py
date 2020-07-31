import argparse
import shutil
from os.path import join, isdir, isfile
from os import makedirs, listdir

#from dataset import VID
from updatenet import UpdateResNet
import torch
from torch.utils.data import dataloader
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import time
#from scipy import io
import pdb

parser = argparse.ArgumentParser(description='Training UpdateNetRes by JackieZhai')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)')

parser.add_argument('--start-epoch', default=5, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='updatenet_trainoutput/lr-4-6/checkpoint5.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', '-s', default='./updatenet_trainoutput/', type=str, help='directory for saving')
parser.add_argument('--save-rate', default=1, type=int, metavar='N', help='number of epochs to save')

args = parser.parse_args()

print(args)
best_loss = 1e6

dataram = dict()
tem_path = 'updatenet_lasot_dataset/'
tem_path_list = listdir(tem_path)
tem_path_list.sort()
first_tem = True
for tem_post in tem_path_list:
    print('Read: ' + tem_post)
    if first_tem:
        dataram['template0'] = np.load(join(tem_path, tem_post, 'template0.npy'))
        dataram['template'] = np.load(join(tem_path, tem_post,'template.npy'))
        dataram['templatei'] = np.load(join(tem_path, tem_post,'templatei.npy'))
        dataram['pre'] = np.load(join(tem_path, tem_post,'pre.npy'))
        dataram['gt'] = np.load(join(tem_path, tem_post,'gt.npy'))
        dataram['init0'] = np.load(join(tem_path, tem_post,'init0.npy'))      
        first_tem = False
    else:
        try:
            dataram['template0'] = np.concatenate((dataram['template0'], np.load(join(tem_path, tem_post, 'template0.npy'))))
            dataram['template'] = np.concatenate((dataram['template'], np.load(join(tem_path, tem_post,'template.npy'))))
            dataram['templatei'] = np.concatenate((dataram['templatei'], np.load(join(tem_path, tem_post,'templatei.npy'))))
            dataram['pre'] = np.concatenate((dataram['pre'], np.load(join(tem_path, tem_post,'pre.npy'))))
            dataram['gt'] = np.concatenate((dataram['gt'], np.load(join(tem_path, tem_post,'gt.npy'))))
            dataram['init0'] = np.concatenate((dataram['init0'], np.load(join(tem_path, tem_post,'init0.npy'))))
        except:
            print('Read .npy Over.')
    # break
dataram['train'] = np.arange(len(dataram['gt']), dtype=np.int)
print(dataram['init0'].shape)

cudnn.benchmark = True

save_path = args.save

def adjust_learning_rate(optimizer, epoch, lr0):
    #lr = np.logspace(-6, -6, num=args.epochs)[epoch]
    lr = np.logspace(-lr0[0], -lr0[1], num=args.epochs)[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, epoch,lr, filename=join(save_path, 'checkpoint.pth.tar')):
    name0 = 'lr-' + str(int(lr[0])) + '-' + str(int(lr[1]))
    epo_path = join(save_path, name0)
    if not isdir(epo_path):
        makedirs(epo_path)
    if (epoch+1) % args.save_rate == 0:
        filename=join(epo_path, 'checkpoint{}.pth.tar'.format(epoch+1))
        torch.save(state, filename)    

lrs = np.array([[4, 6], [4, 4], [3, 4], [3, 3]])

for ii in np.arange(0,lrs.shape[0]):
    # construct model
    model = UpdateResNet()
    model.cuda()
    criterion = nn.MSELoss(size_average=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            # best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #     .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lrs[ii])
        losses = AverageMeter()
        #subset = shuffle(subset)    
        subset = np.random.permutation(dataram['train'])
        for t in range(0, len(subset), args.batch_size):
            
            batchStart = t
            batchEnd = min(t+args.batch_size, len(subset))
            batch = subset[batchStart:batchEnd]
            init_index = dataram['init0'][batch]
            pre_index = dataram['pre'][batch]
            gt_index = dataram['gt'][batch]
            
            # reset diff T0
            for rr in range(len(init_index)):
                if init_index[rr] != 0:
                    init_index[rr] = np.random.choice(init_index[rr],1)

            cur = dataram['templatei'][batch]
            init = dataram['template0'][batch-init_index]
            pre = dataram['template'][batch-pre_index]
            gt = dataram['template0'][batch+gt_index-1]
            #pdb.set_trace() 
            temp = np.concatenate((init, pre, cur), axis=1)
            input_up = torch.Tensor(temp)
            target = torch.Tensor(gt)
            init_inp = Variable(torch.Tensor(init)).cuda()
            input_up = Variable(input_up).cuda()
            target = Variable(target).cuda()
            # compute output
            output = model(input_up, init_inp)
            loss = criterion(output, target)/target.size(0)

            # measure accuracy and record loss            
            losses.update(loss.cpu().data.numpy().tolist())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       str(epoch).zfill(2), str(t).zfill(5), len(subset), loss=losses))     
        save_checkpoint({'state_dict': model.state_dict()}, epoch, lrs[ii])        
