#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from models.simsiam import SimSiam
from setlogger import get_logger
import torch.nn.functional as F

import moco.loader
import moco.builder

#from simsiam
from augmentations import get_aug
from tools import AverageMeter, PlotLogger, knn_monitor, ProgressMeter
from models import get_model
from dataset import get_dataset
from optimizers import get_optimizer, LR_Scheduler

saved_path = os.path.join("logs/R50e100_bs512lr0.1/")#rs56_KDCL_MinLogit_cifar_e250 rs56_5KD_0.4w_cifar_e250
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, 'train.log'))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model', type=str, default='simsiam')
parser.add_argument('--backbone', type=str, default='resnet50')
parser.add_argument('--train_list',  type=str, default='train.txt',
                    help='name of data list file')
parser.add_argument('--test_list', type=str, default='test.txt',
                    help='name of data list file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--base_lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='base_lr')
parser.add_argument('--schedule', default=[700, 750], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='choose from random, stl10, mnist, cifar10, cifar100, imagenet')
parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
parser.add_argument('--proj_layers', type=int, default=None,
                    help="number of projector layers. In cifar experiment, this is set to 2")
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='sgd, lars(from lars paper), lars_simclr(used in simclr and byol), larc(used in swav)')
parser.add_argument('--eval_after_train', type=str, default=None)
parser.add_argument('--test_epoch', type=int, default=10)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
parser.add_argument('--output_dir', type=str, default=os.getenv('OUTPUT'))
parser.add_argument('--hide_progress', action='store_true')
parser.add_argument('--warmup_epochs', type=int, default=0,
                    help='learning rate will be linearly scaled during warm up period')
parser.add_argument('--warmup_lr', type=float, default=0, help='Initial warmup learning rate')
parser.add_argument('--final_lr', type=float, default=0)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    # model = SimSiam(backbone=args.arch)
    model = get_model(args.model, args.arch)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            print('set')
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # define optimizer
    args.lr = args.base_lr*args.batch_size/256
    optimizer = get_optimizer(
        args.optimizer, model,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # if args.aug_plus:
    #     # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    #     augmentation = [
    #         transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    #         transforms.RandomApply([
    #             transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    #         ], p=0.8),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ]
    # else:
    #     # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
    #     augmentation = [
    #         transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    #         transforms.RandomGrayscale(p=0.2),
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize
    #     ]

    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))
    dataset_kwargs = {
        'dataset': args.dataset,
        'data_dir': args.data_dir,
        'download': args.download,
        'debug_subset_size': args.batch_size if args.debug else None
    }
    train_dataset = get_dataset(args,
                          transform=get_aug(args.model, args.image_size, True),
                          train=True,
                          **dataset_kwargs)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.workers,
        'sampler': train_sampler,
    }

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=(train_sampler is None),
        **dataloader_kwargs
    )
    print('(train_sampler is None): ', (train_sampler is None) , 'len(train_loader)', len(train_loader))

    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(args,
            transform=get_aug(args.model, args.image_size, False, train_classifier=False),
            train=True,
            **dataset_kwargs),
        shuffle=False,
        **dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(args,
            transform=get_aug(args.model, args.image_size, False, train_classifier=False),
            train=False,
            **dataset_kwargs),
        shuffle=False,
        **dataloader_kwargs
    )
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.warmup_epochs, args.warmup_lr*args.batch_size/256,
        args.epochs, args.base_lr*args.batch_size/256, args.final_lr*args.batch_size/256,
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.warmup_epochs == 0:
            adjust_learning_rate(optimizer, epoch, args)#TODO(yifan):maybe switch to warmup
        logger.info('Lr:{}{}'.format(optimizer.param_groups[0]['lr'], args.epochs))

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args, lr_scheduler)

        if epoch % args.test_epoch == 0 and not  args.distributed:
            accuracy = knn_monitor(model.features, memory_loader, test_loader,  k=200,
                                   hide_progress=args.hide_progress)
            print('accuracy: ', accuracy)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=saved_path+'checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, optimizer, epoch, args, lr_scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses], logger,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        # output, target = model(im_q=images[0], im_k=images[1])
        # loss = criterion(output, target)
        z1, p1 = model(images[0])
        z2, p2 = model(images[1])
        loss = negcos(p1, z2) / 2 + negcos(p2, z1) / 2

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # top1.update(acc1[0], images[0].size(0))
        # top5.update(acc5[0], images[0].size(0))

        losses.update(loss.item(), images[0].size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr = lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


def negcos(p, z):
    # z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p*z.detach()).sum(dim=1).mean()
    # return - nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()
if __name__ == '__main__':
    main()
