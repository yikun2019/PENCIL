import argparse
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import models
import numpy as np
from PIL import Image
import os
import os.path
import sys
import resnet
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='preact_resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                    metavar='H-P', help='initial learning rate')
parser.add_argument('--lr2', '--learning-rate2', default=0.2, type=float,
                    metavar='H-P', help='initial learning rate of stage3')
parser.add_argument('--alpha', default=0.4, type=float,
                    metavar='H-P', help='the coefficient of Compatibility Loss')
parser.add_argument('--beta', default=0.1, type=float,
                    metavar='H-P', help='the coefficient of Entropy Loss')
parser.add_argument('--lambda1', default=600, type=int,
                    metavar='H-P', help='the value of lambda')
parser.add_argument('--stage1', default=70, type=int,
                    metavar='H-P', help='number of epochs utill stage1')
parser.add_argument('--stage2', default=200, type=int,
                    metavar='H-P', help='number of epochs utill stage2')
parser.add_argument('--epochs', default=320, type=int, metavar='H-P',
                    help='number of total epochs to run')
parser.add_argument('--datanum', default=45000, type=int,
                    metavar='H-P', help='number of train dataset samples')
parser.add_argument('--classnum', default=10, type=int,
                    metavar='H-P', help='number of train dataset classes')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False,dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', dest='gpu', default='0', type=str,
                    help='select gpu')
parser.add_argument('--dir', dest='dir', default='', type=str, metavar='PATH',
                    help='model dir')

best_prec1 = 0

class CIFAR10(torch.utils.data.Dataset):

    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    # val_dataset is from data_batch_5

    val_list = [
        ['val_batch', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=0,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train == 0:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((45000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        elif self.train == 1:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.val_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.val_data = entry['data']
            if 'labels' in entry:
                self.val_labels = entry['labels']
            else:
                self.val_labels = entry['fine_labels']
            fo.close()
            self.val_data = self.val_data.reshape((5000, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == 0:
            img, target = self.train_data[index], self.train_labels[index]
        elif self.train == 1:
            img, target = self.test_data[index], self.test_labels[index]
        else:
            img, target = self.val_data[index], self.val_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train == 0:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        if self.train == 0:
            return len(self.train_data)
        elif self.train == 1:
            return len(self.test_data)
        else:
            return len(self.val_data)




def main():
    global args, best_prec1
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    y_file = args.dir + "y.npy"

    os.makedirs(args.dir)
    os.makedirs(args.dir+'record')

    model = resnet.__dict__[args.arch]()
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    checkpoint_dir = args.dir + "checkpoint.pth.tar"
    modelbest_dir = args.dir + "model_best.pth.tar"

    # optionally resume from a checkpoint
    if os.path.isfile(checkpoint_dir):
        print("=> loading checkpoint '{}'".format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        args.start_epoch = checkpoint['epoch']
        # args.start_epoch = 0
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_dir, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_dir))

    cudnn.benchmark = True

    # Data loading code
    transform1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,4),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.492, 0.482, 0.446), std = (0.247, 0.244, 0.262)),
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.492, 0.482, 0.446), std = (0.247, 0.244, 0.262)),
    ])
    trainset = CIFAR10(root='./', train=0, transform=transform1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True,num_workers=args.workers, pin_memory=True)
    testset = CIFAR10(root='./', train=1,transform=transform2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                              shuffle=False,num_workers=args.workers, pin_memory=True)

    valset = CIFAR10(root='./', train=2, transform=transform2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(testloader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        # load y_tilde
        if os.path.isfile(y_file):
            y = np.load(y_file)
        else:
            y = []

        train(trainloader, model, criterion, optimizer, epoch, y)

        # evaluate on validation set
        prec1 = validate(valloader, model, criterion)
        validate(testloader, model, criterion)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,filename=checkpoint_dir,modelbest=modelbest_dir)

def train(train_loader, model, criterion, optimizer, epoch, y):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    # new y is y_tilde after updating
    new_y = np.zeros([args.datanum,args.classnum])

    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        index = index.numpy()

        target1 = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target1)

        # compute output
        output = model(input_var)

        logsoftmax = nn.LogSoftmax(dim=1).cuda()
        softmax = nn.Softmax(dim=1).cuda()
        if epoch < args.stage1:
            # lc is classification loss
            lc = criterion(output, target_var)
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(target.size(0), 10).scatter_(1, target.view(-1, 1), 10.0)
            onehot = onehot.numpy()
            new_y[index, :] = onehot
        else:
            yy = y
            yy = yy[index,:]
            yy = torch.FloatTensor(yy)
            yy = yy.cuda(async = True)
            yy = torch.autograd.Variable(yy,requires_grad = True)
            # obtain label distributions (y_hat)
            last_y_var = softmax(yy)
            lc = torch.mean(softmax(output)*(logsoftmax(output)-torch.log((last_y_var))))
            # lo is compatibility loss
            lo = criterion(last_y_var, target_var)
        # le is entropy loss
        le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

        if epoch < args.stage1:
            loss = lc
        elif epoch < args.stage2:
            loss = lc + args.alpha * lo + args.beta * le
        else:
            loss = lc

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target1, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= args.stage1 and epoch < args.stage2:
            lambda1 = args.lambda1
            # update y_tilde by back-propagation
            yy.data.sub_(lambda1*yy.grad.data)

            new_y[index,:] = yy.data.cpu().numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    if epoch < args.stage2:
        # save y_tilde
        y = new_y
        y_file = args.dir + "y.npy"
        np.save(y_file,y)
        y_record = args.dir + "record/y_%03d.npy" % epoch
        np.save(y_record,y)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='', modelbest = ''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate"""
    if epoch < args.stage2 :
        lr = args.lr
    elif epoch < (args.epochs - args.stage2)//3 + args.stage2:
        lr = args.lr2
    elif epoch < 2 * (args.epochs - args.stage2)//3 + args.stage2:
        lr = args.lr2//10
    else:
        lr = args.lr2//100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _ , pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
