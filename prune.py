'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time
from collections import OrderedDict

import distiller
import distiller.apputils as apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# Distiller parser
SUMMARY_CHOICES = ['sparsity', 'model']
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model,and exit - options:' +
                         ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--momentum', default=0., type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0., type=float, metavar='W',
                    help='weight decay ( default: 1e-4)')
# parser.add_argument('--lr', type=float, default=20,
#                     help='initial learning rate')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../../data.cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../../data.cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          patience=0, verbose=True, factor=0.5)


# Training
def train(epoch, optimizer, compression_scheduler=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, minibatch_id=batch_idx,
                                                     minibatches_per_epoch=128)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if compression_scheduler:
            compression_scheduler.before_backward_pass(epoch, minibatch_id=batch_idx,
                                                       minibatches_per_epoch=128,loss=loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, minibatch_id=batch_idx,
                                                   minibatches_per_epoch=128)

        if batch_idx % 200 == 0 and batch_idx > 0:
            cur_loss = train_loss / 200
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            msglogger.info(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} '
                '| loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch_idx, len(trainloader) // 200, lr,
                                      elapsed * 1000 / 200, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            stats = ('Performance/Training/',
                     OrderedDict([
                         ('Loss', cur_loss),
                         ('Perplexity', math.exp(cur_loss)),
                         ('LR', lr),
                         ('Batch Time', elapsed * 1000)])
                     )
            steps_completed = batch_idx + 1
            distiller.log_training_progress(stats, net.named_parameters(), epoch, steps_completed,
                                            128, 200, [tflogger])


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return test_loss


# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)

# Distiller loggers
msglogger = apputils.config_pylogger('logging.conf', None)
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)

if args.summary:
    # The last string is the dataset
    distiller.model_summary(net, args.summary, 'coco')
    exit(0)

compression_scheduler = None

if args.compress:
    source = args.compress
    compression_scheduler = distiller.CompressionScheduler(net)
    distiller.config.file_config(net, optimizer, source, compression_scheduler)

try:
    for epoch in range(0, args.epochs):
        total_loss = 0.
        epoch_start_time = time.time()
        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch)

        train(epoch, optimizer, compression_scheduler)
        val_loss = test(epoch)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch)

except KeyboardInterrupt:
    msglogger.info('-' * 89)
    msglogger.info('Exiting from training early')
