import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from config import Config
from load_dataset.load_dataset import fetch_dataloader
from models.resnet18 import resnet18, resnet18_irs, resnet18_test
from src.test_fn import test_fn, test_fn_irs
from src.train_fn import train_fn, train_fn_irs, train_fn_ovf


def train_loop(
        args,
        model,
        device,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        epochs,
        save_path,
        log_path,
):
    best_acc = 0.
    for epoch in range(epochs):
        log_file = open(log_path, 'a')
        log_file.write(f"Starting epoch {epoch + 1}\n")

        if args.type == 'irs':
            train_loss = train_fn_irs(args, model, device, train_loader, optimizer, criterion)
            acc = test_fn_irs(model, device, test_loader, 5, True)
            if acc[-1] > best_acc:
                best_acc = acc[-1]
                log_file.write(f'acc1 is {acc[0]:.2f}%, acc2 is {acc[1]:.2f}%, acc3 is {acc[2]:.2f}%, '
                               f'acc4 is {acc[3]:.2f}%, accb is {acc[3]:.2f}%, loss is {train_loss}\n')
                torch.save(model.state_dict(), save_path)

        elif args.type == 'ovf':
            train_loss = train_fn_ovf(args, model, device, train_loader, optimizer, criterion)
            acc = test_fn(model, device, test_loader)
            if acc > best_acc:
                best_acc = acc
                log_file.write(f'acc is {acc:.2f}%, loss is {train_loss}\n')
                torch.save(model.state_dict(), save_path)

        elif args.type == 'base':
            train_loss = train_fn(model, device, train_loader, optimizer, criterion)
            acc = test_fn(model, device, test_loader)
            if acc > best_acc:
                best_acc = acc
                log_file.write(f'acc is {acc:.2f}%, loss is {train_loss}\n')
                torch.save(model.state_dict(), save_path)

        scheduler.step()
        log_file.close()
    log_file = open(log_path, 'a')
    log_file.write(f'best accuracy is {best_acc:.2f}%\n')
    log_file.close()


def train_part(
        args,
        train_loader,
        test_loader
):
    save_path = f'check_points/res18_{args.type}_{args.dataset}_{args.num}_{args.mark}.pth'

    save_log = f'./log/{args.dataset}'
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    log_path = f'{save_log}/mark{args.mark}_{args.num}.txt'
    log_file = open(log_path, 'a')

    if args.dataset == 'mnist':
        in_channel = 1
        num_classes = 10
    if args.dataset == 'cifar10':
        in_channel = 3
        num_classes = 10
    elif args.dataset == 'cifar100':
        in_channel = 3
        num_classes = 100
    elif args.dataset == 'tiny':
        in_channel = 3
        num_classes = 200

    if args.type == 'base':
        net = resnet18(in_channel, num_classes, noise_backbone=args.var1).to(Config.DEVICE)
        log_file.write('Resnet-18 baseline\n')
    elif args.type == 'irs':
        net = resnet18_irs(in_channel, num_classes, noise_backbone=args.var1, noise_block=args.var2).to(Config.DEVICE)
        log_file.write('Resnet-18 irs\n')
    elif args.type == 'ovf':
        net = resnet18(in_channel, num_classes, noise_backbone=args.var1).to(Config.DEVICE)
        log_file.write('Resnet-18 ovf\n')

    log_file.write(f'Resnet-18 with {args.type}\n')
    log_file.write(f'save model at {save_path}\n')
    log_file.write(f'epoch={Config.EPOCH}, lr={Config.LR}, batch size={Config.BATCH_SIZE}.\n')
    log_file.write(f'weight bit={Config.WEIGHT_BIT}, device bit={Config.DEVICE_BIT}.\n')
    log_file.write(f'noise var={args.var1}.\n')
    log_file.write(f'dataset={args.dataset}, beta={args.beta}.\n')
    log_file.close()

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), Config.LR)
    optimizer = torch.optim.SGD(net.parameters(), Config.LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCH)

    train_loop(args, net, Config.DEVICE, criterion, optimizer, scheduler, train_loader, test_loader, Config.EPOCH, save_path, log_path)
    log_file = open(log_path, 'a')
    log_file.write('******Train END!!!*****\n')
    log_file.close()


def test_part(args, test_loader):
    save_path = f'check_points/res18_{args.type}_{args.dataset}_{args.num}_{args.mark}.pth'

    save_log = f'./log/{args.dataset}'
    if not os.path.exists(save_log):
        os.makedirs(save_log)
    log_path = f'{save_log}/mark{args.mark}_{args.num}.txt'
    log_file = open(log_path, 'a')
    log_file.write('******Test BEGIN!!!*****\n')

    dataset_params = {
        'mnist': (1, 10, torch.arange(0, 0.8 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1]),
        'cifar10': (3, 10, torch.arange(0, 0.4 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1]),
        'cifar100': (3, 100, torch.arange(0, 0.4 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1]),
        'tiny': (3, 200, torch.arange(0, 0.4 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1])
    }

    in_channel, num_classes, inf_variation = dataset_params.get(args.dataset, (None, None, None))

    log_file.write(f'{Config.MC_times} times MC inference with noise on {save_path}.\n')
    log_file.write(f'weight bit={Config.WEIGHT_BIT}, device bit={Config.DEVICE_BIT}.\n')
    log_file.write(f'noise variation in inference:{inf_variation}\n')

    std = []
    acc_list = []
    ci95 = []
    ci99 = []
    for noise_var in inf_variation:
        if noise_var == 0:
            net = resnet18_test(in_channel, num_classes, noise_backbone=noise_var).to(Config.DEVICE)
            net.load_state_dict(torch.load(save_path), strict=False)
            net.epoch_noise()
            accb = test_fn(net, Config.DEVICE, test_loader)
            acc_list.append(accb)
            std.append(0)
            ci95.append(0)
            ci99.append(0)
        else:
            accb_loop = []
            for _ in range(Config.MC_times):
                net = resnet18_test(in_channel, num_classes, noise_backbone=noise_var).to(Config.DEVICE)
                net.load_state_dict(torch.load(save_path), strict=False)
                net.epoch_noise()
                accb = test_fn(net, Config.DEVICE, test_loader)
                accb_loop.append(accb)
            std.append(np.std(accb_loop, ddof=1))
            acc_list.append(np.mean(accb_loop))
            ci95.append((np.std(accb_loop, ddof=1) / np.sqrt(len(accb_loop))) * 1.96)
            ci99.append((np.std(accb_loop, ddof=1) / np.sqrt(len(accb_loop))) * 2.576)
    AC = [float('{:.2f}'.format(i)) for i in acc_list]
    STD = [float('{:.2f}'.format(i)) for i in std]
    CI95 = [float('{:.2f}'.format(i)) for i in ci95]
    CI99 = [float('{:.2f}'.format(i)) for i in ci99]
    log_file.write(f'{AC}\n')
    log_file.write(f'{STD}\n')
    log_file.write(f'{CI95}\n')
    log_file.write(f'{CI99}\n')
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',
                        type=str,
                        help="train / test / tnt")
    parser.add_argument('--type',
                        type=str,
                        help="base / irs / ovf")
    parser.add_argument('--dataset',
                        type=str,
                        help="mnist / cifar10 / cifar100 / tiny")
    parser.add_argument('--var1',
                        type=float,
                        help="the variation of backbone")
    parser.add_argument('--var2',
                        type=float,
                        help="the variation of blocks")
    parser.add_argument('--beta',
                        type=float,
                        help="beta factor")
    parser.add_argument('--num',
                        type=str,
                        help="the number of runs")
    parser.add_argument('--mark',
                        type=float,
                        help="model mark")
    args = parser.parse_args()

    train_loader = fetch_dataloader(args.dataset, train=True)
    test_hard = fetch_dataloader(args.dataset, train=False)
    test_loader = []
    for data, target in test_hard:
        test_loader.append((data, target))

    Config.LR = 1e-2

    if args.mode == 'train':
        train_part(args, train_loader, test_loader)

    if args.mode == 'test':
        test_part(args, test_loader)

    if args.mode == 'tnt':
        train_part(args, train_loader, test_loader)
        test_part(args, test_loader)