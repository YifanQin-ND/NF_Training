import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from config import Config, s_factor, res18_beta
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
):
    best_acc = 0.
    best_ep = 0
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}")

        if args.type == 'irs':
            train_loss = train_fn_irs(args, model, device, train_loader, optimizer, criterion, res18_beta[args.type][args.dataset])
            acc = test_fn_irs(model, device, test_loader, 5, True)
            print(f'acc1 is {acc[0]:.2f}%, acc2 is {acc[1]:.2f}%, acc3 is {acc[2]:.2f}%, '
                  f'acc4 is {acc[3]:.2f}%, accb is {acc[4]:.2f}%, loss is {train_loss}')

            if acc[-1] > best_acc:
                best_acc = acc[-1]
                best_ep = epoch + 1
                torch.save(model.state_dict(), save_path)

        elif args.type == 'ovf':
            train_loss = train_fn_ovf(args, model, device, train_loader, optimizer, criterion, res18_beta[args.type][args.dataset])
            acc = test_fn(model, device, test_loader)
            print(f'acc is {acc:.2f}%, loss is {train_loss}')
            if acc > best_acc:
                best_acc = acc
                best_ep = epoch + 1
                torch.save(model.state_dict(), save_path)

        elif args.type == 'base':
            train_loss = train_fn(model, device, train_loader, optimizer, criterion)
            acc = test_fn(model, device, test_loader)
            print(f'acc is {acc:.2f}%, loss is {train_loss}')
            if acc > best_acc:
                best_acc = acc
                best_ep = epoch + 1
                torch.save(model.state_dict(), save_path)

        scheduler.step()
    print(f'best accuracy is {best_acc:.2f}%')
    print(f'best epoch is {best_ep}')


def train_part(
        args,
        train_loader,
        test_loader
):
    save_path = f'check_points/res18_{args.type}_{args.dataset}_{args.mark}_{args.num}.pth'

    print(f'******Train BEGIN!!!*****')
    print(f'Res-18 with {args.type}')
    print(f'save model at {save_path}')
    print(f'epoch={Config.EPOCH}, lr={Config.LR}, batch size={Config.BATCH_SIZE}.')
    print(f'weight bit={Config.WEIGHT_BIT}, device bit={Config.DEVICE_BIT}.')
    print(f'noise var1={args.var1}, var2={args.var2}.')
    print(f'device={args.device}, s_factor={s_factor[args.device]}.')
    print(f'dataset={args.dataset}, beta={res18_beta[args.type][args.dataset]}.')

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
        net = resnet18(in_channel, num_classes, args.var1, s_factor[args.device]).to(Config.DEVICE)
    elif args.type == 'irs':
        net = resnet18_irs(in_channel, num_classes, args.var1, args.var2, s_factor[args.device], s_factor['RRAM1']).to(Config.DEVICE)
    elif args.type == 'ovf':
        net = resnet18(in_channel, num_classes, args.var1, s_factor[args.device]).to(Config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), Config.LR)
    # optimizer = torch.optim.SGD(net.parameters(), Config.LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCH)

    # print training settings: optimizer
    print(f'optimizer: {optimizer}')

    train_loop(args, net, Config.DEVICE, criterion, optimizer, scheduler, train_loader, test_loader, Config.EPOCH, save_path)


def test_part(args, test_loader):
    save_path = f'check_points/res18_{args.type}_{args.dataset}_{args.mark}_{args.num}.pth'

    print(f'******Test BEGIN!!!*****')
    print(f'Res-18 with {args.type}')
    print(f'load model from {save_path}')
    print(f'weight bit={Config.WEIGHT_BIT}, device bit={Config.DEVICE_BIT}.')
    print(f'noise var1={args.var1}, var2={args.var2}.')
    print(f'device={args.device}, s_factor={s_factor[args.device]}.')
    print(f'dataset={args.dataset}, beta={res18_beta[args.type][args.dataset]}.')
    print(f'{Config.MC_times} times MC inference.')

    dataset_params = {
        'mnist': (1, 10, torch.arange(0, 0.4 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1]),
        'cifar10': (3, 10, torch.arange(0, 0.4 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1]),
        'cifar100': (3, 100, torch.arange(0, 0.4 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1]),
        'tiny': (3, 200, torch.arange(0, 0.4 + 0.05, 0.05).tolist() if args.var1 == 0 else [0, args.var1])
    }

    in_channel, num_classes, inf_variation = dataset_params.get(args.dataset, (None, None, None))
    print(f'noise variation in inference:{inf_variation}')

    std = []
    acc_list = []
    ci95 = []
    ci99 = []
    for noise_var in inf_variation:
        if noise_var == 0:
            net = resnet18_test(in_channel, num_classes, noise_var, s_factor[args.device]).to(Config.DEVICE)
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
                net = resnet18_test(in_channel, num_classes, noise_var, s_factor[args.device]).to(Config.DEVICE)
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

    print(f'accuracy: {AC}')
    print(f'std: {STD}')
    print(f'95% CI: {CI95}')
    print(f'99% CI: {CI99}')


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
    parser.add_argument('--device',
                        type=str,
                        default='RRAM1',
                        help="device type")
    parser.add_argument('--num',
                        type=str,
                        help="the number of runs")
    parser.add_argument('--mark',
                        type=float,
                        help="model mark")
    args = parser.parse_args()

    def format_time(seconds):
        days = int(seconds // 86400)  # 1 天 = 86400 秒
        hours = int((seconds % 86400) // 3600)  # 1 小时 = 3600 秒
        minutes = int((seconds % 3600) // 60)  # 1 分钟 = 60 秒
        seconds = seconds % 60
        return f"{days}d {hours}h {minutes}m {seconds:.2f}s"

    start_time = time.time()
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

    end_time = time.time()
    print(f'Total time: {format_time(end_time - start_time)}')

