import os.path as osp
import argparse
import json
import math
import re
import bisect

import matplotlib.pyplot as plt


# -------------------- utility functions -------------------- #
def append(loss_dict, loss_name, loss_value):
    if loss_name not in loss_dict:
        loss_dict[loss_name] = [loss_value]
    else:
        loss_dict[loss_name].append(loss_value)


def split(pattern, string):
    return re.split(r'\s*{}\s*'.format(pattern), string)


def parse_log(log_file):
    # define patterns
    loss_pattern = r'.*\[epoch:.*\| iter: (\d+).*\] (.*)'
    metric_pattern1 = r'.*Sequence: (.*)'
    metric_pattern2 = r'.*Average'
    metric_pattern3 = r'.*:\s*(.*): (.*) \(x(.*)\)'

    # load log file
    with open(log_file, 'r') as f:
        lines = [line.strip() for line in f]

    # parse log file
    loss_dict = {}    # {'iter': [], 'loss1': [], 'loss2':[], ...}
    metric_dict = {}  # {'iter': [], 'sequence1': {'metric1': [], 'metric2':[]}, ...}
    last_test_sequence = ''
    for line in lines:
        loss_match = re.match(loss_pattern, line)
        if loss_match:
            iter = int(loss_match.group(1))
            append(loss_dict, 'iter', iter)
            for s in split(',', loss_match.group(2)):
                if s:
                    k, v = split(':', s)
                    append(loss_dict, k, float(v))

        metric_match1 = re.match(metric_pattern1, line)
        if metric_match1:
            last_test_sequence = metric_match1.group(1)
            if last_test_sequence not in metric_dict:
                metric_dict[last_test_sequence] = {}

        metric_match2 = re.match(metric_pattern2, line)
        if metric_match2:
            last_test_sequence = 'Average'
            if last_test_sequence not in metric_dict:
                metric_dict[last_test_sequence] = {}

        metric_match3 = re.match(metric_pattern3, line)
        if metric_match3:
            iter = loss_dict['iter'][-1]
            # enforce to add a new iter
            if 'iter' not in metric_dict or metric_dict['iter'][-1] != iter:
                append(metric_dict, 'iter', iter)

            k = metric_match3.group(1)
            v = float(metric_match3.group(2)) * float(metric_match3.group(3))
            append(metric_dict[last_test_sequence], k, v)

    return loss_dict, metric_dict


def parse_json(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)

    metric_dict = {}
    for model_idx, metrics in json_dict.items():
        append(metric_dict, 'iter', int(model_idx.replace('G_iter', '')))
        for metric, val in metrics.items():
            append(metric_dict, metric, float(val))

    return metric_dict


def plot_curve(ax, iter, value, style='-', alpha=1.0, label='', color='seagreen',
               start_iter=0, end_iter=-1, smooth=0, linewidth=1.0):

    assert len(iter) == len(value), \
        'mismatch in <iter> and <value> ({} vs {})'.format(
            len(iter), len(value))
    l = len(iter)

    if smooth:
        for i in range(1, l):
            value[i] = smooth * value[i - 1] + (1 - smooth) * value[i]

    start_index = bisect.bisect_left(iter, start_iter)
    end_index = l if end_iter < 0 else bisect.bisect_right(iter, end_iter)
    ax.plot(
        iter[start_index:end_index], value[start_index:end_index],
        style, alpha=alpha, label=label, linewidth=linewidth, color=color)


def plot_loss_curves(loss_dict, ax, loss_type, start_iter=0, end_iter=-1,
                     smooth=0):

    for model_idx, model_loss_dict in loss_dict.items():
        if loss_type in model_loss_dict:
            plot_curve(
                ax, model_loss_dict['iter'], model_loss_dict[loss_type],
                alpha=1.0, label=model_idx, start_iter=start_iter,
                end_iter=end_iter, smooth=smooth, color='royalblue')
    ax.legend(loc='best', fontsize='small')
    ax.set_ylabel(loss_type)
    ax.set_xlabel('iteration')
    plt.grid(True)


def plot_metric_curves(metric_dict, ax, metric_type, start_iter=0, end_iter=-1):
    """ currently only support to plot average results
    """

    for model_idx, model_metric_dict in metric_dict.items():
        if metric_type in model_metric_dict:
            plot_curve(
                ax, model_metric_dict['iter'], model_metric_dict[metric_type],
                alpha=1.0, label=model_idx, start_iter=start_iter,
                end_iter=end_iter)
    ax.legend(loc='best', fontsize='small')
    ax.set_ylabel(metric_type)
    ax.set_xlabel('iteration')
    plt.grid(True)



# -------------------- model-specific monitor -------------------- #
def monitor_VSR(exp_dir, testset):

    # ================ basic settings ================#
    # define logs to monitor
    log_info_lst = [
        # (model name, experiment index)
        ('FRVSR', '001'),
    ]
    # define which losses to monitor
    loss_lst = [
        'l_pix_G',  # pixel loss
        'l_warp_G', # warping loss
    ]
    # define metrics to monitor
    metric_lst = [
        'PSNR',
    ]
    # other settings
    start_iter = 0
    loss_smooth = 0

    # ================ parse logs ================#
    loss_dict = {}    # {'model1': {'loss1': x1, ...}, ...}
    metric_dict = {}  # {'model1': {'metric1': x1, ...}, ...}
    for log_info in log_info_lst:
        # model_idx = '{} [{}]'.format(*log_info)
        model_idx = 'FRVSR'

        # parse log
        log_file = osp.join(
            exp_dir, '{}/{}/train/train.log'.format(*log_info))
        model_loss_dict, _ = parse_log(log_file)
        loss_dict[model_idx] = model_loss_dict

        # parse json
        json_file = osp.join(
            exp_dir, '{}/{}/test/metrics/{}_avg.json'.format(*log_info, testset))
        model_metric_dict = parse_json(json_file)
        metric_dict[model_idx] = model_metric_dict

    # ================ plot loss curves ================#
    n_loss = len(loss_lst)
    base_figsize = (12, 2 * math.ceil(n_loss / 2))
    fig = plt.figure(1, figsize=base_figsize)
    for i in range(n_loss):
        ax = fig.add_subplot('{}{}{}'.format(math.ceil(n_loss / 2), 2, i + 1))
        plot_loss_curves(
            loss_dict, ax, loss_lst[i], start_iter=start_iter,
            smooth=loss_smooth)

    # ================ plot metric curves ================#
    n_metric = len(metric_lst)
    base_figsize = (12, 2 * math.ceil(n_metric / 2))
    fig = plt.figure(2, figsize=base_figsize)
    for i in range(n_metric):
        ax = fig.add_subplot(
            '{}{}{}'.format(math.ceil(n_metric / 2), 2, i + 1))
        plot_metric_curves(
            metric_dict, ax, metric_lst[i], start_iter=start_iter)

    plt.show()


def monitor_VSRGAN(exp_dir, testset):

    # ================ basic settings ================#
    # define logs to monitor
    log_info_lst = [
        # (model name, experiment index)
        ('TecoGAN', '001'),
    ]
    # define losses to monitor
    loss_lst = [
        'l_pix_G',   # pixel loss
        'l_feat_G',  # perceptual loss
        'l_gan_G',   # generator loss
        'l_gan_D',   # discriminator loss
        'p_real_D',
        'p_fake_D',
        'l_warp_G',  # warping loss
    ]
    # define which metrics to monitor
    metric_lst = [
        'LPIPS',
        'tOF',
        'PSNR',
    ]
    # other settings
    base_figsize = (12, 2*math.ceil(len(loss_lst) / 2))
    start_iter = 0
    loss_smooth = 0

    # ================ parse logs ================#
    loss_dict = {}    # {'model1': {'loss1': x1, ...}, ...}
    metric_dict = {}  # {'model1': {'metric1': x1, ...}, ...}
    for log_info in log_info_lst:
        # model_idx = '{} [{}]'.format(*log_info)
        model_idx = 'VSR GAN'

        # parse log
        log_file = osp.join(
            exp_dir, '{}/{}/train/train.log'.format(*log_info))
        model_loss_dict, _ = parse_log(log_file)
        loss_dict[model_idx] = model_loss_dict

        # parse json
        json_file = osp.join(
            exp_dir, '{}/{}/test/metrics/{}_avg.json'.format(*log_info, testset))
        model_metric_dict = parse_json(json_file)
        metric_dict[model_idx] = model_metric_dict

    # ================ plot loss curves ================#
    n_loss = len(loss_lst)
    fig = plt.figure(1, figsize=base_figsize)
    for i in range(n_loss):
        ax = fig.add_subplot('{}{}{}'.format(math.ceil(n_loss / 2), 2, i + 1))
        plot_loss_curves(
            loss_dict, ax, loss_lst[i], start_iter=start_iter,
            smooth=loss_smooth)

    # ================ plot metric curves ================#
    n_metric = len(metric_lst)
    fig = plt.figure(2, figsize=base_figsize)
    for i in range(n_metric):
        ax = fig.add_subplot(
            '{}{}{}'.format(math.ceil(n_metric / 2), 2, i + 1))
        plot_metric_curves(
            metric_dict, ax, metric_lst[i], start_iter=start_iter)

    plt.show()


if __name__ == '__main__':
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument('--degradation', type=str, required=True,
                        help='BD or BI')
    parser.add_argument('--model', type=str, required=True,
                        help='TecoGAN or FRVSR')
    parser.add_argument('--dataset', type=str, required=True,
                        help='which testset to show')
    args = parser.parse_args()
    
    # select model
    exp_dir = 'experiments_{}'.format(args.degradation)

    if args.model == 'FRVSR':
        monitor_VSR(exp_dir, args.dataset)
    elif args.model == 'TecoGAN':
        monitor_VSRGAN(exp_dir, args.dataset)
    else:
        raise ValueError('Unrecoginzed model: {}'.format(args.model))
