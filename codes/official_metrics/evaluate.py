import os
import os.path as osp
import argparse


if __name__ == '__main__':
    # get agrs
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vespcn_ep0500')
    args = parser.parse_args()

    # keys = args.model.split('_')
    # assert keys[0] in ('TecoGAN', 'FRVSR')
    # assert keys[1] in ('BD', 'BI')

    # setup dirs
    Vid4_GT_dir = 'data/Vid4/GT'
    Vid4_vids = os.listdir(Vid4_GT_dir)

    ToS3_GT_dir = 'data/ToS3/GT'
    ToS3_vids = os.listdir(ToS3_GT_dir)

    Gvt72_GT_dir = 'data/Gvt72/GT'
    Gvt72_vids = os.listdir(Gvt72_GT_dir)

    # Vid4_vids = ['calendar', 'city', 'foliage', 'walk']
    # ToS3_vids = ['bridge', 'face', 'room']

    # Vid4_SR_dir = 'results/Vid4/{}'.format(args.model)
    # ToS3_SR_dir = 'results/ToS3/{}'.format(args.model)

    Vid4_SR = 'results/Vid4/'
    ToS3_SR = 'results/ToS3/'
    Gvt72_SR = 'results/Gvt72/'

    # evalte Gvt72:
    model_list = os.listdir(Gvt72_SR)
    for model in model_list:
        print('test dataset:Gvt72,\tmodel:{}'.format(model))
        Gvt72_SR_dir = os.path.join(Gvt72_SR, model)

        Gvt72_GT_lst = [
            osp.join(Gvt72_GT_dir, vid) for vid in Gvt72_vids]
        Gvt72_SR_lst = [
            osp.join(Gvt72_SR_dir, vid) for vid in Gvt72_vids]
        os.system('python codes/official_metrics/metrics.py --output {} --results {} --targets {}'.format(
            osp.join(Gvt72_SR_dir, 'metric_log'),
            ','.join(Gvt72_SR_lst),
            ','.join(Gvt72_GT_lst)))

    # evaluate Tos3:
    model_list = os.listdir(ToS3_SR)
    for model in model_list:
        print('test dataset:ToS3,\tmodel:{}'.format(model))
        ToS3_SR_dir = os.path.join(ToS3_SR, model)

        ToS3_GT_lst = [
            osp.join(ToS3_GT_dir, vid) for vid in ToS3_vids]
        ToS3_SR_lst = [
            osp.join(ToS3_SR_dir, vid) for vid in ToS3_vids]
        os.system('python codes/official_metrics/metrics.py --output {} --results {} --targets {}'.format(
            osp.join(ToS3_SR_dir, 'metric_log'),
            ','.join(ToS3_SR_lst),
            ','.join(ToS3_GT_lst)))

    # evaluate Vid4
    model_list = os.listdir(Vid4_SR)
    for model in model_list:
        print('test dataset:Vid4,\tmodel:{}'.format(model))
        Vid4_SR_dir = os.path.join(Vid4_SR, model)

        Vid4_GT_lst = [
            osp.join(Vid4_GT_dir, vid) for vid in Vid4_vids]
        Vid4_SR_lst = [
            osp.join(Vid4_SR_dir, vid) for vid in Vid4_vids]
        os.system('python codes/official_metrics/metrics.py --output {} --results {} --targets {}'.format(
            osp.join(Vid4_SR_dir, 'metric_log'),
            ','.join(Vid4_SR_lst),
            ','.join(Vid4_GT_lst)))

