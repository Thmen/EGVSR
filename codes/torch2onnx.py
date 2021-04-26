import os
import os.path as osp
import argparse
import yaml
import time
import torch

from data import create_dataloader, prepare_data
from models import define_model
from models.networks import define_generator
from utils import base_utils, data_utils
from metrics.model_summary import register, profile_model

def test(opt):
    # logging
    logger = base_utils.get_logger('base')
    if opt['verbose']:
        logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
        base_utils.print_options(opt, logger)

    # infer and evaluate performance for each model
    for load_path in opt['model']['generator']['load_path_lst']:
        # setup model index
        model_idx = osp.splitext(osp.split(load_path)[-1])[0]

        # log
        logger.info('=' * 40)
        logger.info('Testing model: {}'.format(model_idx))
        logger.info('=' * 40)

        # create model
        opt['model']['generator']['load_path'] = load_path
        model = define_model(opt)

        # for each test dataset
        for dataset_idx in sorted(opt['dataset'].keys()):
            # use dataset with prefix `test`
            if not dataset_idx.startswith('test'):
                continue

            ds_name = opt['dataset'][dataset_idx]['name']
            logger.info('Testing on {}: {}'.format(dataset_idx, ds_name))

            # create data loader
            test_loader = create_dataloader(opt, dataset_idx=dataset_idx)

            # infer and store results for each sequence
            for i, data in enumerate(test_loader):

                # fetch data
                lr_data = data['lr'][0]
                seq_idx = data['seq_idx'][0]
                frm_idx = [frm_idx[0] for frm_idx in data['frm_idx']]

                # infer
                hr_seq = model.infer(lr_data)  # thwc|rgb|uint8

                # save results (optional)
                if opt['test']['save_res']:
                    res_dir = osp.join(
                        opt['test']['res_dir'], ds_name, model_idx)
                    res_seq_dir = osp.join(res_dir, seq_idx)
                    data_utils.save_sequence(
                        res_seq_dir, hr_seq, frm_idx, to_bgr=True)

            logger.info('-' * 40)

    # logging
    logger.info('Finish testing')
    logger.info('=' * 40)


if __name__ == '__main__':
    # ----------------- parse arguments ----------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default="../../experiments_BI/TecoGAN/001")
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--model', type=str, default="TecoGAN")
    parser.add_argument('--opt', type=str, default="test_onnx.yml")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--lr_size', type=str, default='3x960x540')
    parser.add_argument('--test_speed', action='store_true')
    args = parser.parse_args()

    # ----------------- get options ----------------- #
    print(args.exp_dir)
    with open(osp.join(args.exp_dir, args.opt), 'r') as f:
        opt = yaml.load(f.read(), Loader=yaml.FullLoader)

    # ----------------- general configs ----------------- #
    # experiment dir
    opt['exp_dir'] = args.exp_dir
    # random seed
    base_utils.setup_random_seed(opt['manual_seed'])
    # logger
    base_utils.setup_logger('base')
    opt['verbose'] = opt.get('verbose', False)

    # device
    if args.gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        if torch.cuda.is_available():
            # TODO: torch.backends.cudnn.benchmark setting
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.benchmark = True
            opt['device'] = 'cuda'
        else:
            opt['device'] = 'cpu'
    else:
        opt['device'] = 'cpu'

    # ----------------- test ----------------- #
    # basic configs
    scale = opt['scale']
    device = torch.device(opt['device'])

    # create model
    net_G = define_generator(opt).to(device)

    from models.networks.tecogan_nets import FNet, SRNet

    fnet = FNet(in_nc=opt['model']['generator']['in_nc']).to(device)
    srnet = SRNet(in_nc=opt['model']['generator']['in_nc'], out_nc=3, nf=64, nb=10, upsample_func=None, scale=4).to(device)

    # get dummy input
    lr_size = tuple(map(int, args.lr_size.split('x')))
    dummy_input_dict = net_G.generate_dummy_input(lr_size)
    for key in dummy_input_dict.keys():
        dummy_input_dict[key] = dummy_input_dict[key].to(device)

    lr_curr = dummy_input_dict['lr_curr']
    lr_prev = dummy_input_dict['lr_prev']
    hr_prev = dummy_input_dict['hr_prev']
    hr_prev_warp = torch.rand(1, 3*16, 960, 540, dtype=torch.float32).to(device)

    # test running speed
    n_test = 30
    tot_time = 0

    fnet.eval()

    for i in range(n_test):
        print('run num:', i)
        start_time = time.time()
        with torch.no_grad():
            try:
                # rst = net_G(**dummy_input_dict)
                # rst = fnet(lr_curr, lr_prev)
                rst = srnet(lr_curr, hr_prev_warp)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

        end_time = time.time()
        tot_time += end_time - start_time


    print('Speed (FPS): {:.3f} (averaged for {} runs)'.format(n_test / tot_time, n_test))
    print('-' * 40)

    # torch to onnx
    # input_fnet = (lr_curr, lr_prev)
    # input_srnet = (lr_curr, hr_prev_warp)
    # torch.onnx.export(fnet, input_fnet, "fnet.onnx", verbose=True, opset_version=11)

