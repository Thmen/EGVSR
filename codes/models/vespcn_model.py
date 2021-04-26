from .base_model import BaseModel
from .networks import define_generator
from utils import data_utils


class VESPCNModel(BaseModel):
    def __init__(self, opt):
        super(VESPCNModel, self).__init__(opt)

        if self.verbose:
            self.logger.info('{} Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        # set network
        self.set_network()

    def set_network(self):
        # define net G
        self.net_G = define_generator(self.opt).to(self.device)
        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['generator']['name']) + self.net_G.__str__())

        # load network
        load_path_G = self.opt['model']['generator'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            if self.verbose:
                self.logger.info('Load generator from: {}'.format(load_path_G))


    # def eval(self, inputs, labels=None, **kwargs):
    #     metrics = {}
    #     frames = [x.squeeze(1) for x in inputs[0].split(1, dim=1)]
    #     _frames = [pad_if_divide(x, 4, 'reflect') for x in frames]
    #     a = (_frames[0].size(2) - frames[0].size(2)) * self.scale
    #     b = (_frames[0].size(3) - frames[0].size(3)) * self.scale
    #     slice_h = slice(None) if a == 0 else slice(a // 2, -a // 2)
    #     slice_w = slice(None) if b == 0 else slice(b // 2, -b // 2)
    #     sr, warps, flows = self.vespcn(*_frames)
    #     sr = sr[..., slice_h, slice_w].cpu().detach()
    #     if labels is not None:
    #         targets = torch.split(labels[0], 1, dim=1)
    #         targets = [t.squeeze(1) for t in targets]
    #         hr = targets[self.depth // 2]
    #         metrics['psnr'] = psnr(sr, hr)
    #         writer = get_writer(self.name)
    #         if writer is not None:
    #             step = kwargs['epoch']
    #             writer.image('clean', sr.clamp(0, 1), step=step)
    #             writer.image('warp/0', warps[0].clamp(0, 1), step=step)
    #             writer.image('warp/1', warps[-1].clamp(0, 1), step=step)
    #     return [sr.numpy()], metrics

    def infer(self, lr_data):

        lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor
        lr_data = lr_data.permute(0, 3, 1, 2)  # tchw

        # dual direct temporal padding
        lr_data, n_pad_front = self.pad_sequence(lr_data)

        # infer
        hr_seq = self.net_G.infer_sequence(lr_data, self.device)

        return hr_seq
