from .base_model import BaseModel
from .networks.espcn_nets import ESPNet
from utils import data_utils


class ESPCNModel(BaseModel):
    def __init__(self, opt):
        super(ESPCNModel, self).__init__(opt)

        if self.verbose:
            self.logger.info('{} Model Info {}'.format('=' * 20, '=' * 20))
            self.logger.info('Model: {}'.format(opt['model']['name']))

        # set network
        self.set_network()

    def set_network(self):
        # define net G
        self.net_G = ESPNet(scale=4).to(self.device)
        if self.verbose:
            self.logger.info('Generator: {}\n'.format(
                self.opt['model']['generator']['name']) + self.net_G.__str__())

        # load network
        load_path_G = self.opt['model']['generator'].get('load_path')
        if load_path_G is not None:
            self.load_network(self.net_G, load_path_G)
            if self.verbose:
                self.logger.info('Load generator from: {}'.format(load_path_G))

    def infer(self, lr_data):
        """ Function of inference
            Parameters:
                :param lr_data: a rgb video sequence with shape thwc
                :return: a rgb video sequence with type np.uint8 and shape thwc
        """

        # canonicalize
        lr_data = data_utils.canonicalize(lr_data)  # to torch.FloatTensor
        lr_data = lr_data.permute(0, 3, 1, 2)  # tchw

        lr_data = lr_data[:,0:1,:,:]

        # temporal padding
        lr_data, n_pad_front = self.pad_sequence(lr_data)

        # infer
        hr_seq = self.net_G.infer_sequence(lr_data, self.device)
        hr_seq = hr_seq[n_pad_front:, ...]

        return hr_seq

