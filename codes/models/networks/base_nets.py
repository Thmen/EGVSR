import torch.nn as nn


class BaseSequenceGenerator(nn.Module):
    def __init__(self):
        super(BaseSequenceGenerator, self).__init__()

    def generate_dummy_input(self, lr_size):
        """ use for compute per-step FLOPs and speed
            return random tensors that can be taken as input of <forward>
        """
        return None

    def forward(self, *args, **kwargs):
        """ forward pass for a singe frame
        """
        pass

    def forward_sequence(self, lr_data):
        """ forward pass for a whole sequence (for training)
        """
        pass

    def infer_sequence(self, lr_data, device):
        """ infer for a whole sequence (for inference)
        """
        pass


class BaseSequenceDiscriminator(nn.Module):
    def __init__(self):
        super(BaseSequenceDiscriminator, self).__init__()

    def forward(self, *args, **kwargs):
        """ forward pass for a singe frame
        """
        pass

    def forward_sequence(self, data, args_dict):
        """ forward pass for a whole sequence (for training)
        """
        pass
