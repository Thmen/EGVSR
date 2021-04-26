import torch
import torch.nn as nn
import torchvision


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_indexs=(8, 17, 26, 35)):
        super(VGGFeatureExtractor, self).__init__()

        # init feature layers
        self.features = torchvision.models.vgg19(pretrained=True).features
        for param in self.features.parameters():
            param.requires_grad = False

        # Notes:
        # 1. default feature layers are 8(conv2_2), 17(conv3_4), 26(conv4_4),
        #    35(conv5_4)
        # 2. features are extracted after ReLU activation
        self.feature_indexs = sorted(feature_indexs)

        # register normalization params
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # RGB
        std  = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        # assume input ranges in [0, 1]
        out = (x - self.mean) / self.std

        feature_list = []
        for i in range(len(self.features)):
            out = self.features[i](out)
            if i in self.feature_indexs:
                # clone to prevent overlapping by inplaced ReLU
                feature_list.append(out.clone())

        return feature_list
