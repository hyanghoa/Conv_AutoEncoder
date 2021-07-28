import torch
import torch.nn as nn
import math

import timm
from .unetplusplus import UnetPlusPlusDecoder

EFFICIENTNET_V2_XL = [640, 256, 96, 64, 32, 3]
EFFICIENTNET_V2_L = [640, 224, 96, 64, 32, 3]
EFFICIENTNET_V2_M = [512, 176, 80, 48, 24, 3]
EFFICIENTNET_B4 = [448, 160, 56, 32, 24, 3]
EFFICIENTNET_L2 = [1376, 480, 176, 104, 72, 3]
HRNET_W64 = [1024, 512, 256, 128, 64, 3]
EFFICIENTNET_V2_XL_HRNET_W64 = [1664, 768, 352, 192, 96, 3]
CHANNELS = EFFICIENTNET_V2_XL


def conv3x3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class Upsampling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Upsampling, self).__init__()
        self.up1 = self.upsampling(in_channels, out_channels)
        self.conv = conv3x3(out_channels*2, out_channels)

    def forward(self, x):
        x, fea = x
        x = self.up1(x)
        # x = nn.AdaptiveAvgPool2d((fea.size()[2], fea.size()[3]))(x)
        x = torch.cat([x, fea], 1)
        x = self.conv(x)
        return x

    def upsampling(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                bias=False,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class Unet(nn.Module):

    def __init__(self):
        super(Unet, self).__init__()
        self.up1 = Upsampling(CHANNELS[0], CHANNELS[1])
        self.up2 = Upsampling(CHANNELS[1], CHANNELS[2])
        self.up3 = Upsampling(CHANNELS[2], CHANNELS[3])
        self.up4 = Upsampling(CHANNELS[3], CHANNELS[4])
        self.up5 = Upsampling(CHANNELS[4], CHANNELS[5])
        self.conv = conv3x3(3, 3)
        self.output_head = self.out_head(3, 3)
        self._initialize_weights()

    def forward(self, x, x_origin):
        fea1, fea2, fea3, fea4, fea5 = x
        x1 = self.up1([fea5, fea4])
        x2 = self.up2([x1, fea3])
        x3 = self.up3([x2, fea2])
        x4 = self.up4([x3, fea1])
        x5 = self.up5([x4, x_origin])
        x = self.conv(x5)
        x = self.output_head(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def out_head(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
        )


class EfficientNetV2Unet(nn.Module):

    def __init__(self, efficientnet):
        super(EfficientNetV2Unet, self).__init__()
        self.backbone = timm.create_model(efficientnet, features_only=True, pretrained=True)
        self.unet = UnetPlusPlusDecoder(
            EFFICIENTNET_V2_XL[:-1],
            EFFICIENTNET_V2_XL[:-1],
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
        )

    def forward(self, x):
        x = self.backbone(x)
        # for b in x:
        #     print(b.size())
        x = self.unet(x)
        return x
