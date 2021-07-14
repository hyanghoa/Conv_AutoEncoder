import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = self.encoder_layer()
        # self.flatten = nn.Flatten()
        # self.fc = self.fully_connected_layer()
        self.decoder_1 = self.decoder_layer(in_channels=512, out_channels=256)
        self.decoder_2 = self.decoder_layer(in_channels=256, out_channels=128)
        self.decoder_3 = self.decoder_layer(in_channels=128, out_channels=3)
    
    def forward(self, x):
        x = self.encoder(x)
        # x = self.flatten(x)
        # x = self.fc(x)
        # x = x.reshape(1, 512, 64, 64)
        x = self.decoder_1(x)
        x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        x = self.decoder_2(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
        x = self.decoder_3(x)
        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=True)
        return x

    def encoder_layer(self):
        return nn.Sequential(
            self.batch_layer(in_channels=3, out_channels=128),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            self.batch_layer(in_channels=128, out_channels=128),
            self.batch_layer(in_channels=128, out_channels=128),
            self.batch_layer(in_channels=128, out_channels=256),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            self.batch_layer(in_channels=256, out_channels=256),
            self.batch_layer(in_channels=256, out_channels=256),
            self.batch_layer(in_channels=256, out_channels=512),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            self.batch_layer(in_channels=512, out_channels=512),
            self.batch_layer(in_channels=512, out_channels=512),
        )

    def decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            self.batch_layer(in_channels=in_channels, out_channels=in_channels),
            self.batch_layer(in_channels=in_channels, out_channels=in_channels),
            self.batch_layer(in_channels=in_channels, out_channels=out_channels),
        )

    def batch_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True)
        )

    def fully_connected_layer(self, in_features=2097152, out_features=32):
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(out_features, in_features, bias=False),
        )
        