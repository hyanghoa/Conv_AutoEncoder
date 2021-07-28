import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.down5 = self.down5_layer(3, 32, (192, 256))
        self.down4 = self.down4_layer(32, 64, (96, 128))
        self.down3 = self.down3_layer(64, 128, (48, 64))
        self.down2 = self.down2_layer(128, 256, (24, 32))
        self.down1 = self.down1_layer(256, 512, (12, 16))
        self.fc = self.fully_connected_layer()
        self.up1 = self.upsampling(512, 256)
        self.up2 = self.upsampling(256, 128)
        self.up3 = self.upsampling(128, 64)
        self.up4 = self.upsampling(64, 32)
        self.up5 = self.upsampling(32, 3)

    def forward(self, x_origin):
        x_down5 = self.down5(x_origin)
        x_down4 = self.down4(x_down5)
        x_down3 = self.down3(x_down4)
        x_down2 = self.down2(x_down3)
        x_down1 = self.down1(x_down2)
        x = self.fc(x_down1)
        x = self.up1(x) + x_down2
        x = self.up2(x) + x_down3
        x = self.up3(x) + x_down4
        x = self.up4(x) + x_down5
        x = self.up5(x) + x_origin
        return x

    def down5_layer(self, in_channels, out_channels, output_size):
        return nn.Sequential(
            self.batch_layer(in_channels, out_channels),
            nn.AdaptiveAvgPool2d(output_size),
        )

    def down4_layer(self, in_channels, out_channels, output_size):
        return nn.Sequential(
            self.batch_layers(in_channels, out_channels),
            nn.AdaptiveAvgPool2d(output_size),
        )

    def down3_layer(self, in_channels, out_channels, output_size):
        return nn.Sequential(
            self.batch_layers(in_channels, out_channels),
            nn.AdaptiveAvgPool2d(output_size),
        )

    def down2_layer(self, in_channels, out_channels, output_size):
        return nn.Sequential(
            self.batch_layers(in_channels, out_channels),
            nn.AdaptiveAvgPool2d(output_size),
        )

    def down1_layer(self, in_channels, out_channels, output_size):
        return nn.Sequential(
            self.batch_layers(in_channels, out_channels),
            nn.AdaptiveAvgPool2d(output_size),
        )

    def upsampling(self, in_channels, out_channels, kernel_size=3,
                   stride=2, padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding,
                               bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True)
        )

    def batch_layers(self, in_channels, out_channels):
        return nn.Sequential(
            self.batch_layer(in_channels=in_channels,
                             out_channels=in_channels),
            self.batch_layer(in_channels=in_channels,
                             out_channels=in_channels),
            self.batch_layer(in_channels=in_channels,
                             out_channels=out_channels),
        )

    def batch_layer(self, in_channels, out_channels, kernel_size=3, stride=1,
                    padding=1, bias=False):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True)
        )

    def fully_connected_layer(self, in_features=98304, out_features=256):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, out_features, bias=True),
            nn.Linear(256, 128, bias=True),
            nn.Linear(128, 256, bias=True),
            nn.Linear(256, in_features, bias=True),
            nn.Unflatten(1, torch.Size([512, 12, 16])),
        )

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                m.apply(self._sequential_init_weights)

    def _sequential_init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
