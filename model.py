import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.down5 = self.down5_layer(3, 32)
        self.down4 = self.down4_layer(32, 64)
        self.down3 = self.down3_layer(64, 128)
        self.down2 = self.down2_layer(128, 256)
        self.down1 = self.down1_layer(256, 512)
        self.fc = self.fully_connected_layer()
        self.up1 = self.bn_layers(in_channels=512, out_channels=256)
        self.up2 = self.bn_layers(in_channels=256, out_channels=128)
        self.up3 = self.bn_layers(in_channels=128, out_channels=64)
        self.up4 = self.bn_layers(in_channels=64, out_channels=32)
        self.up5 = self.bn_layers(in_channels=32, out_channels=3)
    
    def forward(self, x_origin):
        x_down5 = self.down5(x_origin)
        x_down4 = self.down4(x_down5)
        x_down3 = self.down3(x_down4)
        x_down2 = self.down2(x_down3)
        x_down1 = self.down1(x_down2)
        x_fc = self.fc(x_down1)
        x_fc = x_fc.reshape(x_fc.size()[0], 512, 24, 32)
        x = self.up1(x_fc)
        x = F.interpolate(x, size=(48, 64), mode='bilinear', align_corners=True)
        x = self.up2(x) + x_down3
        x = F.interpolate(x, size=(96, 128), mode='bilinear', align_corners=True)
        x = self.up3(x) + x_down4
        x = F.interpolate(x, size=(192, 256), mode='bilinear', align_corners=True)
        x = self.up4(x) + x_down5
        x = F.interpolate(x, size=(384, 512), mode='bilinear', align_corners=True)
        x = self.up5(x) + x_origin
        return x

    def down5_layer(self, in_channels, out_channels):
        return nn.Sequential(
            self.batch_layer(in_channels=in_channels, out_channels=out_channels),
            nn.AdaptiveMaxPool2d(output_size=(192, 256)),
        )

    def down4_layer(self, in_channels, out_channels):
        return nn.Sequential(
            self.bn_layers(in_channels, out_channels),
            nn.AdaptiveMaxPool2d(output_size=(96, 128)),
        )

    def down3_layer(self, in_channels, out_channels):
        return nn.Sequential(
            self.bn_layers(in_channels, out_channels),
            nn.AdaptiveMaxPool2d(output_size=(48, 64)),
        )

    def down2_layer(self, in_channels, out_channels):
        return nn.Sequential(
            self.bn_layers(in_channels, out_channels),
            nn.AdaptiveMaxPool2d(output_size=(24, 32)),
        )

    def down1_layer(self, in_channels, out_channels):
        return self.bn_layers(in_channels, out_channels)

    def bn_layers(self, in_channels, out_channels):
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

    def fully_connected_layer(self, in_features=393216, out_features=32):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, out_features, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(out_features, in_features, bias=False),
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
            