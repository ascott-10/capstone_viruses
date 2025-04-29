import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def center_crop(enc_feature, target_feature):
    _, _, H, W = target_feature.shape
    enc_H, enc_W = enc_feature.shape[2], enc_feature.shape[3]
    delta_H = (enc_H - H) // 2
    delta_W = (enc_W - W) // 2
    return enc_feature[:, :, delta_H:delta_H + H, delta_W:delta_W + W]

class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = conv_block(input_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final output
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        p1 = self.pool(s1)

        s2 = self.enc2(p1)
        p2 = self.pool(s2)

        s3 = self.enc3(p2)
        p3 = self.pool(s3)

        s4 = self.enc4(p3)
        p4 = self.pool(s4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        u4 = self.up4(b)
        s4 = center_crop(s4, u4)
        u4 = torch.cat([u4, s4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        s3 = center_crop(s3, u3)
        u3 = torch.cat([u3, s3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        s2 = center_crop(s2, u2)
        u2 = torch.cat([u2, s2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        s1 = center_crop(s1, u1)
        u1 = torch.cat([u1, s1], dim=1)
        d1 = self.dec1(u1)

        outputs = self.final(d1)
        return outputs

