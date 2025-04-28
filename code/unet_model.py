import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def build_unet_functional(input_channels=1, output_channels=1):
    model = nn.ModuleDict()

    model['enc1'] = conv_block(input_channels, 64)
    model['enc2'] = conv_block(64, 128)
    model['enc3'] = conv_block(128, 256)
    model['enc4'] = conv_block(256, 512)
    model['pool'] = nn.MaxPool2d(2)

    model['bottleneck'] = conv_block(512, 1024)

    model['up4'] = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    model['dec4'] = conv_block(1024, 512)
    model['up3'] = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    model['dec3'] = conv_block(512, 256)
    model['up2'] = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    model['dec2'] = conv_block(256, 128)
    model['up1'] = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    model['dec1'] = conv_block(128, 64)

    model['final'] = nn.Conv2d(64, output_channels, kernel_size=1)

    return model

def unet_forward(model, x):
    s1 = model['enc1'](x)
    p1 = model['pool'](s1)

    s2 = model['enc2'](p1)
    p2 = model['pool'](s2)

    s3 = model['enc3'](p2)
    p3 = model['pool'](s3)

    s4 = model['enc4'](p3)
    p4 = model['pool'](s4)

    b = model['bottleneck'](p4)

    u4 = model['up4'](b)
    u4 = torch.cat([u4, s4], dim=1)
    d4 = model['dec4'](u4)

    u3 = model['up3'](d4)
    u3 = torch.cat([u3, s3], dim=1)
    d3 = model['dec3'](u3)

    u2 = model['up2'](d3)
    u2 = torch.cat([u2, s2], dim=1)
    d2 = model['dec2'](u2)

    u1 = model['up1'](d2)
    u1 = torch.cat([u1, s1], dim=1)
    d1 = model['dec1'](u1)

    outputs = model['final'](d1)
    outputs = torch.sigmoid(outputs)

    return outputs
