from .effnet import EfficientNet

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpSample(nn.Module):

    def __init__(self, n_channels, factor=2):
        super(UpSample, self).__init__()
        out_channels = n_channels * factor * factor
        self.proj = nn.Conv2d(n_channels, out_channels, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        x = self.proj(x)
        x = self.up(x)
        return x

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)

class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        return x

class SegmentBranch(nn.Module):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.backbone = EfficientNet.from_pretrained(
            model_name='efficientnet-b0',
            weights_path='weights/efficientnet-b0-355c32eb.pth'
        )

    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        features = [endpoints[f"reduction_{i}"] for i in range(1, 5+1)]
        return features

class BGALayer(nn.Module):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        )
        self.right1 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, groups=128, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, bias=False),
        )
        self.up1 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=4)
        ##TODO: does this really has no relu?
        self.conv = nn.Sequential(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), # not shown in paper
        )

    def forward(self, x_d, x_s):
        dsize = x_d.size()[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * torch.sigmoid(right1)
        right = left2 * torch.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out

class SegmentHead(nn.Module):

    def __init__(self, in_channels, mid_channels, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_channels, mid_channels, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_channels = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_channels
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_channels, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_channels, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.conv_out(x)
        return x

class BiSeNetV2(nn.Module):

    def __init__(self, n_classes, mode='train'):
        super(BiSeNetV2, self).__init__()
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.conv = ConvBNReLU(320, 128)
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)

    def forward(self, x):
        f_detail = self.detail(x)
        f1, f2, f3, f4, f_segment = self.segment(x)
        f_segment = self.conv(f_segment)
        f_head = self.bga(f_detail, f_segment)
        logits = self.head(f_head)
        return logits

if __name__ == '__main__':
    net = BiSeNetV2(n_classes=21)
    inputs = torch.randn((16, 3, 256, 256))
    outputs = net(inputs)
    #print(net)
    print(f"Inputs: {inputs.shape}")
    print(f"Inputs: {inputs.shape}")
    print(f"Outputs: {outputs.shape}")