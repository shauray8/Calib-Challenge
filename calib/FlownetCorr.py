import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from corr import CorrBlock

class FlowNet(nn.Module):
    def __init__(self, bn=True):
        super(FlowNet, self).__init__()
## --------------------- Illustration of the whole arch in the README --------------------- ##

        self.batchnorm = bn
        
        self.conv1 = self.ConvBlock(self.batchnorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = self.ConvBlock(self.batchnorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = self.ConvBlock(self.batchnorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = self.ConvBlock(self.batchnorm, 256, 32, kernel_size=1, stride=1)

        self.conv3_1 = self.ConvBlock(self.batchnorm, 473, 256)
        self.conv4 = self.ConvBlock(self.batchnorm, 256, 512, stride=2)
        self.conv4_1 = self.ConvBlock(self.batchnorm, 512, 512)
        self.conv5 = self.ConvBlock(self.batchnorm, 512, 512)
        self.conv5_1 = self.ConvBlock(self.batchnorm, 512, 512)
        self.conv6 = self.ConvBlock(self.batchnorm, 512, 1024, stride=2)
        self.conv6_1 = self.ConvBlock(self.batchnorm, 1024, 1024)

        self.ConvTrans5 = self.ConvTrans(1024,512)
        self.ConvTrans4 = self.ConvTrans(1026,256)
        self.ConvTrans3 = self.ConvTrans(770,128)
        self.ConvTrans2 = self.ConvTrans(386,64)

        self.flow6 = self.flow(1024)
        self.flow5 = self.flow(1026)
        self.flow4 = self.flow(770)
        self.flow3 = self.flow(386)
        self.flow2 = self.flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.yaw_block = [nn.Flatten(),
            nn.Linear(136*320, 1000),
            nn.Linear(1000, 640),
            nn.Linear(640, 320),
            nn.Linear(320, 160),
            nn.Linear(160, 70),
            ]

        self.pitch_block = [nn.Flatten(),
            nn.Linear(136*320, 1000),
            nn.Linear(1000, 640),
            nn.Linear(640, 320),
            nn.Linear(320, 160),
            nn.Linear(160, 70),
            ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def ConvBlock(self, batchnorm, in_channels, out_channels, kernel_size=3, stride=1):
        if batchnorm:
            return nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel-1)//2, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.1,inplace=True)
                    )
        else:
            return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                    nn.LeakyReLU(.1, inplace=True)
                    )

    def flow(self, in_channels):
        return nn.Conv2d(in_channels,2,kernel_size=3,stride=1,padding=1,bias=False)

    def correlate(self, fmap1, fmap2):
        return CorrBlock(fmap1, fmap2)

    def ConvTrans(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1,inplace=True)
        )

    def crop_like(self, input, target):
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :, :target.size(2), :target.size(3)]


    def forward(self, x):
## --------------------- seperating both the images --------------------- ##

        x1 = x[:,:3]
        x2 = x[:,3:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        out_conv_redir = self.conv_redir(out_conv3a)

## --------------------- I don't think the corr block will work --------------------- ##
## --------------------- If not i will change it a tiny bit--------------------- ##

        out_correlation = correlate(out_conv3a,out_conv3b)

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation], dim=1)

        out_conv3 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

## --------------------- FLOWS and CONCATENATIONS --------------------- ##

        flow6       = self.flow6(out_conv6)
        flow6_up    = self.crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_ConvTrans5 = self.crop_like(self.ConvTrans5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_ConvTrans5,flow6_up),1)
        flow5       = self.flow5(concat5)
        flow5_up    = self.crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_ConvTrans4 = self.crop_like(self.ConvTrans4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_ConvTrans4,flow5_up),1)
        flow4       = self.flow4(concat4)
        flow4_up    = self.crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_ConvTrans3 = self.crop_like(self.ConvTrans3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_ConvTrans3,flow4_up),1)
        flow3       = self.flow3(concat3)
        flow3_up    = self.crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        out_ConvTrans2 = self.crop_like(self.ConvTrans2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a,out_ConvTrans2,flow3_up),1)
        flow2 = self.flow2(concat2)

## --------------------- Returning the last block containing Linear Layers --------------------- ##
        return self.yaw_block(flow2), self.pitch_block(flow2)


def flownetc(data=None):
    """FlowNetS model architecture from the

    Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    """
    model = FlowNet(bn=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

def flownetc_bn(data=None):
    """FlowNetS model architecture from the

    Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    """
    model = FlowNet(bn=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
