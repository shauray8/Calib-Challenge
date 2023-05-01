import torch
import torch.nn as nn
import torch.nn.functional as F


class U_net(nn.Module):
    def __init__(self, n_channel, n_classes, bilinear=True):
        super(U_net, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.bilinear = bilinear
        
        self.ini = Conv_Layer(n_channel, 64)
        self.downSam1 = Down_Sampling(64, 128)
        self.downSam2 = Down_Sampling(128, 256)
        self.downSam3 = Down_Sampling(256, 512)

        factor = 2 if bilinear else 1

        self.downSam4 = Down_Sampling(512, 1024 // factor)
        self.upSam1 = Up_Sampling(1024, 512 // factor, bilinear)
        self.upSam2 = Up_Sampling(512, 256 // factor, bilinear)
        self.upSam3 = Up_Sampling(256, 128 // factor, bilinear)
        self.upSam4 = Up_Sampling(128, 64, bilinear)
        
        self.output = Out_Conv(64, n_classes)

    def forward(self, x):
        x1 = self.ini(x)
        x2 = self.downSam1(x1)
        x3 = self.downSam2(x2)
        x4 = self.downSam3(x3)
        x5 = self.downSam4(x4)
        x = self.upSam1(x5, x4)
        x = self.upSam2(x, x3)
        x = self.upSam3(x, x2)
        x = self.upSam4(x, x1)
        gamma = self.output(x)
        return gamma


class Conv_Layer(nn.Module):
    def __init__(self, in_channel, out_channel, mid = None):
        super(Conv_Layer, self).__init__()
        if not mid:
            mid = out_channel

        self.convolution = nn.Sequential(
                nn.Conv2d(in_channel, mid, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))

    def forward(self, tou):
        return self.convolution(tou)

class Down_Sampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down_Sampling, self).__init__()

        self.Down = nn.Sequential(
                nn.MaxPool2d(2),
                Conv_Layer(in_channel, out_channel))

    def forward(self, tou):
        return self.Down(tou)

class Up_Sampling(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear = True):
        super(Up_Sampling, self).__init__()

        if bilinear:
           self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
           self.conv = Conv_Layer(in_channel, out_channel, in_channel // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = Double_conv(in_channel, out_channel)
        
    def forward(self,alpha, beta):
        alpha = self.up(alpha)

        diffY = beta.size()[2] - alpha.size()[2]
        diffX = beta.size()[3] - alpha.size()[3]

        alpha = F.pad(alpha, [diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2])

        x = torch.cat([alpha, beta],  dim=1)

        return self.conv(x)

class Out_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Out_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, tou):
        return self.conv(tou)



if __name__ == "__main__":
    Unet = U_net(4,4)
    print(Unet)
