import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- U square Net architecture here the research paper https://arxiv.org/pdf/2005.09007v2.pdf --------------- #


def _upsample(self):
        pass

    
class Green_block(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, rate=1):
        super(Green_block, self).__init__()

        self.conv1_g = nn.Conv2d(in_channel, out_channel, 3, padding=dirate, dilation=dirate)
        self.bn_g = nn.BatchNorm2d(out_channel)
        self.relu_g = nn.Relu(inplace=True)

    def forward(self, x):

        return self.relu_g(self.bn_g(self.conv_g(x)))


class En_De_1(nn.Module):    
    def __init__(self, in_channel=3, out_channel=3, mid_channel=12):
        super(En_De_1, self).__init__()

        self.conv_input = Green_block(in_channel, out_channel, rate=1)

        self.conv_1 = Green_block(in_channel, mid_channel, rate=1)
        self.pool_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_2 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_3 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_4 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_5 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_6 = Green_block(mid_channel, mid_channel, rate=1)

        self.conv_7 = Green_block(mid_channel, mid_channel, rate=2)

        self.conv_6d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_5d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_4d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_3d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_2d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_1d = Green_block(mid_channel*2, mid_channel, rate=1)

    def forward(self, x):
        hx_in = self.conv_input(x)

        hx_1 = self.conv_1(hx_in)
        hx = self.pool_1(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool_2(hx_2)
        
        hx_3 = self.conv_3(hx)
        hx = self.pool_3(hx_3)
        
        hx_4 = self.conv_4(hx)
        hx = self.pool_4(hx_4)

        hx_5 = self.conv_5(hx)
        hx = self.pool_5(hx_5)
        
        hx_6 = self.conv_6(hx)

        hx_7 = self.conv_7(hx_6)

        hx_6d = self.conv_6d(torch.cat((hx_7, hx_6),1))
        hx_6d_up = _upsample(hx_6d, hx5)

        hx_5d =self.conv_5d(torch.cat((hx_6d_up, hx_5),1))
        hx_5d_up = _upsample(hx_5d, hx4)

        hx_4d = self.conv_4d(torch.cat((hx_5d_up, hx_4),1))
        hx_4d_up = _upsample(hx_4d, hx3)

        hx_3d = self.conv_3d(torch.cat((hx_4d_up, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_1d + hx_in


class En_De_2(nn.Module):    
    def __init__(self, in_channel=3, out_channel=3, mid_channel=12):
        super(En_De_2, self).__init__()

        self.conv_input = Green_block(in_channel, out_channel, rate=1)

        self.conv_1 = Green_block(out_channel, mid_channel, rate=1)
        self.pool_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_2 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_3 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_4 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_5 = Green_block(mid_channel, mid_channel, rate=1)

        self.conv_6 = Green_block(mid_channel, mid_channel, rate=2)

        self.conv_5d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_4d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_3d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_2d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_1d = Green_block(mid_channel*2, out_channel, rate=1)

    def forward(self, x):
        hx_in = self.conv_input(x)

        hx_1 = self.conv_1(hx_in)
        hx = self.pool_1(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool_2(hx_2)
        
        hx_3 = self.conv_3(hx)
        hx = self.pool_3(hx_3)
        
        hx_4 = self.conv_4(hx)
        hx = self.pool_4(hx_4)

        hx_5 = self.conv_5(hx)
        
        hx_7 = self.conv_7(hx_5)

        hx_5d =self.conv_5d(torch.cat((hx_6, hx_5),1))
        hx_5d_up = _upsample(hx_5d, hx4)

        hx_4d = self.conv_4d(torch.cat((hx_5d_up, hx_4),1))
        hx_4d_up = _upsample(hx_4d, hx3)

        hx_3d = self.conv_3d(torch.cat((hx_4d_up, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_1d + hx_in




class En_De_3(nn.Module):    
    def __init__(self, in_channel=3, out_channel=3, mid_channel=12):
        super(En_De_3, self).__init__()

        self.conv_input = Green_block(in_channel, out_channel, rate=1)

        self.conv_1 = Green_block(out_channel, mid_channel, rate=1)
        self.pool_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_2 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_3 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_4 = Green_block(mid_channel, mid_channel, rate=1)

        self.conv_5 = Green_block(mid_channel, mid_channel, rate=2)

        self.conv_4d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_3d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_2d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_1d = Green_block(mid_channel*2, out_channel, rate=1)

    def forward(self, x):
        hx_in = self.conv_input(x)

        hx_1 = self.conv_1(hx_in)
        hx = self.pool_1(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool_2(hx_2)
        
        hx_3 = self.conv_3(hx)
        hx = self.pool_3(hx_3)
        
        hx_4 = self.conv_4(hx)

        hx_5 = self.conv_7(hx_4)

        hx_4d = self.conv_4d(torch.cat((hx_5, hx_4),1))
        hx_4d_up = _upsample(hx_4d, hx3)

        hx_3d = self.conv_3d(torch.cat((hx_4d_up, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_1d + hx_in




class En_De_4(nn.Module):    
    def __init__(self, in_channel=3, out_channel=3, mid_channel=12):
        super(En_De_4, self).__init__()

        self.conv_input = Green_block(in_channel, out_channel, rate=1)

        self.conv_1 = Green_block(out_channel, mid_channel, rate=1)
        self.pool_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_2 = Green_block(mid_channel, mid_channel, rate=1)
        self.pool_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_3 = Green_block(mid_channel, mid_channel, rate=1)

        self.conv_4 = Green_block(mid_channel, mid_channel, rate=2)

        self.conv_3d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_2d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_1d = Green_block(mid_channel*2, out_channel, rate=1)

    def forward(self, x):
        hx_in = self.conv_input(x)

        hx_1 = self.conv_1(hx_in)
        hx = self.pool_1(hx_1)

        hx_2 = self.conv_2(hx)
        hx = self.pool_2(hx_2)
        
        hx_3 = self.conv_3(hx)
        
        hx_4 = self.conv_7(hx_3)


        hx_3d = self.conv_3d(torch.cat((hx_4, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_1d + hx_in


class En_De_4F(nn.Module):    
    def __init__(self, in_channel=3, out_channel=3, mid_channel=12):
        super(En_De_1, self).__init__()

        self.conv_input = Green_block(in_channel, out_channel, rate=1)

        self.conv_1 = Green_block(out_channel, mid_channel, rate=1)

        self.conv_2 = Green_block(mid_channel, mid_channel, rate=2)

        self.conv_3 = Green_block(mid_channel, mid_channel, rate=4)

        self.conv_4 = Green_block(mid_channel, mid_channel, rate=8)

        self.conv_3d = Green_block(mid_channel*2, mid_channel, rate=4)
        self.conv_2d = Green_block(mid_channel*2, mid_channel, rate=2)
        self.conv_1d = Green_block(mid_channel*2, out_channel, rate=1)

    def forward(self, x):
        hx_in = self.conv_input(x)

        hx_1 = self.conv_1(hx_in)
        hx_2 = self.conv_2(hx_1)
        hx_3 = self.conv_3(hx_2)

        hx_4 = self.conv_7(hx_3)

        hx_3d = self.conv_3d(torch.cat((hx_4, hx_3),1))
        hx_2d = self.conv_2d(torch.cat((hx_3d, hx_2),1))
        hx_1d = self.conv_1d(torch.cat((hx_2d, hx_1),1))

        return hx_1d + hx_in


class Unet_square(nn.Module):
    def __init__(self):
        super(Unet_square, self).__init__()
        pass


if __name__ == "__main__":
    ## Using U square net for segmentations 
    ## getting the data from comma's some repo
    model = Unet_square()
