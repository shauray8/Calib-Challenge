import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- U square Net architecture here the research paper https://arxiv.org/pdf/2005.09007v2.pdf --------------- #


def _upsample(src, tar):
    return F.interpolate(src,size=tar.shape[2:],mode='bilinear')
    
class Green_block(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, rate=1):
        super(Green_block, self).__init__()

        self.conv1_g = nn.Conv2d(in_channel, out_channel, 3, padding=rate, dilation=rate)
        self.bn_g = nn.BatchNorm2d(out_channel)
        self.relu_g = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.relu_g(self.bn_g(self.conv1_g(x)))


class En_De_1(nn.Module):    
    def __init__(self, in_channel=3, mid_channel=12, out_channel=3):
        super(En_De_1, self).__init__()

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
        self.pool_5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv_6 = Green_block(mid_channel, mid_channel, rate=1)

        self.conv_7 = Green_block(mid_channel, mid_channel, rate=2)

        self.conv_6d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_5d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_4d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_3d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_2d = Green_block(mid_channel*2, mid_channel, rate=1)
        self.conv_1d = Green_block(mid_channel*2, out_channel, rate=1)

    def forward(self, x):

        hx = x
        hx_in = self.conv_input(hx)

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
        hx_6d_up = _upsample(hx_6d, hx_5)

        hx_5d =self.conv_5d(torch.cat((hx_6d_up, hx_5),1))
        hx_5d_up = _upsample(hx_5d, hx_4)

        hx_4d = self.conv_4d(torch.cat((hx_5d_up, hx_4),1))
        hx_4d_up = _upsample(hx_4d, hx_3)

        hx_3d = self.conv_3d(torch.cat((hx_4d_up, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx_2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx_1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_in+hx_1d


class En_De_2(nn.Module):    
    def __init__(self, in_channel=3, mid_channel=12, out_channel=3):
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
        
        hx_6 = self.conv_6(hx_5)

        hx_5d =self.conv_5d(torch.cat((hx_6, hx_5),1))
        hx_5d_up = _upsample(hx_5d, hx_4)

        hx_4d = self.conv_4d(torch.cat((hx_5d_up, hx_4),1))
        hx_4d_up = _upsample(hx_4d, hx_3)

        hx_3d = self.conv_3d(torch.cat((hx_4d_up, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx_2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx_1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_1d + hx_in




class En_De_3(nn.Module):    
    def __init__(self, in_channel=3, mid_channel=12, out_channel=3):
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

        hx_5 = self.conv_5(hx_4)

        hx_4d = self.conv_4d(torch.cat((hx_5, hx_4),1))
        hx_4d_up = _upsample(hx_4d, hx_3)

        hx_3d = self.conv_3d(torch.cat((hx_4d_up, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx_2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx_1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_1d + hx_in




class En_De_4(nn.Module):    
    def __init__(self, in_channel=3, mid_channel=12, out_channel=3):
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
        
        hx_4 = self.conv_4(hx_3)


        hx_3d = self.conv_3d(torch.cat((hx_4, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx_2)

        hx_2d = self.conv_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx_1)

        hx_1d = self.conv_1d(torch.cat((hx_2d_up, hx_1),1))

        return hx_1d + hx_in


class En_De_4F(nn.Module):    
    def __init__(self, in_channel=3, mid_channel=12, out_channel=3):
        super(En_De_4F, self).__init__()

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

        hx_4 = self.conv_4(hx_3)

        hx_3d = self.conv_3d(torch.cat((hx_4, hx_3),1))
        hx_2d = self.conv_2d(torch.cat((hx_3d, hx_2),1))
        hx_1d = self.conv_1d(torch.cat((hx_2d, hx_1),1))

        return hx_1d + hx_in


class Unet_square(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(Unet_square, self).__init__()

        self.cube_1 = En_De_1(in_channel, 32, 64)
        self.pool_12 = nn.MaxPool2d(2, stride=2, ceil_mode = True)

        self.cube_2 = En_De_2(64, 32, 128)
        self.pool_23 = nn.MaxPool2d(2, stride=2, ceil_mode = True)

        self.cube_3 = En_De_3(128, 64, 256)
        self.pool_34 = nn.MaxPool2d(2, stride=2, ceil_mode = True)

        self.cube_4 = En_De_4(256, 128, 512)
        self.pool_45 = nn.MaxPool2d(2, stride=2, ceil_mode = True)

        self.cube_5 = En_De_4F(512, 256, 512)
        self.pool_56 = nn.MaxPool2d(2, stride=2, ceil_mode = True)

        self.cube_6 = En_De_4F(512, 256, 512)


        self.cube_5d = En_De_4F(1024,256,512)
        self.cube_4d = En_De_4(1024,128,256)
        self.cube_3d = En_De_3(512,64,128)
        self.cube_2d = En_De_2(256,32,64)
        self.cube_1d = En_De_1(128,16,64)

        self.side_1 = nn.Conv2d(64,out_channel,3,padding=1)
        self.side_2 = nn.Conv2d(64,out_channel,3,padding=1)
        self.side_3 = nn.Conv2d(128,out_channel,3,padding=1)
        self.side_4 = nn.Conv2d(256,out_channel,3,padding=1)
        self.side_5 = nn.Conv2d(512,out_channel,3,padding=1)
        self.side_6 = nn.Conv2d(512,out_channel,3,padding=1)

        self.outconv = nn.Conv2d(6*out_channel, out_channel,1)


    def forward(self, x):

        hx_1 = self.cube_1(x)
        hx = self.pool_12(hx_1)

        hx_2 = self.cube_2(hx)
        hx = self.pool_23(hx_2)

        hx_3 = self.cube_3(hx)
        hx = self.pool_34(hx_3)

        hx_4 = self.cube_4(hx)
        hx = self.pool_45(hx_4)

        hx_5 = self.cube_5(hx)
        hx = self.pool_56(hx_5)

        hx_6 = self.cube_6(hx)
        hx_6_up = _upsample(hx_6,hx_5)



        hx_5d = self.cube_5d(torch.cat((hx_6_up, hx_5),1))
        hx_5d_up = _upsample(hx_5d, hx_4)

        hx_4d = self.cube_4d(torch.cat((hx_5d_up, hx_4),1))
        hx_4d_up = _upsample(hx_4d, hx_3)

        hx_3d = self.cube_3d(torch.cat((hx_4d_up, hx_3),1))
        hx_3d_up = _upsample(hx_3d, hx_2)

        hx_2d = self.cube_2d(torch.cat((hx_3d_up, hx_2),1))
        hx_2d_up = _upsample(hx_2d, hx_1)

        hx_1d = self.cube_1d(torch.cat((hx_2d_up, hx_1),1))



        d1 = self.side_1(hx_1d)

        d2 = self.side_2(hx_2d)
        d2 = _upsample(d2, d1)

        d3 = self.side_3(hx_3d)
        d3 = _upsample(d3, d1)

        d4 = self.side_4(hx_4d)
        d4 = _upsample(d4, d1)

        d5 = self.side_5(hx_5d)
        d5 = _upsample(d5, d1)

        d6 = self.side_6(hx_6)
        d6 = _upsample(d6, d1)

        d_not = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6),1))

        return F.sigmoid(d_not), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)


if __name__ == "__main__":
    ## Using U square net for segmentations 
    ## getting the data from comma's some repo
    model = Unet_square(3,5)
    print(model)
