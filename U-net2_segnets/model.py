import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------- U square Net architecture here the research paper https://arxiv.org/pdf/2005.09007v2.pdf --------------- #

class Unet_square(nn.Module):
    def __init__(self):
        super(Unet_square, self).__init__()
        pass

    def _upsample(self):
        pass

    
class Green_block(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, dirate=1):
        super(Green_block, self).__init__()

        self.conv1_g = nn.Conv2d(in_channel, out_channel, 3, padding=dirate, dilation=dirate)
        self.bn_g = nn.BatchNorm2d(out_channel)
        self.relu_g = nn.Relu(inplace=True)

    def forward(self, x):

        return self.relu_g(self.bn_g(self.conv_g(x)))


class En_De_1(nn.Module):    
    def __init__(self):
        super(En_De_1, self).__init__()
        pass

    def forward(self, x):
        pass

class En_De_2(nn.module):
    def __init__(self):
        super(En_De_2, self).__init__()
        pass

    def forward(self, x):
        pass

class En_De_3(nn.module):
    def __init__(self):
        super(En_De_3, self).__init__()
        pass

    def forward(self, x):
        pass

class En_De_4(nn.module):
    def __init__(self):
        super(En_De_4, self).__init__()
        pass

    def forward(self, x):
        pass



if __name__ == "__main__":
    ## Using U square net for segmentations 
    ## getting the data from comma's some repo
    model = Unet_square()
