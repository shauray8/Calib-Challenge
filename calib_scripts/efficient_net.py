"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
"""

import torch
import torch.nn as nn

class Efficient_net_v4(nn.Module):
    def __init__(self):
        super(Efficient_net_v4, self).__init__()
        pass

    def forward(self, x):
        pass



if __name__ == "__main__":
    net = Efficient_net_v4()
    print(net)
