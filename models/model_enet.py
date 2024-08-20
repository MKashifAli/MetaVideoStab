import torch.nn as nn
import torch
from torchvision.models import vgg19
import torchvision 

'''
EnhanceNet Implementation in PyTorch by Erik Quintanilla 
Single Image Super Resolution 
https://arxiv.org/abs/1612.07919/
This program assumes GPU.
'''

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        ip_ = x
        return torch.add(self.conv_block(x), ip_)
        
        
class Generator(nn.Module):
    def __init__(self, in_channels=8, out_channels=2, residual_blocks=64):
        super(Generator, self).__init__()
        self.merge = torch.cat
        self.add = torch.add
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), 
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
                nn.ReLU())
        #Residual blocks
        residuals = []
        for _ in range(residual_blocks):
            residuals.append(ResidualBlock(64))
        self.residuals = nn.Sequential(*residuals)
        
        #nearest neighbor upsample 
        self.seq = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residuals(out)
        out = self.conv3(out)
        #out = self.add(out, ip_)
        out = self.conv4(out) 

        return out
        
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        layers = []
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        self.output_shape = (1, 8, 8)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        in_filters = in_channels
        for i, out_filters in enumerate([16, 32,64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
        
class Vgg_Features(nn.Module):
    def __init__(self, pool_layer_num = 9):
        '''
        To capture bothlow-level and high-level features, 
        we use a combination ofthe second and fifth pooling 
        layers and compute the MSEon their feature activations. 
        
        - Sajjadi et al.
        '''
        
        #we have maxpooling layers at [4,9,18,27,36]
        super(Vgg_Features, self).__init__()
        model = vgg19(pretrained=True)
        self.features = nn.Sequential(*list(model.features.children())[:pool_layer_num])

    def forward(self, img):
        return self.features(img)