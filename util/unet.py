# Source code that builds UNet
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
  
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Encoder(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 1
        super (Encoder, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = nn.Dropout(p=0.2)(x1)
        x2 = self.layer2(x1)
        x2 = nn.Dropout(p=0.2)(x2)
        x3 = self.layer3(x2)
        x3 = nn.Dropout(p=0.2)(x3)
        x4 = self.layer4(x3)
        x4 = nn.Dropout(p=0.2)(x4)
        return x, x1, x2, x3, x4
    
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
    
class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        #print(np.shape(x))
        x = torch.cat([x, down_x], 1)
        #print(np.shape(x))
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, remove_skip=0):
        super(Decoder,self).__init__()
        up_blocks = []
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(UpBlock(in_channels=128 + 128, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlock(in_channels=64 + 64, out_channels=128,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        up_blocks.append(UpBlock(in_channels=1 + 64, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.remove_skip = remove_skip

    def forward(self, x0, x1, x2, x3, x4):
        x_list = [x0, x1, x2, x3, x4]
        
        if self.remove_skip != 0:
            for j in range(self.remove_skip):
                #print(j, "testing")
                x_list[j] = torch.zeros_like(x_list[j])
        
        x = x_list[-1]
        for i, block in enumerate(self.up_blocks):
            #print(i)
            #print(np.shape(x))
            x = block(x, x_list[-1 - i - 1])
            
            x = nn.Dropout(p=0.2)(x)
            
        x = self.out(x)
        x = nn.ReLU()(x)
        
        return x

class FBP_with_CNN(nn.Module):
    """ FBP+CNN
    """
    def __init__(self, img_size=320, patch_size=32, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, remove_skip=0):
        super().__init__()

        # --------------------------------------------------------------------------
        self.encoder = Encoder(Bottleneck, [3, 3, 3, 3])
        self.decoder = Decoder(remove_skip=remove_skip)

    def forward_loss(self, gd, pred):
        """
        imgs: [N, 1, H, W]
        pred: [N, 1, H, W]
        """

        loss = nn.MSELoss()(pred, gd)
        
        return loss

    def forward(self, imgs, mask_ratio=0.75):
     
        FBP = imgs[:, 0:1, :, :]
        gd = imgs[:, 1:2, :, :]
        
        x0, x1, x2, x3, x4 = self.encoder(FBP)
        pred = self.decoder(x0, x1, x2, x3, x4)
        
        loss = self.forward_loss(gd, pred)
        
        mask = 0
        
        return loss, pred, mask

def FBP_CNN0(**kwargs):
    model = FBP_with_CNN(img_size=128, in_chans=1, remove_skip=0, **kwargs)
    return model

def FBP_CNN1(**kwargs):
    model = FBP_with_CNN(img_size=128, in_chans=1, remove_skip=1, **kwargs)
    return model

def FBP_CNN2(**kwargs):
    model = FBP_with_CNN(img_size=128, in_chans=1, remove_skip=2, **kwargs)
    return model

def FBP_CNN3(**kwargs):
    model = FBP_with_CNN(img_size=128, in_chans=1, remove_skip=3, **kwargs)
    return model

def FBP_CNN4(**kwargs):
    model = FBP_with_CNN(img_size=128, in_chans=1, remove_skip=4, **kwargs)
    return model

# set recommended archs
FBP_CNN_model0 = FBP_CNN0
FBP_CNN_model1 = FBP_CNN1
FBP_CNN_model2 = FBP_CNN2
FBP_CNN_model3 = FBP_CNN3
FBP_CNN_model4 = FBP_CNN4
