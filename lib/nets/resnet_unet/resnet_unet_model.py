import torch
import torch.nn as nn
from .resnet import resnet18
import torch.nn.functional as F
import numpy as np


import copy

def convrelu(in_channels, out_channels, kernel, padding, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, stride=stride),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class, lastop='tanh', scale=1):
        super(ResNetUNet,self).__init__()

        self.lastop = lastop
        self.scale = scale

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, flag=0):
    
        if (flag == 0): #'im-input'
        
          # Image Encoder
          x_original = self.conv_original_size0(input)
          x_original = self.conv_original_size1(x_original)

          layer0 = self.layer0(input)
          layer1 = self.layer1(layer0)
          layer2 = self.layer2(layer1)
          layer3 = self.layer3(layer2)
          layer4 = self.layer4(layer3)
          
          # Normal decoder
          layer4_1 = self.layer4_1x1(layer4)                 
          x_1 = self.upsample(layer4_1)
          layer3_1 = self.layer3_1x1(layer3)
          x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer3_1)], dim=1)           
          x_1 = self.conv_up3(x_1)

          x_1 = self.upsample(x_1)
          layer2_1 = self.layer2_1x1(layer2)
          x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer2_1)], dim=1)
          x_1 = self.conv_up2(x_1)

          x_1 = self.upsample(x_1)
          layer1_1 = self.layer1_1x1(layer1)
          x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer1_1)], dim=1)
          x_1 = self.conv_up1(x_1)

          x_1 = self.upsample(x_1)
          layer0_1 = self.layer0_1x1(layer0)
          x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer0_1)], dim=1)
          x_1 = self.conv_up0(x_1)

          x_1 = self.upsample(x_1)
          x_1 = torch.cat([x_1[:,:128,:,:], torch.max(x_1[:,128:,:,:] , x_original)], dim=1)          
          x_1 = self.conv_original_size2(x_1)

          out_1 = self.conv_last(x_1)

          if self.lastop == 'sigmoid':
              out_1 = F.sigmoid(out_1)*self.scale
          elif self.lastop == 'tanh':
              out_1 = F.tanh(out_1)*self.scale
          else:
              out_1 = out_1*self.scale
                  
          return out_1 


class ResNetEnc(nn.Module):
    def __init__(self, outsize):
        super(ResNetEnc,self).__init__()

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256 + 512, 1, 0)


        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.avgpool = nn.AvgPool2d(8, stride=1)

    def forward(self, input, flag=0):
    
        if (flag == 0): #'im-input'
        
          # Image Encoder
          x_original = self.conv_original_size0(input)
          x_original = self.conv_original_size1(x_original)

          layer0 = self.layer0(input)
          layer1 = self.layer1(layer0)
          layer2 = self.layer2(layer1)
          layer3 = self.layer3(layer2)
          layer4 = self.layer4(layer3)
          
          x = self.avgpool(layer4)
          x = x.view(x.size(0), -1)                  
          return x 


class ResNetEncDec(nn.Module):
    def __init__(self, n_class):
        super(ResNetEncDec,self).__init__()

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, flag=0):
    
        if (flag == 0): #'im-input'
        
          # Image Encoder
          x_original = self.conv_original_size0(input)
          x_original = self.conv_original_size1(x_original)

          layer0 = self.layer0(input)
          layer1 = self.layer1(layer0)
          layer2 = self.layer2(layer1)
          layer3 = self.layer3(layer2)
          layer4 = self.layer4(layer3)
          
          # Image decoder
          x_2 = self.upsample_2(layer4)  
          x_2 = self.conv_up3_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up2_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up1_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up0_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_original_size2_2(x_2)

          out_2 = self.conv_last_2(x_2)
          out_2 = F.tanh(out_2)*0.01
                  
          return out_2 



class CondResNetEncDec(nn.Module):
    def __init__(self, n_class, n_cond=53):
        super(CondResNetEncDec,self).__init__()

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512+n_cond, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, cond, flag=0):
    
        if (flag == 0): #'im-input'
        
          # Image Encoder
          x_original = self.conv_original_size0(input)
          x_original = self.conv_original_size1(x_original)

          layer0 = self.layer0(input)
          layer1 = self.layer1(layer0)
          layer2 = self.layer2(layer1)
          layer3 = self.layer3(layer2)
          layer4 = self.layer4(layer3)

          # cond
          layer4_cond = torch.cat([layer4, cond[:,:,None,None].repeat(1,1,layer4.shape[2],layer4.shape[3])], dim=1)
          
          # Image decoder
          x_2 = self.upsample_2(layer4_cond)  
          x_2 = self.conv_up3_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up2_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up1_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up0_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_original_size2_2(x_2)

          out_2 = self.conv_last_2(x_2)
          out_2 = F.tanh(out_2)*0.01
                  
          return out_2 

class TexResNetEncDec(nn.Module):
    def __init__(self, n_class):
        super(TexResNetEncDec,self).__init__()

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = convrelu(3+3, 64, 7, 3, 2) # size=(N, 64, x.H/2, x.W/2)
        # self.layer0 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3+3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, flag=0):
    
        if (flag == 0): #'im-input'
        
          # Image Encoder
          x_original = self.conv_original_size0(input)
          x_original = self.conv_original_size1(x_original)

          layer0 = self.layer0(input)
          layer1 = self.layer1(layer0)
          layer2 = self.layer2(layer1)
          layer3 = self.layer3(layer2)
          layer4 = self.layer4(layer3)
          
          # Image decoder
          x_2 = self.upsample_2(layer4)  
          x_2 = self.conv_up3_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up2_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up1_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up0_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_original_size2_2(x_2)

          out_2 = self.conv_last_2(x_2)
          out_2 = F.sigmoid(out_2)
                  
          return out_2 



########
class CondResNetUNet(nn.Module):
    def __init__(self, n_class=1, n_cond=53):
        super(CondResNetUNet,self).__init__()
        self.n_cond = n_cond

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512+n_cond, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, cond):
        ## cond: [bz, 53]
            
        # Image Encoder
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)  # size=(N, 512, x.H/32, x.W/32)
        
        # Normal decoder
        layer4_cond = torch.cat([layer4[:,:512,:,:], cond[:,:,None,None].repeat(1,1,layer4.shape[2],layer4.shape[3])], dim=1)
        layer4_1 = self.layer4_1x1(layer4_cond)                 
        x_1 = self.upsample(layer4_1)
        layer3_1 = self.layer3_1x1(layer3)
        x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer3_1)], dim=1)           
        x_1 = self.conv_up3(x_1)

        x_1 = self.upsample(x_1)
        layer2_1 = self.layer2_1x1(layer2)
        x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer2_1)], dim=1)
        x_1 = self.conv_up2(x_1)

        x_1 = self.upsample(x_1)
        layer1_1 = self.layer1_1x1(layer1)
        x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer1_1)], dim=1)
        x_1 = self.conv_up1(x_1)

        x_1 = self.upsample(x_1)
        layer0_1 = self.layer0_1x1(layer0)
        x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer0_1)], dim=1)
        x_1 = self.conv_up0(x_1)

        x_1 = self.upsample(x_1)
        x_1 = torch.cat([x_1[:,:128,:,:], torch.max(x_1[:,128:,:,:] , x_original)], dim=1)          
        x_1 = self.conv_original_size2(x_1)

        out_1 = self.conv_last(x_1)
        out_1 = F.tanh(out_1)*0.01
                
        return out_1 


class StrongCondResNetUNet(nn.Module):
    def __init__(self, n_class=1, n_cond=53):
        super(StrongCondResNetUNet,self).__init__()
        self.n_cond = n_cond

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512+n_cond, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512 + n_cond, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512 + n_cond,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256 + n_cond,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256 + n_cond,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, cond):
        ## cond: [bz, 53]
            
        # Image Encoder
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)  # size=(N, 512, x.H/32, x.W/32)
        
        # Normal decoder
        layer4_cond = torch.cat([layer4[:,:512,:,:], cond[:,:,None,None].repeat(1,1,layer4.shape[2],layer4.shape[3])], dim=1)
        layer4_1 = self.layer4_1x1(layer4_cond)                 
        x_1 = self.upsample(layer4_1)
        layer3_1 = self.layer3_1x1(layer3)
        x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer3_1), cond[:,:,None,None].repeat(1,1,x_1.shape[2],x_1.shape[3])], dim=1)           
        x_1 = self.conv_up3(x_1)

        x_1 = self.upsample(x_1)
        layer2_1 = self.layer2_1x1(layer2)
        x_1 = torch.cat([x_1[:,:512,:,:], torch.max(x_1[:,512:,:,:] , layer2_1), cond[:,:,None,None].repeat(1,1,x_1.shape[2],x_1.shape[3])], dim=1)
        x_1 = self.conv_up2(x_1)

        x_1 = self.upsample(x_1)
        layer1_1 = self.layer1_1x1(layer1)
        x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer1_1), cond[:,:,None,None].repeat(1,1,x_1.shape[2],x_1.shape[3])], dim=1)
        x_1 = self.conv_up1(x_1)

        x_1 = self.upsample(x_1)
        layer0_1 = self.layer0_1x1(layer0)
        x_1 = torch.cat([x_1[:,:256,:,:], torch.max(x_1[:,256:,:,:] , layer0_1), cond[:,:,None,None].repeat(1,1,x_1.shape[2],x_1.shape[3])], dim=1)
        x_1 = self.conv_up0(x_1)

        x_1 = self.upsample(x_1)
        x_1 = torch.cat([x_1[:,:128,:,:], torch.max(x_1[:,128:,:,:] , x_original)], dim=1)          
        x_1 = self.conv_original_size2(x_1)

        out_1 = self.conv_last(x_1)
        out_1 = F.tanh(out_1)*0.01
                
        return out_1 





####################
def load_local_mask(image_size=256, mode='bbx'):
    if mode == 'bbx':
        # UV space face attributes bbx in size 2048 (l r t b)
        # face = np.array([512, 1536, 512, 1536]) #
        face = np.array([400, 1648, 400, 1648])
        if image_size == 512:
            # face = np.array([400, 400+512*2, 400, 400+512*2])
            face = np.array([512, 512+512*2, 512, 512+512*2])

        forehead = np.array([550, 1498, 430, 700+50])
        eye_nose = np.array([490, 1558, 700, 1050+50])
        mouth = np.array([574, 1474, 1050, 1550])
        ratio = image_size / 2048.
        face = (face * ratio).astype(np.int)
        forehead = (forehead * ratio).astype(np.int)
        eye_nose = (eye_nose * ratio).astype(np.int)
        mouth = (mouth * ratio).astype(np.int)
        regional_mask = np.array([face, forehead, eye_nose, mouth])

    return regional_mask

def texture2patch(texture, regional_mask, new_size=None):
    patch_list = []
    for pi in range(len(regional_mask)):
        patch = texture[:, :, regional_mask[pi][2]:regional_mask[pi][3], regional_mask[pi][0]:regional_mask[pi][1]]
        if new_size is not None:
            patch = F.interpolate(patch, [new_size, new_size], mode='bilinear')
        patch_list.append(patch)
    return patch_list

def patch2texture(texture, patch_list, regional_mask):
    for pi in range(len(regional_mask)):
        patch = patch_list[pi]
        patch = F.interpolate(patch, [regional_mask[pi][3]-regional_mask[pi][2], regional_mask[pi][1]-regional_mask[pi][0]], mode='bilinear')
        texture[:, :, regional_mask[pi][2]:regional_mask[pi][3], regional_mask[pi][0]:regional_mask[pi][1]] = patch
    return texture
    
class ResNetEncDecPatch(nn.Module):
    def __init__(self, n_class):
        super(ResNetEncDecPatch,self).__init__()
        self.encdec_list = \
        [ResNetEncDec(n_class=1).cuda(), ResNetEncDec(n_class=1).cuda(), ResNetEncDec(n_class=1).cuda(), ResNetEncDec(n_class=1).cuda()]
        # self.
        self.encdec_all = ResNetEncDec(n_class=1).cuda()
        self.mask = load_local_mask(image_size=256)
        # 
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, input, flag=0):
        patch_list = texture2patch(input, self.mask, new_size=256)
        out_list = []
        for i in range(len(patch_list)):
            out_list.append(self.encdec_list[i](patch_list[i]))
        out_all = self.encdec_all(input)
        # combine
        out_com = patch2texture(out_all, out_list, self.mask)
        ## 
        out_final = self.final_conv(out_com)*0.01       
        return out_final, out_list

class ResNetDec(nn.Module):
    def __init__(self, n_class):
        super(ResNetDec,self).__init__()

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 256 + 512, 1, 0)
              
        self.layer0_2 = copy.deepcopy(self.layer0)
        self.layer1_2 = copy.deepcopy(self.layer1)
        self.layer2_2 = copy.deepcopy(self.layer2)
        self.layer3_2 = copy.deepcopy(self.layer3)
        self.layer4_2 = copy.deepcopy(self.layer4)                


        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 128 + 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512,  64 + 256, 3, 1)
        self.conv_up1 = convrelu( 64 + 256,  64 + 256, 3, 1)
        self.conv_up0 = convrelu( 64 + 256,  64 + 128, 3, 1)
        
        self.conv_up3_2 = convrelu(512, 512, 3, 1)
        self.conv_up2_2 = convrelu(512, 256, 3, 1)
        self.conv_up1_2 = convrelu(256, 256, 3, 1)
        self.conv_up0_2 = convrelu(256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_original_size0_2 = convrelu(3, 64, 3, 1)
        self.conv_original_size1_2 = convrelu(64, 64, 3, 1)
        self.conv_original_size2_2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        self.conv_last_2 = nn.Conv2d(64, n_class, 1)

    def forward(self, input, flag=0):
    
        if (flag == 0): #'im-input'
        
          # Image Encoder
          x_original = self.conv_original_size0(input)
          x_original = self.conv_original_size1(x_original)

          layer0 = self.layer0(input)
          layer1 = self.layer1(layer0)
          layer2 = self.layer2(layer1)
          layer3 = self.layer3(layer2)
          layer4 = self.layer4(layer3)
          
          # Image decoder
          x_2 = self.upsample_2(layer4)  
          x_2 = self.conv_up3_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up2_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up1_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_up0_2(x_2)

          x_2 = self.upsample_2(x_2)
          x_2 = self.conv_original_size2_2(x_2)

          out_2 = self.conv_last_2(x_2)
          out_2 = F.tanh(out_2)*0.01
                  
          return out_2 