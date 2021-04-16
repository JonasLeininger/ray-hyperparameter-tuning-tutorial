import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import ZeroPad2d, Conv2d, MaxPool2d, BatchNorm2d, Linear

class ResNet18(nn.Module):
    def __init__(self, config):
        super(ResNet18, self).__init__()

        self.classes = config['num_classes']
        self.in_shape = config['input_shape']
        self.channels = 3

        self.pad1 = ZeroPad2d(padding=3)
        self.conv1 = Conv2d(self.channels, 64, kernel_size=(7, 7), stride=2)
        self.bn1 = BatchNorm2d(64)
        self.pad2 = ZeroPad2d(padding=1)
        self.pool1 = MaxPool2d(kernel_size=3, stride=2)

        # stage 2
        self.conv_block_s2 = ResNetConv(64, filters=64, stride=1)
        self.identity_s2_1 = ResNetIdentity(64, filters=64)
        self.identity_s2_2 = ResNetIdentity(64, filters=64)

        # stage 3
        self.conv_block_s3 = ResNetConv(64, filters=128)
        self.identity_s3_1 = ResNetIdentity(128, filters=128)
        self.identity_s3_2 = ResNetIdentity(128, filters=128)

        # stage 4
        self.conv_block_s4 = ResNetConv(128, filters=256)
        self.identity_s4_1 = ResNetIdentity(256, filters=256)
        self.identity_s4_2 = ResNetIdentity(256, filters=256)

        # stage 5
        self.conv_block_s5 = ResNetConv(256, filters=512)
        self.identity_s5_1 = ResNetIdentity(512, filters=512)
        self.identity_s5_2 = ResNetIdentity(512, filters=512)

        # final stage
        self.linear = Linear(512, self.classes)
    
    def forward(self, x):

        x = self.pad1(x)
        x = self.conv1(x)
        x = self.pad2(x)
        x = self.pool1(x)
        
        # stage 2
        x = self.conv_block_s2(x)
        x = self.identity_s2_1(x)
        x = self.identity_s2_2(x)
        
        # stage 3
        x = self.conv_block_s3(x)
        x = self.identity_s3_1(x)
        x = self.identity_s3_2(x)
        
        # stage 4
        x = self.conv_block_s4(x)
        x = self.identity_s4_1(x)
        x = self.identity_s4_2(x)
        
        # stage 5
        x = self.conv_block_s5(x)
        x = self.identity_s5_1(x)
        x = self.identity_s5_2(x)
        
        # final stage
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        return x



class ResNetConv(nn.Module):

    def __init__(self, in_channels, filters, stride=2):
        super(ResNetConv, self).__init__()

        # todo: padding
        self.conv1 = Conv2d(in_channels, filters, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = BatchNorm2d(filters)
        self.conv2 = Conv2d(filters, filters, kernel_size=(3, 3), padding=1)
        self.bn2 = BatchNorm2d(filters)
        self.conv3 = Conv2d(in_channels, filters, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn3 = BatchNorm2d(filters)

    def forward(self, x):

        x_skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.conv3(x_skip)
        shortcut = self.bn3(shortcut)

        x += shortcut
        x = F.leaky_relu(x)

        return x


class ResNetIdentity(nn.Module):

    def __init__(self, in_channels, filters, stride=1):
        super(ResNetIdentity, self).__init__()

        self.conv1 = Conv2d(in_channels, filters, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = BatchNorm2d(filters)
        self.conv2 = Conv2d(filters, filters, kernel_size=(3, 3), padding=1)
        self.bn2 = BatchNorm2d(filters)

    def forward(self, x):

        x_skip = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += x_skip
        x = F.leaky_relu(x)

        return x


def main():
    config = {'classes': 10, 'in_shape': (224, 224, 3)}
    device = torch.device("cuda:0")
    tensor = torch.rand(1, 3, 224, 224).cuda()
    model = ResNet18(config).cuda()
    print(model.parameters())

    for i in range(100):
        test = model(tensor)
        print('Performing step {}'.format(i + 1))

if __name__ == '__main__':
    main()