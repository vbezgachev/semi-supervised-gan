'''
Defines the Generative Adversarial Network (GAN) for semi-supervised learning using PyTorch library
The code from https://github.com/yunjey/mnist-svhn-transfer partially was used
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


class _netG(nn.Module):
    '''
    GAN generator
    '''
    def __init__(self, num_noise_channels, size_mult, lrelu_alpha, num_output_channels):
        super(_netG, self).__init__()
        self.lrelu_alpha = lrelu_alpha

        # noise is going into a convolution
        self.deconv1 = deconv(
            c_in=num_noise_channels,
            c_out=size_mult * 4,
            k_size=4,
            stride=1,
            pad=0)
        # (size_mult * 4) x 4 x 4

        self.deconv2 = deconv(
            c_in=size_mult * 4,
            c_out=size_mult * 2,
            k_size=4)
        # (size_mult * 2) x 8 x 8

        self.deconv3 = deconv(
            c_in=size_mult * 2,
            c_out=size_mult * 1,
            k_size=4)
        # (size_mult) x 16 x 16

        self.deconv4 = deconv(
            c_in=size_mult,
            c_out=num_output_channels,
            k_size=4,
            bn=False)
        # (num_output_channels) x 16 x 16

    def forward(self, inputs):
        out = F.leaky_relu(self.deconv1(inputs), self.lrelu_alpha)
        out = F.leaky_relu(self.deconv2(out), self.lrelu_alpha)
        out = F.leaky_relu(self.deconv3(out), self.lrelu_alpha)
        out = F.tanh(self.deconv4(out))
        return out


class _netD(nn.Module):
    '''
    GAN discruminator
    '''
    def __init__(self, size_mult, lrelu_alpha, number_channels, drop_rate, num_classes):
        super(_netD, self).__init__()
        self.drop_rate = drop_rate
        self.lrelu_alpha = lrelu_alpha
        self.size_mult = size_mult
        self.num_classes = num_classes

        # input is (number_channels) x 32 x 32
        self.conv1 = conv(
            c_in=number_channels,
            c_out=size_mult,
            k_size=3,
            bn=False
        )
        # (size_mult) x 16 x 16

        self.conv2 = conv(
            c_in=size_mult,
            c_out=size_mult,
            k_size=3,
        )
        # (size_mult) x 8 x 8

        self.conv3 = conv(
            c_in=size_mult,
            c_out=size_mult,
            k_size=3,
        )
        # (size_mult) x 4 x 4

        self.conv4 = conv(
            c_in=size_mult,
            c_out=size_mult * 2,
            k_size=3,
            stride=1
        )
        # (size_mult * 2) x 4 x 4

        self.conv5 = conv(
            c_in=size_mult * 2,
            c_out=size_mult * 2,
            k_size=3,
            stride=1
        )
        # (size_mult * 2) x 4 x 4

        self.conv6 = conv(
            c_in=size_mult * 2,
            c_out=size_mult * 2,
            k_size=3,
            stride=1,
            pad=0,
            bn=False
        )
        # (size_mult * 2) x 2 x 2

        self.features = nn.AvgPool2d(kernel_size=2)

        self.class_logits = nn.Linear(
            in_features=(size_mult * 2) * 1 * 1,
            out_features=num_classes)

    def forward(self, inputs):
        out = F.dropout2d(inputs, p=self.drop_rate/2.5)

        out = F.leaky_relu(self.conv1(out), self.lrelu_alpha)
        out = F.dropout2d(out, p=self.drop_rate)

        out = F.leaky_relu(self.conv2(out), self.lrelu_alpha)

        out = F.leaky_relu(self.conv3(out), self.lrelu_alpha)
        out = F.dropout2d(out, p=self.drop_rate)

        out = F.leaky_relu(self.conv4(out), self.lrelu_alpha)

        out = F.leaky_relu(self.conv5(out), self.lrelu_alpha)

        out = F.leaky_relu(self.conv6(out), self.lrelu_alpha)

        features = self.features(out)
        features = features.squeeze()

        class_logits = self.class_logits(features)

        # calculate gan logits
        max_val, _ = torch.max(class_logits, 1, keepdim=True)
        stable_class_logits = class_logits - max_val
        max_val = torch.squeeze(max_val)
        gan_logits = torch.log(torch.sum(torch.exp(stable_class_logits), 1)) + max_val

        out = F.softmax(class_logits, dim=0)

        return out, class_logits, gan_logits, features
