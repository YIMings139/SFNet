from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import torch

from layers import *
from timm.models.layers import trunc_normal_


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=(0, 0), dilation=(1, 1), groups=1, bn_act=False, bias=False):

        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class DeConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=(0, 0), dilation=(1, 1), groups=1, bn_act=False, bias=False):

        super().__init__()

        self.bn_act = bn_act

        self.deconv = nn.ConvTranspose2d(nIn, nOut, kernel_size=kSize,
                                         stride=stride, padding=padding,
                                         dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.deconv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output


class channelAttention(nn.Module):
    """ Channel attention module"""

    def __init__(self, k_size=5):
        super(channelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        indentity = x.clone()

        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)
        y_1 = self.conv1(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_2 = self.conv2(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        attn = self.sigmoid(y_1 + y_2)
        return indentity * attn.expand_as(indentity)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.vertial_pool = nn.AvgPool2d(kernel_size=(kernel_size, 1), stride=1,
                                         padding=(int((kernel_size - 1) / 2), 0))
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        indentity = x.clone()
        x = self.vertial_pool(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        attn = self.sigmoid(x)
        return indentity * attn.expand_as(indentity)


class MFAM(nn.Module):
    '''
    Multiscale Feature Aggregation Module.
    '''

    def __init__(self, dim_in, dim_out, Ksize):
        super(MFAM, self).__init__()
        self.conv1x1 = Conv(dim_in, dim_out, 1, 1, bn_act=True)

        self.conv_x = Conv(dim_out, dim_out, kSize=(Ksize, 1), stride=1, padding=(int((Ksize - 1) / 2), 0),
                           groups=dim_out)
        self.conv_y = Conv(dim_out, dim_out, kSize=(1, Ksize), stride=1, padding=(0, int((Ksize - 1) / 2)),
                           groups=dim_out)
        self.conv_refine = Conv(dim_out, dim_out, 3, 1, 1, bn_act=True)
        self.conv5x5 = Conv(dim_out, dim_out, 5, 1, 2, bn_act=False)

        self.spa = SpatialAttention()
        self.cha = channelAttention()

    def forward(self, x):
        identity = x
        x = self.conv1x1(x)
        x_1 = self.conv_x(x)
        x_2 = self.conv_y(x)
        x = self.conv5x5(self.conv_refine(x_1 + x_2))
        x = self.cha(x)
        x = self.spa(x)
        out = x + identity
        return out


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(3), num_output_channels=1):

        super().__init__()
        self.scales = scales[::-1]
        self.num_output_channels = num_output_channels
        self.scales = scales
        self.pool_k = [5, 7, 11]
        self.num_ch_enc = num_ch_enc

        self.num_ch_dec = num_ch_enc[::-1]

        self.convs = OrderedDict()
        for i in range(2, -1, -1):

            self.convs[("upconv", i, 0)] = Conv(self.num_ch_enc[i], int(self.num_ch_enc[i] / 4), kSize=1, stride=1,
                                                bn_act=False)
            self.convs[("skipconv", i)] = Conv(self.num_ch_enc[i], int(self.num_ch_enc[i] / 4), kSize=1, stride=1,
                                               bn_act=False)
            if i != 0:
                self.convs[("upconv", i, 1)] = Conv(self.num_ch_enc[i], self.num_ch_enc[i - 1], kSize=1, stride=1,
                                                    bn_act=False)

        self.down_sample_f1_2 = Conv(nIn=self.num_ch_enc[0], nOut=int(self.num_ch_enc[1] / 4), kSize=3, stride=2,
                                     padding=1,
                                     bn_act=True)
        self.down_sample_f1_3 = nn.Sequential(
            Conv(nIn=self.num_ch_enc[0], nOut=self.num_ch_enc[1], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(nIn=self.num_ch_enc[1], nOut=int(self.num_ch_enc[2] / 4), kSize=3, stride=2, padding=1, bn_act=True))
        self.down_sample_f2_3 = Conv(nIn=self.num_ch_enc[1], nOut=int(self.num_ch_enc[2] / 4), kSize=3, stride=2,
                                     padding=1,
                                     bn_act=True)

        self.up_sample_f3_2 = DeConv(nIn=self.num_ch_enc[2], nOut=int(self.num_ch_enc[1] / 4), kSize=4, stride=2,
                                     padding=1, bn_act=True)
        self.up_sample_f3_1 = nn.Sequential(
            DeConv(nIn=self.num_ch_enc[2], nOut=self.num_ch_enc[1], kSize=4, stride=2, padding=1, bn_act=True),
            DeConv(nIn=self.num_ch_enc[1], nOut=int(self.num_ch_enc[0] / 4), kSize=4, stride=2, padding=1, bn_act=True))
        self.up_sample_f2_1 = DeConv(nIn=self.num_ch_enc[1], nOut=int(self.num_ch_enc[0] / 4), kSize=4, stride=2,
                                     padding=1,
                                     bn_act=True)

        self.blocks = nn.ModuleList([
            MFAM(self.num_ch_dec[0], self.num_ch_dec[0], Ksize=5),
            MFAM(self.num_ch_dec[1], self.num_ch_dec[1], Ksize=7),
            MFAM(self.num_ch_dec[2], self.num_ch_dec[2], Ksize=11)
        ])

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_enc[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}

        f1, f2, f3, f4 = input_features
        x_3 = upsample(f4)
        x_3 = self.convs[("upconv", 2, 0)](x_3)
        f33 = self.convs[("skipconv", 2)](f3)

        x_13 = self.down_sample_f1_3(f1)
        x_23 = self.down_sample_f2_3(f2)
        x_3 = torch.cat([f33, x_23, x_13, x_3], dim=1)
        x_3 = self.blocks[0](x_3)
        f = upsample(self.convs[("dispconv", 2)](x_3), mode='bilinear')
        self.outputs[("disp", 2)] = self.sigmoid(f)

        x_2 = upsample(self.convs[("upconv", 2, 1)](x_3))
        x_2 = self.convs[("upconv", 1, 0)](x_2)
        f22 = self.convs[("skipconv", 1)](f2)
        x_12 = self.down_sample_f1_2(f1)
        x_32 = self.up_sample_f3_2(f3)

        x_2 = torch.cat([x_2, x_12, x_32, f22], dim=1)
        x_2 = self.blocks[1](x_2)
        f = upsample(self.convs[("dispconv", 1)](x_2), mode='bilinear')
        self.outputs[("disp", 1)] = self.sigmoid(f)

        x_1 = upsample(self.convs[("upconv", 1, 1)](x_2))

        x_1 = self.convs[("upconv", 0, 0)](x_1)
        f11 = self.convs[("skipconv", 0)](f1)
        x_21 = self.up_sample_f2_1(f2)
        x_31 = self.up_sample_f3_1(f3)
        x_1 = torch.cat([x_1, x_21, x_31, f11], dim=1)
        x_1 = self.blocks[2](x_1)
        f = upsample(self.convs[("dispconv", 0)](x_1), mode='bilinear')
        self.outputs[("disp", 0)] = self.sigmoid(f)

        return self.outputs
