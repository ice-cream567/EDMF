'''
EFNet
@inproceedings{sun2022event,
      author = {Sun, Lei and Sakaridis, Christos and Liang, Jingyun and Jiang, Qi and Yang, Kailun and Sun, Peng and Ye, Yaozu and Wang, Kaiwei and Van Gool, Luc},
      title = {Event-Based Fusion for Motion Deblurring with Cross-modal Attention},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = 2022
      }
'''

import torch
import torch.nn as nn
import math
from basicsr.models.archs.arch_util import EventImage_ChannelAttentionTransformerBlock,HF,CALayer
from torch.nn import functional as F


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class MFHI(nn.Module):
    def __init__(self, in_chn=3, ev_chn=6, wf=64, depth=3, fuse_before_downsample=True, relu_slope=0.2,num_heads=None
                ):
        super(MFHI, self).__init__()
        self.depth = depth
        self.fuse_before_downsample = fuse_before_downsample

        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        # event
        self.down_path_ev = nn.ModuleList()
        self.conv_ev1 = nn.Conv2d(ev_chn, wf, 3, 1, 1)

        prev_channels = self.get_input_chn(wf)

        for i in range(depth):
            downsample = True if (i + 1) < depth else False

            self.down_path_1.append(
                UNetConvBlock(prev_channels, (2 ** i) * wf, downsample, relu_slope,num_heads=1))
            self.down_path_2.append(
                UNetConvBlock(prev_channels, (2 ** i) * wf, downsample, relu_slope, use_emgc=downsample))
            # ev encoder
            if i < self.depth:
                self.down_path_ev.append(UNetEVConvBlock(prev_channels, (2 ** i) * wf, downsample, relu_slope))

            prev_channels = (2 ** i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2 ** i) * wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2 ** i) * wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2 ** i) * wf, (2 ** i) * wf, 3, 1, 1))
            prev_channels = (2 ** i) * wf
        self.sam12 = SAM(prev_channels)

        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x, event, mask=None):
        image = x
        ev = []
        # EVencoder
        e1 = self.conv_ev1(event)
        for i, down in enumerate(self.down_path_ev):
            if i < self.depth - 1:
                e1, e1_up = down(e1, self.fuse_before_downsample)
                if self.fuse_before_downsample:
                    ev.append(e1_up)
                else:
                    ev.append(e1)
            else:
                e1 = down(e1, self.fuse_before_downsample)
                ev.append(e1)

        # stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        masks = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:

                x1, x1_up = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)
                encs.append(x1_up)

                if mask is not None:
                    masks.append(F.interpolate(mask, scale_factor=0.5 ** i))

            else:
                x1 = down(x1, event_filter=ev[i], merge_before_downsample=self.fuse_before_downsample)

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)
        sam_feature, out_1 = self.sam12(x1, image)

        # stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                if mask is not None:
                    x2, x2_up = down(x2, encs[i], decs[-i - 1], mask=masks[i])
                else:
                    x2, x2_up = down(x2, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))

        out_2 = self.last(x2)
        out_2 = out_2 + image

        return [out_1, out_2]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False, num_heads=None):  # cat
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc
        self.num_heads = num_heads

        # self.att = CALayer(out_size)
        # self.alpha=nn.Parameter(torch.ones(1))
        # self.beta=nn.Parameter(torch.ones(1))
        # self.alise = nn.Conv2d(out_size,out_size,1,1,0,bias=False)  # one_module(n_feats)
        # self.alise2 = nn.Conv2d(out_size*2,out_size,3,1,1,bias=False)  # one_module(n_feats)
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        if self.num_heads is not None:
            # self.image_event_transformer = EventImage_ChannelAttentionTransformerBlock(out_size,
            #                                                                            num_heads=self.num_heads,
            #                                                                            ffn_expansion_factor=4,
            #                                                                            bias=False,
            #                                                                            LayerNorm_type='WithBias')
            self.sf = SF(out_size)
            self.hf = HF(out_size)
            self.fusion = Fusion(out_size)


    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):

        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)



        #频率增强 event_filter(2,64,256,256)=img    img_hight+event_filter=gaopin    img_low=dipin 首先将高频进行transformerzengqiang1   最后 频率融合


        if enc is not None and dec is not None and mask is not None:
            assert self.use_emgc
            out_enc = self.emgc_enc(enc) + self.emgc_enc_mask((1 - mask) * enc)
            out_dec = self.emgc_dec(dec) + self.emgc_dec_mask(mask * dec)
            out = out + out_enc + out_dec

        if event_filter is not None and merge_before_downsample:
            # b, c, h, w = out.shape
            # 频率分离
            img_high, img_low = self.sf(out)
            # fea_high = self.image_event_transformer(img_high, event_filter) #增强
            fea_high = self.hf(img_high, event_filter)
            # 频率融合
            out = self.fusion(img_low,fea_high)+self.identity(x)
            # out=self.alise(self.att(self.alise2(torch.cat([img_low,fea_high], dim=1))))

        if self.downsample:

            out_down = self.downsample(out)
            # if not merge_before_downsample:
                # out_down = self.image_event_transformer(out_down, event_filter)

            return out_down, out

        else:
            if merge_before_downsample:
                return out
            else:
                out = self.image_event_transformer(out, event_filter)


class UNetEVConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_emgc=False):
        super(UNetEVConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_emgc = use_emgc

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        self.conv_before_merge = nn.Conv2d(out_size, out_size, 1, 1, 0)
        if downsample and use_emgc:
            self.emgc_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_enc_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.emgc_dec_mask = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, merge_before_downsample=True):
        out = self.conv_1(x)

        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))

        out = out_conv2 + self.identity(x)

        if self.downsample:

            out_down = self.downsample(out)

            if not merge_before_downsample:

                out_down = self.conv_before_merge(out_down)
            else:
                out = self.conv_before_merge(out)
            return out_down, out

        else:

            out = self.conv_before_merge(out)
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class SF(nn.Module):

    def __init__(self, in_channel):
        super(SF, self).__init__()
        self.updown = Updownblock(in_channel)

    def forward(self, img):
        img_high, img_low = self.updown(img)
        return img_high, img_low


class Updownblock(nn.Module):
    def __init__(self, n_feats, relu_slope=0.02 ):
        super(Updownblock, self).__init__()
        self.decoder_high = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        )
        self.decoder_low = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_slope, inplace=False),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        )
        self.down = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x1 = self.down(x)  # 低频
        low = F.interpolate(x1, size=x.size()[-2:], mode='bilinear', align_corners=True)
        high = x - low  # 得到了图片的高频信息
        high = self.decoder_high(high)
        low = self.decoder_low(low)
        return high, low


class Fusion(nn.Module):
    def __init__(self, out_size, use_bn=True, use_relu=True):
        super(Fusion, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        self.alise = nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.alise2 = nn.Conv2d(out_size * 2, out_size, kernel_size=3, stride=1, padding=1, bias=False)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_size)
            self.bn2 = nn.BatchNorm2d(out_size)

        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

        self.att = CALayer(out_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(out_size, out_size, 1, 1, 0)


    def forward(self, img_low, fea_high):
        x1 = self.alise2(torch.cat([img_low, fea_high], dim=1))

        if self.use_bn:
            x1 = self.bn1(x1)

        if self.use_relu:
            x1 = self.relu(x1)

        emerge = self.gap(x1)
        fea1 = self.alise(emerge)
        fea2 = self.alise(emerge)
        attention_vectors = torch.cat([fea1, fea2], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        low_att, high_att = torch.chunk(attention_vectors, 2, dim=1)
        high = fea_high * high_att
        low = img_low * low_att
        out1 = self.out(high + low)

        x2 = self.att(out1)
        x3 = self.alise(x2)

        if self.use_bn:
            x3 = self.bn2(x3)

        return x3

if __name__ == "__main__":
    pass
