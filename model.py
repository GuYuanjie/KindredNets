import torch
import torch.nn as nn

import numpy as np

def save_feature(feature):
    import cv2
    C = feature.shape[1]
    for i in range(C):
        map = feature[0,i,:,:].cpu().numpy() * 255.0
        map = np.uint8(map.clip(0, 255))
        cv2.applyColorMap(map, cv2.COLORMAP_JET)
        cv2.imwrite('./map/{}.jpg'.format(i),map)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享权重的MLP
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self,planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        out = self.relu(out)
        return out

class LightenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(LightenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code+offset
        out = self.conv_Decoder(code_lighten)
        return out

class DarkenBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DarkenBlock, self).__init__()
        codedim=output_size//2
        self.conv_Encoder = ConvBlock(input_size, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Offset = ConvBlock(codedim, codedim, 3, 1, 1,isuseBN=False)
        self.conv_Decoder = ConvBlock(codedim, output_size, 3, 1, 1,isuseBN=False)

    def forward(self, x):
        code= self.conv_Encoder(x)
        offset = self.conv_Offset(code)
        code_lighten = code + offset
        out = self.conv_Decoder(code_lighten)
        return out

class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.outlayer(y)
        return y


class LBP(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(LBP, self).__init__()
        self.fusion = FusionLayer(input_size,output_size)
        self.conv1_1 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DarkenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = LightenBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2_1 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x=self.fusion(x)
        hr = self.conv1_1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1_1(x) + lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2_1(hr)
        return hr_weight + h_residue


class PLN(nn.Module):
    def __init__(self, input_dim=3, dim=16):
        super(DLN, self).__init__()
        inNet_dim = input_dim+1
        # 1:brightness
        self.feat_d1_1 = ConvBlock(inNet_dim, 2 * dim, 3, 1, 1)
        self.feat_d1_2 = ConvBlock(2 * dim, dim, 3, 1, 1)

        self.feat_d2_1 = ConvBlock(inNet_dim, 2 * dim, 3, 2, 1)
        self.feat_d2_2 = ConvBlock(2 * dim, dim, 3, 1, 1)

        self.feat_d4_1 = ConvBlock(inNet_dim, 2 * dim, 3, 4, 1)
        self.feat_d4_2 = ConvBlock(2 * dim, dim, 3, 1, 1)

        self.feat_d1_out_1 = LBP(input_size=dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.bam_d1_1 = CBAM(dim)
        self.feat_d1_out_2 = LBP(input_size=2* dim, output_size=dim, kernel_size=3, stride=1,
                              padding=1)
        self.bam_d1_2 = CBAM(dim)
        self.feat_d1_out_3 = LBP(input_size=3* dim, output_size=dim, kernel_size=3, stride=1,
                              padding=1)

        self.feat_d2_out_1 = LBP(input_size=dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.bam_d2_1 = CBAM(dim)
        self.feat_d2_out_2 = LBP(input_size=2 * dim, output_size=dim, kernel_size=3, stride=1,
                                 padding=1)
        self.bam_d2_2 = CBAM(dim)
        self.feat_d2_out_3 = LBP(input_size=3 * dim, output_size=dim, kernel_size=3, stride=1,
                                 padding=1)
        self.up_d2 = nn.Sequential(
                nn.PixelShuffle(2)
        )

        self.up_d2_res = nn.Sequential(
            nn.PixelShuffle(2)
        )

        self.feat_d4_out_1 = LBP(input_size= dim, output_size=dim, kernel_size=3, stride=1, padding=1)
        self.bam_d4_1 = CBAM(dim)
        self.feat_d4_out_2 = LBP(input_size=2 * dim, output_size=dim, kernel_size=3, stride=1,
                                 padding=1)
        self.bam_d4_2 = CBAM(dim)
        self.feat_d4_out_3 = LBP(input_size=3 * dim, output_size=dim, kernel_size=3, stride=1,
                                 padding=1)
        self.up_d4 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
        )
        self.up_d4_res = nn.Sequential(
            nn.PixelShuffle(2),

        )
        self.up_d8_res = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),

        )

        self.feature = ConvBlock(input_size=(4 * dim +dim+int(dim/4)), output_size=dim, kernel_size=3, stride=1, padding=1)
        self.bam_tol = CBAM(dim)

        self.out = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_ori, tar=None):
        # data gate
        x = torch.pow(torch.abs(x_ori + 0.001),0.5)
        x_bright, _ = torch.max(x_ori, dim=1, keepdim=True)
        x_in = torch.cat((x, x_bright), 1)

        #x_in = x_ori
        # feature extraction d4-path3
        feature = self.feat_d4_1(x_in)
        feature_d4_1_in = self.feat_d4_2(feature)
        # ------------------------------------------Level3_LBP1
        #level3_to_level2_lbp1 = self.up_d4_res(feature_d4_1_in)
        #level3_to_level1_lbp1 = self.up_d8_res(feature_d4_1_in)
        feature_d4_1_out = self.feat_d4_out_1(feature_d4_1_in)


        feature_d4_2_in = torch.cat([feature_d4_1_in, feature_d4_1_out], dim=1)

        # ------------------------------------------Level3_LBP2
        #level3_to_level2_lbp2 = self.up_d4_res(feature_d4_2_in)
        #level3_to_level1_lbp2 = self.up_d8_res(feature_d4_2_in)
        feature_d4_2_out = self.feat_d4_out_2(feature_d4_2_in)
        feature_d4_2_out = self.bam_d4_1(feature_d4_2_out)


        feature_d4_3_in = torch.cat([feature_d4_1_in, feature_d4_1_out, feature_d4_2_out], dim=1)

        # ------------------------------------------Level3_LBP3
        #level3_to_level2_lbp3 = self.up_d4_res(feature_d4_3_in)
       # level3_to_level1_lbp3 = self.up_d8_res(feature_d4_3_in)
        feature_d4_3_out = self.feat_d4_out_3(feature_d4_3_in)
        feature_d4_3_out = self.bam_d4_2(feature_d4_3_out)


        feature_d4 = torch.cat([feature_d4_1_in, feature_d4_1_out, feature_d4_2_out, feature_d4_3_out], dim=1)
        feature_d4 = self.up_d4(feature_d4)

        # feature extraction d2-path2
        feature = self.feat_d2_1(x_in)
        feature_d2_1_in = self.feat_d2_2(feature)
        # ------------------------------------------Level2_LBP1
        #level2_to_level1_lbp1 = self.up_d4_res(feature_d2_1_in)
        #feature_d2_1_in = torch.cat([feature_d2_1_in,level3_to_level2_lbp1],dim=1)
        feature_d2_1_out = self.feat_d2_out_1(feature_d2_1_in)

        feature_d2_2_in = torch.cat([feature_d2_1_in, feature_d2_1_out], dim=1)

        # ------------------------------------------Level2_LBP2
        #level2_to_level1_lbp2 = self.up_d4_res(feature_d2_2_in)
        #feature_d2_2_in = torch.cat([feature_d2_2_in,level3_to_level2_lbp2],dim=1)

        feature_d2_2_out = self.feat_d2_out_2(feature_d2_2_in)
        feature_d2_2_out = self.bam_d2_1(feature_d2_2_out)
        feature_d2_3_in = torch.cat([feature_d2_1_in, feature_d2_1_out, feature_d2_2_out], dim=1)

        # ------------------------------------------Level2_LBP3
       # level2_to_level1_lbp3 = self.up_d4_res(feature_d2_3_in)
        #feature_d2_3_in = torch.cat([feature_d2_3_in, level3_to_level2_lbp3], dim=1)
        feature_d2_3_out = self.feat_d2_out_3(feature_d2_3_in)
        feature_d2_3_out = self.bam_d2_2(feature_d2_3_out)
        feature_d2 = torch.cat([feature_d2_1_in, feature_d2_1_out, feature_d2_2_out, feature_d2_3_out], dim=1)
        feature_d2 = self.up_d2(feature_d2)

        # feature extraction d1-path1
        feature = self.feat_d1_1(x_in)
        feature_d1_1_in = self.feat_d1_2(feature)
        # ------------------------------------------Level1_LBP1
        #feature_d1_1_in = torch.cat([feature_d1_1_in,level2_to_level1_lbp1,level3_to_level1_lbp1],dim=1)
        feature_d1_1_out = self.feat_d1_out_1(feature_d1_1_in)
        feature_d1_2_in = torch.cat([feature_d1_1_in, feature_d1_1_out], dim=1)

        # ------------------------------------------Level1_LBP2
        #feature_d1_2_in = torch.cat([feature_d1_2_in, level2_to_level1_lbp2, level3_to_level1_lbp2], dim=1)
        feature_d1_2_out = self.feat_d1_out_2(feature_d1_2_in)
        feature_d1_2_out = self.bam_d1_1(feature_d1_2_out)
        feature_d1_3_in = torch.cat([feature_d1_1_in, feature_d1_1_out, feature_d1_2_out], dim=1)

        # ------------------------------------------Level1_LBP3
        #feature_d1_3_in = torch.cat([feature_d1_3_in, level2_to_level1_lbp3, level3_to_level1_lbp3], dim=1)
        feature_d1_3_out = self.feat_d1_out_3(feature_d1_3_in)
        feature_d1_3_out = self.bam_d1_2(feature_d1_3_out)
        feature_d1 = torch.cat([feature_d1_1_in, feature_d1_1_out, feature_d1_2_out, feature_d1_3_out], dim=1)

        # ------------------------------------------fusion
        feature_in = torch.cat([feature_d1,feature_d2,feature_d4],dim=1)
        feature_out = self.feature(feature_in)
        feature_out = self.bam_tol(feature_out)
        #save_feature(feature_out)
        pred = self.out(feature_out) + x_ori

        return pred


############################################################################################
# Base models
############################################################################################

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()
        self.convx4 = ConvBlock(input_size, input_size*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.up = nn.PixelShuffle(2)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        x =self.convx4(x)
        out = self.up(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ResnetBlock(output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) + lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ResnetBlock(output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) + hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out
