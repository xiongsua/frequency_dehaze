from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from helper import *
from unet_parts import *
from torch.nn.modules.activation import Hardtanh
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
from torch.nn import init
from networks import define_G, define_D
from torch.autograd import Variable
import functools
import torchvision.models as models
import matplotlib.pyplot as plt
import cv2

# -------------------------------------------------------------------------- #
#  Network Architecture
# -------------------------------------------------------------------------- #
class DecompModel(nn.Module):
    def __init__(self):
        super(DecompModel, self).__init__()
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        self.fognet = FogNet(33)
        self.refined_T= UnetGenerator(4, 3, 7, 64, norm_layer=norm_layer, use_dropout=False)
        self.radiux = [2, 4, 8, 16, 32]
        self.trans = DCPDehazeGenerator()
        self.DehazeNet = DeRain_v2()
        # self.trans_net = Dense()
        self.conv_J_1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.conv_J_2 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)
        self.conv_down = nn.Conv2d(30, 3, 5, 1, 2)
        self.eps_list = [0.001, 0.0001]
        self.ref = RefUNet(3, 3)
        self.eps = 0.001
        self.gf = None
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        lf, hf = self.decomposition(x)

        atm, A = self.fognet(torch.cat([x, lf], dim=1))
        trans = self.trans(lf)
        dehaze = self.DehazeNet(torch.cat([x, lf], dim=1))
        # plt.figure()
        # test_x = (x[0, :, :, :]*255).to(int).permute(1, 2, 0).data.cpu().numpy()
        # test = (trans[0, :, :, :]*255).to(int).permute(1, 2, 0).data.cpu().numpy()
        # test_hf = (hf[0, :, :, :]*255).to(int).permute(1, 2, 0).data.cpu().numpy()
        # test_lf = (lf[0, :, :, :]*255).mean(dim=0).to(int).data.cpu().numpy()
        # plt.subplot(2, 2, 1)
        # plt.imshow(test_x)
        # plt.subplot(2, 2, 2)
        # plt.imshow(test, cmap='gray')
        # plt.subplot(2, 2, 3)
        # plt.imshow(test_lf, cmap='gray')
        # plt.subplot(2, 2, 4)
        # plt.imshow(test_hf)
        # plt.show()
        # '''
        # applay PCA
        #
        # '''
        # new_hf = torch.reshape(hf,(-1,hf.shape[1]))
        # pca = PCA(n_components=3,whiten=True)
        # new_hf = pca.fit_transform(new_hf.to('cpu'))
        # new_hf = np.reshape(new_hf,(hf.shape[0], 3, hf.shape[2], hf.shape[3]))
        # new_hf = torch.from_numpy(new_hf).cuda()
        refine_T = self.refined_T(torch.cat((x, trans), 1))
        # new_hf = new_hf.to(torch.float)
        # dehaze = (x - (1 - refine_T) * atm) / (refine_T + self.eps) + hf
        # dehaze = dehaze.to(torch.float)
        clean = dehaze+hf
        # dehaze = (dehaze[0, :, :, :]*255).to(int).permute(1, 2, 0).data.cpu().numpy()
        # plt.imshow(dehaze)
        # plt.show()
        # clean = self.relu(self.ref(dehaze, A))
        rec_I = clean * refine_T + (1-refine_T)* atm
        # rec_I = (rec_I[0, :, :, :]*255).to(int).permute(1, 2, 0).data.cpu().numpy()
        # plt.imshow(rec_I)
        # plt.show()
        # rec_I = rec_I.to(torch.float)
        return trans, atm, clean, rec_I  # trans, atm, clean,streak,

    def forward_test(self, x, A, mode='A'):
        if mode=='run':
            lf, hf = self.decomposition(x)
            atm, A = self.fognet(torch.cat([x, lf], dim=1))
            trans = self.trans(lf)
            # new_hf = torch.reshape(hf, (-1, hf.shape[1]))
            # pca = PCA(n_components=3, whiten=True)
            # new_hf = pca.fit_transform(new_hf.to('cpu'))
            # new_hf = np.reshape(new_hf, (hf.shape[0], 3, hf.shape[2], hf.shape[3]))
            # new_hf = torch.from_numpy(new_hf).cuda()
            # new_hf = new_hf.to(torch.float)
            refine_T = self.refined_T(torch.cat((x, trans), 1))

            dehaze = (x - (1 - refine_T) * atm) / (refine_T + self.eps) + hf
            # clean = self.relu(self.ref(dehaze, A))
            rec_I = dehaze * refine_T + (1 - refine_T) * atm
            return trans, atm, dehaze, rec_I

        else:
            Amean = self.predict_atm(x)
            return Amean

    def predict_atm(self, x):
        _, c, h, w = x.size()
        lf, hf = self.decomposition(x)
        sum_A = torch.zeros(1,3,1,1, dtype=torch.float32).cuda()
        count = 0
        ph = 256
        pw = 256
        rows = int(np.ceil(float(h)/float(ph)))
        cols = int(np.ceil(float(w)/float(pw)))
        for i in range(rows):
            for j in range(cols):
                ratey = 1
                ratex = 1
                if (i+1)*256 > h:
                    dy = h - 256
                    ratey = h - i*256
                else:
                    dy = i * 256
                if (j+1) * 256 > w:
                    dx = w - 256
                    ratex = w - j*256
                else:
                    dx = j * 256
                _, A = self.fognet(torch.cat([x, lf], dim=1), dx, dy)
                sum_A = sum_A + A * ratey * ratex
                count+=1 * ratey * ratex
        return sum_A/count

    def synthesize_fog(J, t, A=None):
            """
            Synthesize hazy image base on optical model
            I = J * t + A * (1 - t)
            """

            if A is None:
                A = 1

            return J * t + A * (1 - t)

    def decomposition(self, x):
        LF_list = []
        HF_list = []
        res = get_residue(x)
        res = res.repeat(1, 3, 1, 1)
        for radius in self.radiux:
            for eps in self.eps_list:
                self.gf = GuidedFilter(radius, eps)
                LF = self.gf(res, x)
                HF= x-LF
                LF_list.append(LF)
                HF_list.append(x - LF)
        LF1 = torch.cat(LF_list, dim=1)
        HF1 = torch.cat(HF_list, dim=1)
        return LF1, HF
class DCPDehazeGenerator(nn.Module):
    """Create a DCP Dehaze generator"""
    def __init__(self, win_size=5, r=15, eps=1e-3):
        super(DCPDehazeGenerator, self).__init__()

        self.guided_filter = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega = 0.95

    def get_dark_channel(self, img, neighborhood_size):
        shape = img.shape
        if len(shape) == 4:
            img_min,_ = torch.min(img, dim=1)

            padSize = np.int(np.floor(neighborhood_size/2))
            if neighborhood_size % 2 == 0:
                pads = [padSize, padSize-1 ,padSize ,padSize-1]
            else:
                pads = [padSize, padSize ,padSize ,padSize]

            img_min = F.pad(img_min, pads, mode='constant', value=1)
            dark_img = -F.max_pool2d(-img_min, kernel_size=neighborhood_size, stride=1)
        else:
            raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

        dark_img = torch.unsqueeze(dark_img, dim=1)
        return dark_img

    def atmospheric_light(self, img, dark_img):
        num,chl,height,width = img.shape
        topNum = np.int(0.01*height*width)

        A = torch.Tensor(num,chl,1,1)
        if img.is_cuda:
            A = A.cuda()

        for num_id in range(num):
            curImg = img[num_id,...]
            curDarkImg = dark_img[num_id,0,...]

            _, indices = curDarkImg.reshape([height*width]).sort(descending=True)
            #curMask = indices < topNum

            for chl_id in range(chl):
                imgSlice = curImg[chl_id,...].reshape([height*width])
                A[num_id,chl_id,0,0] = torch.mean(imgSlice[indices[0:topNum]])

        return A


    def forward(self, x):
        if x.shape[1] > 1:
            # rgb2gray
            guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
        else:
            guidance = x
        # rescale to [0,1]
        guidance = (guidance + 1)/2
        guidance = torch.unsqueeze(guidance, dim=1)
        imgPatch = (x + 1)/2

        num,chl,height,width = imgPatch.shape

        # dark_img and A with the range of [0,1]
        dark_img = self.get_dark_channel(imgPatch, self.neighborhood_size)
        A = self.atmospheric_light(imgPatch, dark_img)

        map_A = A.repeat(1,1,height,width)
        # make sure channel of trans_raw == 1
        trans_raw = 1 - self.omega*self.get_dark_channel(imgPatch/map_A, self.neighborhood_size)

        # get initial results
        T_DCP = self.guided_filter(guidance, trans_raw)
        return T_DCP

class DeRain_v2(nn.Module):
    def __init__(self):
        super(DeRain_v2, self).__init__()
        self.baseWidth = 12  # 4#16
        self.cardinality = 8  # 8#16
        self.scale = 6  # 4#5
        self.stride = 1
            ############# Block1-scale 1.0  ##############
        self.conv_input = nn.Conv2d(33, 16, 3, 1, 1)
        self.dense_block1 = Bottle2neckX(16, 16, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')

            ############# Block2-scale 0.50  ##############
        self.trans_block2 = TransitionBlock1(32, 32)
        self.dense_block2 = Bottle2neckX(32, 32, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block2_o = TransitionBlock3(64, 32)

            ############# Block3-scale 0.250  ##############
        self.trans_block3 = TransitionBlock1(32, 32)
        self.dense_block3 = Bottle2neckX(32, 32, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block3_o = TransitionBlock3(64, 64)

            ############# Block4-scale 0.25  ##############
        self.trans_block4 = TransitionBlock1(64, 128)
        self.dense_block4 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block4_o = TransitionBlock3(256, 128)

        ############# Block5-scale 0.25  ##############
        self.dense_block5 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block5_o = TransitionBlock3(256, 128)

            ############# Block6-scale 0.25  ##############
        self.dense_block6 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block6_o = TransitionBlock3(256, 128)

            ############# Block7-scale 0.25  ############## 7--3 skip connection
        self.trans_block7 = TransitionBlock(32, 64)
        self.dense_block7 = Bottle2neckX(128, 128, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block7_o = TransitionBlock3(256, 32)

            ############# Block8-scale 0.5  ############## 8--2 skip connection
        self.trans_block8 = TransitionBlock(32, 32)
        self.dense_block8 = Bottle2neckX(64, 64, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block8_o = TransitionBlock3(128, 32)

            ############# Block9-scale 1.0  ############## 9--1 skip connection
        self.trans_block9 = TransitionBlock(32, 32)
        self.dense_block9 = Bottle2neckX(80, 80, self.baseWidth, self.cardinality, self.stride, downsample=None,
                                             scale=self.scale, stype='normal')
        self.trans_block9_o = TransitionBlock3(160, 16)

        self.conv_refin = nn.Conv2d(16, 16, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.refine3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.zout = nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
            # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.refineclean2 = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)


    def forward(self, xin):
        x = self.conv_input(xin)
            # Size - 1.0
        x1 = (self.dense_block1(x))

            # Size - 0.5
        x2_i = self.trans_block2(x1)
        x2_i = self.dense_block2(x2_i)
        x2 = self.trans_block2_o(x2_i)

            # Size - 0.25
        x3_i = self.trans_block3(x2)
        x3_i = self.dense_block3(x3_i)
        x3 = self.trans_block3_o(x3_i)

            # Size - 0.125
        x4_i = self.trans_block4(x3)
        x4_i = self.dense_block4(x4_i)
        x4 = self.trans_block4_o(x4_i)

        x5_i = self.dense_block5(x4)
        x5 = self.trans_block5_o(x5_i)

        x6_i = self.dense_block6(x5)
        x6 = self.trans_block6_o(x6_i)
        z = self.zout(self.relu(x6))

        x7_i = self.trans_block7(z)
            # print(x4.size())
            # print(x7_i.size())
        x7_i = self.dense_block7(torch.cat([x7_i, x3], 1))
        x7 = self.trans_block7_o(x7_i)

        x8_i = self.trans_block8(x7)
        x8_i = self.dense_block8(torch.cat([x8_i, x2], 1))
        x8 = self.trans_block8_o(x8_i)

        x9_i = self.trans_block9(x8)
        x9_i = self.dense_block9(torch.cat([x9_i, x1, x], 1))
        x9 = self.trans_block9_o(x9_i)

        x11 = x - self.relu((self.conv_refin(x9)))
        residual = self.tanh(self.refine3(x11))
        clean = residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.sig(self.refineclean2(clean))
        return clean


# class Dense(nn.Module):
#     def __init__(self):
#         super(Dense, self).__init__()
#
#
#
#
#         ############# 256-256  ##############
#         haze_class = models.densenet121(pretrained=True)
#
#         self.conv0=haze_class.features.conv0
#         self.norm0=haze_class.features.norm0
#         self.relu0=haze_class.features.relu0
#         self.pool0=haze_class.features.pool0
#
#         ############# Block1-down 64-64  ##############
#         self.dense_block1=haze_class.features.denseblock1
#         self.trans_block1=haze_class.features.transition1
#
#         ############# Block2-down 32-32  ##############
#         self.dense_block2=haze_class.features.denseblock2
#         self.trans_block2=haze_class.features.transition2
#
#         ############# Block3-down  16-16 ##############
#         self.dense_block3=haze_class.features.denseblock3
#         self.trans_block3=haze_class.features.transition3
#
#         ############# Block4-up  8-8  ##############
#         self.dense_block4=BottleneckBlock(512,256)
#         self.trans_block4=TransitionBlock(768,128)
#
#         ############# Block5-up  16-16 ##############
#         self.dense_block5=BottleneckBlock(384,256)
#         self.trans_block5=TransitionBlock(640,128)
#
#         ############# Block6-up 32-32   ##############
#         self.dense_block6=BottleneckBlock(256,128)
#         self.trans_block6=TransitionBlock(384,64)
#
#
#         ############# Block7-up 64-64   ##############
#         self.dense_block7=BottleneckBlock(64,64)
#         self.trans_block7=TransitionBlock(128,32)
#
#         ## 128 X  128
#         ############# Block8-up c  ##############
#         self.dense_block8=BottleneckBlock(32,32)
#         self.trans_block8=TransitionBlock(64,16)
#
#         self.conv_refin=nn.Conv2d(19,20,3,1,1)
#         self.tanh=nn.Tanh()
#
#
#         self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
#         self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
#         self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
#         self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
#
#         self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)
#         # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)
#
#         self.upsample = F.upsample_nearest
#
#         self.relu=nn.LeakyReLU(0.2, inplace=True)
#
#     def forward(self, x):
#         ## 256x256
#         x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))
#
#         ## 64 X 64
#         x1=self.dense_block1(x0)
#         # print x1.size()
#         x1=self.trans_block1(x1)
#
#         ###  32x32
#         x2=self.trans_block2(self.dense_block2(x1))
#         # print  x2.size()
#
#
#         ### 16 X 16
#         x3=self.trans_block3(self.dense_block3(x2))
#
#         # x3=Variable(x3.data,requires_grad=True)
#
#         ## 8 X 8
#         x4=self.trans_block4(self.dense_block4(x3))
#
#         x42=torch.cat([x4,x2],1)
#         ## 16 X 16
#         x5=self.trans_block5(self.dense_block5(x42))
#
#         x52=torch.cat([x5,x1],1)
#         ##  32 X 32
#         x6=self.trans_block6(self.dense_block6(x52))
#
#         ##  64 X 64
#         x7=self.trans_block7(self.dense_block7(x6))
#
#         ##  128 X 128
#         x8=self.trans_block8(self.dense_block8(x7))
#
#         # print x8.size()
#         # print x.size()
#
#         x8=torch.cat([x8,x],1)
#
#         # print x8.size()
#
#         x9=self.relu(self.conv_refin(x8))
#
#         shape_out = x9.data.size()
#         # print(shape_out)
#         shape_out = shape_out[2:4]
#
#         x101 = F.avg_pool2d(x9, 32)
#         x102 = F.avg_pool2d(x9, 16)
#         x103 = F.avg_pool2d(x9, 8)
#         x104 = F.avg_pool2d(x9, 4)
#
#         x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
#         x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
#         x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
#         x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)
#
#         dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
#         dehaze = self.tanh(self.refine3(dehaze))
#
#         return dehaze


class FogNet(nn.Module):
    def __init__(self, input_nc):
        super(FogNet, self).__init__()
        self.atmconv1x1 = nn.Conv2d(input_nc, input_nc, kernel_size=1, stride=1, padding=0)
        self.atmnet = define_D(input_nc, 64, 'n_estimator', n_layers_D=5, norm='batch', use_sigmoid=True, gpu_ids=[])
        # self.transnet = TransUNet(input_nc, 1)
        self.htanh = nn.Hardtanh(0, 1)
        self.relu1 = ReLU1()
        self.relu = nn.ReLU()

    def forward(self, x, dx=0, dy=0):
        _, c, h, w = x.size()

        A = self.relu(self.atmnet(self.atmconv1x1(x[:, :, dy:dy+256, dx:dx+256])))
        # trans = self.relu(self.transnet(x))
        atm = A.repeat(1, 1, h, w)
        # trans = trans.repeat(1, 3, 1, 1)
        return atm, A


class TransUNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(TransUNet, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.inc = inconv(in_channel, 64)
        self.image_size = 256
        self.down1 = down(64, 128)  # 112x112 | 256x256 | 512x512
        self.down2 = down(128, 256)  # 56x56   | 128x128 | 256x256
        self.down3 = down(256, 512)  # 28x28   | 64x64  | 128x128
        self.down4 = down(512, 512)  # 14x14   | 32x32  | 64x64

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1x1(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # decoder
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8= self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x10 = self.relu(self.outc(x9))
        return x10


class GlobalLocalUnet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(GlobalLocalUnet, self).__init__()
        self.inc = inconv(in_channel, 64)
        self.image_size = 256
        self.down1 = down(64, 128)  # 112x112 | 256x256 | 512x512
        self.down2 = down(128, 256)  # 56x56   | 128x128 | 256x256
        self.down3 = down(256, 512)  # 28x28   | 64x64  | 128x128
        self.down4 = down(512, 512)  # 14x14   | 32x32  | 64x64
        self.down5 = down(512, 512)  # 32x32

        self.up1 = up(1024, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.up5 = up(128, 64)
        self.outc = outconv(64, n_classes)
        # for global
        self.gdown5 = down(512, 512)  # 14x14  | 32x32  | 64x64
        self.gdown6 = down(512, 512)  # 7x7x512   | 16x16x512  | 32x32x512
        self.digest = nn.Conv2d(1024, 512, 1, 1, 0)
        self.l1 = nn.Linear(self.image_size / 64 * self.image_size / 64 * 512, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # local branch
        lx5 = self.down5(x5)

        # global branch
        gx5 = self.gdown5(x5)
        gx6 = self.gdown6(gx5)
        gx6vec = gx6.view(gx6.size(0), -1)
        gx7vec = self.relu(self.l1(gx6vec))
        gx7conv = gx7vec.view(gx7vec.size(0), -1, 1, 1).repeat(1, 1, self.image_size / 32, self.image_size / 32)

        # decoder
        u0 = torch.cat((lx5, gx7conv), dim=1)
        u0 = self.relu(self.digest(u0))
        x = self.up1(u0, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.relu(self.outc(x))
        return x


class RefUNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(RefUNet, self).__init__()
        self.inc = inconv(in_channel, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024+32, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.leakyRelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.Aup1 = nn.ConvTranspose2d(3,16,2,stride=2)
        self.Aup2 = nn.ConvTranspose2d(16,32,2,stride=2)
        # self.Aup3 = up(32,64,bilinear=False)

    def forward(self, x, A):
        input_img = x
        _, c, h, w = x.size()
        A1 = A.repeat(1, 1, int(h/64), int(w/64))
        A2 = self.Aup1(A1)  # 4x4
        A3 = self.Aup2(A2)  # 8x8 A3=16x16
        # A4 = self.Aup3(A3) # 16x16 A4=16x16
        x1 = self.inc(x)
        x2 = self.down1(x1)  # 256x256
        x3 = self.down2(x2)  # 128x128
        x4 = self.down3(x3)  # 64x64
        x5 = self.down4(x4)  # 32x32  x5=16x16
        x5 = torch.cat([x5, A3], dim=1)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        out = self.relu(self.outc(x9))
        # out = x + input_img
        return out


class DepthGuidedD(nn.Module):
    def __init__(self, nc):
        super(DepthGuidedD, self).__init__()
        self.conv1 = nn.Conv2d(nc, 8, 5,1,2)
        self.conv2 = nn.Conv2d(8, 16, 5,1,2)
        self.conv3 = nn.Conv2d(16, 64, 5,1,2)
        self.conv4 = nn.Conv2d(64, 128,5,1,2)
        self.conv5 = nn.Conv2d(128,128,5,1,2)
        self.conv6 = nn.Conv2d(128,128,5,1,2)
        self.convdepth = nn.Conv2d(128,1,5,1,2)
        self.conv7 = nn.Conv2d(128,64,5,4,1)
        self.conv8 = nn.Conv2d(64,32,5,4,1)
        self.conv9 = nn.Conv2d(32, 16, 5, 4, 1)
        self.fc = nn.Linear(32*16*16,1024)
        self.fc2 = nn.Linear(1024,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        depth = self.convdepth(x)
        x = self.relu(self.conv7(x*depth))
        x = self.relu(self.conv8(x))
        #x = self.relu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        return depth, self.sigmoid(x)


    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
# -------------------------------------------------------------------------- #
#  Sub-Networks
# -------------------------------------------------------------------------- #
class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class Bottle2neckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist block of a new stage.
        """
        super(Bottle2neckX, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C * scale, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(D * C * scale)
        self.SE = SEBlock(inplanes, C)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False))
            bns.append(nn.BatchNorm2d(D * C))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(D * C * scale, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.width = D * C
        self.stype = stype
        self.scale = scale

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.relu(self.bns[i](sp))
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        # out = self.SE(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # pdb.set_trace()
        out += residual

        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock1, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock2(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock2, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,
                               padding=3, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

# class TransitionBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, dropRate=0.0):
#         super(TransitionBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.droprate = dropRate
#     def forward(self, x):
#         out = self.conv1(self.relu(self.bn1(x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)

class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch, dil=1, middle_planes=32):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=dil, dilation=dil)
        self.layer1 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.layer2 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.layer3 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.layer4 = ResidualBlockStraight(32, 32, dilation=1, last=False)
        self.conv_out = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_out(x)
        return x


class ResidualBlockStraight(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, last=False):
        super(ResidualBlockStraight, self).__init__()
        assert (in_channels == out_channels)
        self.conv1 = res_conv(in_channels, out_channels // 4, dil=dilation)
        self.conv2 = res_conv(out_channels // 4, out_channels // 4)
        self.conv3 = res_conv(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, last=True):
        super(ResidualBlockDown, self).__init__()
        self.conv1 = res_conv(in_channels, in_channels, dil=dilation)
        self.conv2 = res_conv(in_channels, in_channels // 2)
        self.conv3 = res_conv(in_channels // 2, in_channels // 4)
        self.conv_out = nn.Conv2d(in_channels + in_channels // 4, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = torch.cat((out, residual), dim=1)
        out = self.relu(out)
        out = self.conv_out(out)
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super(ResidualBlockUp, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, stride=1, padding=2)
        self.conv1 = res_conv(out_channels // 4, out_channels // 4)
        self.conv2 = res_conv(out_channels // 4, out_channels // 4)
        self.conv3 = res_conv(out_channels // 4, out_channels)
        self.conv_in = nn.Conv2d(out_channels // 4, out_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.last = last

    def forward(self, x):
        x = self.relu(self.conv0(x))
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        residual = self.conv_in(residual)
        out = out + residual
        if self.last:
            out = self.tanh(out)
        else:
            out = self.relu(out)
        return out
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

# ---------------------------------------------------------------------------- #
# Sub-Modules
# ---------------------------------------------------------------------------- #


class ReLU1(Hardtanh):
    def __init__(self, inplace=False):
        super(ReLU1, self).__init__(0, 1, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


def res_conv(in_ch, out_ch, k=3, s=1, dil=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, dilation=dil, padding=1, bias=bias)


class SimpUnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(SimpUnet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            # bn = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            # bn = True if i> 0 and i < depth-2 else False
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x


class RainUnet(nn.Module):
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        super(RainUnet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.ConvRefine = nn.Conv2d(outs, 32, 3, 1, 1)
        self.RefineClean1 = nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0)
        self.RefineClean2 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.RefineClean3 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1)
        self.LRelu = nn.LeakyReLU(0.2, inplace=True)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        input_img = x
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.

        x9 = self.LRelu((self.ConvRefine(x)))

        residual = F.tanh(self.RefineClean3(x9))
        clean1 = self.LRelu(self.RefineClean1(residual))
        clean2 = F.tanh(self.RefineClean2(clean1))
        return clean2


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
#  Network Architecture
# -------------------------------------------------------------------------- #
def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, dropout_rate=0.5):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x

        if self.pooling:
            x = self.pool(x)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)

        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose', dropout_rate=0.5):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.dropout_rate = dropout_rate

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        # layer 1
        x = F.relu(self.conv1(x))
        # layer 2
        x = F.relu(self.conv2(x))
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate)

        return x

# ============================================================================
