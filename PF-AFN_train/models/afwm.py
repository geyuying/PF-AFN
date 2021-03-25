import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions
from .correlation import correlation # the custom cost volume layer
opt = TrainOptions().parse()

def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...]
        for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0
        for grid, size in zip(grid_list, reversed(sizes))] 

    return torch.stack(grid_list, dim=-1)


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))


# backbone 
class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x) + x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.block=  nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)



class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, chns=[64,128,256,256,256]):
        # in_channels = 3 for images, and is larger (e.g., 17+1+1) for agnositc representation
        super(FeatureEncoder, self).__init__()
        self.encoders = []
        for i, out_chns in enumerate(chns):
            if i == 0:
                encoder = nn.Sequential(DownSample(in_channels, out_chns),
                                        ResBlock(out_chns),
                                        ResBlock(out_chns))
            else:
                encoder = nn.Sequential(DownSample(chns[i-1], out_chns),
                                         ResBlock(out_chns),
                                         ResBlock(out_chns))
            
            self.encoders.append(encoder)

        self.encoders = nn.ModuleList(self.encoders)


    def forward(self, x):
        encoder_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_features.append(x)
        return encoder_features

class RefinePyramid(nn.Module):
    def __init__(self, chns=[64,128,256,256,256], fpn_dim=256):
        super(RefinePyramid, self).__init__()
        self.chns = chns

        # adaptive 
        self.adaptive = []
        for in_chns in list(reversed(chns)):
            adaptive_layer = nn.Conv2d(in_chns, fpn_dim, kernel_size=1)
            self.adaptive.append(adaptive_layer)
        self.adaptive = nn.ModuleList(self.adaptive)
        # output conv
        self.smooth = []
        for i in range(len(chns)):
            smooth_layer = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.smooth.append(smooth_layer)
        self.smooth = nn.ModuleList(self.smooth)

    def forward(self, x):
        conv_ftr_list = x
        
        feature_list = []
        last_feature = None
        for i, conv_ftr in enumerate(list(reversed(conv_ftr_list))):
            # adaptive
            feature = self.adaptive[i](conv_ftr)
            # fuse
            if last_feature is not None:
                feature = feature + F.interpolate(last_feature, scale_factor=2, mode='nearest')
            # smooth
            feature = self.smooth[i](feature)
            last_feature = feature
            feature_list.append(feature)

        return tuple(reversed(feature_list))


class AFlowNet(nn.Module):
    def __init__(self, num_pyramid, fpn_dim=256):
        super(AFlowNet, self).__init__()
        self.netMain = []
        self.netRefine = []
        for i in range(num_pyramid):
            netMain_layer = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )

            netRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(2 * fpn_dim, out_channels=128, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
            )
            self.netMain.append(netMain_layer)
            self.netRefine.append(netRefine_layer)

        self.netMain = nn.ModuleList(self.netMain)
        self.netRefine = nn.ModuleList(self.netRefine)


    def forward(self, x, x_edge, x_warps, x_conds, warp_feature=True):
        last_flow = None
        last_flow_all = []
        delta_list = []
        x_all = []
        x_edge_all = []
        cond_fea_all = []
        delta_x_all = []
        delta_y_all = []
        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3,2,0,1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        for i in range(len(x_warps)):
              x_warp = x_warps[len(x_warps) - 1 - i]
              x_cond = x_conds[len(x_warps) - 1 - i]
              cond_fea_all.append(x_cond)

              if last_flow is not None and warp_feature:
                  x_warp_after = F.grid_sample(x_warp, last_flow.detach().permute(0, 2, 3, 1),
                       mode='bilinear', padding_mode='border')
              else:
                  x_warp_after = x_warp

              tenCorrelation = F.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=x_warp_after, tenSecond=x_cond, intStride=1), negative_slope=0.1, inplace=False)
              flow = self.netMain[i](tenCorrelation)
              delta_list.append(flow)
              flow = apply_offset(flow)
              if last_flow is not None:
                  flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')
              else:
                  flow = flow.permute(0, 3, 1, 2)

              last_flow = flow
              x_warp = F.grid_sample(x_warp, flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
              concat = torch.cat([x_warp,x_cond],1)
              flow = self.netRefine[i](concat)
              delta_list.append(flow)
              flow = apply_offset(flow)
              flow = F.grid_sample(last_flow, flow, mode='bilinear', padding_mode='border')

              last_flow = F.interpolate(flow, scale_factor=2, mode='bilinear')
              last_flow_all.append(last_flow)
              cur_x = F.interpolate(x, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
              cur_x_warp = F.grid_sample(cur_x, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='border')
              x_all.append(cur_x_warp)
              cur_x_edge = F.interpolate(x_edge, scale_factor=0.5**(len(x_warps)-1-i), mode='bilinear')
              cur_x_warp_edge = F.grid_sample(cur_x_edge, last_flow.permute(0, 2, 3, 1),mode='bilinear', padding_mode='zeros')
              x_edge_all.append(cur_x_warp_edge)
              flow_x,flow_y = torch.split(last_flow,1,dim=1)
              delta_x = F.conv2d(flow_x, self.weight)
              delta_y = F.conv2d(flow_y,self.weight)
              delta_x_all.append(delta_x)
              delta_y_all.append(delta_y)

        x_warp = F.grid_sample(x, last_flow.permute(0, 2, 3, 1),
                     mode='bilinear', padding_mode='border')
        return x_warp, last_flow, cond_fea_all, last_flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all


class AFWM(nn.Module):

    def __init__(self, opt, input_nc):
        super(AFWM, self).__init__()
        num_filters = [64,128,256,256,256]
        self.image_features = FeatureEncoder(3, num_filters) 
        self.cond_features = FeatureEncoder(input_nc, num_filters)
        self.image_FPN = RefinePyramid(num_filters)
        self.cond_FPN = RefinePyramid(num_filters)
        self.aflow_net = AFlowNet(len(num_filters))
        self.old_lr = opt.lr
        self.old_lr_warp = opt.lr*0.2

    def forward(self, cond_input, image_input, image_edge):
        cond_pyramids = self.cond_FPN(self.cond_features(cond_input)) # maybe use nn.Sequential
        image_pyramids = self.image_FPN(self.image_features(image_input))

        x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = self.aflow_net(image_input, image_edge, image_pyramids, cond_pyramids)

        return x_warp, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all


    def update_learning_rate(self,optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self,optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr

