import numpy as np
import torch
import torch.nn as nn
from transformer.vit import ViT

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.activation = nn.LeakyReLU(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    def forward(self, x,):
        x = self.upsample(x)
        return x

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        # num_levels = len(layer_nums)
        # c_in_list = [input_channels, *num_filters[:-1]]
        # self.blocks = nn.ModuleList()
        # self.deblocks = nn.ModuleList()
        # for idx in range(num_levels):
        #     cur_layers = [
        #         nn.ZeroPad2d(1),
        #         nn.Conv2d(
        #             c_in_list[idx], num_filters[idx], kernel_size=3,
        #             stride=layer_strides[idx], padding=0, bias=False
        #         ),
        #         nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #         nn.ReLU()
        #     ]
        #     for k in range(layer_nums[idx]):
        #         cur_layers.extend([
        #             nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
        #             nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
        #             nn.ReLU()
        #         ])
        #     self.blocks.append(nn.Sequential(*cur_layers))
        #     if len(upsample_strides) > 0:
        #         stride = upsample_strides[idx]
        #         if stride >= 1:
        #             self.deblocks.append(nn.Sequential(
        #                 nn.ConvTranspose2d(
        #                     num_filters[idx], num_upsample_filters[idx],
        #                     upsample_strides[idx],
        #                     stride=upsample_strides[idx], bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #         else:
        #             stride = np.round(1 / stride).astype(np.int)
        #             self.deblocks.append(nn.Sequential(
        #                 nn.Conv2d(
        #                     num_filters[idx], num_upsample_filters[idx],
        #                     stride,
        #                     stride=stride, bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))

        c_in = sum(num_upsample_filters)
        # if len(upsample_strides) > num_levels:
        #     self.deblocks.append(nn.Sequential(
        #         nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
        #         nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
        #         nn.ReLU(),
        #     ))

        self.num_bev_features = c_in
        # # ##began
        # # self.v = ViT(
        # #     image_size = 256,
        # #     patch_size = 32,
        # #     num_classes = 1000,
        # #     dim = 1024,
        # #     depth = 6,
        # #     heads = 16,
        # #     mlp_dim = 2048,
        # #     dropout = 0.1,
        # #     emb_dropout = 0.1
        # # )
        # # ##end
        # self.v = ViT(496,432,4)
        # begin res_block_1
        self.conv1_1 = BasicConv(64, 64, 3)
        self.conv1_2 = BasicConv(32, 32, 3)
        self.conv1_3 = BasicConv(32, 32, 3)
        self.conv1_4 = BasicConv(64, 64, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])
        self.conv1_5 = BasicConv(128, 192, 1)
        # end res_block_1
        # begin res_block_2
        self.conv2_1 = BasicConv(128, 128, 3)
        self.conv2_2 = BasicConv(64, 64, 3)
        self.conv2_3 = BasicConv(64, 64, 3)
        self.conv2_4 = BasicConv(128, 128, 1)
        self.maxpool_1 = nn.MaxPool2d([2, 2], [2, 2])
        # end res_block_2
        # begin res_block_3
        self.conv3_1 = BasicConv(256, 256, 3)
        self.conv3_2 = BasicConv(128, 128, 3)
        self.conv3_3 = BasicConv(128, 128, 3)
        self.conv3_4 = BasicConv(256, 256, 1)
        # self.maxpool = nn.MaxPool2d([2, 2], [2, 2])
        # end res_block_3
        self.upsample = Upsample(512, 192)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        print("input shape",x.shape)


        # # self.v(x)
        # # ##began
        # # print("OpenPCDet/pcdet/models/backbones_2d/base_bev_backbone.py",x.shape)
        # # img = torch.randn(1, 3, 256, 256)
        # # preds = self.v(img) # (1, 1000)
        # # print(preds.shape)
        # # ##end
        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #     stride = int(spatial_features.shape[2] / x.shape[2])
        #     ret_dict['spatial_features_%dx' % stride] = x
        #     if len(self.deblocks) > 0:
        #         ups.append(self.deblocks[i](x))
        #     else:
        #         ups.append(x)

        # if len(ups) > 1:
        #     x = torch.cat(ups, dim=1)
        # elif len(ups) == 1:
        #     x = ups[0]

        # if len(self.deblocks) > len(self.blocks):
        #     x = self.deblocks[-1](x)

        x = self.conv1_1(x)
        route = x
        x = torch.split(x, 32, dim=1)[1]
        x = self.conv1_2(x)
        route1 = x
        x = self.conv1_3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv1_4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.maxpool(x)
        feat1 = x
        feat12 = self.conv1_5(feat1)
        x = self.conv2_1(x)
        route = x
        x = torch.split(x, 64, dim=1)[1]
        x = self.conv2_2(x)
        route1 = x
        x = self.conv2_3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv2_4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.maxpool_1(x)
        x = self.conv3_1(x)
        route = x
        x = torch.split(x, 128, dim=1)[1]
        x = self.conv3_2(x)
        route1 = x
        x = self.conv3_3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv3_4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, feat12], axis=1)


        
        print("output shape",x.shape)

        data_dict['spatial_features_2d'] = x

        return data_dict
