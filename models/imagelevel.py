'''
Function:
    Implementation of ImageLevelContext
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .selfattention import SelfAttentionBlock
# from ...backbones import BuildActivation, BuildNormalization
'''image-level context module'''
class ImageLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False, **kwargs): #feats_channels为512,transform_channels为256
        super(ImageLevelContext, self).__init__()
        norm_cfg, act_cfg, self.align_corners = kwargs['norm_cfg'], kwargs['act_cfg'], kwargs['align_corners']
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1)) #自适应全局池化到 (1,1)
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2, #1024
            query_in_channels=feats_channels,
            transform_channels=transform_channels,#256
            out_channels=feats_channels,#输出还是512维度
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if concat_input: # 为 True
            self.bottleneck = nn.Sequential(
                nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(feats_channels),
                nn.ReLU(inplace=True),
            )
    '''forward'''
    def forward(self, x):
        x_global = self.global_avgpool(x) # 将进行全局池化，得到全局的特征
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='bilinear', align_corners=self.align_corners) #然后进行上采样，这里不需要角像素点的中心对齐
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1)) # 先将全局特征与原来的特征进行拼接，然后计算原来的特征与拼接之后的特征的相似性。
        # feats_il是 512维度的
        if hasattr(self, 'bottleneck'):#将x与产生的全局的语义信息进行拼接，然后卷积之后输出
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il #输出的尺寸是1024