'''
Function:
    Implementation of SemanticLevelContext
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .selfattention import SelfAttentionBlock


'''semantic-level context module'''
class SemanticLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=False, **kwargs):# 512 256 True
        super(SemanticLevelContext, self).__init__()
        norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels, # 512维度
            query_in_channels=feats_channels, # 512维度
            transform_channels=transform_channels, #256维度
            out_channels=feats_channels, #512维度
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
        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.SyncBatchNorm(feats_channels),
                nn.ReLU(inplace=True),
            )
    '''forward'''
    def forward(self, x, preds, feats_il):# x是backbone输出的高维度的特征, preds是 x通过解码之后产生的概率分布,feats_il是将原来的特征图与图像语义级别的图进行堆叠
        inputs = x #输入的特征
        batch_size, num_channels, h, w = x.size() #维度 是 512
        num_classes = preds.size(1) #获得类的个数
        feats_sl = torch.zeros(batch_size, h*w, num_channels).type_as(x) #将前者的数据类型转换为x的数据类型
        for batch_idx in range(batch_size):
            # (C, H, W), (num_classes, H, W) --> (H*W, C), (H*W, num_classes)
            feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
            feats_iter, preds_iter = feats_iter.reshape(num_channels, -1), preds_iter.reshape(num_classes, -1)
            feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
            # (H*W, )
            argmax = preds_iter.argmax(1) #获得每个点的类别agrmax的尺寸是 (H*W) * 1  表示的是哪个点属于哪个类
            for clsid in range(num_classes):
                mask = (argmax == clsid)
                if mask.sum() == 0: continue #如果当前的类没有像素点 则接着下一个类的操作
                feats_iter_cls = feats_iter[mask] #获取这些特定类别点的特征
                preds_iter_cls = preds_iter[:, clsid][mask] #获得这些点的概率分布的大小
                weight = F.softmax(preds_iter_cls, dim=0)
                feats_iter_cls = feats_iter_cls * weight.unsqueeze(-1) #扩充了一维度
                feats_iter_cls = feats_iter_cls.sum(0) #获得总的语义信息
                feats_sl[batch_idx][mask] = feats_iter_cls
        feats_sl = feats_sl.reshape(batch_size, h, w, num_channels)
        feats_sl = feats_sl.permute(0, 3, 1, 2).contiguous()
        feats_sl = self.correlate_net(inputs, feats_sl)# 输入是512维度，feats_sl是512维度，输出也是512维度的，计算注意力机制
        if hasattr(self, 'bottleneck'):
            feats_sl = self.bottleneck(torch.cat([feats_il, feats_sl], dim=1)) #对图像进行堆叠
        return feats_sl #返回语义的信息