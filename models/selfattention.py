'''
Function:
    Implementation of SelfAttentionBlock
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
'''self attention block'''
#计算自注意力机制，计算两个特征图之间的相似性
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query,
                 query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm,
                 value_out_norm, matmul_norm, with_out_project, **kwargs):
        super(SelfAttentionBlock, self).__init__()
        norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        # key project
        self.key_project = self.buildproject( # 使用两个卷积，将通道从1024 --> 256
            in_channels=key_in_channels,# 1024
            out_channels=transform_channels, # 256
            num_convs=key_query_num_convs, # 2
            use_norm=key_query_norm, # True
            norm_cfg=norm_cfg,# 异步的批处理正则化
            act_cfg=act_cfg, # 激活函数
        )
        # query project
        if share_key_query: #False
            assert key_in_channels == query_in_channels
            self.query_project = self.key_project
        else:
            self.query_project = self.buildproject( # 使用两个卷积将特征从512 --> 256
                in_channels=query_in_channels, #512
                out_channels=transform_channels,#256
                num_convs=key_query_num_convs,#2
                use_norm=key_query_norm,#True
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        # value project
        self.value_project = self.buildproject( # 使用1个卷积将特征从1024 -->256
            in_channels=key_in_channels, # 1024
            out_channels=transform_channels if with_out_project else out_channels, # 256
            num_convs=value_out_num_convs, # 1
            use_norm=value_out_norm, #True
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        # out project
        self.out_project = None
        if with_out_project: # True
            self.out_project = self.buildproject( #使用一个卷积将通道从256 --> 512
                in_channels=transform_channels,
                out_channels=out_channels,
                num_convs=value_out_num_convs, # 1
                use_norm=value_out_norm, #使用 norm
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        # downsample
        self.query_downsample = query_downsample # none
        self.key_downsample = key_downsample # none
        self.matmul_norm = matmul_norm # True
        self.transform_channels = transform_channels # 256

    '''forward'''
    def forward(self, query_feats, key_feats):
        batch_size = query_feats.size(0)#原来的特征向量
        query = self.query_project(query_feats) #原来的x作为特征向量，维度从512--->256,得到查询的256维度特征
        if self.query_downsample is not None: query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)  # 查询
        query = query.permute(0, 2, 1).contiguous()
        key = self.key_project(key_feats) #将全局信息和原来的特征拼接在一起作为键 维度从1024 ---> 256
        value = self.value_project(key_feats) #value就是对应的值， 维度是从1024 --> 256
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous() #进行了转置的操作
        sim_map = torch.matmul(query, key) # 将query和key进行乘积即可得到
        if self.matmul_norm:
            sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # 这一步得到了图像级别特征的相似性矩阵
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:]) #将语义信息的尺寸变成了 256 *(H/8)*(W/8)
        if self.out_project is not None:
            context = self.out_project(context) #尺度 从 256 --> 512
        return context
    '''build project'''
    def buildproject(self, in_channels, out_channels, num_convs, use_norm, norm_cfg, act_cfg):
        if use_norm:
            convs = [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.SyncBatchNorm(out_channels),
                    nn.ReLU(inplace=True),
                )
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.SyncBatchNorm(out_channels),
                        nn.ReLU(inplace=True),
                    )
                )
        else:
            convs = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]