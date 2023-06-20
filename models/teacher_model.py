import math, time
from itertools import chain
from urllib.request import proxy_bypass
import torch
import copy
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.helpers import set_trainable
from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
from utils.losses import CE_loss
import torch.distributed as dist
from models.semanticlevel import SemanticLevelContext
from models.imagelevel import ImageLevelContext
from models.backbones.resnet38_SEAM import Net
class MMFA_Teacher(BaseModel):
    def __init__(self, num_classes, arg, conf, train_logger,sup_loss=None, cons_w_unsup=None, ignore_index=None, testing=False,
            pretrained=True, use_weak_lables=False, weakly_loss_w=0.4):
        self.train_logger = train_logger
        self.arg = arg
        if not testing:
            #断言宣称语句，如果下面的条件有一个不对，会出现异常抛出
            assert (ignore_index is not None) and (sup_loss is not None) and (cons_w_unsup is not None)

        super(MMFA_Teacher, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        else:
            self.mode = 'semi' #半监督学习

        if arg.local_rank==0:
            self.train_logger.info("Now the %s is starting"%(self.mode))

        # Supervised and unsupervised losses
        self.ignore_index = ignore_index
        if conf['un_loss'] == "KL":
            self.unsuper_loss = softmax_kl_loss
        elif conf['un_loss'] == "MSE": #选用均方误差
            self.unsuper_loss = softmax_mse_loss
        elif conf['un_loss'] == "JS":
            self.unsuper_loss = softmax_js_loss
        else:
            raise ValueError(f"Invalid supervised loss {conf['un_loss']}")

        self.unsup_loss_w = cons_w_unsup #表示的是非监督学习损失的权重，服从一个正态分布
        self.sup_loss_w = conf['supervised_w']#表示的是监督学习损失的权重
        self.softmax_temp = conf['softmax_temp'] # 1
        self.sup_loss = sup_loss  #表示的是监督学习的损失函数，即CE
        self.sup_type = conf['sup_loss'] #监督的损失函数类型

        # Use weak labels
        self.use_weak_lables = use_weak_lables #是否使用弱监督的图片
        self.weakly_loss_w = weakly_loss_w  #弱监督学习的权重是0.4
        # pair wise loss (sup mat)
        self.aux_constraint = conf['aux_constraint']
        self.aux_constraint_w = conf['aux_constraint_w'] # 1
        # confidence masking (sup mat)
        self.confidence_th = conf['confidence_th'] #表示的是设置置信图，
        self.confidence_masking = conf['confidence_masking'] # True


        # Create the model
        self.encoder = Encoder(pretrained=pretrained) #设置编码器

        #########
        norm_cfg={'type': 'syncbatchnorm', 'opts': {}}
        act_cfg={'type': 'relu', 'opts': {'inplace': True}}
        # build image-level context module
        ilc_cfg = {
            'feats_channels': 512,  # 512
            'transform_channels': 256,  # 256
            'concat_input':True,  # True
            'norm_cfg': copy.deepcopy(norm_cfg),  # 批处理正则化的类型
            'act_cfg': copy.deepcopy(act_cfg),  # 激活函数的类型
            'align_corners': False  # False
        }
        self.ilc_net = ImageLevelContext(**ilc_cfg)  # **表示的是将字典中的键和值拿出来，ImageLevelContext有对应的键与ilc_cfg对应，从而将值给它
        # build semantic-level context module
        slc_cfg = {
            'feats_channels': 512,  # 512
            'transform_channels': 256,  # 256
            'concat_input': True,  # True
            'norm_cfg': copy.deepcopy(norm_cfg),  # 批处理正则化的类型
            'act_cfg': copy.deepcopy(act_cfg),  # 激活函数的类型
        }
        self.slc_net = SemanticLevelContext(**slc_cfg)  # 建立语义级别信息的网络
        decoder_cfg = {
                 'in_channels': 512,
                 'out_channels': 512,
                 'dropout': 0.1,
        }
        self.decoder_stage1 = nn.Sequential(  # in_channels输入是512 out_channels输出也是512
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.SyncBatchNorm(decoder_cfg['out_channels']),
            nn.ReLU(inplace=True),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], 21, kernel_size=1, stride=1, padding=0)  ### 数据集的类别的个数一定要注意
        )


        # The main encoder
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        self.main_decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes) # 主解码器

        # The auxilary decoders
        if self.mode == 'semi' or self.mode == 'weakly_semi':
            vat_decoder = [VATDecoder(upscale, decoder_in_ch, num_classes, xi=conf['xi'],
            							eps=conf['eps']) for _ in range(conf['vat'])]
            drop_decoder = [DropOutDecoder(upscale, decoder_in_ch, num_classes,
            							drop_rate=conf['drop_rate'], spatial_dropout=conf['spatial'])
            							for _ in range(conf['drop'])]
            cut_decoder = [CutOutDecoder(upscale, decoder_in_ch, num_classes, erase=conf['erase'])
            							for _ in range(conf['cutout'])]
            context_m_decoder = [ContextMaskingDecoder(upscale, decoder_in_ch, num_classes)
            							for _ in range(conf['context_masking'])]
            object_masking = [ObjectMaskingDecoder(upscale, decoder_in_ch, num_classes)
            							for _ in range(conf['object_masking'])]
            feature_drop = [FeatureDropDecoder(upscale, decoder_in_ch, num_classes)
            							for _ in range(conf['feature_drop'])]
            feature_noise = [FeatureNoiseDecoder(upscale, decoder_in_ch, num_classes,
            							uniform_range=conf['uniform_range'])
            							for _ in range(conf['feature_noise'])]

            self.aux_decoders = nn.ModuleList([*vat_decoder, *drop_decoder, *cut_decoder,
                                    *context_m_decoder, *object_masking, *feature_drop, *feature_noise])

    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, curr_iter=None, epoch=None,mode='train'):
        input_size = (x_ul.size(2), x_ul.size(3))
        pred_out_ul={}
        feats_ul = self.encoder(x_ul)
        pred_mi_ul ,pred_ul = self.main_decoder(feats_ul,probb=True)
        
        pred_out_ul['feat'] = feats_ul
        pred_out_ul["pred"]=pred_mi_ul

        if x_l!=None:
            pred_out_l={}
            feats_l = self.encoder(x_l,select=True)
            pred_out_l['feat'] = feats_l
            
            ##############
            # 在这个后面加语义模块和全局模块
            feats_il = self.ilc_net(feats_l)  # 输入特征是512维度的，输出也是512维度的
            preds_stage1 = self.decoder_stage1(feats_l)  # 进行解码操作，得到对应的概率图 512  是对 backbone产生的高维特征进行
            preds = preds_stage1
            if preds_stage1.size()[2:] != feats_l.size()[2:]:
                preds = F.interpolate(preds_stage1, size=feats_l.size()[2:], mode='bilinear', align_corners=False)
            # 语义类别的信息也是512维度的
            feats_sl = self.slc_net(feats_l, preds, feats_il) #输入backbone产生的最后一维特征feats, preds是预测出来的概率分布，feats_il是图像级别的语义信息和原来的特征图的拼接
            # preds_stage2 = self.decoder_stage1(feats_sl)  # 第二阶段的解码, 预测出第二个阶段的结果，feat_sl是增强后的结果。
            output_l = self.main_decoder(feats_l) #先由编码器,然后再经过主解析器，产生结果
            pred_mi_l,output_l_sl= self.main_decoder(feats_sl,probb = True)  # 先由编码器,然后再经过主解析器，产生结果
        

            if output_l.shape != x_l.shape:
                output_l = F.interpolate(output_l, size=input_size, mode='bilinear', align_corners=True)

            if output_l_sl.shape != x_l.shape:#整合语义之后产生的特征图
                output_l_sl = F.interpolate(output_l_sl, size=input_size, mode='bilinear', align_corners=True)
            pred_out_l["pred"] = pred_mi_l

            return pred_out_l,pred_out_ul

        


        return pred_out_ul




        
