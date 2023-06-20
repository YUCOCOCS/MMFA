import os
import requests
import datetime
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import numpy as np
import math
import PIL
import cv2 
from matplotlib import colors
from matplotlib import pyplot as plt
import matplotlib.cm as cmx
from utils import pallete
import torch.distributed as dist
from torch.nn import functional as F


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def dir_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
                    palette.append(0)
    palette[-3:] = [255, 255, 255]
    new_mask = PIL.Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b

def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c)>0:
        for l in c:
            apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0 #全部设置为0
    return mask.long()

def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][
        : len(labels) // 2
    ]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()

def generate_unsup_data(data, target, logits, mode="cutout"):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        if mode == "cutout":
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
            target[i][(1 - mix_mask).bool()] = 255

            new_data.append((data[i] * mix_mask).unsqueeze(0))
            new_target.append(target[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue

        if mode == "cutmix": #使用的是这种的cutmix进行增强
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device) #产生了掩码
        if mode == "classmix":
            mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append(
            (
                data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_target.append(
            (
                target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )
        new_logits.append(
            (
                logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)
            ).unsqueeze(0)
        )

    new_data, new_target, new_logits = (
        torch.cat(new_data),
        torch.cat(new_target),
        torch.cat(new_logits),
    )
    return new_data, new_target.long(), new_logits


def compute_unsupervised_loss(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        # drop pixels with high entropy
        prob = torch.softmax(pred_teacher, dim=1) #先对教师级别的网络得到对应的概率
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1) # 计算的熵

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        ) #选择对应的阈值
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool() #大于阈值的全部去掉

        target[thresh_mask] = 255 #大于阈值的，对应的标签设置为255，忽略掉这方面产生的因素
        weight = batch_size * h * w / torch.sum(target != 255) #这是无标记损失的权值

    loss = weight * F.cross_entropy(predict, target, ignore_index=255)  # [10, 321, 321]

    return loss

@torch.no_grad()
def gather_together(data):
    dist.barrier()

    world_size = dist.get_world_size()
    gather_data = [None for _ in range(world_size)]
    dist.all_gather_object(gather_data, data)

    return gather_data


def label_onehot(inputs, num_segments): # 输入的是标签图 ,
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    # scatter_方法  第一个0表示的是按照第0维度将数据插进去，inputs_temp.unsqueeze(1)记录的是对应的index,按照这些index将数据插到outputs中
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)

@torch.no_grad()
def dequeue_and_enqueue(keys, queue, queue_ptr, queue_size):
    # gather keys before updating queue
    keys = keys.detach().clone().cpu()
    gathered_list = gather_together(keys) #这里用于获得分布式的数据,返回一个列表,
    keys = torch.cat(gathered_list, dim=0).cuda()

    batch_size = keys.shape[0]

    ptr = int(queue_ptr)

    queue[0] = torch.cat((queue[0], keys.cpu()), dim=0) # 加到内存里面,
    if queue[0].shape[0] >= queue_size: #如果超出内存了, 就前面的一部分出队列,
        queue[0] = queue[0][-queue_size:, :]
        ptr = queue_size
    else:
        ptr = (ptr + batch_size) % queue_size  # move pointer

    queue_ptr[0] = ptr

    return batch_size

def compute_contra_memobank_loss(rep,label_l,label_u,prob_l,prob_u,low_mask,high_mask,cfg,memobank,queue_prtlis,queue_size,rep_teacher,momentum_prototype=None,i_iter=0):
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    current_class_threshold = cfg["current_class_threshold"] #0.3  这是用来选择锚框像素的
    current_class_negative_threshold = cfg["current_class_negative_threshold"] # 1
    low_rank, high_rank = cfg["low_rank"], cfg["high_rank"]# 3,20
    temp = cfg["temperature"] # 0.5
    num_queries = cfg["num_queries"] # 256
    num_negatives = cfg["num_negatives"] # 50

    num_feat = rep.shape[1]
    num_labeled = label_l.shape[0]
    num_segments = label_l.shape[1]

    # 对于有标签的数据，我们选择除去值为255的，而对应无标记的标签，我们根据阈值进行选择
    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask #低阈值的像素筛选出来, 尺寸是 2B * h * w,
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask #高阈值的像素，不可靠的像素值

    rep = rep.permute(0, 2, 3, 1) #这个是学生网络产生的像素表示 转换成 B * h * w * c
    rep_teacher = rep_teacher.permute(0, 2, 3, 1) #这个是教师网络产生的像素特征表示

    seg_feat_all_list = [] #存储的是分割的特征
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class
    seg_proto_list = []  # the center of each class

    # torch.sort() 返回的是两个值，第一个值是排序之后的tensor值 ,第二个值是排序结果之后的下标索引
    _, prob_indices_l = torch.sort(prob_l, 1, True) #对有标签元素的概率进行排序
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)  # (num_unlabeled, h, w, num_cls)

    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    valid_classes = []
    new_keys = []

    ## 锚框是由
    for i in range(num_segments): #分割的类别种类
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        prob_seg = prob[:, i, :, :] #获得对应类的分割概率
        rep_mask_low_entropy = (prob_seg > current_class_threshold) * low_valid_pixel_seg.bool() #符合条件的像素点的表示
        rep_mask_high_entropy = (prob_seg < current_class_negative_threshold) * high_valid_pixel_seg.bool()

        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy]) # 选择这个候选的第i类的锚像素, 锚框的像素是由学生模型产生的

        # positive sample: center of the class  由老师模型产生的正样本，取每个类的锚框的像素表示的中心点
        seg_proto_list.append(torch.mean(rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True)) #每个类的像素的平均表示的中心

        # generate class mask for unlabeled data
        # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
        class_mask_u = torch.sum(prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3).bool()

        # generate class mask for labeled data
        # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0)
        # prob_i_classes = prob_indices_l[label_l_mask]
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat((class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0)

        negative_mask = rep_mask_high_entropy * class_mask  #分别从有标记的数据和无标记的数据中获得负样本对

        keys = rep_teacher[negative_mask].detach()#这是对应的负样本的像素表示，由教师模型产生
        new_keys.append(
            #这里是把 教师级网络产生的负样本的像素表示加入到内存库里面
            dequeue_and_enqueue(# 对每一个batch中的类的属性进行入库操作，如果满了，则出库操作。返回一个batchsizes,
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )

        if low_valid_pixel_seg.sum() > 0: # 存在 满足条件的有效的像素
            seg_num_list.append(int(low_valid_pixel_seg.sum().item())) # seg_num_list存储的是各个类别的有效的像素的个数
            valid_classes.append(i)

    if (len(seg_num_list) <= 1):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum()
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum()
    else:
        reco_loss = torch.tensor(0.0).cuda()
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]  存储每个类的正样本 (每个类锚框的中心)
        valid_seg = len(seg_num_list)  # number of valid classes

        prototype = torch.zeros((prob_indices_l.shape[-1], num_queries, 1, num_feat)).cuda()

        for i in range(valid_seg):
        ##  判断在对应的类是否存在锚框
            if (len(seg_feat_low_entropy_list[i]) > 0 and memobank[valid_classes[i]][0].shape[0] > 0):
                # select anchor pixel
                seg_low_entropy_idx = torch.randint(len(seg_feat_low_entropy_list[i]), size=(num_queries,)) #随机选择锚框
                anchor_feat = (seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda())
            else:
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()

                high_entropy_idx = torch.randint(len(negative_feat), size=(num_queries * num_negatives,))
                negative_feat = negative_feat[high_entropy_idx]
                negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat) #这个表示的是负样本的特征表示
                positive_feat = (seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1).cuda())  # (num_queries, 1, num_feat)  这个是正样本的特征表示

                if momentum_prototype is not None:
                    if not (momentum_prototype == 0).all():
                        ema_decay = min(1 - 1 / i_iter, 0.999)
                        positive_feat = (
                            1 - ema_decay
                        ) * positive_feat + ema_decay * momentum_prototype[
                            valid_classes[i]
                        ]

                    prototype[valid_classes[i]] = positive_feat.clone()

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )  # (num_queries, 1 + num_negative, num_feat)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)  #这里是计算余弦的相似度度量

            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().cuda())  #这里开始求对抗损失值

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg  #返回的是平均的对抗损失值
        else:
            return prototype, new_keys, reco_loss / valid_seg



