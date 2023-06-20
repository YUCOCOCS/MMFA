import copy
import math
import os
import os.path
import random
from utils import pallete
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import augmentation as psp_trsform
from .base import BaseDataset
from .sampler import DistributedGivenIterationSampler


class city_dset(BaseDataset):
    def __init__(self, data_root, data_list,logger, trs_form, seed, n_sup, split="val"):
        super(city_dset, self).__init__(data_list,logger)
        self.num_classes=19
        self.palette = pallete.get_voc_pallete(self.num_classes)
        self.ignore_index = 255
        self.data_root = data_root
        self.transform = trs_form
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup) # 从self.list_sample中随机的裁剪出n_sup的长度序列样本出来
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample)) #这里他进行了重复
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample #验证集

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB") # 图片转换成RGB文件
        label = self.img_loader(label_path, "L")   #转行灰度图
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long() # 返回经过预处理之后的图片

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())  #1
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std)) # 2
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False): #对的
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"): #4
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False): # 5
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    if cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))

    return psp_trsform.Compose(trs_form)


def build_cityloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 2975)

    # build transform
    trs_form = build_transfrom(cfg)
    dset = city_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    # build sampler
    sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_city_semi_loader(split, all_cfg, logger,seed=0):
    if split=='train':
        cfg_sup = all_cfg["train_supervised"]
        cfg = copy.deepcopy(cfg_sup)
        n_sup = 2975 - cfg.get("n_sup",2975)
        trs_form = build_transfrom(cfg) #这是一系列的变换
        trs_form_unsup = build_transfrom(cfg) # 对无标记数据的一系列的变换
        dset_sup = city_dset(cfg["data_dir"], cfg["data_list"], logger,trs_form, seed, n_sup, split)
        cfg_sup = all_cfg["train_unsupervised"]
        dset_unsup = city_dset(cfg_sup["data_dir"], cfg_sup["data_list"], logger,trs_form, seed, n_sup, split)
        return dset_sup,dset_unsup
    else:
        cfg_val = all_cfg["val_loader"]
        n_sup = 2975 - cfg_val.get("n_sup",2975)
        trs_val = build_transfrom(cfg_val)
        dset_val = city_dset(cfg_val["data_dir"], cfg_val["data_list"], logger,trs_val, seed, n_sup, split)
        return dset_val




    # cfg_dset = all_cfg[split]

    # cfg = copy.deepcopy(cfg_dset)
    # #cfg.update(cfg.get(split,{}))

    # workers = cfg.get("workers", 2)
    # batch_size = cfg.get("batch_size", 1)
    # n_sup = 2975 -  cfg.get("n_sup", 2975)

    # # build transform
    # trs_form = build_transfrom(cfg) #这是一系列的变换
    # trs_form_unsup = build_transfrom(cfg) # 对无标记数据的一系列的变换
    # dset_sup = city_dset(cfg["data_dir"], cfg["data_list"], logger,trs_form, seed, n_sup, split) #这里面开始设置


    # if split == "val": #如果是验证集，则直接输出
    #     return dset_sup

    # else:
    #     # build sampler for unlabeled set
    #     data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
    #     dset_unsup = city_dset(
    #         cfg["data_dir"], data_list_unsup,logger, trs_form_unsup, seed, n_sup, split
    #     )
    #     return dset_sup, dset_unsup
