from base import BaseDataSet, BaseDataLoader
from utils import pallete
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import json

class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 21 #voc中有21个类别
        self.palette = pallete.get_voc_pallete(self.num_classes)
        super(VOCDataset, self).__init__(**kwargs)

    def _set_files(self):
        # self.root = os.path.join(self.root, 'VOCdevkit/VOC2012') #根路径
        self.root = os.path.join(self.root, 'VOC2012') #根路径
        if self.split == "val": #验证集上面操作
            file_list = os.path.join("dataloaders/voc_splits", f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join("dataloaders/voc_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list)) # 左边是图像，右边是标签

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:]) #获取图片的路径
        image = np.asarray(Image.open(image_path), dtype=np.float32) #
        image_id = self.files[index].split("/")[-1].split(".")[0] # 获取图片的id
        if self.use_weak_lables: #根据列表加载弱监督的标签
            label_path = os.path.join(self.weak_labels_output, image_id+".png")
        else:
            label_path = os.path.join(self.root, self.labels[index][1:]) #获取图片的对应的标签路径
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        return image, label, image_id #返回的是numpy类型的数据


class VOC(BaseDataLoader):
    def __init__(self, kwargs):

        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN

        kwargs['std'] = self.STD
        kwargs['ignore_index'] = 255 #忽略像素值为255的像素点
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = VOCDataset(**kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)
