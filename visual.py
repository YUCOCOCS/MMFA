import argparse
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import json
import models
import dataloaders
from utils.helpers import colorize_mask
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
from utils.losses import abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha
from utils.train_loger import logger_config
class testDataset(Dataset):
    def __init__(self, images):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        images_path = Path(images)
        self.filelist = list(images_path.glob("*.jpg"))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        image_path = self.filelist[index]
        image_id = str(image_path).split("/")[-1].split(".")[0]
        image = Image.open(image_path)
        image = self.normalize(self.to_tensor(image))
        return image, image_id

def multi_scale_predict(model, image, scales, num_classes, flip=True):
    H, W = (image.size(2), image.size(3))
    upsize = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w = upsize[0] - H, upsize[1] - W
    image = F.pad(image, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image.shape[2], image.shape[3]))

    for scale in scales:
        scaled_img = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(x_l = scaled_img,mode="val"))

        if flip:
            fliped_img = scaled_img.flip(-1)
            fliped_predictions = upsample(model(x_l = fliped_img,mode="val"))
            scaled_prediction = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)

    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

def main():
    args = parse_arguments()

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    # DATA
    testdataset = testDataset(args.images)
    loader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=1)
    num_classes = 21
    palette = get_voc_pallete(num_classes)
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs']) #ramp_up=0.1，经过80代 这里乘的是非监督损失的权重
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=570,
                                        rampup_ends=rampup_ends)  #这里对应的是一致性的权重
    logger_handle = logger_config(log_path="/home/y212202015/CCT/11.log",logging_name="train.log")
    torch.distributed.init_process_group("nccl", world_size=1, rank = args.local_rank) #初始化进程组
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True  # 加快 gpu 的速度

    # MODEL
    config['model']['supervised'] = True; config['model']['semi'] = False
    model = models.CCT(num_classes=21, 
                       arg = args,
                       conf=config['model'],
                       train_logger=logger_handle,
    				   sup_loss=CE_loss, 
                       cons_w_unsup=cons_w_unsup,
    				   weakly_loss_w=config['weakly_loss_w'], 
                       use_weak_lables=config['use_weak_lables'],
                       ignore_index=255,
                    ) # weakly_loss_w是0.4  use_weak_lables为false
    model = model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])
    checkpoint = torch.load("/home/y212202015/CCT/saved/CCT/VOC_74.70.pth",map_location="cpu")
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model = model.cuda()

    if args.save and not os.path.exists('outputs'):
        os.makedirs('outputs')

    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    labels, predictions = [], []

    for index, data in enumerate(tbar):
        image, image_id = data
        image = image.cuda()

        # PREDICT
        with torch.no_grad():
            output = multi_scale_predict(model, image, scales, num_classes)
        prediction = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)

        # SAVE RESULTS
        prediction_im = colorize_mask(prediction, palette)
        # prediction_im.save('outputs/'+image_id[0]+'.png')
        prediction_im.save("/home/y212202015/CCT/result/"+image_id[0][-11:] + '.png')

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='configs/config.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '--save', action='store_true', help='Save images')
    parser.add_argument('--images', default="/home/y212202015/CCT/test", type=str,
                        help='Test images for Pascal VOC')
    parser.add_argument('-r', '--resume', default="/home/y212202015/CCT/saved/CCT/VOC_74.70.pth", type=str,
                        help='Path to the .pth model checkpoint to resume training') #这里面保存着已经训练好的模型
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')  #gpu的数量
    #parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed testing', default=0,
                        type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=4,
                        type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()


