import logging

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, d_list, logger,**kwargs):
        # parse the input list
        self.parse_input_list(d_list,logger,**kwargs) #这里的d_list是train的数据样本路径,train.txt

    def parse_input_list(self, d_list,logger, max_sample=-1, start_idx=-1, end_idx=-1):
        assert isinstance(d_list, str)
        if "cityscapes" in d_list: #路劲中包含cityscapes
            self.list_sample = [   #这里面记录的是标签的路径，这里面是两个，第一个是图片的地址，第二个是标签的地址
                [
                     line.strip(), #移除头和尾部的空格或者换行符
                    "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
                ]
                for line in open(d_list, "r")
            ]
        elif "pascal" in d_list or "VOC" in d_list:
            self.list_sample = [
                [
                    "JPEGImages/{}.jpg".format(line.strip()),
                    "SegmentationClassAug/{}.png".format(line.strip()),
                ]
                for line in open(d_list, "r")
            ]
        else:
            raise "unknown dataset!"

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        logger.info("# Cityscapes samples length: %s"%(self.num_sample)) #对应的是训练集的总长度
    
        # self.logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert(mode)

    def __len__(self):
        return self.num_sample
