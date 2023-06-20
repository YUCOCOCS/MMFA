import os
import json
import argparse
import torch
#import dataloaders
from dataloaders.voc import VOCDataset
import models
import math
from utils import Logger
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha
from utils.Loggerr import LLogger
from utils.train_loger import logger_config
from utils.selectDatasets import selectDataset
def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume,arg,logger): # 第一个配置是配置文件，第二个配置是保存训练的模型
    torch.manual_seed(42)  #这里面设置了随机种子，表示的是每次训练网络的时候，网络的权重都是一样的，
    train_logger = Logger() #定义一个日志
    
    supervised_loader,unsupervised_loader,val_loader = selectDataset(config,logger,arg)

    iter_per_epoch = len(unsupervised_loader) #非监督的训练数据
    if arg.local_rank==0:
        logger.info("iter_per_epoch:%s"%(iter_per_epoch))
    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss #对于全监督的训练，使用计算交叉熵损失
    elif config['model']['sup_loss'] == 'FL':
        alpha = get_alpha(supervised_loader) # calculare class occurences
        sup_loss = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5)
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                                num_classes=val_loader.dataset.num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs']) #ramp_up=0.1，经过80代 这里乘的是非监督损失的权重
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                        rampup_ends=rampup_ends)  #这里对应的是一致性的权重
    #加载CCT的模型
    model = models.MMFA(num_classes=val_loader.dataset.num_classes, 
                       arg = arg,
                       conf=config['model'],
                       train_logger=logger,
    				   sup_loss=sup_loss, 
                       cons_w_unsup=cons_w_unsup,
    				   weakly_loss_w=config['weakly_loss_w'], 
                       use_weak_lables=config['use_weak_lables'],
                       ignore_index=val_loader.dataset.ignore_index,
                    ) # weakly_loss_w是0.4  use_weak_lables为false
    

    print(f'\n{model}\n')
    #logger.info("当前的进程id为%s"%(arg.local_rank))
    # 建立分布式模型
    model = model.cuda(arg.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[arg.local_rank])

    model_teacher = models.MMFA_Teacher(num_classes=val_loader.dataset.num_classes, 
                       arg = arg,
                       conf=config['model'],
                       train_logger=logger,
    				   sup_loss=sup_loss, 
                       cons_w_unsup=cons_w_unsup,
    				   weakly_loss_w=config['weakly_loss_w'], 
                       use_weak_lables=config['use_weak_lables'],
                       ignore_index=val_loader.dataset.ignore_index,
                    )
    model_teacher = model_teacher.cuda(arg.local_rank)
    model_teacher = torch.nn.parallel.DistributedDataParallel(model_teacher,device_ids=[arg.local_rank])
    for p in model_teacher.parameters():
        p.requires_grad = False
    
    # build class-wise memory bank
    memobank = []  #表示的是memory bank
    queue_ptrlis = [] #表示的是
    queue_size = []
    for i in range(config["num_class"]): #对每一个类都有一个memory bank
        memobank.append([torch.zeros(0, 512)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            config["num_class"],#类别的个数
            config["contrastive"]["num_queries"], # 256
            1,
            256,
        )
    ).cuda()

    # TRAINING
    trainer = Trainer(
        model=model,
        model_teacher = model_teacher,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,#每一轮需要迭代的次数
        train_logger_yes=logger,
        local_rank=arg.local_rank,
        train_logger=train_logger,
        memobank=memobank,
        queue_ptrlis=queue_ptrlis,
        queue_size=queue_size 
      )

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_cityscapes copy.json',type=str,
                        help='Path to the config file') #这里是配置文件
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training') #这里面保存着已经训练好的模型
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')  #gpu的数量
    #parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--local_rank', dest='local_rank', help='node rank for distributed testing', default=0,
                        type=int)
    parser.add_argument('--nproc_per_node', dest='nproc_per_node', help='number of process per node', default=8,
                        type=int)
    args = parser.parse_args()
    n_gpus = 8  # gpu的数量
    # batchsize为10,,num_worker为8
    config = json.load(open(args.config)) #打开配置文件
    #logger_handle = LLogger(config['log_dir'])
    logger_handle = logger_config(log_path=config['log_dir'],logging_name="train.log")
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank = args.local_rank) #初始化进程组
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True  # 加快 gpu 的速度
    main(config, args.resume,args,logger_handle)
