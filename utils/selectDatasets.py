from dataloaders.voc import VOCDataset
from dataloaders.cityscapesloader import build_city_semi_loader
import torch


def selectDataset(config,logger,arg):
    if config['dataset']=="voc2012":
        config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples'] #将标签数据的数量给监督学习
        config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples'] #把标签数据的数量给无监督学习
        config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']  #把弱标签的数据数量给无监督学习
        supervised_dataset = VOCDataset(**config['train_supervised'])
        unsupervised_dataset = VOCDataset(**config['train_unsupervised'])
        val_dataset = VOCDataset(**config['val_loader'])
        if arg.local_rank==0:
            logger.info("dataset的长度为%s  %s"%(len(supervised_dataset),len(unsupervised_dataset)))

        supervised_sampler = torch.utils.data.distributed.DistributedSampler(supervised_dataset, shuffle=True)
        supervised_loader = torch.utils.data.DataLoader(supervised_dataset,
                                                        batch_size=config['train_supervised']['batch_size']//(arg.nproc_per_node),
                                                        num_workers=0,
                                                        shuffle=False,
                                                        pin_memory=False,
                                                        drop_last=True,
                                                        sampler=supervised_sampler)

        unsupervised_sampler = torch.utils.data.distributed.DistributedSampler(unsupervised_dataset, shuffle=True)
        unsupervised_loader = torch.utils.data.DataLoader(unsupervised_dataset,
                                                        batch_size=config['train_unsupervised']['batch_size']//(arg.nproc_per_node),
                                                        num_workers=0,
                                                        shuffle=False,
                                                        pin_memory=False,
                                                        drop_last=True,
                                                        sampler=unsupervised_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
        val_loader =  torch.utils.data.DataLoader(val_dataset,
                                            batch_size=config['val_loader']['batch_size'],
                                            num_workers=0,
                                            shuffle=False,
                                            pin_memory=False,
                                            drop_last=True,
                                            sampler=val_sampler)
        
        return supervised_loader,unsupervised_loader,val_loader
    
    elif config['dataset']=="Cityscapes":
        supervised_dataset , unsupervised_dataset = build_city_semi_loader("train", config, logger,seed=0)
        val_dataset =  build_city_semi_loader("val", config, logger,seed=0)
        if arg.local_rank==0:
            logger.info("dataset的长度为%s  %s"%(len(supervised_dataset),len(unsupervised_dataset)))
            logger.info("批处理大小为:%s"%(config['train_supervised']['batch_size']//(arg.nproc_per_node)))

        supervised_sampler = torch.utils.data.distributed.DistributedSampler(supervised_dataset, shuffle=True)
        supervised_loader = torch.utils.data.DataLoader(supervised_dataset,
                                                        batch_size=config['train_supervised']['batch_size']//(arg.nproc_per_node),
                                                        num_workers=0,
                                                        shuffle=False,
                                                        pin_memory=False,
                                                        drop_last=True,
                                                        sampler=supervised_sampler)

        unsupervised_sampler = torch.utils.data.distributed.DistributedSampler(unsupervised_dataset, shuffle=True)
        unsupervised_loader = torch.utils.data.DataLoader(unsupervised_dataset,
                                                        batch_size=config['train_unsupervised']['batch_size']//(arg.nproc_per_node),
                                                        num_workers=0,
                                                        shuffle=False,
                                                        pin_memory=False,
                                                        drop_last=True,
                                                        sampler=unsupervised_sampler)

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
        val_loader =  torch.utils.data.DataLoader(val_dataset,
                                            batch_size=config['val_loader']['batch_size'],
                                            num_workers=0,
                                            shuffle=False,
                                            pin_memory=False,
                                            drop_last=True,
                                            sampler=val_sampler)
        
        return supervised_loader,unsupervised_loader,val_loader
    

