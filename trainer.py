from cv2 import EMD
import torch
import time, random, cv2, sys 
from math import ceil
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseTrainer
from utils import losses
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
from utils.helpers import DeNormalize
import torch.distributed as dist
from utils.helpers import generate_unsup_data,compute_unsupervised_loss,label_onehot,compute_contra_memobank_loss

class Trainer(BaseTrainer):
    def __init__(self, model,model_teacher, resume, config, supervised_loader, unsupervised_loader, iter_per_epoch,train_logger_yes,local_rank,memobank,queue_ptrlis,queue_size,
                val_loader=None, train_logger=None):
        super(Trainer, self).__init__(model, resume, config, iter_per_epoch, train_logger_yes,local_rank,train_logger)
        
        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.model_teacher = model_teacher
        self.memobank = memobank
        self.queue_ptrlis =queue_ptrlis
        self.queue_size = queue_size
        self.val_loader = val_loader
        self.local_rank = local_rank
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.train_logger_yes = train_logger_yes
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.val_loader.batch_size)))
        if config['trainer']['log_per_iter']:
            self.log_step = int(self.log_step / self.val_loader.batch_size) + 1

        self.num_classes = self.val_loader.dataset.num_classes
        self.mode = self.model.module.mode

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            #DeNormalize(self.val_loader.MEAN, self.val_loader.STD),
            DeNormalize(config['train_supervised']['mean'],config['train_supervised']['std']),
            transforms.ToPILImage()]) #将tensor的数据转换成PIL图像文件，以显示图像
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])

        self.start_time = time.time()
    
    def reduce_tensor(self,inp):
        """
        Reduce the loss from all processes so that
        process with rank 0 has the averaged results.
        """
        world_size = torch.distributed.get_world_size()
        if world_size < 2:
            return inp
        with torch.no_grad():
            reduced_inp = inp
            dist.reduce(reduced_inp, dst=0) #将每个显卡算的损失值全部统计到0号显卡中，
        return reduced_inp/world_size



    def _train_epoch(self, epoch):
        self.html_results.save()
        if self.local_rank==0:
            self.train_logger_yes.info("The %sth training is starting "%(epoch))
        
        self.logger.info('\n')
        self.model.train() #开始训练
        self.supervised_loader.sampler.set_epoch(epoch)#每次迭代的时候都将数据打乱，
        self.unsupervised_loader.sampler.set_epoch(epoch)#每次迭代的时候都将数据打乱
        if self.mode == 'supervised':
            dataloader = iter(self.supervised_loader)
            tbar = tqdm(range(len(self.supervised_loader)), ncols=135) #这里是进度条,ncols表示的是进度条的宽度，nrows表示的是进度条的高度
        else:
            #半监督学习的时候,将监督数据和非监督数据一起传进来 cycle是对self.supervised_loader进行一个循环的重复
            dataloader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=135)

        self._reset_metrics() #重置量度
        
        cur_losses={}
        ema_decay_origin = self.config["ema_decay"]
        for batch_idx in tbar:

            iterr  =  epoch*len(self.unsupervised_loader) + batch_idx
            (input_l, target_l), (input_ul, target_ul) = next(dataloader)
            # input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
            input_ul, target_ul = input_ul.cuda(self.local_rank), target_ul.cuda(self.local_rank)

            # input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            input_l, target_l = input_l.cuda(self.local_rank), target_l.cuda(self.local_rank)
            self.optimizer.zero_grad() #优化器的梯度参数首先设置为0
            B,c,h,w = input_l.size()
            if epoch < self.config["sup_only_epoch"]:
                # total_loss为总的损失，cur_loss是各个损失的字典，outputs是监督产生的标签图与无监督产生的标签图
                cur_losses,outputs,rep_output = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                        curr_iter=batch_idx, target_ul=target_ul, epoch=epoch)
                rep_sup = rep_output["sup"]
                rep_unsup = rep_output["unsup"]
                self.model_teacher.train()
                contra_loss = 0 * rep_sup.sum() + 0 * rep_unsup.sum()
                cur_losses["loss_contra"] = contra_loss
                cur_losses["loss_unsup"] = contra_loss

            else: #对比学习开始
                if epoch == self.config["sup_only_epoch"]:
                    with torch.no_grad():
                        for t_params,s_params in zip(self.model_teacher.parameters(),self.model.parameters()):
                            t_params.data = s_params.data
                
                
                #使用cutmix产生伪标签
                self.model_teacher.eval()
                pred_u_teacher = self.model_teacher(x_ul = input_ul )["pred"]
                pred_u_teacher = F.interpolate(pred_u_teacher, (h, w), mode="bilinear", align_corners=True)
                pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
                logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

                if np.random.uniform(0,1)<0.5:
                    image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(input_ul,label_u_aug.clone(),logits_u_aug.clone(),mode="cutmix")
                else:
                    image_u_aug = input_ul
                
                cur_losses,outputs,rep_output = self.model(x_l=input_l, target_l=target_l, x_ul=input_ul,
                                                        curr_iter=batch_idx, target_ul=target_ul, epoch=epoch-1)
                rep_sup = rep_output["sup"]
                rep_unsup = rep_output["unsup"]
                rep_all = torch.cat((rep_sup,rep_unsup),dim = 0) #学生产生的特征表示
                pred_l = outputs["pred_min_l"]
                pred_u = outputs["pred_min_ul"]
                pred_all = torch.cat((pred_l,pred_u),dim=0)
                pred_l_large = F.interpolate(pred_l, size=(h, w), mode="bilinear", align_corners=True)
                pred_u_large = F.interpolate(pred_u, size=(h, w), mode="bilinear", align_corners=True) 


                # 这里是教师级网络产生特征
                self.model_teacher.train()
                with torch.no_grad():
                    pred_out_l,pred_out_ul = self.model_teacher(x_l=input_l, target_l=target_l, x_ul=image_u_aug,curr_iter=batch_idx, target_ul=target_ul, epoch=epoch-1)
                    rep_all_teacher = torch.cat((pred_out_l["feat"],pred_out_ul["feat"]),dim = 0)
                    prob_u_teacher = F.softmax(pred_out_ul["pred"],dim = 1)
                    prob_l_teacher = F.softmax(pred_out_l["pred"],dim = 1)
                    pred_u_large_teacher=F.interpolate(prob_u_teacher, size=(h, w), mode="bilinear", align_corners=True)
                
                drop_percent = self.config["drop_percent"]
                percent_unreliable = (100-drop_percent)*(1-epoch/self.config["trainer"]["epochs"])
                drop_percent = 100 - percent_unreliable
                unsup_loss = (
                     compute_unsupervised_loss( #在这里进行无监督损失的计算，根据对应的一定比例的熵的值进行挑选，然后求得权值，
                        pred_u_large, # 对应的是学生网络对cutmix之后的图像预测的结果的概率分布
                        label_u_aug.clone(), #对应的教师网络产生的伪标签
                        drop_percent,
                        pred_u_large_teacher.detach(), #计算的是对应的经过教师级网络产生的无标签的概率分布
                    ) #计算的是非标签的数据的损失
                )

                cur_losses["loss_unsup"] = unsup_loss

                cfg_contra = self.config["contrastive"]
                alpha_t = cfg_contra["low_entropy_threshold"] * (1 - epoch / self.config["trainer"]["epochs"]) 

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1) #对应的是每个像素点的熵 尺寸为 B*h*w

                     #获得对应的前alpha_t的阈值
                    low_thresh = np.percentile(entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t)   #得到的是低的阈值
                    low_entropy_mask = (entropy.le(low_thresh).float() * (label_u_aug != 255).bool())   #低域值的掩码

                    #获得对应的前 100 - alpha_t的阈值
                    high_thresh = np.percentile(entropy[label_u_aug != 255].cpu().numpy().flatten(),100 - alpha_t,) #得到的是高阈值

                    high_entropy_mask = (entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()) #高阈值的掩码 为 B * H * W

                    low_mask_all = torch.cat( # 转换成四个维度的 尺寸为 2个 B * 1 * h * w
                        (
                            (target_l.unsqueeze(1) != 255).float(),#这里是有标签的数据
                            low_entropy_mask.unsqueeze(1),# n*1*h*w
                        )
                    ) # 计算的是阈值小的，低熵的像素

                    low_mask_all = F.interpolate(low_mask_all, size=pred_all.shape[2:], mode="nearest" ) #将低熵的图的尺寸变为预测分类的尺寸
                    high_mask_all = torch.cat(
                            (
                                (target_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    high_mask_all = F.interpolate(high_mask_all, size=pred_all.shape[2:], mode="nearest")  # down sample

                    label_l_small = F.interpolate(label_onehot(target_l, self.config["num_class"]),size=pred_all.shape[2:],mode="nearest",)

                    label_u_small = F.interpolate(label_onehot(label_u_aug, self.config["num_class"]),size=pred_all.shape[2:],mode="nearest",)
                    new_keys, contra_loss = compute_contra_memobank_loss( #计算的对比损失值
                            rep_all, #表示的是学生网络产生的所有的像素表示 (n1+n2) * 512 * h1 * w1
                            label_l_small.long(), # 表示将真实的标签进行下采样与预测图大小一样 B * h1 * h1
                            label_u_small.long(), # 将cutmix之后得到的教师级网络产生的标签图下采样和预测图一样大小
                            prob_l_teacher.detach(), #尺寸为 B * C * h1 * w1
                            prob_u_teacher.detach(),
                            low_mask_all,#这里存储的是低熵的像素点的掩码  2个 B * 1 * h1 * w1
                            high_mask_all,
                            cfg_contra,
                            self.memobank,
                            self.queue_ptrlis,
                            self.queue_size,
                            rep_all_teacher.detach(),#教师级网络产生的所有像素点的表示
                        )
                    dist.all_reduce(contra_loss)
                    #contra_loss = self.reduce_tensor(contra_loss) #求均值
                    # contra_loss = contra_loss * 0.1
                    cur_losses["loss_contra"] = contra_loss * 0.1


            total_loss = cur_losses["loss_sup"] + cur_losses["loss_unsup"] + cur_losses["loss_aux"] + contra_loss * 0.1 #取平均值
            # if self.local_rank==0:
            #     self.train_logger_yes.info("mean total_loss is %s"%(total_loss))
            total_loss.backward()
            self.optimizer.step()

            #更新教师级的网络参数
            if epoch >= self.config["sup_only_epoch"]:
                with torch.no_grad():
                    ema_decay = min(1 - 1/(iterr - len(self.unsupervised_loader) + 1),ema_decay_origin)

                    for t_params,s_params in zip(self.model_teacher.parameters(),self.model.parameters()):
                        t_params.data = (ema_decay*t_params.data + (1-ema_decay)*s_params.data)

            cur_losses["loss_sup"] = self.reduce_tensor(cur_losses["loss_sup"])
            cur_losses["loss_unsup"] = self.reduce_tensor(cur_losses["loss_unsup"])
            cur_losses["loss_aux"] = self.reduce_tensor(cur_losses["loss_aux"])
            

            self._update_losses(cur_losses) #更新各个损失的值
            self._compute_metrics(outputs, target_l, target_ul, epoch-1) #计算损失
            logs = self._log_values(cur_losses) #批处理的均值
            
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.unsupervised_loader) + batch_idx
                self._write_scalars_tb(logs)

            if batch_idx % int(len(self.unsupervised_loader)*0.9) == 0:
                self._write_img_tb(input_l, target_l, input_ul, target_ul, outputs, epoch)

            #删除以下的信息
            del input_l, target_l, input_ul, target_ul
            del total_loss, cur_losses, outputs

            
            tbar.set_description('T ({}) | Ls {:.2f} Lu {:.2f} Lw {:.2f} PW {:.2f} m1 {:.2f} m2 {:.2f}|'.format(
                epoch, self.loss_sup.average, self.loss_unsup.average, self.loss_aux.average,
                self.loss_contra.average, self.mIoU_l, self.mIoU_ul))
            
            if self.local_rank==0 and batch_idx%20==0:
                self.train_logger_yes.info("T : %s |Ls: %s  Lu:%s  Laux:%s  Lcontra:%s  m1:%s   m2:%s"%(epoch, self.loss_sup.average, self.loss_unsup.average, 
                self.loss_aux.average,self.loss_contra.average, self.mIoU_l, self.mIoU_ul))

            self.lr_scheduler.step(epoch=epoch-1)

        return logs



    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')
        self.val_loader.sampler.set_epoch(epoch) #每次验证都将数据打乱
        self.model.eval()
        self.wrt_mode = 'val'
        total_loss_val = AverageMeter()
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0

        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []

            for batch_idx, (data, target) in enumerate(tbar):
                target, data = target.cuda(self.local_rank), data.cuda(self.local_rank)

                H, W = target.size(1), target.size(2)
                up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
                pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
                data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')#这个对应的是左右上下，
                output = self.model(data,mode='val')#进入了CCT模块
                output = output[:, :, :H, :W] #只截取对应的区域

                # LOSS
                loss = F.cross_entropy(output, target, ignore_index=self.ignore_index) # 计算的是验证的损失
                total_loss_val.update(loss.item())#加入到averagemeter中，计算的是均值

                # 计算的是各个类别的交并比
                correct, labeled, inter, union = eval_metrics(output, target, self.num_classes, self.ignore_index)
                total_inter, total_union = total_inter+inter, total_union+union
                total_correct, total_label = total_correct+correct, total_label+labeled

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    if isinstance(data, list): data = data[0]
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
                IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                mIoU = IoU.mean() # 计算的是每个类的平均交并比
                seg_metrics = {"epoch":epoch,"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                                "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))}
                mIoU = np.round(mIoU,3)
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                total_loss_val.average, pixAcc, mIoU))
                mIoU = torch.tensor(mIoU)
                mIoU = mIoU.cuda(self.local_rank)
            self._add_img_tb(val_visual, 'val')
            #self.train_logger_yes.info("local_rank:%s  miou:%s"%(self.local_rank,mIoU))
            def reduce_tensor(inp):
                """
                Reduce the loss from all processes so that
                process with rank 0 has the averaged results.
                """
                world_size = torch.distributed.get_world_size()
                if world_size < 2:
                    return inp
                with torch.no_grad():
                    reduced_inp = inp
                    dist.reduce(reduced_inp, dst=0) #将每个显卡算的损失值全部统计到0号显卡中，
                return reduced_inp
            miou = reduce_tensor(mIoU) / torch.distributed.get_world_size()
            if self.local_rank==0:
                self.train_logger_yes.info("************************************")
                self.train_logger_yes.info("epoch:%s  miou:%s"%(epoch,miou))
                self.train_logger_yes.info("************************************")

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', total_loss_val.average, self.wrt_step)
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': total_loss_val.average,
                'miou':miou,
                **seg_metrics
            }
            self.html_results.add_results(epoch=epoch, seg_resuts=log)
            self.html_results.save()

            # if (time.time() - self.start_time) / 3600 > 22:
            #     self._save_checkpoint(epoch, save_best=self.improved)
        return log



    def _reset_metrics(self):
        self.loss_sup = AverageMeter()
        self.loss_unsup  = AverageMeter()
        self.loss_contra = AverageMeter()
        self.loss_aux = AverageMeter()
        self.total_inter_l, self.total_union_l = 0, 0
        self.total_correct_l, self.total_label_l = 0, 0
        self.total_inter_ul, self.total_union_ul = 0, 0
        self.total_correct_ul, self.total_label_ul = 0, 0
        self.mIoU_l, self.mIoU_ul = 0, 0
        self.pixel_acc_l, self.pixel_acc_ul = 0, 0
        self.class_iou_l, self.class_iou_ul = {}, {}



    def _update_losses(self, cur_losses):
        if "loss_sup" in cur_losses.keys():
            self.loss_sup.update(cur_losses['loss_sup'].mean().item())
        if "loss_unsup" in cur_losses.keys():
            self.loss_unsup.update(cur_losses['loss_unsup'].mean().item())
        if "loss_aux" in cur_losses.keys():
            self.loss_aux.update(cur_losses['loss_aux'].mean().item())
        if "loss_contra" in cur_losses.keys():
            self.loss_contra.update(cur_losses['loss_contra'].mean().item())



    def _compute_metrics(self, outputs, target_l, target_ul, epoch): #计算的是有标签与无标签的交并比
        # 返回的数字依次是预测正确的像素个数，总的类别的标签个数，图像区域的交，图像区域的并
        seg_metrics_l = eval_metrics(outputs['sup_pred'], target_l, self.num_classes, self.ignore_index)
        self._update_seg_metrics(*seg_metrics_l, True)
        seg_metrics_l = self._get_seg_metrics(True)
        self.pixel_acc_l, self.mIoU_l, self.class_iou_l = seg_metrics_l.values()

        if self.mode == 'semi':
            seg_metrics_ul = eval_metrics(outputs['unsup_pred'], target_ul, self.num_classes, self.ignore_index)
            self._update_seg_metrics(*seg_metrics_ul, False)
            seg_metrics_ul = self._get_seg_metrics(False)
            self.pixel_acc_ul, self.mIoU_ul, self.class_iou_ul = seg_metrics_ul.values()
            


    def _update_seg_metrics(self, correct, labeled, inter, union, supervised=True):
        if supervised:
            self.total_correct_l += correct
            self.total_label_l += labeled
            self.total_inter_l += inter
            self.total_union_l += union
        else:#非监督的
            self.total_correct_ul += correct
            self.total_label_ul += labeled
            self.total_inter_ul += inter
            self.total_union_ul += union



    def _get_seg_metrics(self, supervised=True):
        if supervised:
            # np.spacing(1) 表示的是产生一个无穷小的随机数
            pixAcc = 1.0 * self.total_correct_l / (np.spacing(1) + self.total_label_l)
            IoU = 1.0 * self.total_inter_l / (np.spacing(1) + self.total_union_l)
        else:
            pixAcc = 1.0 * self.total_correct_ul / (np.spacing(1) + self.total_label_ul)
            IoU = 1.0 * self.total_inter_ul / (np.spacing(1) + self.total_union_ul)
        mIoU = IoU.mean() #这计算的是各个类的交并比，取平均就是平均交并比，
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }



    def _log_values(self, cur_losses):
        logs = {}
        if "loss_sup" in cur_losses.keys():
            logs['loss_sup'] = self.loss_sup.average
        if "loss_unsup" in cur_losses.keys():
            logs['loss_unsup'] = self.loss_unsup.average
        if "loss_aux" in cur_losses.keys():
            logs['loss_aux'] = self.loss_aux.average
        if "loss_contra" in cur_losses.keys():
            logs['loss_contra'] = self.loss_contra.average

        logs['mIoU_labeled'] = self.mIoU_l
        logs['pixel_acc_labeled'] = self.pixel_acc_l
        if self.mode == 'semi':
            logs['mIoU_unlabeled'] = self.mIoU_ul
            logs['pixel_acc_unlabeled'] = self.pixel_acc_ul
        return logs


    def _write_scalars_tb(self, logs):
        for k, v in logs.items():
            if 'class_iou' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'train/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
        current_rampup = self.model.module.unsup_loss_w.current_rampup
        self.writer.add_scalar('train/Unsupervised_rampup', current_rampup, self.wrt_step)



    def _add_img_tb(self, val_visual, wrt_mode):
        val_img = []
        palette = self.val_loader.dataset.palette
        for imgs in val_visual:
            imgs = [self.restore_transform(i) if (isinstance(i, torch.Tensor) and len(i.shape) == 3) 
                        else colorize_mask(i, palette) for i in imgs]
            imgs = [i.convert('RGB') for i in imgs]
            imgs = [self.viz_transform(i) for i in imgs]
            val_img.extend(imgs)
        val_img = torch.stack(val_img, 0)
        val_img = make_grid(val_img.cpu(), nrow=val_img.size(0)//len(val_visual), padding=5)
        self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)



    def _write_img_tb(self, input_l, target_l, input_ul, target_ul, outputs, epoch):
        outputs_l_np = outputs['sup_pred'].data.max(1)[1].cpu().numpy()
        targets_l_np = target_l.data.cpu().numpy()
        imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_l, outputs_l_np, targets_l_np)]
        self._add_img_tb(imgs, 'supervised')

        if self.mode == 'semi':
            outputs_ul_np = outputs['unsup_pred'].data.max(1)[1].cpu().numpy()
            targets_ul_np = target_ul.data.cpu().numpy()
            imgs = [[i.data.cpu(), j, k] for i, j, k in zip(input_ul, outputs_ul_np, targets_ul_np)]
            self._add_img_tb(imgs, 'unsupervised')

