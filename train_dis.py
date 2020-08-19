from model.dgn_model import DGN
import torch
import numpy as np
import random
from torch.backends import cudnn
from config import get_args
from utils.logger import setup_logger
from utils import *
from model.dgn_model import DGN
from dataloader import make_dataloader
import logging
from utils.meter import AverageMeter
from utils.metrics import R1_mAP,R1_mAP_eval,R1_mAP_Pseudo,R1_mAP_query_mining
from utils.visualtools import visualize_gan_results
from torchvision.utils import  save_image



def divide_images_into_two_parts(imgs,pids,num_identities,num_instances):
    assert num_instances % 2 ==0, 'num of instances for each identity should be even'
    num_instance_division = num_instances //2
    batch_size = imgs.shape[0]//2
    #print('The generator batch size',batch_size)
    num_channels, width, height = imgs.shape[1],imgs.shape[2],imgs.shape[3]
    images1 = torch.zeros(size=[batch_size,num_channels,width,height]).cuda()
    images2 = torch.zeros(size=[batch_size,num_channels,width,height]).cuda()
    pids1 = torch.zeros(size= [batch_size],dtype= torch.int64).cuda()
    pids2 = torch.zeros(size= [batch_size],dtype=torch.int64).cuda()
    for i in range(num_identities):
        start = i* num_instances
        images1[start//2 : start//2 + num_instance_division] = imgs[start:start+num_instance_division]
        images2[start//2:start//2+num_instance_division] = imgs[start+num_instance_division:start+2*num_instance_division]
        pids1[start//2 :start//2 +num_instance_division] = pids[start: start+num_instance_division]
        pids2[start//2:start//2+num_instance_division] = pids[start+num_instance_division:start+2*num_instance_division]

    return images1,pids1,images2,pids2

def train_disentangler(config,epoch,model,train_loader_gen,logger):
    for n_iter, (imgs, vids) in enumerate(train_loader_gen):
        model.gen_optimizer.zero_grad()
        model.dis_optimizer.zero_grad()
        log_period = config.log_interval
        imgs = imgs.to(model.device)
        targets = vids.to(model.device)

        imgs1, pids1, imgs2, pids2 = divide_images_into_two_parts(imgs, targets,
                                                                  config.batch_size_gen // config.num_instances_gen,
                                                                  config.num_instances_gen)

        imgs1grey = to_edge(imgs1)
        print(imgs1.shape,imgs1grey.shape)
        exit()
        id_global_feats1, style_feature_maps1, id_scores1 = model.encode(imgs1)
        id_global_feats2, style_feature_maps2, id_scores2 = model.encode(imgs2)


        # same image generation
        recons_images1 = model.decode(id_global_feats1, style_feature_maps1)
        recons_images2 = model.decode(id_global_feats2, style_feature_maps2)

        # print(torch.equal(id_global_feats1,id_global_feats2))

        # cross image with same identity generation
        recons_images_cross1 = model.decode(id_global_feats2, style_feature_maps1)
        recons_images_cross2 = model.decode(id_global_feats1, style_feature_maps2)

        # losses which include "reid loss" and "reconstuction loss" and cross_reconstruction loss
        loss_classfication = model.id_loss(id_scores1, pids1) + model.id_loss(id_scores2, pids2)
        # loss_triplet_loss = model.triplet_loss(id_global_feats1, pids1)[0] + \
        #                     model.triplet_loss(id_global_feats2, pids2)[0]
        # loss_reid = loss_classfication + loss_triplet_loss

        loss_recons = model.reconst_loss(imgs1, recons_images1) + model.reconst_loss(imgs2, recons_images2)

        loss_recons_cross = model.reconst_loss(imgs1, recons_images_cross1) + model.reconst_loss(imgs2,
                                                                                                 recons_images_cross2)

        loss_total =   loss_recons + loss_recons_cross+ loss_classfication

        loss_total.backward()
        model.gen_optimizer.step()

        acc1 = (id_scores1.max(1)[1] == pids1).float().mean()
        acc2 = (id_scores2.max(1)[1] == pids2).float().mean()
        acc = (acc1 + acc2) / 2.0

        loss_gen_class_meter.update(loss_classfication.item(), imgs.shape[0])
        loss_recons_meter.update(loss_recons.item(), imgs.shape[0])
        loss_recon_cross_meter.update(loss_recons_cross.item(), imgs.shape[0])
        acc_meter.update(acc, 1)

        if n_iter % log_period == 0:
            logger.info(
                "Train Generator - Epoch[{}]  Iteration[{}/{}] , Loss_class: {:.3f}, Loss_recon: {:.3f},Loss_recon_cross: {:.3f} Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, (n_iter + 1), len(train_loader_gen), loss_gen_class_meter.avg,
                        loss_recons_meter.avg,
                        loss_recon_cross_meter.avg, acc_meter.avg, model.gen_lr_scheduler.get_lr()[0]))
    return imgs1, imgs2, recons_images1, recons_images2


def train_normal_reid(config,epoch,model,train_loader_reid,logger):
    for n_iter, (imgs, vids) in enumerate(train_loader_reid):
        model.gen_optimizer.zero_grad()
        model.dis_optimizer.zero_grad()
        log_period = config.log_interval

        imgs = imgs.to(model.device)
        targets = vids.to(model.device)
        scores, global_feat = model.id_encoder(imgs)
        loss_classification = model.id_loss(scores,targets)
        loss_reid = model.triplet_loss(global_feat,targets)[0]
        loss_total = loss_classification + loss_reid

        loss_total.backward()
        model.gen_optimizer.step()

        acc = (scores.max(1)[1] == targets).float().mean()
        loss_reid_meter.update(loss_reid.item(),imgs.shape[0])
        loss_total_meter.update(loss_total.item(),imgs.shape[0])
        acc_meter1.update(acc,1)

        if n_iter % log_period == 0:
            logger.info(
                "Train reid - Epoch[{}]  Iteration[{}/{}] , Loss_total: {:.3f}, Loss_reid: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, (n_iter + 1), len(train_loader_reid), loss_total_meter.avg,
                        loss_reid_meter.avg, acc_meter1.avg, model.gen_lr_scheduler.get_lr()[0]))





def main():
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    # CONFIG PARSER
    config = get_args()
    output_path = config.output_path
    logger = setup_logger('disentangle',output_path, if_train=True)
    train_loader,train_loader_gen, val_loader, num_query, num_classes = make_dataloader(config)
    model = DGN(num_classes,config)


    checkpoint_period = config.save_model_interval
    eval_period = config.test_interval

    logger = logging.getLogger("disentangle")
    logger.info('start training')


    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm='yes')

    if config.resume:
        resume_epoch_number = config.resume_epoch_number
        model_save_path = config.save_models_path
        root, dirs, files = os_walk(model_save_path)
        can_resume_model = False
        if 'models_{}'.format(resume_epoch_number) in dirs:
            can_resume_model = True
            print('Can resume mode from epoch {}'.format(resume_epoch_number))
        if can_resume_model:
            print('Start resuming from model from epoch {}'.format(resume_epoch_number))
            model.resume_model(resume_epoch_number)

    start_epoch = int(config.resume_epoch_number) if config.resume else 0
    for epoch in range(start_epoch,config.train_epoches):
        start_time = time.time()
        loss_reid_meter.reset()
        loss_recons_meter.reset()
        loss_recon_cross_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.lr_scheduler_step()
        model.set_train()
        make_dirs(config.save_images_path)
        if epoch < config.train_generator_epoch:
            imgs1,imgs2,recons_images1,recons_images2,=train_disentangler(config,epoch,model,train_loader_gen,logger)
            view_img_tensor = torch.cat([imgs1, imgs2], dim=0)
            save_image(view_img_tensor, config.save_images_path + "/" + "origin_imgs_{}.jpg".format(epoch))

            view_recons_tensor = torch.cat([recons_images1, recons_images2], dim=0)
            save_image(view_recons_tensor, config.save_images_path + '/' + 'recons_img_{}.jpg'.format(epoch))

        else:
            train_normal_reid(config,epoch,model,train_loader,logger)


        end_time = time.time()
        time_per_epoch =end_time - start_time
        logger.info("Epoch {} done. Time for this epoch: {:.3f}"
                    .format(epoch, time_per_epoch))



        if epoch % checkpoint_period == 0:
            make_dirs(config.save_models_path)
            model.save_model(epoch)

        if epoch % eval_period == 0:
            model.set_eval()
            for n_iter, (img, vid, camid, _, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(model.device)
                    feat = model.id_encoder(img)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


loss_total_meter = AverageMeter()
loss_reid_meter = AverageMeter()
loss_recons_meter = AverageMeter()
loss_recon_cross_meter = AverageMeter()
loss_gen_class_meter = AverageMeter()
acc_meter = AverageMeter()
acc_meter1 = AverageMeter()
main()