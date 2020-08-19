import os
import numpy as np
from torch.backends import cudnn
from utils.logger import setup_logger
import os
import argparse
import torch
import torch.nn as nn
import torchvision
import random
from config import *
from utils import *
from dataloader import make_dataloader
from model import Backbone
from  losses import make_loss
from solver import make_optimizer,WarmupMultiStepLR
import logging
from utils.meter import AverageMeter
from utils.metrics import R1_mAP,R1_mAP_eval,R1_mAP_Pseudo,R1_mAP_query_mining

if __name__ == '__main__':
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    #CONFIG PARSER
    config = get_args()
    output_path = config.output_path
    make_dirs(output_path)
    logger = setup_logger('reid_baseline',output_path,if_train=True)

    train_loader, val_loader, num_query, num_classes = make_dataloader(config)
    model = Backbone(num_classes,config)
    # if config.pretrain:
    #     model.load_param_finetune(config.m_pretrain_path)

    loss_func, center_criterion = make_loss(config, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer( model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, [40,70], 0.1,
                                  0.01,
                                  10, 'linear')

    log_period = config.log_interval
    checkpoint_period = config.save_model_interval
    eval_period = config.test_interval

    device = "cuda"
    epochs = 80

    logger = logging.getLogger("reid_baseline.train")
    logger.info('start training')

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm='yes')
    model.base._freeze_stages()
    logger.info('Freezing the stages number:{}'.format(-1))
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            score, feat = model(img, target)
            loss = loss_func(score, feat, target)

            loss.backward()
            optimizer.step()
            acc = (score.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            make_dirs(config.save_models_path)
            torch.save(model.state_dict(), os.path.join(config.save_models_path, config.model_name + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _,_) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    feat = model(img)
                    evaluator.update((feat, vid, camid))

            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))



