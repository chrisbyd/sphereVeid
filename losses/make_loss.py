# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from  .sphere_loss import OhemSphereLoss,SphereLoss


def make_loss(cfg,num_classes):    # modified by gu
    SAMPLER = 'softmax_triplet'
    print(SAMPLER)
    feat_dim = 2048
    not_use_margin = True
    MARGIN = 0.3
    IF_LABELSMOOTH = 'off'
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    sphere_criterion = OhemSphereLoss(feat_dim,num_classes,thresh=0.8)
    sphere_criterion.cuda()
    if not_use_margin:
        triplet = TripletLoss()
        print("using soft triplet loss for training")
    else:
        triplet = TripletLoss(MARGIN)  # triplet loss
        print("using triplet loss with margin:{}".format(MARGIN))


    if IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)


    if  SAMPLER == 'softmax_triplet':
        def loss_func( feat, target):
                return sphere_criterion(feat,target) + triplet(feat, target)[0]

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


