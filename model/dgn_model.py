from model import Backbone
import os
import torch
import torch.optim as optim
from .networks import ID_encoder,Style_encoder,Discriminator,weights_init,MLP,F_Decoder
from solver.lr_scheduler import WarmupMultiStepLR
from utils import make_dirs,os_walk
import shutil
import torch.nn as nn
from losses.triplet_loss import TripletLoss
from utils import time_now


class DGN(object):
    """
    The model which incorporates identity shuffing and reconstruction loss
    """
    def __init__(self,num_classes,config):
        self.config = config
        self.num_classes = num_classes
        self.device  =  torch.device('cuda')
        self._init_networks()
        self._init_optimizers()
        self._init_criterion()



    def _init_networks(self):
        #init models
        self.id_encoder = ID_encoder(self.num_classes,self.config).to(self.device)
        self.style_encoder = Style_encoder(self.config).to(self.device)
        self.decoder = F_Decoder(2,2,self.style_encoder.output_dim,3,0,'adain','relu','reflect').to(self.device)
        self.discriminator = Discriminator(n_layer=4, middle_dim=32, num_scales=2).to(self.device)
        self.discriminator.apply(weights_init('gaussian'))
        self.mlp = MLP(2048, self.get_num_adain_params(self.decoder), 256, 3, norm='none', activ='relu').to(self.device)

        self.model_list = []
        self.model_list.append(self.id_encoder)
        self.model_list.append(self.style_encoder)
        self.model_list.append(self.discriminator)

    def _init_optimizers(self):
        id_params = list(self.id_encoder.parameters())
        style_params = list(self.style_encoder.parameters())
        dis_params = list(self.discriminator.parameters())

        if self.config.optimizer_name == 'sgd':
            self.gen_optimizer = optim.SGD(id_params+style_params, lr=1e-3, weight_decay = 0.0005, momentum= 0.9)
            self.dis_optimizer = optim.SGD(dis_params, lr=1e-3, weight_decay = 0.0005, momentum=0.9)
        else:
            self.gen_optimizer = optim.Adam(id_params+style_params,lr=1e-3,betas =[0.9,0.999],weight_decay=5e-4)
            self.dis_optimizer = optim.Adam(dis_params,lr= 1e-3, betas =[0.9,0.999],weight_decay=5e-4)

        self.gen_lr_scheduler = WarmupMultiStepLR(self.gen_optimizer, [40,70], 0.1,
                                  0.01,
                                  10, 'linear')
        self.dis_lr_scheduler = WarmupMultiStepLR(self.dis_optimizer, [40,70], 0.1,
                                  0.01,
                                  10, 'linear')

    def _init_criterion(self):
        self.id_loss = nn.CrossEntropyLoss()
        self.reconst_loss = nn.L1Loss()
        self.triplet_loss = TripletLoss(0.5)

    def lr_scheduler_step(self):
        self.gen_lr_scheduler.step()
        self.dis_lr_scheduler.step()

    def encode(self, images):
        # encode  an image to foreground vector and background vector
        id_scores,id_global_feat = self.id_encoder(images)
        style_feature_maps = self.style_encoder(images)
        return id_global_feat,style_feature_maps,id_scores


    def decode(self, id, style):
        adain_params = self.mlp(id)
        self.assign_adain_params(adain_params, self.decoder)
        images = self.decoder(style)
        return images

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]


    def save_model(self, save_epoch):
        # save model
        for ii, _ in enumerate(self.model_list):
            model_dir_path = self.config.save_models_path + 'models_{}'.format(save_epoch)
            if os.path.exists(self.config.save_models_path):
                make_dirs(model_dir_path)
            else:
                make_dirs(self.config.save_models_path)
                make_dirs(model_dir_path)
            torch.save(self.model_list[ii].state_dict(),
                       os.path.join(model_dir_path, 'model-{}_{}.pkl'.format(ii, save_epoch)))

        if self.config.max_save_model_num > 0:
            root, dirs, files = os_walk(self.config.save_models_path)
            total_save_models = len(dirs)
            if total_save_models > self.config.max_save_model_num:
                delet_index = total_save_models - self.config.max_save_model_num
                for to_delet in dirs[:delet_index]:
                    shutil.rmtree(self.config.save_models_path + to_delet)

    def resume_model(self, resume_epoch):
        for i, _ in enumerate(self.model_list):
            self.model_list[i].load_state_dict(
                torch.load(os.path.join(self.config.save_models_path + 'models_{}'.format(resume_epoch),
                                        'model-{}_{}.pkl'.format(i, resume_epoch)))
            )
        print('Time:{}, successfully resume model from {}'.format(time_now(), resume_epoch))

    def resume_model_from_path(self, path, resume_epoch):
        for i, _ in enumerate(self.model_list):
            self.model_list[i].load_state_dict(
                torch.load(os.path.join(path + 'models_{}'.format(resume_epoch)), 'model-{}_{}'.format(i, resume_epoch))
            )

        # set the model into training mode

    def set_train(self):
        for i, _ in enumerate(self.model_list):
            self.model_list[i] = self.model_list[i].train()

    def set_eval(self):
        for i, _ in enumerate(self.model_list):
            self.model_list[i] = self.model_list[i].eval()




