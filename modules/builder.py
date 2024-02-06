import timm
import torch
import torch.nn as nn
from torchvision import models
from timm.models.nfnet import nfnet_f0
from timm.models.nfnet import dm_nfnet_f0
from utils.func import print_msg, select_out_features

def generate_model(cfg):
    model = build_model(cfg)

    if cfg.config_train.config_checkpoint:
        weights = torch.load(cfg.config_train.config_checkpoint)
        # for k in list(weights.keys()):
        #     if "head" in k:
        #         del weights[k]
        print(model.load_state_dict(weights, strict=False))
        # model.load_state_dict(weights, strict=False)
        print_msg('Load weights form {}'.format(cfg.config_train.config_checkpoint))


    model = model.to(cfg.config_base.config_device)

    return model


def build_model(cfg):
    network = cfg.config_train.config_network
    out_features = select_out_features(
        cfg.config_data.config_num_classes,
        cfg.config_train.config_criterion
    )

    if cfg.config_train.config_backend == 'timm':
        model = timm.create_model(
            cfg.config_train.config_network,
            out_features,
            cfg.config_train.pretrained
            )

    elif cfg.config_train.config_backend == 'torchvision':
        model = build_torchvision_model(
            cfg.config_train.config_network,
            out_features,
            cfg.config_train.config_pretrained
        )
    return model



def build_torchvision_model(network, num_classes, pretrained=False):
    model = BUILDER[network](pretrained=pretrained)
    if 'efficientnet' in network:
        print("Model: EfficentNetV2",)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )
    else:
        raise NotImplementedError('Not implemented network.')

    return model


BUILDER = {'efficientnet_v2_m':models.efficientnet_v2_m}