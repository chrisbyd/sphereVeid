import torch


def make_optimizer( model, center_criterion):
    params = []
    BASE_LR = 1e-2
    WEIGHT_DECAY = 0.0005
    BIAS_LR_FACTOR= 1
    WEIGHT_DECAY_BIAS = 0.0005
    OPTIMIZER_NAME = 'SGD'
    MOMENTUM = 0.9
    CENTER_LR = 0.5
    LARGE_FC_LR = False
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = BASE_LR
        weight_decay = WEIGHT_DECAY
        if "bias" in key:
            lr = BASE_LR * BIAS_LR_FACTOR
            weight_decay = WEIGHT_DECAY_BIAS
        if LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if  OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, OPTIMIZER_NAME)(params, momentum=MOMENTUM)
    else:
        optimizer = getattr(torch.optim, OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=CENTER_LR)

    return optimizer, optimizer_center
