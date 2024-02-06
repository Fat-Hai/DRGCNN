import os
import sys
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.distributed import all_reduce, all_gather, ReduceOp
from utils.func import *
from modules.loss import *
from modules.scheduler import *
from utils.metrics import PerformanceEvaluator
from Encoder.my_dataset import generate_dataset
from modules.builder import generate_model

def train(cfg, model, train_dataset, val_dataset, estimator, logger=None):
    device = cfg.config_base.config_device
    optimizer = initialize_optimizer(cfg, model)
    train_sampler, val_sampler = initialize_sampler(cfg, train_dataset, val_dataset)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(cfg, optimizer)
    loss_function, loss_weight_scheduler = initialize_loss(cfg, train_dataset)
    train_loader, val_loader = initialize_dataloader(cfg, train_dataset, val_dataset, train_sampler, val_sampler)

    # start training
    model.train()
    max_indicator = 0
    avg_loss, avg_acc, avg_kappa = 0, 0, 0
    for epoch in range(1, cfg.config_train.config_epochs + 1):
        # resampling weight update

        if train_sampler:
            train_sampler.step()

        # update loss weights
        if loss_weight_scheduler:
            weight = loss_weight_scheduler.step()
            loss_function.weight = weight.to(device)

        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss = 0
        estimator.reset()
        progress = tqdm(enumerate(train_loader)) if cfg.config_base.config_progress else enumerate(train_loader)
        for step, train_data in progress:
            X, y = train_data
            X = X.to(device)
            y = y.to(device)
            y = select_target_type(y, cfg.config_train.config_criterion)

            # forward
            y_pred = model(X)
            loss = loss_function(y_pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)
            estimator.update_metrics(y_pred, y)
            avg_acc = estimator.get_accuracy(6)
            avg_kappa = estimator.get_kappa(6)


            message = 'epoch: [{} / {}], loss: {:.6f}, acc: {:.4f}, kappa: {:.4f}'\
                        .format(epoch, cfg.config_train.config_epochs, avg_loss, avg_acc, avg_kappa)

            progress.set_description(message)

        if not cfg.config_base.config_progress:
            print(message)


        # validation performance
        if epoch % cfg.config_train.config_eval_interval == 0:
            eval(cfg, model, val_loader, cfg.config_train.config_criterion, estimator, device)
            acc = estimator.get_accuracy(6)
            kappa = estimator.get_kappa(6)
            print('valid Acc: {}, kappa: {}'.format(acc, kappa))


            # save model
            indicator = kappa if cfg.config_train.config_kappa_prior else acc
            if  indicator > max_indicator:
                save_weights(model, os.path.join(cfg.config_base.config_save_path, 'best_epoch.pt'))
                max_indicator = indicator
                print_msg('Best in valida. Model save at {}'.format(cfg.config_base.config_save_path))

        if epoch % cfg.config_train.config_save_interval == 0:
            save_weights(model, os.path.join(cfg.config_base.config_save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            lr_scheduler.step()

    # save final model

    save_weights(model, os.path.join(cfg.config_base.config_save_path, 'last_epoch.pt'))




def evaluate(cfg, model, test_dataset, estimator):
    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=cfg.config_train.config_batch_size,
        num_workers=cfg.config_train.config_num_workers,
        pin_memory=cfg.config_train.config_pin_memory
    )

    print('Running on Test set...')
    eval(cfg, model, test_loader, cfg.config_train.config_criterion, estimator, cfg.config_base.config_device)


    print('========================================')
    print('Finished! test acc: {}'.format(estimator.get_accuracy(6)))
    print('Confusion Matrix:')
    print(estimator.conf_mat)
    print('quadratic kappa: {}'.format(estimator.get_kappa(6)))
    print('========================================')


def eval(cfg, model, dataloader, criterion, estimator, device):
    model.eval()
    torch.set_grad_enabled(False)

    estimator.reset()
    for test_data in dataloader:
        X, y = test_data
        X = X.to(device)
        y = y.to(device)
        y = select_target_type(y, criterion)

        y_pred = model(X)

        estimator.update_metrics(y_pred, y)

    model.train()
    torch.set_grad_enabled(True)


# define weighted_sampler
def initialize_sampler(cfg, train_dataset, val_dataset):

    sampling_strategy = cfg.config_data.config_sampling_strategy

    val_sampler = None
    if sampling_strategy == 'class_balanced':
        train_sampler = ScheduledWeightedSampler(train_dataset, 1)
    elif sampling_strategy == 'progressively_balanced':
        train_sampler = ScheduledWeightedSampler(train_dataset, cfg.data.sampling_weights_decay_rate)
    elif sampling_strategy == 'instance_balanced':
        train_sampler = None
    else:
        raise NotImplementedError('Not implemented resampling strategy.')

    return train_sampler, val_sampler


# define data loader
def initialize_dataloader(cfg, train_dataset, val_dataset, train_sampler, val_sampler):
    batch_size = cfg.config_train.config_batch_size
    num_workers = cfg.config_train.config_num_workers
    pin_memory = cfg.config_train.config_pin_memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# define loss and loss weights scheduler
def initialize_loss(cfg, train_dataset):
    criterion = cfg.config_train.config_criterion
    criterion_args = cfg.config_criterion_args[criterion]

    weight = None
    loss_weight_scheduler = None
    loss_weight = cfg.config_train.config_loss_weight
    if criterion == 'cross_entropy':
        if loss_weight == 'balance':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, 1)
        elif loss_weight == 'dynamic':
            loss_weight_scheduler = LossWeightsScheduler(train_dataset, cfg.config_train.config_loss_weight_decay_rate)
        elif isinstance(loss_weight, list):
            assert len(loss_weight) == len(train_dataset.classes)
            weight = torch.as_tensor(loss_weight, dtype=torch.float32, device=cfg.config_base.config_device)
        loss = nn.CrossEntropyLoss(weight=weight, **criterion_args)
    elif criterion == 'mean_square_error':
        loss = nn.MSELoss(**criterion_args)
    elif criterion == 'mean_absolute_error':
        loss = nn.L1Loss(**criterion_args)
    elif criterion == 'focal_loss':
        loss = FocalLoss(**criterion_args)
    else:
        raise NotImplementedError('Not implemented loss function.')

    loss_function = WarpedLoss(loss, criterion)
    return loss_function, loss_weight_scheduler


# define optmizer
def initialize_optimizer(cfg, model):
    optimizer_strategy = cfg.config_solver.config_optimizer
    learning_rate = cfg.config_solver.config_learning_rate
    weight_decay = cfg.config_solver.config_weight_decay
    momentum = cfg.config_solver.config_momentum
    nesterov = cfg.config_solver.config_nesterov

    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(cfg, optimizer):
    warmup_epochs = cfg.config_train.config_warmup_epochs
    learning_rate = cfg.config_solver.config_learning_rate
    scheduler_strategy = cfg.config_solver.config_lr_scheduler

    if not scheduler_strategy:
        lr_scheduler = None
    else:
        scheduler_args = cfg.config_scheduler_args[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        else:
            raise NotImplementedError('Not implemented learning rate scheduler.')

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
