import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from utils.func import *
from train import train, evaluate
from utils.metrics import PerformanceEvaluator
from Encoder.my_dataset import generate_dataset
from modules.builder import generate_model


def main():
    args = parse_configuration()
    cfg = load_config(args.config)
    save_path = cfg.config_base.config_save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    copy_config(args.config, cfg.config_base.config_save_path)
    worker(cfg)


def worker(cfg):
    # train
    model = generate_model(cfg)
    total_param = 0
    for param in model.parameters():
        total_param+=param.numel()
    print("Parameter: %.2fM"%(total_param/1e6))
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)
    estimator = PerformanceEvaluator(cfg.config_train.config_criterion, cfg.config_data.config_num_classes)
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,

    )

    # test
    print('This is the performance of the best validation model:')
    checkpoint = os.path.join(cfg.config_base.config_save_path, 'best_validation_weights.pt')
    cfg.config_train.config_checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)

    print('This is the performance of the final model:')
    checkpoint = os.path.join(cfg.config_base.config_save_path, 'final_weights.pt')
    cfg.config_train.config_checkpoint = checkpoint
    model = generate_model(cfg)
    evaluate(cfg, model, test_dataset, estimator)

if __name__ == '__main__':
    main()
