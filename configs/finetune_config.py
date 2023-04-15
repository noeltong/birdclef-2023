from ml_collections.config_dict import ConfigDict
from configs.pretrain_config import get_config as default_config


def get_config():

    cfg = default_config()

    # ----------------
    # Train
    # ----------------

    training = cfg.training
    training.num_epochs = 100
    training.batch_size = 32
    # 32 -> 8G, 64 -> 12G
    training.save_ckpt_freq = 10
    training.eval_freq = 1

    # ----------------
    # Finetune
    # ----------------

    finetune = cfg.finetune = ConfigDict()
    finetune.ckpt_path = '/public/home/tongshq/kaggle/birdclef/workspace/convnext_small/pretrain/ckpt/30_loss_0.22.pth'
    finetune.num_classes = 264

    # ----------------
    # Model
    # ----------------

    model = cfg.model


    # ----------------
    # Optimization
    # ----------------

    optim = cfg.optim
    optim.optimizer = 'AdamW'
    optim.schedule = 'CosineAnnealingLR'
    optim.grad_clip = 1.
    optim.initial_lr = 1e-3
    optim.weight_decay = 0.0001
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = None
    optim.label_smoothing = 0.1


    data = cfg.data
    data.csv_tune_path = '/public/home/tongshq/kaggle/birdclef/model_data/finetune_data.csv'
    data.num_workers = 4
    data.prefetch_factor = 1


    cfg.seed = 42
    cfg.distributed = True
    cfg.use_deterministic_algorithms = True
    cfg.debug = False


    return cfg
