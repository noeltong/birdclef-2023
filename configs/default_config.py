from ml_collections.config_dict import ConfigDict


def default_config():

    cfg = ConfigDict()

    pretrain = cfg.pretrain = ConfigDict()
    pretrain.num_epochs = 500


    # ----------------
    # Optimization
    # ----------------

    cfg.optim = optim = ConfigDict()
    optim.optimizer = 'LAMB'
    optim.schedule = 'CosineAnnealingLR'
    optim.grad_clip = 1.
    optim.initial_lr = 2.5e-4
    optim.weight_decay = 0.1
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = 25


    data = cfg.data = ConfigDict()
    data.csv_path = '/public/home/tongshq/kaggle/birdclef/model_data/data_all.csv'
    data.num_workers = 4


    cfg.seed = 42


    return cfg
