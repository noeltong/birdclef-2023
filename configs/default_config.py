from ml_collections.config_dict import ConfigDict


def get_config():

    cfg = ConfigDict()

    # ----------------
    # Train
    # ----------------

    training = cfg.training = ConfigDict()
    training.num_epochs = 500
    training.batch_size = 16
    training.save_ckpt_freq = 100
    training.eval_freq = 50

    # ----------------
    # Model
    # ----------------

    model = cfg.model = ConfigDict()
    model.depths = [3, 3, 27, 3]
    model.dims = [128, 256, 512, 1024]
    model.drop_path_rate = 0.1
    model.head_out_dim = 2048
    model.pred_dim = 512
    model.clip_grad_norm = 1.


    # ----------------
    # Optimization
    # ----------------

    cfg.optim = optim = ConfigDict()
    optim.optimizer = 'LARS'
    optim.schedule = 'CosineAnnealingLR'
    optim.grad_clip = 1.
    optim.initial_lr = 2e-3
    optim.weight_decay = 0.1
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = 25


    data = cfg.data = ConfigDict()
    data.csv_path = '/public/home/tongshq/kaggle/birdclef/model_data/data_all.csv'
    data.num_workers = 4


    cfg.seed = 42
    cfg.distributed = True
    cfg.use_deterministic_algorithms = True


    return cfg
