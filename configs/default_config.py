from ml_collections.config_dict import ConfigDict


def default_config():

    cfg = ConfigDict()

    pretrain = cfg.pretrain = ConfigDict()
    pretrain.num_epochs = 500


    return cfg
