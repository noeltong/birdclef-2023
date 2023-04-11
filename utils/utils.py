import random
import os
import numpy as np
import torch
from models.lars import LARS
import pandas as pd
import sklearn.metrics


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_optim(model, config):
    init_lr = config.optim.initial_lr * config.training.batch_size / 256

    if config.optim.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=init_lr,
            weight_decay=config.optim.weight_decay,
        )
    elif config.optim.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=init_lr,
            weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer.lower() == 'lars':
        optimizer = LARS(
            model.parameters(),
            lr=init_lr,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(
            f'{config.optim.optimizer} is not supported.')

    if config.optim.schedule.lower() is not None and config.optim.schedule.lower() == 'cosineannealinglr':
        if config.optim.warmup_epochs is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.optim.min_lr,
            )
        else:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.optim.initial_lr / config.optim.warmup_epochs,
                total_iters=config.optim.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs - config.optim.warmup_epochs,
                eta_min=config.optim.min_lr
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.optim.warmup_epochs]
            )
    else:
        raise ValueError(f'{config.optim.schedule} is not supported.')
    
    return optimizer, scheduler


def padded_cmap(solution, submission, padding_factor=5):
    solution = solution.drop(['row_id'], axis=1, errors='ignore')
    submission = submission.drop(['row_id'], axis=1, errors='ignore')
    new_rows = []
    for i in range(padding_factor):
        new_rows.append([1 for i in range(len(solution.columns))])
    new_rows = pd.DataFrame(new_rows)
    new_rows.columns = solution.columns
    padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
    padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
    score = sklearn.metrics.average_precision_score(
        padded_solution.values,
        padded_submission.values,
        average='macro',
    )
    return score