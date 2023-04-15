import torch
from torch import nn
from torch.nn import functional as F
from utils.data import get_dataloader
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from utils.time import time_calculator
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from utils.prefetch import prefetcher
from models.lars import LARS
from models.convnext import ConvNeXt
from utils.utils import seed_everything
from models.builder import SimSiam
from utils.utils import get_optim
from models.ema import ExponentialMovingAverage
# from sklearn.metrics import average_precision_score
from utils.utils import padded_cmap
import numpy as np
import pandas as pd


def train(config, workdir, train_dir='pretrain'):
    """Runs the training pipeline.

    Args:
    config: ml_collections.ConfigDict(), config of the project
    workdir: directory to store files.
    """

    assert config.distributed, "Distributed train is needed!"
    torch.backends.cudnn.benchmark = True
    workdir = os.path.join(workdir, train_dir)

    # -------------------
    # Initialize DDP
    # -------------------

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # -------------------
    # seeds
    # -------------------

    seed_everything(config.seed + rank)

    if config.use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    # -----------------------------
    # Create directories for data
    # -----------------------------

    log_dir = os.path.join(workdir, 'tensorboard')
    ckpt_dir = os.path.join(workdir, 'ckpt')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------
    # Loggers
    # -------------------

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s: %(message)s')

    fh = logging.FileHandler(os.path.join(
        workdir, 'train_log.log'), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # -------------------
    # Load data
    # -------------------

    if rank == 0:
        logger.info('Loading data...')

    train_loader, test_loader, train_sampler, test_sampler = get_dataloader(config, mode='train')
    dist.barrier()

    if rank == 0:
        logger.info('Data loaded.')

    # -------------------
    # Initialize model
    # -------------------

    if rank == 0:
        logger.info('Begin model initialization...')

    model = SimSiam(config=config)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[rank])

    model_without_ddp = model.module

    if config.model.ema:
        adjust = config.training.batch_size * \
            config.model.ema_steps / config.training.num_epochs
        alpha = 1 - config.model.ema_rate
        alpha = 1 - min(1.0, alpha * adjust)
        model_ema = ExponentialMovingAverage(
            model_without_ddp, device=device, decay=alpha)

    if rank == 0:
        logger.info("Models initialized.")

    dist.barrier()

    # -------------------
    # define optimization
    # -------------------

    if rank == 0:
        logger.info('Handling optimizations...')

    optimizer, scheduler = get_optim(model, config)
    
    criterion = nn.CosineSimilarity(dim=1).cuda()

    if rank == 0:
        logger.info('Completed.')

    # -------------------
    # training loop
    # -------------------

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    time_logger = time_calculator()

    best_loss = 999999999.
    iters_per_epoch = len(train_loader)

    dist.barrier()
    torch.cuda.empty_cache()

    for epoch in range(config.training.num_epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        model.train()
        train_loss_epoch = 0

        if rank == 0:
            logger.info(f'Start training epoch {epoch + 1}.')

        # ----------------------------
        # initialize data prefetcher
        # ----------------------------

        train_prefetcher = prefetcher(train_loader, rank, mode='train')
        x1, x2 = train_prefetcher.next()
        i = 0

        # ----------------------------
        # run the training process
        # ----------------------------

        while x1 is not None:
            with torch.cuda.amp.autocast(enabled=True):
                p1, p2, z1, z2 = model(x1, x2)
                loss = 1 - (criterion(p1, z2).mean() + criterion(p2, z1).mean()) / 2.

            train_loss_epoch += loss.item()

            if rank == 0:
                writer.add_scalar("Loss", loss.item(),
                                  epoch * iters_per_epoch + i)

            logger.info(
                f'Epoch: {epoch + 1}/{config.training.num_epochs}, Iter: {i + 1}/{iters_per_epoch}, Loss: {loss.item():.6f}, Time: {time_logger.time_length()}, Device: {rank}')

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if config.model.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.model.clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            if config.model.ema and i % config.model.ema_steps == 0:
                model_ema.update_parameters(model)

            x1, x2 = train_prefetcher.next()
            i += 1

        scheduler.step()

        avg_train_loss_epoch = train_loss_epoch / iters_per_epoch
        dist.barrier()
        if rank == 0:
            logger.info(
                f'Epoch: {epoch + 1}/{config.training.num_epochs}, Avg loss: {avg_train_loss_epoch}, Time: {time_logger.time_length()}')

        # save snapshot periodically

        if (epoch + 1) % config.training.save_ckpt_freq == 0:
            if rank == 0:
                logger.info(f'Saving snapshot at epoch {epoch + 1}')
                snapshot = {
                    'epoch': epoch + 1,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict()
                }
                if config.model.ema:
                    snapshot['model_ema'] = model_ema.state_dict()
                torch.save(snapshot, os.path.join(
                    ckpt_dir, f'{epoch+1}_loss_{avg_train_loss_epoch:.2f}.pth'))

        is_best = avg_train_loss_epoch < best_loss
        if is_best:
            best_loss = avg_train_loss_epoch
            if rank == 0:
                logger.info(
                    f'Saving best model state dict at epoch {epoch + 1}.')
                torch.save(model_ema.state_dict() if config.model.ema else model_without_ddp.state_dict,
                           os.path.join(ckpt_dir, 'best.pth'))

        # Report loss on eval dataset periodically

        if (epoch + 1) % config.training.eval_freq == 0:
            if rank == 0:
                logger.info(f'Start evaluate at epoch {epoch + 1}.')

            eval_model = model_ema if config.model.ema else model_without_ddp
            # eval_model = model_without_ddp
            with torch.inference_mode():
                eval_model.eval()
                iters_per_eval = len(test_loader)
                loss_sum = 0

                # ----------------------------
                # initialize data prefetcher
                # ----------------------------

                test_prefetcher = prefetcher(test_loader, rank, mode='train')
                x1, x2 = test_prefetcher.next()
                i = 0

                while x1 is not None:
                    with torch.cuda.amp.autocast(enabled=True):
                        p1, p2, z1, z2 = model(x1, x2)
                        loss = 1 - (criterion(p1, z2).mean() + criterion(p2, z1).mean()) / 2.

                    loss_sum += loss.item()
                    logger.info(
                        f'Epoch: {epoch + 1}/{config.training.num_epochs}, Iter: {i + 1}/{iters_per_eval}, Loss: {loss.item():.6f}, Time: {time_logger.time_length()}, Device: {rank}')

                    x1, x2 = test_prefetcher.next()
                    i += 1

                avg_eval_loss_epoch = loss_sum / iters_per_eval
                if rank == 0:
                    writer.add_scalar('Eval loss', avg_eval_loss_epoch, epoch)

            if rank == 0:
                logger.info(
                    f'Epoch: {epoch + 1}/{config.training.num_epochs}, Avg eval loss: {avg_eval_loss_epoch}, Time: {time_logger.time_length()}')

        dist.barrier()

    if rank == 0:
        logger.info(
            f'Training complete.\nTotal time:, {time_logger.time_length()}\nFinal loss:, {avg_train_loss_epoch}\nBest loss:, {best_loss}\nFinal eval loss:, {avg_eval_loss_epoch}')


def tune(config, workdir, tune_dir):
    """Runs the finetune pipeline.

    Args:
    config: ml_collections.ConfigDict(), config of the project
    workdir: directory to store files.
    """

    assert config.distributed, "Distributed train is needed!"
    torch.backends.cudnn.benchmark = True
    workdir = os.path.join(workdir, tune_dir)

    # -------------------
    # Initialize DDP
    # -------------------

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # -------------------
    # seeds
    # -------------------

    seed_everything(config.seed + rank)

    if config.use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    # -----------------------------
    # Create directories for data
    # -----------------------------

    log_dir = os.path.join(workdir, 'tensorboard')
    ckpt_dir = os.path.join(workdir, 'ckpt')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # -------------------
    # Loggers
    # -------------------

    logger = logging.getLogger('tune')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s: %(message)s')

    fh = logging.FileHandler(os.path.join(
        workdir, 'tune_log.log'), encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # -------------------
    # Load data
    # -------------------

    if rank == 0:
        logger.info('Loading data...')

    train_loader, test_loader, train_sampler, test_sampler = get_dataloader(config, mode='tune')
    dist.barrier()

    ckpt = torch.load(config.finetune.ckpt_path, map_location='cpu')

    if rank == 0:
        logger.info('Data loaded.')

    # -------------------
    # Initialize model
    # -------------------

    if rank == 0:
        logger.info('Begin model initialization...')

    model = SimSiam(config=config)
    model.load_state_dict(ckpt['model'] if not config.model.ema else ckpt['model_ema'], strict=False)
    model = model.encoder
    model.head = nn.Linear(config.model.dims[-1], config.finetune.num_classes)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    model = DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        logger.info("Models initialized. Loaded pretrained checkpoints.")

    dist.barrier()

    # -------------------
    # define optimization
    # -------------------

    if rank == 0:
        logger.info('Handling optimizations...')

    optimizer, scheduler = get_optim(model, config)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.optim.label_smoothing).cuda()

    if rank == 0:
        logger.info('Completed.')

    # -------------------
    # training loop
    # -------------------

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=log_dir)
    time_logger = time_calculator()

    best_loss = 999999999.
    iters_per_epoch = len(train_loader)

    dist.barrier()
    torch.cuda.empty_cache()

    for epoch in range(config.training.num_epochs):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        model.train()
        train_loss_epoch = 0
        best_eval_score = 0.

        if rank == 0:
            logger.info(f'Start training epoch {epoch + 1}.')

        # ----------------------------
        # initialize data prefetcher
        # ----------------------------

        train_prefetcher = prefetcher(train_loader, rank, mode='tune')
        x, y = train_prefetcher.next()
        i = 0

        # ----------------------------
        # run the training process
        # ----------------------------

        while x is not None:
            with torch.cuda.amp.autocast(enabled=True):
                out = model(x)
                loss = criterion(out, y)

            train_loss_epoch += loss.item()

            if rank == 0:
                writer.add_scalar("Loss", loss.item(),
                                  epoch * iters_per_epoch + i)

            logger.info(
                f'Epoch: {epoch + 1}/{config.training.num_epochs}, Iter: {i + 1}/{iters_per_epoch}, Loss: {loss.item():.6f}, Time: {time_logger.time_length()}, Device: {rank}')

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            if config.model.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), config.model.clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            x, y = train_prefetcher.next()
            i += 1

        scheduler.step()

        avg_train_loss_epoch = train_loss_epoch / iters_per_epoch
        dist.barrier()
        if rank == 0:
            logger.info(
                f'Epoch: {epoch + 1}/{config.training.num_epochs}, Avg loss: {avg_train_loss_epoch:.4f}, Time: {time_logger.time_length()}')

        # save snapshot periodically
        
        if (epoch + 1) % config.training.save_ckpt_freq == 0:
            if rank == 0:
                logger.info(f'Saving snapshot at epoch {epoch + 1}')
                snapshot = {
                    'epoch': epoch + 1,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict()
                }
                torch.save(snapshot, os.path.join(
                    ckpt_dir, f'{epoch+1}_loss_{avg_train_loss_epoch:.2f}.pth'))

        # Report loss on eval dataset periodically

        if (epoch + 1) % config.training.eval_freq == 0:
            if rank == 0:
                logger.info(f'Start evaluate at epoch {epoch + 1}.')

            # eval_model = model_without_ddp

            with torch.inference_mode():
                model.eval()
                iters_per_eval = len(test_loader)
                loss_sum = 0

                # ----------------------------
                # initialize data prefetcher
                # ----------------------------

                test_prefetcher = prefetcher(test_loader, rank, mode='tune')
                x, y = test_prefetcher.next()
                i = 0

                preds, gts = [], []

                while x is not None:
                    with torch.cuda.amp.autocast(enabled=True):
                        out = model(x)
                        loss = criterion(out, y)

                    preds.append(F.softmax(out.detach()).cpu().numpy().squeeze())
                    gts.append(np.eye(config.finetune.num_classes)[y.detach().cpu().numpy(), :].squeeze())

                    loss_sum += loss.item()
                    logger.info(
                        f'Epoch: {epoch + 1}/{config.training.num_epochs}, Iter: {i + 1}/{iters_per_eval}, Loss: {loss.item():.6f}, Time: {time_logger.time_length()}, Device: {rank}')

                    x, y = test_prefetcher.next()
                    i += 1

            preds = pd.DataFrame(np.concatenate(preds, axis=0))
            gts = pd.DataFrame(np.concatenate(gts, axis=0))

            # score = average_precision_score(gts, preds, average='macro')
            score = padded_cmap(gts, preds)
            dist.barrier()
            if score > best_eval_score:
                logger.info(f'Obtained new highest score {score} against previous best at {best_eval_score}.')
                best_eval_score = score
                if rank == 0:
                    logger.info(
                        f'Saving best model state dict at epoch {epoch + 1}.')
                    torch.save(model.module.state_dict(), os.path.join(ckpt_dir, 'best.pth'))

            avg_eval_loss_epoch = loss_sum / iters_per_eval
            if rank == 0:
                writer.add_scalar('Eval loss', avg_eval_loss_epoch, epoch)
                writer.add_scalar('Score', score, epoch)

            dist.barrier()
            if rank == 0:
                logger.info(
                    f'Epoch: {epoch + 1}/{config.training.num_epochs}, Avg eval loss: {avg_eval_loss_epoch:4f}, Eval Score: {score:.4f} Time: {time_logger.time_length()}')
            
            dist.barrier()

        dist.barrier()

    if rank == 0:
        logger.info(
            f'Training complete.\nTotal time:, {time_logger.time_length()}\nFinal loss:, {avg_train_loss_epoch}\nBest loss:, {best_loss}\nFinal eval loss:, {avg_eval_loss_epoch}')


def eval(config, workdir):
    pass