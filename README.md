# Contrastive Pre-Training for BirdCLEF Classification

## Train

```shell
torchrun --nproc_per_node=4 main.py
    --mode train
    --config [path to config file]
    --workdir [name of work directory]
```

## Finetune

```shell
torchrun --nproc_per_node=4 main.py
    --mode tune
    --config [path to config file]
    --workdir [name of work directory]
    --tunedir [name of finetune directory]
```

## Evaluation

Evaluation codes are ignored due to kaggle competetion.