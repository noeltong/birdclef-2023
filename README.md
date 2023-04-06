# Contrastive Pre-Training for BirdCLEF Classification

This is a simple baseline for pre-training of BirdCLEF classification. The official website of this competetion is on this [kaggle page](https://www.kaggle.com/competitions/birdclef-2023/overview). We used publically available BirdCLEF 2021 and BirdCLEF 2022 dataset for contrastive pretraining using [SimSiam](https://github.com/facebookresearch/simsiam).

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