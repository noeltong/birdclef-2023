import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm


def list_classes():
    idx = []

    idx.extend(os.listdir('/storage/data/tongshq/kaggle/birdclef/birdclef-2021/train_short_audio'))
    idx.extend(os.listdir('/storage/data/tongshq/kaggle/birdclef/birdclef-2022/train_audio'))

    result = []

    for i in tqdm(idx, total=len(idx)):
        if i not in result:
            result.append(i)

    result.sort()
    os.makedirs('model_data', exist_ok=True)
    pd.DataFrame(result).to_csv('model_data/classes.csv', header=None, index=None)


def list_paths_classes():

    classes = []

    with open('/public/home/tongshq/kaggle/birdclef/model_data/classes.csv', 'r') as f:
        for idx in f.readlines():
            classes.append(idx.strip())

    def path_label_replace(df, dir):
        path = os.path.join(dir, df['primary_label'], df['filename'])
        label = classes.index(df['primary_label'])

        return [label, path]

    df1 = pd.read_csv('/storage/data/tongshq/kaggle/birdclef/birdclef-2021/train_metadata.csv')
    df1 = df1.loc[:, ['primary_label', 'filename']]
    df1 = df1.apply(path_label_replace, dir='/storage/data/tongshq/kaggle/birdclef/birdclef-2021/train_short_audio', axis=1, result_type='expand')
    df1.columns = ['label', 'path']

    df2 = pd.read_csv('/storage/data/tongshq/kaggle/birdclef/birdclef-2022/train_metadata.csv')
    df2 = df2.loc[:, ['primary_label', 'filename']]
    df2 = df2.apply(path_label_replace, dir='/storage/data/tongshq/kaggle/birdclef/birdclef-2022/train_audio', axis=1, result_type='expand')
    df2.columns = ['label', 'path']

    result = pd.concat([df1, df2], axis=0)
    print(f'Number of samples: {result.shape[0]}')
    result.to_csv('model_data/data_all.csv', index=None)



if __name__ == '__main__':
    list_classes()
    list_paths_classes()
