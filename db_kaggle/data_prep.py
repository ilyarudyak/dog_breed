from fastai.vision import *


def get_data(bs=64, size=224):
    path = Path.home() / 'data/db_kaggle'
    path_test = path / 'test'
    data = ImageDataBunch.from_csv(path=path,
                                   folder='train',
                                   csv_labels='labels.csv',
                                   ds_tfms=get_transforms(),
                                   suffix='.jpg',
                                   test=path_test,
                                   size=size,
                                   bs=bs,
                                   num_workers=0)
    data.normalize(imagenet_stats)
    return path, path_test, data

