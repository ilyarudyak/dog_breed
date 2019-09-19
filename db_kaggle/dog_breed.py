from fastai.vision import *
import pandas as pd

from data_prep import get_data
from submission_prep import prepare_submission


def train_learner(arch=models.resnet101):
    _, path_test, data = get_data(bs=32, size=299)
    learn = cnn_learner(data, base_arch=arch, metrics=accuracy)
    learn.fit_one_cycle(10)
    prepare_submission(data, learn, path_test)


if __name__ == '__main__':
    train_learner()
