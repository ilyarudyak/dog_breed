from fastai.vision import *
from fastai.metrics import error_rate

from data_prep import get_data_fastai


def get_learner():
    data = get_data_fastai()
    return cnn_learner(data, models.resnet50, metrics=accuracy)


def train():
    learn = get_learner()
    learn.fit_one_cycle(5)
    learn.save('stage-1')


if __name__ == '__main__':
    train()
