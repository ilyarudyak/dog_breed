from fastai.vision import *
from fastai.metrics import error_rate

from data_prep import get_data_fastai
from dog_breeds import test_model


def get_learner():
    data = get_data_fastai()
    return cnn_learner(data, models.resnet50, metrics=accuracy)


def train():
    learn = get_learner()
    learn.fit_one_cycle(5)
    learn.save('stage-1')


def test_model_fastai(model_type=models.resnet50, stage='stage-1'):
    data = get_data_fastai()
    learn = cnn_learner(data, model_type, metrics=accuracy)
    learn.load('stage-1')
    test_model(learn.model)


if __name__ == '__main__':
    # train()
    test_model_fastai()