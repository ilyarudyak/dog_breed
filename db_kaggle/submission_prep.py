from fastai.vision import *
import pandas as pd


def get_numeric_part(file_path, path_test):
    low = len(str(path_test)) + 1
    return str(file_path)[low:-4]


def get_predictions(learn):
    preds, _ = learn.get_preds(DatasetType.Test)
    return preds


def prepare_submission(data, learn, path_test):
    preds = get_predictions(learn)
    df = pd.DataFrame(preds.numpy(), columns=data.classes)
    filenames = [get_numeric_part(fp, path_test) for fp in data.test_ds.items]
    df.insert(0, "id", filenames)
    df_sorted = df.sort_values(by='id')
    df_sorted.to_csv('submission.csv', index=False)



