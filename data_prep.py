import numpy as np
from glob import glob

from pathlib import Path
import os
home = str(Path.home())


def get_files():
    human_files = np.array(glob(os.path.join(home, "data/dog_breed/lfw/*/*")))
    dog_files = np.array(glob(os.path.join(home, "data/dog_breed/dogImages/*/*/*")))
    return human_files, dog_files


if __name__ == '__main__':
    human_files, dog_files = get_files()
    print('There are %d total human images.' % len(human_files))
    print('There are %d total dog images.' % len(dog_files))