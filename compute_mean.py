import os

import numpy as np
from PIL import Image


def compute_mean(data_path, data_index):
    fid = open(data_index, 'r')
    mean = 0

    for item in fid.readlines():
        id = item.strip().split(' ')[0]
        img = os.path.join(data_path, id)
        img_data = np.asarray(Image.open(img))
        img_slice = img_data.reshape(-1, 3)
        mean += np.mean(img_slice, axis=0)
    # data mean & std
    data_mean = mean / 1449
    print('Img mean: {}'.format(data_mean))


if __name__ == '__main__':
    DATA_PATH = ''
    DATA_INDEX = ''
    compute_mean(DATA_PATH, DATA_INDEX)
