from __future__ import print_function

import concurrent.futures
import glob
import os
import time

import numpy as np
from PIL import Image

from pallete import colours_cityscapes, colours_voc12, colours_context

NUM_CLASSES = 19
NUM_TOTAL = 1525
DATA_LIST_PATH = ''
DATA_DIRECTORY = ''
SAVE_DIR = ''


def decode_labels(mask, num_images=1, num_classes=19):
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = colours_cityscapes[k]
        outputs[i] = np.array(img)
    return outputs


def visualize(mask):
    gary_label = np.array(Image.open(mask))
    id = mask.split('/')[-1][:-4]
    gary_label = gary_label[np.newaxis, :, :, np.newaxis]
    color_label = decode_labels(gary_label, num_classes=NUM_CLASSES)
    im = Image.fromarray(color_label[0])
    im.save(SAVE_DIR + id + '.png')
    return id


def main():
    # processing options
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if DATA_LIST_PATH:
        # with .txt
        lst_labels = open(DATA_LIST_PATH, 'w')
    else:
        # without .txt
        lst_labels = glob.glob(os.path.join(DATA_DIRECTORY, '*.png'))
        lst_labels.sort()

    assert NUM_TOTAL == len(lst_labels)

    # multiple processes
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        index = 0
        for lst_labels, thumbnail_file in zip(lst_labels, executor.map(visualize, lst_labels)):
            index += 1
            print("step {}: saved {}".format(index, thumbnail_file))
    # cal processing time
    end = time.time()
    print('Total time: {}s'.format(end - start))


if __name__ == '__main__':
    main()
