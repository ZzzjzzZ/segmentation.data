import glob
import os
import time

import numpy as np
from PIL import Image

import cityscapes_labels


def reverse_id(root):
    all_images = glob.glob(os.path.join(root, '*/*.png'))
    all_images.sort()
    index = 0
    for p in all_images:
        # get image information
        id = p.split('/')[-1]
        name = id.replace('leftImg8bit', 'gtFine_labelIds')
        save_path = os.path.join(root, '')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # read image
        mask = np.asarray(Image.open(p)).astype(np.uint8)
        # reverse id
        cvt_label = np.zeros(mask.shape)
        for l in cityscapes_labels.labels:
            cvt_label[mask == l.trainId] = cityscapes_labels.trainId2label[l.trainId].id

        label = Image.fromarray(cvt_label.astype(np.uint8))
        label.save(save_path + name)
        index += 1
        print('Finished {} figures'.format(index))


def reverse_id_for_cityscapes(data_path, save_path):
    # id mapping
    train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    ori_id = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    # save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    lst_labels = glob.glob(os.path.join(data_path, '*.png'))
    lst_labels.sort()
    info = 0
    for item in lst_labels:
        start_time = time.time()
        # process save name
        id = item.split('/')[-1][:-4]

        ori_label = np.array(Image.open(item))
        cvt_temp = np.zeros(ori_label.shape).astype(np.uint8)
        index = 0
        for l in ori_id:
            indices = np.where(ori_label == l)
            cvt_temp[indices] = train_id[index]
            index += 1
        # save converted label
        info += 1
        cvt_label = Image.fromarray(cvt_temp)
        cvt_label.save(os.path.join(save_path, id + '.png'))
        end_time = time.time()
        print('converted {} items, time current {}s'.format(info, end_time-start_time))


if __name__ == '__main__':
    ROOT = ''
    reverse_id(ROOT)
