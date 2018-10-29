import glob
import os

import numpy as np
from PIL import Image


def boundary(raw_input, save_path, save_name):
    """
    calculate boundary mask & save

    :param raw_input: *instanceIds image
    :param save_path: city name
    :param save_name: boundary mask name
    :return:
    """
    # process instance mask
    instance_mask = Image.open(raw_input)
    width = instance_mask.size[0]
    height = instance_mask.size[1]
    mask_array = np.array(instance_mask)

    # define the boundary mask
    boundary_mask = np.zeros((height, width), dtype=np.uint8)  # 0-255

    # perform boundary calculate: the center pixel_id is differ from the 4-nearest pixels_id
    for i in range(1, height-1):
        for j in range(1, width-1):
            if mask_array[i, j] != mask_array[i - 1, j] \
                    or mask_array[i, j] != mask_array[i + 1, j] \
                    or mask_array[i, j] != mask_array[i, j - 1] \
                    or mask_array[i, j] != mask_array[i, j + 1]:
                boundary_mask[i, j] = 255
    boundary_image = Image.fromarray(np.uint8(boundary_mask))
    # boundary_image.show()
    boundary_image.save(os.path.join(RAW_INPUT_PATH, save_path, save_name))


def process(root):
    """
    read instanceIds masks

    :param root: root path for cityscapes gtFine (train, val)
    :return:
    """
    index = 0
    instance_suffix = 'gtFine_instanceIds'
    boundary_suffix = 'boundary'

    # all instance masks train-2975 val-500 test-1525
    all_instance_masks = glob.glob(os.path.join(root, '*/*gtFine_instanceIds.png'))
    for instance_mask in all_instance_masks:
        instance_name = instance_mask.split('/')[-1]
        boundary_save_path = instance_mask.split('/')[-2]
        boundary_name = instance_name.replace(instance_suffix, boundary_suffix)
        boundary(instance_mask, boundary_save_path, boundary_name)
        index += 1
        print('{} instance_map -> boundary_map'.format(index))


if __name__ == '__main__':
    RAW_INPUT_PATH = ''
    process(RAW_INPUT_PATH)
