import glob
import os
import shutil


def gen_lst_from_fig(data_path, list):
    ids = glob.glob(os.path.join(data_path, '*.png'))
    ids.sort()
    lst = []
    for item in ids:
        item = item.strip()
        id = item.split('/')[-1]
        lst.append([id, ])

    # write list
    target = open(list, 'w')
    for line in lst:
        print('\t'.join(line), file=target)

    # tag
    print('Done!')


def gen_lst_from_txt(source_lst, target_lst):
    fid = open(source_lst, 'r')
    lst = []
    for item in fid.readlines():
        img, label = item.strip().split(' ')
        lst.append([label, ])

    # write list
    target = open(target_lst, 'w')
    for line in lst:
        print('\t'.join(line), file=target)

    # tag
    print('Done!')


def copy_desired_imgs(data_path, save_path, lst):
    fid = open(lst, 'r')
    # save path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for line in fid.readlines():
        id = line.strip()
        source_path = os.path.join(data_path, id)
        target_path = os.path.join(save_path, id)
        shutil.copy(source_path, target_path)
    # tag
    print('Done!')


if __name__ == '__main__':
    DATA_PATH = ''
    LIST = ''
    SAVE_PATH = ''
    TARGET = ''
    # gen_lst_from_fig(DATA_PATH, LIST)
    # copy_desired_imgs(DATA_PATH, SAVE_PATH, LIST)
    # gen_lst_from_txt(LIST, TARGET)
