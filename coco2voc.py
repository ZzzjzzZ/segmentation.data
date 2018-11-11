import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools import mask
from pycocotools.coco import COCO
from tqdm import trange

"""
NUM_CHANNEL = 91
[] background
[5] airplane
[2] bicycle
[16] bird
[9] boat
[44] bottle
[6] bus
[3] car
[17] cat
[62] chair
[21] cow
[67] dining table
[18] dog
[19] horse
[4] motorcycle
[1] person
[64] potted plant
[20] sheep
[63] couch
[7] train
[72] tv
"""


class COCOSegmentation(data.Dataset):
    NUM_CLASS = 21
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]

    def __init__(self, root='/path/to/coco/dataset/', split='train'):

        if split == 'train':
            print('train set')
            ann_file = os.path.join(root, 'annotations/instances_train2017.json')
            ids_file = os.path.join(root, 'annotations/train_ids.pth')
            self.root = os.path.join(root, 'train2017')
        else:
            print('val set')
            ann_file = os.path.join(root, 'annotations/instances_val2017.json')
            ids_file = os.path.join(root, 'annotations/val_ids.pth')
            self.root = os.path.join(root, 'val2017')
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        img = np.asarray(Image.open(os.path.join(self.root, path)).convert('RGB'))
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        mask = self._gen_seg_mask(cocotarget, img_metadata['height'], img_metadata['width'])
        return img, mask, path

    def __len__(self):
        return len(self.ids)

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                      img_metadata['width'])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description('Doing: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))
        print('Found number of qualified images: ', len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids


if __name__ == '__main__':
    SAVE_PATH = '/path/to/save/converted_labels/'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    trainloader = data.DataLoader(COCOSegmentation(root='/path/to/coco/dataset/', split='train'),
                                  batch_size=1, num_workers=0)

    for i, data in enumerate(trainloader):
        imgs, labels, name = data
        n, h, w = labels.shape
        data = labels.cpu().data.numpy()  # .transpose(0, 2, 3, 1)
        assert np.max(data[0]) in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        im = Image.fromarray(data[0, :, :].astype(np.uint8))
        # im.show()
        im.save(os.path.join(SAVE_PATH, name[0].replace('.jpg', '.png')))
        print('prcessing {}th images ...'.format(i))
