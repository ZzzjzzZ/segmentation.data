# DataProcessing

The project is for processing dataset, including [Cityscapes](https://www.cityscapes-dataset.com/) and [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/)

## Usage

## Prerequisites:
1. python 3
2. numpy
3. PIL

## Visualization using multi-process
It consists of ```multi-process_visual``` & ```pallete```
* run ```multi-process_visual.py``` for converting gray predictions to colors. 
* it will use all the cpu are avaliable.
* ```pallete.py``` provides palletes of different datasets, you can custom it yourself.

## Converting  index of Cityscapes labels
It consists of ```reverse_idx``` & ```cityscapes_labels```
* ```reverse_idx.py``` provides two functions for converting the ```idx```.
* ```cityscapes_labels```is based on [cityscapesScripts](https://github.com/mcordts/cityscapesScripts)

## Extra
* ```contour```is for computing the boundary maps used in [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) based on instance labels.
* ```scripts```is for coping desired images from files and generating lists of dataset (ie. w/ lst, w/o lst)

## TODO

- [ ] Converting scripts for [PASCAL Context dataset](https://cs.stanford.edu/~roozbeh/pascal-context/)
- [ ] Scripts for [ADE20k](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
