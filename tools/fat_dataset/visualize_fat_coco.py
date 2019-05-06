#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from convert_fat_coco import *

image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/'
annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_2018.json'

example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['square'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]

print(image_data)

plt.figure()
image = io.imread(image_directory + image_data['file_name'])
plt.imshow(image); plt.axis('off')

plt.figure()
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)
print(annotations)

for annotation in annotations:
    viewpoint_xyz = get_viewpoint_from_id(annotation['viewpoint_id'])
    r, theta, phi = cart2polar(viewpoint_xyz)
    inplate_rotation_angle = get_inplane_rotation_from_id(annotation['inplane_rotation_id'])
    print(viewpoint_xyz)
    print(inplate_rotation_angle)

plt.show()