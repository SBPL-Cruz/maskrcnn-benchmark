
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from PIL import Image
import numpy as np
import torch
import json
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


from maskrcnn_benchmark.config import cfg
from pycocotools.coco import COCO
import skimage.io as io
import pylab



image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'
coco_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/coco_results.pth')
all_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/predictions.pth')
example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
print(coco_predictions)
print(all_predictions[:5])

# ## Load Image from COCO Dataset


category_ids = example_coco.getCatIds(catNms=['square'])
image_ids = example_coco.getImgIds(catIds=category_ids)
# image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]
image_data = example_coco.loadImgs(image_ids[100])[0]
viewpoints_xyz = np.array(example_coco.dataset['viewpoints'])
inplane_rotations = np.array(example_coco.dataset['inplane_rotations'])
fixed_transforms_dict = example_coco.dataset['fixed_transforms']
camera_intrinsics = example_coco.dataset['camera_intrinsic_settings']

print(image_data)

# plt.figure()
img_path = os.path.join(image_directory, image_data['file_name'])
image = io.imread(img_path)
# plt.imshow(image); plt.axis('off')

# # Running model on image

from predictor import COCODemo

cfg_file = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/configs/fat_pose/e2e_mask_rcnn_R_50_FPN_1x_test_cocostyle.yaml'
args = {
    'config_file' : cfg_file,
    'confidence_threshold' : 0.7,
    'min_image_size' : 750,
    'masks_per_dim' : 10,
    'show_mask_heatmaps' : False
}
cfg.merge_from_file(args['config_file'])
cfg.freeze()
    
coco_demo = COCODemo(
    cfg,
    confidence_threshold=args['confidence_threshold'],
    show_mask_heatmaps=args['show_mask_heatmaps'],
    masks_per_dim=args['masks_per_dim'],
    min_image_size=args['min_image_size'],
    categories = category_names,
    viewpoints_xyz = viewpoints_xyz,
    inplane_rotations = inplane_rotations,
    fixed_transforms_dict = fixed_transforms_dict,
    camera_intrinsics = camera_intrinsics
)

from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
img = cv2.imread(img_path)
composite, result, img_list = coco_demo.run_on_opencv_image(img)
fig = plt.figure(1, (4., 4.))
grid = ImageGrid(fig, 111,  
                 nrows_ncols=(1, len(img_list)+1),
                 axes_pad=0.1, 
                 )

# plt.imshow(composite); plt.axis('off')
# cv2.imwrite('composite.png', composite)
grid[0].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
grid[0].axis("off")

for i in range(len(img_list)):
    image_file = os.path.join(
        "{}-color.png".format(i),
    )
    rgb_gl = img_list[i][0]
    # cv2.imwrite(image_file, rgb_gl)
    grid[i+1].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
    grid[i+1].axis("off")

plt.savefig('output.png')
plt.show()
    # print(i[0])



