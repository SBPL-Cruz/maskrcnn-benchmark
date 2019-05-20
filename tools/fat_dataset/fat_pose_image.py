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
from dipy.core.geometry import cart2sphere, sphere2cart
from convert_fat_coco import *
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2

# coco_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/coco_results.pth')
# all_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/predictions.pth')

class FATImage:
    def __init__(self, coco_annotation_file=None, coco_image_directory=None):
        self.image_directory = image_directory
        self.example_coco = COCO(coco_annotation_file)
        example_coco = self.example_coco
        self.categories = example_coco.loadCats(example_coco.getCatIds())
        self.category_names = [category['name'] for category in self.categories]
        print('Custom COCO categories: \n{}\n'.format(' '.join(self.category_names)))
        # print(coco_predictions)
        # print(all_predictions[:5])

        # ## Load Image from COCO Dataset


        self.category_ids = example_coco.getCatIds(catNms=['square'])
        self.image_ids = example_coco.getImgIds(catIds=self.category_ids)
        
        self.viewpoints_xyz = np.array(example_coco.dataset['viewpoints'])
        self.inplane_rotations = np.array(example_coco.dataset['inplane_rotations'])
        self.fixed_transforms_dict = example_coco.dataset['fixed_transforms']
        self.camera_intrinsics = example_coco.dataset['camera_intrinsic_settings']

    def get_random_image(self):
        # image_data = example_coco.loadImgs(self.image_ids[np.random.randint(0, len(self.image_ids))])[0]
        image_data = self.example_coco.loadImgs(self.image_ids[100])[0]
        # print(image_data)
        annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
        annotations = self.example_coco.loadAnns(annotation_ids)
        self.example_coco.showAnns(annotations)
        # print(annotations)

        return image_data, annotations
    
    def visualize_image_annotations(self, image_data, annotations):
        img_path = os.path.join(self.image_directory, image_data['file_name'])
        image = io.imread(img_path)
        count = 1
        fig = plt.figure(2, (4., 4.))
        plt.axis("off")
        grid = ImageGrid(fig, 111,  
                        nrows_ncols=(1, len(annotations)+1),
                        axes_pad=0.1, 
                        )

        grid[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        grid[0].axis("off")

        for annotation in annotations:
            print("Annotated viewpoint_id : {}".format(annotation['viewpoint_id']))
            theta, phi = get_viewpoint_rotations_from_id(viewpoints_xyz, annotation['viewpoint_id'])
            inplane_rotation_angle = get_inplane_rotation_from_id(
                self.inplane_rotations, annotation['inplane_rotation_id']
            )
            xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
            class_name = self.categories[annotation['category_id']]['name']
            print("*****{}*****".format(class_name))
            print("Recovered rotation : {}".format(xyz_rotation_angles))
            quat = annotation['quaternion_xyzw']
            print("Actual rotation : {}".format(RT_transform.quat2euler(get_wxyz_quaternion(quat))))
            fixed_transform = self.fixed_transforms_dict[class_name]
            rgb_gl, depth_gl = render_pose(
                class_name, fixed_transform, self.camera_intrinsics, xyz_rotation_angles, annotation['location']
            )
            grid[count].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
            grid[count].axis("off")

            count += 1
        plt.savefig('image_annotations_output.png')

    def visualize_model_output(self, image_data):
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
            categories = self.category_names,
            viewpoints_xyz = self.viewpoints_xyz,
            inplane_rotations = self.inplane_rotations,
            fixed_transforms_dict = self.fixed_transforms_dict,
            camera_intrinsics = self.camera_intrinsics
        )

        
        img = cv2.imread(img_path)
        composite, result, img_list = coco_demo.run_on_opencv_image(img)
        fig = plt.figure(1, (4., 4.))
        plt.axis("off")
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

        plt.savefig('model_output.png')
    # plt.show()
    # print(i[0])

if __name__ == '__main__':
    
    # example_coco = COCO(annotation_file)

    # categories = example_coco.loadCats(example_coco.getCatIds())
    # category_names = [category['name'] for category in categories]
    # print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
    # print(coco_predictions)
    # print(all_predictions[:5])

    # # ## Load Image from COCO Dataset


    # category_ids = example_coco.getCatIds(catNms=['square'])
    # image_ids = example_coco.getImgIds(catIds=category_ids)
    # # image_data = example_coco.loadImgs(image_ids[np.random.randint(0, len(image_ids))])[0]
    # image_data = example_coco.loadImgs(image_ids[100])[0]
    # viewpoints_xyz = np.array(example_coco.dataset['viewpoints'])
    # inplane_rotations = np.array(example_coco.dataset['inplane_rotations'])
    # fixed_transforms_dict = example_coco.dataset['fixed_transforms']
    # camera_intrinsics = example_coco.dataset['camera_intrinsic_settings']

    # print(image_data)
    # annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
    # annotations = example_coco.loadAnns(annotation_ids)
    # example_coco.showAnns(annotations)

    # visualize_model_output(image_data)
    # visualize_annotations(image_data, annotations)
    image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'

    fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory)
    image_data, annotations = fat_image.get_random_image()
    fat_image.visualize_image_annotations(image_data, annotations)
    fat_image.visualize_model_output(image_data)




