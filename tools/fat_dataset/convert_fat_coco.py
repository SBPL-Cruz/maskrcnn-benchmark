#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:15:34 2019

@author: aditya
"""

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from pathlib import Path
import skimage
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import pylab
from pyquaternion import Quaternion
from transformations import euler_from_matrix
import math
from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points
from tqdm import tqdm, trange
from lib.pair_matching import RT_transform
from dipy.core.geometry import cart2sphere, sphere2cart, sphere_distance
from lib.render_glumpy.render_py import Render_Py
from lib.utils.mkdir_if_missing import mkdir_if_missing

ROOT_DIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
IMAGE_DIR_LIST = [
        os.path.join(ROOT_DIR, "kitchen_0"), 
        os.path.join(ROOT_DIR, "kitchen_1"),
        # os.path.join(ROOT_DIR, "kitchen_2"),
        # os.path.join(ROOT_DIR, "kitchen_3"),
        # os.path.join(ROOT_DIR, "kitchen_4"),
        # os.path.join(ROOT_DIR, "kitedemo_0"),
        # os.path.join(ROOT_DIR, "kitedemo_1"),
        # os.path.join(ROOT_DIR, "kitedemo_2"),
        # os.path.join(ROOT_DIR, "kitedemo_3"),
        # os.path.join(ROOT_DIR, "kitedemo_4"),
        # os.path.join(ROOT_DIR, "temple_0"),
        # os.path.join(ROOT_DIR, "temple_1"),
        # os.path.join(ROOT_DIR, "temple_2"),
        # os.path.join(ROOT_DIR, "temple_3"),
        # os.path.join(ROOT_DIR, "temple_4")
]


ng = 1000
print ( '' )
print ( '  Number of points NG = %d' % ( ng ) )

viewpoints_xyz = sphere_fibonacci_grid_points(ng)
# inplane_rot_angles = np.linspace(-math.pi/4, math.pi/4, 19)
inplane_rot_angles = np.linspace(-math.pi, math.pi, 68)



# ROOT_OUTDIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/train'
# OUTFILE_NAME = 'instances_fat_train_pose_limited_2018'
OUTFILE_NAME = 'instances_fat_val_pose_2018'

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]
LM6d_root = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/"

def render_pose(class_name, fixed_transform, camera_intrinsics, rotation_angles, location):
    width = 960
    height = 540
    K = np.array([[camera_intrinsics['fx'], 0, camera_intrinsics['cx']], 
                  [0, camera_intrinsics['fy'], camera_intrinsics['cy']], 
                  [0, 0, 1]])
    # Check these TODO
    ZNEAR = 0.1
    ZFAR = 20
    depth_factor = 1000
    model_dir = os.path.join(LM6d_root, "models", class_name)
    render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)

    fixed_transform = np.transpose(np.array(fixed_transform))
    fixed_transform[:3,3] = [i/100 for i in fixed_transform[:3,3]]
    object_world_transform = np.zeros((4,4))
    object_world_transform[:3,:3] = RT_transform.euler2mat(rotation_angles[0],rotation_angles[1],rotation_angles[2])
    object_world_transform[:,3] = [i/100 for i in location] + [1]

    total_transform = np.matmul(object_world_transform, fixed_transform)
    pose_rendered_q = RT_transform.mat2quat(total_transform[:3,:3]).tolist() + total_transform[:3,3].flatten().tolist()
    
    rgb_gl, depth_gl = render_machine.render(
        pose_rendered_q[:4], np.array(pose_rendered_q[4:])
    )
    rgb_gl = rgb_gl.astype("uint8")

    depth_gl = (depth_gl * depth_factor).astype(np.uint16)
    return rgb_gl, depth_gl

def cart2polar(point):
    r = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    theta = math.acos(point[2]/r)
    phi = math.atan2(point[1], point[0])
    return [r, theta, phi]

def polar2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

def filter_for_jpeg(root, files):
    file_types = ['*.left.jpeg', '*.left.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.seg.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def filter_for_labels(root, files, image_filename):
    file_types = ['*.json']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def get_viewpoint_from_id(viewpoints_xyz, id):
    return viewpoints_xyz[id, :]

def get_inplane_rotation_from_id(inplane_rot_angles, id):
    return inplane_rot_angles[id]

def find_viewpoint_id(sphere_points, point):
    # print(np.array(point))
    # distances = np.linalg.norm(sphere_points - point, axis=1)
    distances = sphere_distance(sphere_points, np.array(point), radius=1)
    viewpoint_index = np.argmin(distances)
    # print(distances[viewpoint_index])
    return viewpoint_index
    # print(distances.shape[0])

def find_inplane_rotation_id(inplane_rot_angles, angle):
    distances = abs(inplane_rot_angles - angle)
    viewpoint_index = np.argmin(distances)
    # print("Angle : {}, All angles : {}, Closest : {}".format(
    #     angle, inplane_rot_angles, inplane_rot_angles[viewpoint_index])
    # )
    return viewpoint_index
    # print(distances.shape[0])

def get_wxyz_quaternion(quat_xyzw):
    return [quat_xyzw[3]]+quat_xyzw[0:3]

def euler2sphere(theta, phi):
    # phi = phi

    # theta should be rotation from z axis but when z is pointing up, so we minus 90 degree
    theta = math.pi/2 - theta

    # theta = angles[1]
    # phi = math.pi - angles[2]
    # if theta < 0:
    #     theta += 2*math.pi
    # if phi < 0:
    #     phi += 2*math.pi

    return theta, phi

def sphere2euler(theta, phi):
    theta = math.pi/2 - theta
    # if phi < 0:
    #     phi += 2*math.pi
    return theta, phi

def main():


    
    object_settings_file = Path(os.path.join(IMAGE_DIR_LIST[0], "_object_settings.json"))
    camera_settings_file = Path(os.path.join(IMAGE_DIR_LIST[0], "_camera_settings.json"))
    if object_settings_file.is_file():
        with open(object_settings_file) as file:
            object_settings_data = json.load(file)
            CLASSES = object_settings_data['exported_object_classes']
            CATEGORIES = [{
                'id': i,
                'name': CLASSES[i].replace('_16k', '').replace('_16K', ''),
                'supercategory': 'shape',
            } for i in range(0,len(CLASSES))]

            FIXED_TRANSFORMS = {}
            for i in range(0,len(CLASSES)):
                class_name = object_settings_data['exported_objects'][i]['class'].replace('_16k', '').replace('_16K', '')
                transform = object_settings_data['exported_objects'][i]['fixed_model_transform']
                FIXED_TRANSFORMS[class_name] = transform
                    
        
            SEGMENTATION_DATA =  object_settings_data['exported_objects']    
    else:
        raise Exception("Object settings file not found")

    if camera_settings_file.is_file():
        with open(camera_settings_file) as file:
            camera_settings_data = json.load(file)       
            CAMERA_INTRINSICS = camera_settings_data['camera_settings'][0]['intrinsic_settings']
    else:
        raise Exception("Camera settings file not found")

    VIEWPOINTS = [viewpoints_xyz[i].tolist() for i in range(0, len(viewpoints_xyz))]

    INPLANE_ROTATIONS = [inplane_rot_angles[i] for i in range(0, len(inplane_rot_angles))]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "viewpoints" : VIEWPOINTS,
        "inplane_rotations" : INPLANE_ROTATIONS,
        "camera_intrinsic_settings": CAMERA_INTRINSICS,
        "fixed_transforms": FIXED_TRANSFORMS,
        "images": [],
        "annotations": []
    }

    image_global_id = 1
    segmentation_global_id = 1

    # filter for jpeg images
    for IMAGE_DIR in IMAGE_DIR_LIST:
        # for root, _, files in os.walk(IMAGE_DIR):
        all_dir_files = os.listdir(IMAGE_DIR)
        image_files = filter_for_jpeg(IMAGE_DIR, all_dir_files)
        dir_name = os.path.basename(IMAGE_DIR)

        # go through each image
        for ii in trange(len(image_files)):
            image_filename = image_files[ii]
            image_out_filename =  os.path.join(dir_name, os.path.basename(image_filename))
            img_size = (960,540)
            image_info = pycococreatortools.create_image_info(
                image_global_id, image_out_filename, img_size
            )
            # plt.figure()
            # skimage.io.imshow(skimage.io.imread(image_filename))
            # plt.show()
            # filter for associated png annotations
            # for root, _, files in os.walk(IMAGE_DIR):
            segmentation_image_files = filter_for_annotations(IMAGE_DIR, all_dir_files, image_filename)
            label_files = filter_for_labels(IMAGE_DIR, all_dir_files, image_filename)
            boxes = []
            labels = []
            segmentation_ids = []
            label_filename = label_files[0]
            # go through each associated json file containing objects data
            # for label_filename in label_files:
            # print("File %d - %s"% (image_global_id, label_filename))
            my_file = Path(label_filename)
            segmentation_image = skimage.io.imread(segmentation_image_files[0])

            if my_file.is_file():
                with open(label_filename) as file:
                    label_data = json.load(file)
                    all_objects_yaw_only = True
                    for i in range(0, len(label_data['objects'])):
                        class_name = label_data['objects'][i]['class']
                        class_bounding_box = label_data['objects'][i]['bounding_box']
                        quat = label_data['objects'][i]['quaternion_xyzw']
                        
                        angles = RT_transform.quat2euler(get_wxyz_quaternion(quat))
                        # This function gives angles with this convention of euler - https://en.wikipedia.org/wiki/Euler_angles#Signs_and_ranges (geometric definition)

                        if np.isclose(angles[1], 0):
                            print("Test")
                        theta, phi = euler2sphere(angles[1], angles[0])
                        actual_angles = np.array([1, theta, phi])
                        xyz_coord = sphere2cart(1, theta, phi)
                        
                        viewpoint_id = find_viewpoint_id(viewpoints_xyz, xyz_coord)
                        r_xyz = get_viewpoint_from_id(viewpoints_xyz, viewpoint_id)
                        recovered_angles = np.array(cart2sphere(r_xyz[0], r_xyz[1], r_xyz[2]))

                        inplane_rotation_id = find_inplane_rotation_id(inplane_rot_angles, angles[2])
                        # inplate_rotation_angle = get_inplane_rotation_from_id(INPLANE_ROTATIONS, inplane_rotation_id)


                        if np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == False:
                            print("Mismatch in : {}".format(label_filename))
                            print("sphere2cart angles : {}".format(actual_angles))
                            print("cart2sphere angles : {}".format(recovered_angles))
                        # elif np.all(np.isclose(actual_angles, recovered_angles, atol=0.4)) == True:
                        #     print("Match")

                        # print(inplate_rotation_angle)
                        class_label = [x['id'] for x in CATEGORIES if x['name'] in class_name][0]
                        segmentation_id = [x['segmentation_class_id'] for x in SEGMENTATION_DATA if x['class'] in class_name][0]

                        boxes.append(class_bounding_box['top_left'] + class_bounding_box['bottom_right'])
                        labels.append(class_label)     
                        segmentation_ids.append([x['segmentation_class_id'] for x in SEGMENTATION_DATA if x['class'] in class_name][0])

                        # Create binary masks from segmentation image for every object
                        # for segmentation_image_file in segmentation_image_files:
                        # segmentation_image = skimage.io.imread(segmentation_image_file)
                        binary_mask = np.copy(segmentation_image)
                        binary_mask[binary_mask != segmentation_id] = 0
                        binary_mask[binary_mask == segmentation_id] = 1
                        # skimage.io.imshow(binary_mask, cmap=plt.cm.gray)
                        # plt.show()
                        # TODO : check if its actually a crowd in case of multiple instances of one object type
                        # class_label = [x['class'] for x in SEGMENTATION_DATA if x['segmentation_class_id'] in segmentation_id][0]
                        category_info = {'id': class_label, 'is_crowd': 0}

                        
                        annotation_info = pycococreatortools.create_annotation_info(
                            segmentation_global_id, image_global_id, category_info, binary_mask,
                            img_size, tolerance=2)
                        
                        # print(annotation_info)

                        if annotation_info is not None:
                            annotation_info['viewpoint_id'] = int(viewpoint_id)
                            annotation_info['inplane_rotation_id'] = int(inplane_rotation_id)
                            annotation_info['camera_pose'] = label_data['camera_data']
                            annotation_info['location'] = label_data['objects'][i]['location']
                            annotation_info['quaternion_xyzw'] = quat
                            coco_output["annotations"].append(annotation_info)
                            coco_output["images"].append(image_info)
                        else:
                            tqdm.write("File %s doesn't have boxes or labels in json file" % image_filename)
                        segmentation_global_id = segmentation_global_id + 1
            else:
                tqdm.write("File %s doesn't have a label file" % image_filename)
                    

            image_global_id = image_global_id + 1

        with open('{}/{}.json'.format(ROOT_DIR, OUTFILE_NAME), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()