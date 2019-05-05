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

ROOT_DIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
IMAGE_DIR_LIST = [
#        os.path.join(ROOT_DIR, "kitchen_0"), 
#        os.path.join(ROOT_DIR, "kitchen_1"),
        os.path.join(ROOT_DIR, "kitchen_2"),
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
#IMAGE_DIR_LIST = [
#"002_master_chef_can_16k/kitchen_0",
#
#"002_master_chef_can_16k/kitchen_1",
#
#"002_master_chef_can_16k/kitchen_2",
#
#"002_master_chef_can_16k/kitchen_3",
#
#"002_master_chef_can_16k/kitchen_4",
#
#"002_master_chef_can_16k/kitedemo_0",
#
#"002_master_chef_can_16k/kitedemo_1",
#
#"002_master_chef_can_16k/kitedemo_2",
#
#"002_master_chef_can_16k/kitedemo_3",
#
#"002_master_chef_can_16k/kitedemo_4",
#
#"002_master_chef_can_16k/temple_0",
#
#"002_master_chef_can_16k/temple_1",
#
#"002_master_chef_can_16k/temple_2",
#
#"002_master_chef_can_16k/temple_3",
#
#"002_master_chef_can_16k/temple_4",
#
#"003_cracker_box_16k/kitchen_0",
#
#"003_cracker_box_16k/kitchen_1",
#
#"003_cracker_box_16k/kitchen_2",
#
#"003_cracker_box_16k/kitchen_3",
#
#"003_cracker_box_16k/kitchen_4",
#
#"003_cracker_box_16k/kitedemo_0",
#
#"003_cracker_box_16k/kitedemo_1",
#
#"003_cracker_box_16k/kitedemo_2",
#
#"003_cracker_box_16k/kitedemo_3",
#
#"003_cracker_box_16k/kitedemo_4",
#
#"003_cracker_box_16k/temple_0",
#
#"003_cracker_box_16k/temple_1",
#
#"003_cracker_box_16k/temple_2",
#
#"003_cracker_box_16k/temple_3",
#
#"003_cracker_box_16k/temple_4",
#
#"004_sugar_box_16k/kitchen_0",
#
#"004_sugar_box_16k/kitchen_1",
#
#"004_sugar_box_16k/kitchen_2",
#
#"004_sugar_box_16k/kitchen_3",
#
#"004_sugar_box_16k/kitchen_4",
#
#"004_sugar_box_16k/kitedemo_0",
#
#"004_sugar_box_16k/kitedemo_1",
#
#"004_sugar_box_16k/kitedemo_2",
#
#"004_sugar_box_16k/kitedemo_3",
#
#"004_sugar_box_16k/kitedemo_4",
#
#"004_sugar_box_16k/temple_0",
#
#"004_sugar_box_16k/temple_1",
#
#"004_sugar_box_16k/temple_2",
#
#"004_sugar_box_16k/temple_3",
#
#"004_sugar_box_16k/temple_4",
#
#"005_tomato_soup_can_16k/kitchen_0",
#
#"005_tomato_soup_can_16k/kitchen_1",
#
#"005_tomato_soup_can_16k/kitchen_2",
#
#"005_tomato_soup_can_16k/kitchen_3",
#
#"005_tomato_soup_can_16k/kitchen_4",
#
#"005_tomato_soup_can_16k/kitedemo_0",
#
#"005_tomato_soup_can_16k/kitedemo_1",
#
#"005_tomato_soup_can_16k/kitedemo_2",
#
#"005_tomato_soup_can_16k/kitedemo_3",
#
#"005_tomato_soup_can_16k/kitedemo_4",
#
#"005_tomato_soup_can_16k/temple_0",
#
#"005_tomato_soup_can_16k/temple_1",
#
#"005_tomato_soup_can_16k/temple_2",
#
#"005_tomato_soup_can_16k/temple_3",
#
#"005_tomato_soup_can_16k/temple_4",
#
#"006_mustard_bottle_16k/kitchen_0",
#
#"006_mustard_bottle_16k/kitchen_1",
#
#"006_mustard_bottle_16k/kitchen_2",
#
#"006_mustard_bottle_16k/kitchen_3",
#
#"006_mustard_bottle_16k/kitchen_4",
#
#"006_mustard_bottle_16k/kitedemo_0",
#
#"006_mustard_bottle_16k/kitedemo_1",
#
#"006_mustard_bottle_16k/kitedemo_2",
#
#"006_mustard_bottle_16k/kitedemo_3",
#
#"006_mustard_bottle_16k/kitedemo_4",
#
#"006_mustard_bottle_16k/temple_0",
#
#"006_mustard_bottle_16k/temple_1",
#
#"006_mustard_bottle_16k/temple_2",
#
#"006_mustard_bottle_16k/temple_3",
#
#"006_mustard_bottle_16k/temple_4",
#
#"007_tuna_fish_can_16k/kitchen_0",
#
#"007_tuna_fish_can_16k/kitchen_1",
#
#"007_tuna_fish_can_16k/kitchen_2",
#
#"007_tuna_fish_can_16k/kitchen_3",
#
#"007_tuna_fish_can_16k/kitchen_4",
#
#"007_tuna_fish_can_16k/kitedemo_0",
#
#"007_tuna_fish_can_16k/kitedemo_1",
#
#"007_tuna_fish_can_16k/kitedemo_2",
#
#"007_tuna_fish_can_16k/kitedemo_3",
#
#"007_tuna_fish_can_16k/kitedemo_4",
#
#"007_tuna_fish_can_16k/temple_0",
#
#"007_tuna_fish_can_16k/temple_1",
#
#"007_tuna_fish_can_16k/temple_2",
#
#"007_tuna_fish_can_16k/temple_3",
#
#"007_tuna_fish_can_16k/temple_4",
#
#"008_pudding_box_16k/kitchen_0",
#
#"008_pudding_box_16k/kitchen_1",
#
#"008_pudding_box_16k/kitchen_2",
#
#"008_pudding_box_16k/kitchen_3",
#
#"008_pudding_box_16k/kitchen_4",
#
#"008_pudding_box_16k/kitedemo_0",
#
#"008_pudding_box_16k/kitedemo_1",
#
#"008_pudding_box_16k/kitedemo_2",
#
#"008_pudding_box_16k/kitedemo_3",
#
#"008_pudding_box_16k/kitedemo_4",
#
#"008_pudding_box_16k/temple_0",
#
#"008_pudding_box_16k/temple_1",
#
#"008_pudding_box_16k/temple_2",
#
#"008_pudding_box_16k/temple_3",
#
#"008_pudding_box_16k/temple_4",
#
#"009_gelatin_box_16k/kitchen_0",
#
#"009_gelatin_box_16k/kitchen_1",
#
#"009_gelatin_box_16k/kitchen_2",
#
#"009_gelatin_box_16k/kitchen_3",
#
#"009_gelatin_box_16k/kitchen_4",
#
#"009_gelatin_box_16k/kitedemo_0",
#
#"009_gelatin_box_16k/kitedemo_1",
#
#"009_gelatin_box_16k/kitedemo_2",
#
#"009_gelatin_box_16k/kitedemo_3",
#
#"009_gelatin_box_16k/kitedemo_4",
#
#"009_gelatin_box_16k/temple_0",
#
#"009_gelatin_box_16k/temple_1",
#
#"009_gelatin_box_16k/temple_2",
#
#"009_gelatin_box_16k/temple_3",
#
#"009_gelatin_box_16k/temple_4",
#
#"010_potted_meat_can_16k/kitchen_0",
#
#"010_potted_meat_can_16k/kitchen_1",
#
#"010_potted_meat_can_16k/kitchen_2",
#
#"010_potted_meat_can_16k/kitchen_3",
#
#"010_potted_meat_can_16k/kitchen_4",
#
#"010_potted_meat_can_16k/kitedemo_0",
#
#"010_potted_meat_can_16k/kitedemo_1",
#
#"010_potted_meat_can_16k/kitedemo_2",
#
#"010_potted_meat_can_16k/kitedemo_3",
#
#"010_potted_meat_can_16k/kitedemo_4",
#
#"010_potted_meat_can_16k/temple_0",
#
#"010_potted_meat_can_16k/temple_1",
#
#"010_potted_meat_can_16k/temple_2",
#
#"010_potted_meat_can_16k/temple_3",
#
#"010_potted_meat_can_16k/temple_4",
#
#"011_banana_16k/kitchen_0",
#
#"011_banana_16k/kitchen_1",
#
#"011_banana_16k/kitchen_2",
#
#"011_banana_16k/kitchen_3",
#
#"011_banana_16k/kitchen_4",
#
#"011_banana_16k/kitedemo_0",
#
#"011_banana_16k/kitedemo_1",
#
#"011_banana_16k/kitedemo_2",
#
#"011_banana_16k/kitedemo_3",
#
#"011_banana_16k/kitedemo_4",
#
#"011_banana_16k/temple_0",
#
#"011_banana_16k/temple_1",
#
#"011_banana_16k/temple_2",
#
#"011_banana_16k/temple_3",
#
#"011_banana_16k/temple_4",
#
#"019_pitcher_base_16k/kitchen_0",
#
#"019_pitcher_base_16k/kitchen_1",
#
#"019_pitcher_base_16k/kitchen_2",
#
#"019_pitcher_base_16k/kitchen_3",
#
#"019_pitcher_base_16k/kitchen_4",
#
#"019_pitcher_base_16k/kitedemo_0",
#
#"019_pitcher_base_16k/kitedemo_1",
#
#"019_pitcher_base_16k/kitedemo_2",
#
#"019_pitcher_base_16k/kitedemo_3",
#
#"019_pitcher_base_16k/kitedemo_4",
#
#"019_pitcher_base_16k/temple_0",
#
#"019_pitcher_base_16k/temple_1",
#
#"019_pitcher_base_16k/temple_2",
#
#"019_pitcher_base_16k/temple_3",
#
#"019_pitcher_base_16k/temple_4",
#
#"021_bleach_cleanser_16k/kitchen_0",
#
#"021_bleach_cleanser_16k/kitchen_1",
#
#"021_bleach_cleanser_16k/kitchen_2",
#
#"021_bleach_cleanser_16k/kitchen_3",
#
#"021_bleach_cleanser_16k/kitchen_4",
#
#"021_bleach_cleanser_16k/kitedemo_0",
#
#"021_bleach_cleanser_16k/kitedemo_1",
#
#"021_bleach_cleanser_16k/kitedemo_2",
#
#"021_bleach_cleanser_16k/kitedemo_3",
#
#"021_bleach_cleanser_16k/kitedemo_4",
#
#"021_bleach_cleanser_16k/temple_0",
#
#"021_bleach_cleanser_16k/temple_1",
#
#"021_bleach_cleanser_16k/temple_2",
#
#"021_bleach_cleanser_16k/temple_3",
#
#"021_bleach_cleanser_16k/temple_4",
#
#"024_bowl_16k/kitchen_0",
#
#"024_bowl_16k/kitchen_1",
#
#"024_bowl_16k/kitchen_2",
#
#"024_bowl_16k/kitchen_3",
#
#"024_bowl_16k/kitchen_4",
#
#"024_bowl_16k/kitedemo_0",
#
#"024_bowl_16k/kitedemo_1",
#
#"024_bowl_16k/kitedemo_2",
#
#"024_bowl_16k/kitedemo_3",
#
#"024_bowl_16k/kitedemo_4",
#
#"024_bowl_16k/temple_0",
#
#"024_bowl_16k/temple_1",
#
#"024_bowl_16k/temple_2",
#
#"024_bowl_16k/temple_3",
#
#"024_bowl_16k/temple_4",
#
#"025_mug_16k/kitchen_0",
#
#"025_mug_16k/kitchen_1",
#
#"025_mug_16k/kitchen_2",
#
#"025_mug_16k/kitchen_3",
#
#"025_mug_16k/kitchen_4",
#
#"025_mug_16k/kitedemo_0",
#
#"025_mug_16k/kitedemo_1",
#
#"025_mug_16k/kitedemo_2",
#
#"025_mug_16k/kitedemo_3",
#
#"025_mug_16k/kitedemo_4",
#
#"025_mug_16k/temple_0",
#
#"025_mug_16k/temple_1",
#
#"025_mug_16k/temple_2",
#
#"025_mug_16k/temple_3",
#
#"025_mug_16k/temple_4",
#
#"035_power_drill_16k/kitchen_0",
#
#"035_power_drill_16k/kitchen_1",
#
#"035_power_drill_16k/kitchen_2",
#
#"035_power_drill_16k/kitchen_3",
#
#"035_power_drill_16k/kitchen_4",
#
#"035_power_drill_16k/kitedemo_0",
#
#"035_power_drill_16k/kitedemo_1",
#
#"035_power_drill_16k/kitedemo_2",
#
#"035_power_drill_16k/kitedemo_3",
#
#"035_power_drill_16k/kitedemo_4",
#
#"035_power_drill_16k/temple_0",
#
#"035_power_drill_16k/temple_1",
#
#"035_power_drill_16k/temple_2",
#
#"035_power_drill_16k/temple_3",
#
#"035_power_drill_16k/temple_4",
#
#"036_wood_block_16k/kitchen_0",
#
#"036_wood_block_16k/kitchen_1",
#
#"036_wood_block_16k/kitchen_2",
#
#"036_wood_block_16k/kitchen_3",
#
#"036_wood_block_16k/kitchen_4",
#
#"036_wood_block_16k/kitedemo_0",
#
#"036_wood_block_16k/kitedemo_1",
#
#"036_wood_block_16k/kitedemo_2",
#
#"036_wood_block_16k/kitedemo_3",
#
#"036_wood_block_16k/kitedemo_4",
#
#"036_wood_block_16k/temple_0",
#
#"036_wood_block_16k/temple_1",
#
#"036_wood_block_16k/temple_2",
#
#"036_wood_block_16k/temple_3",
#
#"036_wood_block_16k/temple_4",
#
#"037_scissors_16k/kitchen_0",
#
#"037_scissors_16k/kitchen_1",
#
#"037_scissors_16k/kitchen_2",
#
#"037_scissors_16k/kitchen_3",
#
#"037_scissors_16k/kitchen_4",
#
#"037_scissors_16k/kitedemo_0",
#
#"037_scissors_16k/kitedemo_1",
#
#"037_scissors_16k/kitedemo_2",
#
#"037_scissors_16k/kitedemo_3",
#
#"037_scissors_16k/kitedemo_4",
#
#"037_scissors_16k/temple_0",
#
#"037_scissors_16k/temple_1",
#
#"037_scissors_16k/temple_2",
#
#"037_scissors_16k/temple_3",
#
#"037_scissors_16k/temple_4",
#
#"040_large_marker_16k/kitchen_0",
#
#"040_large_marker_16k/kitchen_1",
#
#"040_large_marker_16k/kitchen_2",
#
#"040_large_marker_16k/kitchen_3",
#
#"040_large_marker_16k/kitchen_4",
#
#"040_large_marker_16k/kitedemo_0",
#
#"040_large_marker_16k/kitedemo_1",
#
#"040_large_marker_16k/kitedemo_2",
#
#"040_large_marker_16k/kitedemo_3",
#
#"040_large_marker_16k/kitedemo_4",
#
#"040_large_marker_16k/temple_0",
#
#"040_large_marker_16k/temple_1",
#
#"040_large_marker_16k/temple_2",
#
#"040_large_marker_16k/temple_3",
#
#"040_large_marker_16k/temple_4",
#
#"051_large_clamp_16k/kitchen_0",
#
#"051_large_clamp_16k/kitchen_1",
#
#"051_large_clamp_16k/kitchen_2",
#
#"051_large_clamp_16k/kitchen_3",
#
#"051_large_clamp_16k/kitchen_4",
#
#"051_large_clamp_16k/kitedemo_0",
#
#"051_large_clamp_16k/kitedemo_1",
#
#"051_large_clamp_16k/kitedemo_2",
#
#"051_large_clamp_16k/kitedemo_3",
#
#"051_large_clamp_16k/kitedemo_4",
#
#"051_large_clamp_16k/temple_0",
#
#"051_large_clamp_16k/temple_1",
#
#"051_large_clamp_16k/temple_2",
#
#"051_large_clamp_16k/temple_3",
#
#"051_large_clamp_16k/temple_4",
#
#"052_extra_large_clamp_16k/kitchen_0",
#
#"052_extra_large_clamp_16k/kitchen_1",
#
#"052_extra_large_clamp_16k/kitchen_2",
#
#"052_extra_large_clamp_16k/kitchen_3",
#
#"052_extra_large_clamp_16k/kitchen_4",
#
#"052_extra_large_clamp_16k/kitedemo_0",
#
#"052_extra_large_clamp_16k/kitedemo_1",
#
#"052_extra_large_clamp_16k/kitedemo_2",
#
#"052_extra_large_clamp_16k/kitedemo_3",
#
#"052_extra_large_clamp_16k/kitedemo_4",
#
#"052_extra_large_clamp_16k/temple_0",
#
#"052_extra_large_clamp_16k/temple_1",
#
#"052_extra_large_clamp_16k/temple_2",
#
#"052_extra_large_clamp_16k/temple_3",
#
#"052_extra_large_clamp_16k/temple_4",
#
#"061_foam_brick_16k/kitchen_0",
#
#"061_foam_brick_16k/kitchen_1",
#
#"061_foam_brick_16k/kitchen_2",
#
#"061_foam_brick_16k/kitchen_3",
#
#"061_foam_brick_16k/kitchen_4",
#
#"061_foam_brick_16k/kitedemo_0",
#
#"061_foam_brick_16k/kitedemo_1",
#
#"061_foam_brick_16k/kitedemo_2",
#
#"061_foam_brick_16k/kitedemo_3",
#
#"061_foam_brick_16k/kitedemo_4",
#
#"061_foam_brick_16k/temple_0",
#
#"061_foam_brick_16k/temple_1",
#
#"061_foam_brick_16k/temple_2",
#
#"061_foam_brick_16k/temple_3",
#
#"061_foam_brick_16k/temple_4"
#]
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "kitchen_0")
from pyquaternion import Quaternion
from transformations import euler_from_matrix
import math
from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points
from tqdm import tqdm, trange

def polar2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

ROOT_OUTDIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/train'
OUTFILE_NAME = 'instances_fat_train_pose_2018'

#ROOT_DIR = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/val'
#IMAGE_DIR = os.path.join(ROOT_DIR, "kitchen_3")
#ANNOTATION_DIR = os.path.join(ROOT_DIR, "kitchen_3")
#OUTFILE_NAME = 'instances_fat_val2018'

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

def find_viewpoint_id(sphere_points, point):
    distances = np.linalg.norm(sphere_points - point, axis=1)
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

def main():

    ng = 642
    print ( '' )
    print ( '  Number of points NG = %d' % ( ng ) )

    viewpoints_xyz = sphere_fibonacci_grid_points ( ng )
    # inplane_rot_angles = np.linspace(-math.pi/4, math.pi/4, 19)
    inplane_rot_angles = np.linspace(-math.pi, math.pi, 68)
    
    object_settings_file = Path(os.path.join(IMAGE_DIR_LIST[0], "_object_settings.json"))
    if object_settings_file.is_file():
        with open(object_settings_file) as file:
            object_settings_data = json.load(file)
            CLASSES = object_settings_data['exported_object_classes']
            CATEGORIES = [{
                'id': i+1,
                'name': CLASSES[i],
                'supercategory': 'shape',
            } for i in range(0,len(CLASSES))]
        
            SEGMENTATION_DATA =  object_settings_data['exported_objects']    
    else:
        raise Exception("Object settings file not found")
            

    VIEWPOINTS = [{
                'id': i,
                'name': viewpoints_xyz[i].tolist(),
            } for i in range(0, len(viewpoints_xyz))]

    INPLANE_ROTATIONS = [{
                'id': i,
                'name': inplane_rot_angles[i],
            } for i in range(0, len(inplane_rot_angles))]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "viewpoints" : VIEWPOINTS,
        "inplate_rotations" : INPLANE_ROTATIONS,
        "images": [],
        "annotations": []
    }

    image_global_id = 1
    segmentation_global_id = 1

    # filter for jpeg images
    for IMAGE_DIR in IMAGE_DIR_LIST:
        for root, _, files in os.walk(IMAGE_DIR):
            image_files = filter_for_jpeg(root, files)
            dir_name = os.path.basename(IMAGE_DIR)
    
            # go through each image
            for ii in trange(len(image_files)):
                image_filename = image_files[ii]
                image_out_filename =  os.path.join(dir_name, os.path.basename(image_filename))
                img_size = (960,540)
                image_info = pycococreatortools.create_image_info(
                    image_global_id, image_out_filename, img_size
                )
#               plt.figure()
#               skimage.io.imshow(skimage.io.imread(image_filename))
#               plt.show()
                # filter for associated png annotations
                for root, _, files in os.walk(IMAGE_DIR):
                    segmentation_image_files = filter_for_annotations(root, files, image_filename)
                    label_files = filter_for_labels(root, files, image_filename)
                    boxes = []
                    labels = []
                    segmentation_ids = []
                    # go through each associated json file containing objects data
                    for label_filename in label_files:
                        # print("File %d - %s"% (image_global_id, label_filename))
                        my_file = Path(label_filename)
    
                        if my_file.is_file():
                            with open(label_filename) as file:
                                label_data = json.load(file)
                                for i in range(0, len(label_data['objects'])):
                                    # print(label_data['objects'][i].keys())
                                    class_name = label_data['objects'][i]['class']
                                    class_bounding_box = label_data['objects'][i]['bounding_box']
                                    quat = label_data['objects'][i]['quaternion_xyzw']
                                    q = Quaternion(quat[3], quat[0], quat[1], quat[2])
                                    # print("angles : {}".format(q.angle))
                                    
                                    angles = euler_from_matrix(q.rotation_matrix)
                                    xyz_coord = (polar2cart(1, angles[1], angles[2]))

                                    viewpoint_id = find_viewpoint_id(viewpoints_xyz, xyz_coord)
                                    inplane_rotation_id = find_inplane_rotation_id(inplane_rot_angles, angles[0])

                                    class_label = [x['id'] for x in CATEGORIES if x['name'] in class_name][0]
                                    segmentation_id = [x['segmentation_class_id'] for x in SEGMENTATION_DATA if x['class'] in class_name][0]
    
                                    boxes.append(class_bounding_box['top_left'] + class_bounding_box['bottom_right'])
                                    labels.append(class_label)     
                                    segmentation_ids.append([x['segmentation_class_id'] for x in SEGMENTATION_DATA if x['class'] in class_name][0])

                                    # Create binary masks from segmentation image
                                    for segmentation_image_file in segmentation_image_files:
                                        segmentation_image = skimage.io.imread(segmentation_image_file)
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