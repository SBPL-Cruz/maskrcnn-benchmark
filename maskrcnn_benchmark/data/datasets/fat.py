import os

import torch
import torch.utils.data
from PIL import Image
import sys
import glob
import json

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList
from pathlib import Path


class FATDataset(torch.utils.data.Dataset):

    CLASSES = [
        "__background__ ",
		"002_master_chef_can_16k",
		"003_cracker_box_16k",
		"004_sugar_box_16k",
		"005_tomato_soup_can_16k",
		"006_mustard_bottle_16K",
		"007_tuna_fish_can_16k",
		"008_pudding_box_16k",
		"009_gelatin_box_16k",
		"010_potted_meat_can_16k",
		"011_banana_16k",
		"019_pitcher_base_16k",
		"021_bleach_cleanser_16k",
		"024_bowl_16k",
		"025_mug_16k",
		"035_power_drill_16k",
		"036_wood_block_16k",
		"037_scissors_16k",
		"040_large_marker_16k",
		"051_large_clamp_16k",
		"052_extra_large_clamp_16k",
		"061_foam_brick_16k"
    ]

    # SCENES = [
    #     "mixed/kitchen_0",
    #     "mixed/kitchen_1",
    #     "mixed/kitchen_2",
    #     "mixed/kitchen_3",
    #     "mixed/kitchen_4",
    #     "mixed/kitedemo_0",
    #     "mixed/kitedemo_1",
    #     "mixed/kitedemo_2",
    #     "mixed/kitedemo_3",
    #     "mixed/kitedemo_4",
    #     "mixed/temple_0",
    #     "mixed/temple_1",
    #     "mixed/temple_2",
    #     "mixed/temple_3",
    #     "mixed/temple_4"
    # ]


    def get_img_list(self, root):
        # all_image_paths = []
        # for img_dir_path in dir_list:
        all_image_paths = glob.glob(root + "/" + "*" + '/*.left.jpg')
        # print(len[all_image_paths])
        return all_image_paths

    def __init__(self, data_dir, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        self.image_lists = []
        self.image_boxes = []
        self.image_labels = []
        
        all_images = self.get_img_list(self.root)
        for imgpath in all_images:
            labpath = imgpath.replace('.jpg', '.json').replace('.png','.json')
            my_file = Path(labpath)
            boxes = []
            labels = []
        
            if my_file.is_file():
                with open(labpath) as file:
                    label_data = json.load(file)
                    for i in range(0, len(label_data['objects'])):
                        class_name = label_data['objects'][i]['class']
                        class_bounding_box = label_data['objects'][i]['bounding_box']
                        class_label = FATDataset.CLASSES.index(class_name)
                        boxes.append(class_bounding_box['top_left'] + class_bounding_box['bottom_right'])
                        labels.append(class_label)
                if len(boxes) > 0 and len(labels) > 0 and len(boxes) == len(labels):
                    self.image_boxes.append(boxes)
                    self.image_labels.append(labels) 
                    self.image_lists.append(imgpath)
                else:
                    print("File %s doesn't have boxes or labels in json file" % imgpath)
            else:
                print("File %s doesn't have a label file" % imgpath)
        print("********************************************")
        print("Numbed of images with labels : " + str(len(self.image_lists)))
        print("********************************************")

    def __getitem__(self, item):
        imgpath = self.image_lists[item]
        image = Image.open(imgpath).convert("RGB")
#        labpath = imgpath.replace('.jpg', '.json').replace('.png','.json')
        print(imgpath)
        boxes = self.image_boxes[item]
        labels = self.image_labels[item]
#        print(labpath)
#        boxes = []
#        labels = []
#        try:
#            with open(labpath) as file:
#            	label_data = json.load(file)
#            	for i in range(0, len(label_data['objects'])):
#                    class_name = label_data['objects'][i]['class']
#                    class_bounding_box = label_data['objects'][i]['bounding_box']
##                    print(class_name)
##                    print(class_bounding_box)
#                    class_label = FATDataset.CLASSES.index(class_name)
##                    print(class_bounding_box['top_left'])
#                    boxes.append(class_bounding_box['top_left'] + class_bounding_box['bottom_right'])
#                    labels.append(class_label)
#
#        except TypeError as e:
#            print(labpath)
#            print(e)
        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        # boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        # labels = torch.tensor([10, 20])

        # create a BoxList from the boxes
        # print(boxes)
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        target.add_field("labels", torch.tensor(labels))
        target = target.clip_to_image(remove_empty=True)

        if self.transforms:
            image, target = self.transforms(image, target)

        # return the image, the boxlist and the idx in your dataset
        return image, target, item

    def __len__(self):
        return len(self.image_lists)

    def get_img_info(self, item):
        """
        Return the image dimensions for the image, without
        loading and pre-processing it
        """
        return {"height": 540, "width": 960}
