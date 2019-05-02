# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time
import sys
sys.path.append('..')
from tools import convert_shapestacks_coco
import random
import os
from PIL import Image

# /data/sarthak_data/dataset/jenga_recordings/env_jenga-h=5-id=02798-n=12-r=5/rgb-w=0-f=0-l=0-c=original-cam_5-r=5-mono-0.png
# /data/sarthak_data/dataset/jenga_recordings/env_jenga-h=7-id=02539-n=16-r=7/rgb-w=0-f=0-l=0-c=original-cam_8-r=7-mono-0.png

# /data/sarthak_data/dataset/jenga_recordings/env_jenga-h=8-id=02014-n=16-r=11/rgb-w=0-f=0-l=0-c=original-cam_7-r=11-mono-0.png
# /data/sarthak_data/dataset/jenga_recordings/env_jenga-h=6-id=03246-n=12-r=10/rgb-w=0-f=0-l=0-c=original-cam_12-r=10-mono-0.png
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=10,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    img_seg_files_dict = convert_shapestacks_coco.filter_for_jpeg()
    img_list = list(img_seg_files_dict.keys())

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    # img_path = os.path.join(convert_shapestacks_coco.IMG_DIR, img_list[random.randint(0,len(img_list))])
    img_path = '/data/sarthak_data/dataset/jenga_recordings/env_jenga-h=8-id=02014-n=16-r=11/rgb-w=0-f=0-l=0-c=original-cam_7-r=11-mono-0.png'
    print(img_path)
    img = cv2.imread(img_path)

    composite, mask_list = coco_demo.run_on_opencv_image(img)
    if not os.path.exists('demo/output/'):
        os.makedirs('demo/output/')

    
    im_name = "rgb-w=0-f=0-l=0-c=original-cam_8-r=12-mono-0"
    cv2.imwrite('demo/output/{}.png'.format(im_name), cv2.resize(img,(224,224)))
    cv2.imwrite('demo/output/composite-{}.png'.format(im_name), cv2.resize(composite,(224,224)))

    for i in range(len(mask_list)):
        im_name = "vseg-w=0-f=0-l=0-c=original-cam_8-seg-{}".format(i)
        cv2.imwrite('demo/output/visualize-{}.png'.format(im_name), cv2.resize(mask_list[i], (224,224)))
        Image.fromarray(mask_list[i]).resize((224,224)).convert('1').save('demo/output/{}.png'.format(im_name))

    # cv2.imshow("COCO detections", composite)
    # if cv2.waitKey(1) == 27:
    #     break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
