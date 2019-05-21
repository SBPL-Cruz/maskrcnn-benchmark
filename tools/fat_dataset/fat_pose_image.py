import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from PIL import Image
import numpy as np
import torch
import json
import sys



from maskrcnn_benchmark.config import cfg
from pycocotools.coco import COCO
import skimage.io as io
import pylab
from dipy.core.geometry import cart2sphere, sphere2cart
from convert_fat_coco import *
from mpl_toolkits.axes_grid1 import ImageGrid




class FATImage:
    def __init__(self, coco_annotation_file=None, coco_image_directory=None):
        self.coco_image_directory = coco_image_directory
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
        # image_data = self.example_coco.loadImgs(self.image_ids[np.random.randint(0, len(self.image_ids))])[0]
        image_data = self.example_coco.loadImgs(self.image_ids[22])[0]
        # print(image_data)
        annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
        annotations = self.example_coco.loadAnns(annotation_ids)
        self.example_coco.showAnns(annotations)
        # print(annotations)

        return image_data, annotations
    
    def visualize_image_annotations(self, image_data, annotations):
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2
        img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        image = io.imread(img_path)
        count = 1
        fig = plt.figure(2, (4., 4.), dpi=1000)
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

    def get_ros_pose(self, location, quat):
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        p = Pose()
        p.position.x, p.position.y, p.position.z = [i/100 for i in location]
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat[0], quat[1], quat[2], quat[3]
        return p

    def visualize_pose_ros(self, image_data, annotations, frame='camera'):
        print("ROS visualizing")
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
            sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import rospy
        import rospkg
        import rosparam
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge, CvBridgeError

        rospy.init_node('fat_pose')
        self.ros_rate = rospy.Rate(5)
        self.objects_pose_pub = rospy.Publisher('fat_image/objects_pose', PoseArray, queue_size=1, latch=True)
        self.camera_pose_pub = rospy.Publisher('fat_image/camera_pose', PoseStamped, queue_size=1, latch=True)
        self.scene_image_pub = rospy.Publisher("fat_image/scene_image", Image)
        self.bridge = CvBridge()
        import cv2
            
        img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        # cv_scene_image = cv2.imread(img_path)
        # cv2.imshow(cv_scene_image)
        # image = io.imread(img_path)
        # plt.imshow(image); plt.axis('off')
        # plt.show()

        object_pose_msg = PoseArray()
        object_pose_msg.header.frame_id = frame
        object_pose_msg.header.stamp = rospy.Time.now()

        camera_pose_msg = PoseStamped()
        camera_pose_msg.header.frame_id = frame
        camera_pose_msg.header.stamp = rospy.Time.now()

        fat_world_to_world = {}
        fat_world_to_world['location'] = [0,0,0]
        # fat_world_to_world['quaternion_xyzw'] = [-0.658, 0.259, 0.259, 0.658]
        fat_world_to_world['quaternion_xyzw'] = [0.707, 0, 0, -0.707]

        while not rospy.is_shutdown():
            for annotation in annotations:
                class_name = self.categories[annotation['category_id']]['name']

                if frame == 'camera':
                    location, quat = annotation['location'], annotation['quaternion_xyzw']

                if frame == 'fat_world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'])
                    camera_location, camera_quat = get_camera_pose_in_world(annotation['camera_pose'])      

                if frame == 'world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'], fat_world_to_world)
                    camera_location, camera_quat = get_camera_pose_in_world(annotation['camera_pose'], fat_world_to_world)      

                object_pose_msg.poses.append(self.get_ros_pose(location, quat))

                if frame != 'camera':
                    camera_pose_msg.pose = self.get_ros_pose(camera_location, camera_quat)
                    self.camera_pose_pub.publish(camera_pose_msg)

                print("Rotation for {} : {}".format(class_name, RT_transform.quat2euler(get_xyzw_quaternion(quat))))
                    
                # print(location, quat)
                # try:
                #     self.scene_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_scene_image, "bgr8"))
                # except CvBridgeError as e:
                #     print(e)
            self.objects_pose_pub.publish(object_pose_msg)
            self.ros_rate.sleep()

    def visualize_model_output(self, image_data):
        # plt.figure()
        img_path = os.path.join(image_directory, image_data['file_name'])
        image = io.imread(img_path)
        # plt.imshow(image); plt.axis('off')

        # # Running model on image

        from predictor import COCODemo
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2

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
        fig = plt.figure(1, (8., 8.), dpi=6000)
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
            # image_file = os.path.join(
            #     "{}-color.png".format(i),
            # )
            rgb_gl = img_list[i][0]
            # cv2.imwrite(image_file, rgb_gl)
            grid[i+1].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
            grid[i+1].axis("off")

        plt.savefig('model_output.eps', format='eps', dpi=6000)
        plt.show()

if __name__ == '__main__':
    
    # coco_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/coco_results.pth')
    # all_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/predictions.pth')

    image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'

    fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory)
    image_data, annotations = fat_image.get_random_image()
    # fat_image.visualize_image_annotations(image_data, annotations)
    fat_image.visualize_model_output(image_data)
    # fat_image.visualize_pose_ros(image_data, annotations, frame='fat_world')




