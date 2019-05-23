import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from PIL import Image
import numpy as np
import torch
import json
import sys
from tqdm import tqdm, trange


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
        self.depth_factor = 1000

        self.world_to_fat_world = {}
        self.world_to_fat_world['location'] = [0,0,0]
        self.world_to_fat_world['quaternion_xyzw'] = [0.853, -0.147, -0.351, -0.357]
        self.model_dir = "/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/"

        # self.world_to_fat_world['quaternion_xyzw'] = [0.7071, 0, 0, -0.7071]

    def get_random_image(self):
        # image_data = self.example_coco.loadImgs(self.image_ids[np.random.randint(0, len(self.image_ids))])[0]
        # image_data = self.example_coco.loadImgs(self.image_ids[22])[0]
        image_data = self.example_coco.loadImgs(self.image_ids[91])[0]
        # print(image_data)
        annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
        annotations = self.example_coco.loadAnns(annotation_ids)
        self.example_coco.showAnns(annotations)
        # print(annotations)

        return image_data, annotations
    
    def get_renderer(self, class_name):
        width = 960
        height = 540
        K = np.array([[self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']], 
                    [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']], 
                    [0, 0, 1]])
        ZNEAR = 0.1
        ZFAR = 20
        model_dir = os.path.join(self.model_dir, "models", class_name)
        render_machine = Render_Py(model_dir, K, width, height, ZNEAR, ZFAR)
        return render_machine

    def render_pose(self, class_name, render_machine, rotation_angles, location):
        fixed_transform = np.transpose(np.array(self.fixed_transforms_dict[class_name]))
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

        depth_gl = (depth_gl * self.depth_factor).astype(np.uint16)
        return rgb_gl, depth_gl

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

    def update_coordinate_max_min(self, max_min_dict, location):
        location = [i/100 for i in location]
        if location[0] > max_min_dict['xmax']:
            max_min_dict['xmax'] = location[0]
        if location[1] > max_min_dict['ymax']:
            max_min_dict['ymax'] = location[1]
        if location[2] > max_min_dict['zmax']:
            max_min_dict['zmax'] = location[2]
        
        if location[0] < max_min_dict['xmin']:
            max_min_dict['xmin'] = location[0]
        if location[1] < max_min_dict['ymin']:
            max_min_dict['ymin'] = location[1]
        if location[2] < max_min_dict['zmin']:
            max_min_dict['zmin'] = location[2]

        return max_min_dict

    def visualize_pose_ros(self, image_data, annotations, frame='camera'):
        print("ROS visualizing")
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
            sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import rospy
        import rospkg
        import rosparam
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        from sensor_msgs.msg import Image
        # if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
        #     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        # from cv_bridge import CvBridge, CvBridgeError

        rospy.init_node('fat_pose')
        self.ros_rate = rospy.Rate(5)
        self.objects_pose_pub = rospy.Publisher('fat_image/objects_pose', PoseArray, queue_size=1, latch=True)
        self.camera_pose_pub = rospy.Publisher('fat_image/camera_pose', PoseStamped, queue_size=1, latch=True)
        self.scene_image_pub = rospy.Publisher("fat_image/scene_image", Image)
        # self.bridge = CvBridge()
        # import cv2
            
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

        max_min_dict = {
            'xmin' : np.inf,
            'ymin' : np.inf,
            'zmin' : np.inf,
            'xmax' : -np.inf,
            'ymax' : -np.inf,
            'zmax' : -np.inf
        }
        # while not rospy.is_shutdown():
        for i in range(10):
            for annotation in annotations:
                class_name = self.categories[annotation['category_id']]['name']

                if frame == 'camera':
                    location, quat = annotation['location'], annotation['quaternion_xyzw']

                if frame == 'fat_world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'])
                    camera_location, camera_quat = get_camera_pose_in_world(annotation['camera_pose'])      

                if frame == 'world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'], self.world_to_fat_world)
                    camera_location, camera_quat = get_camera_pose_in_world(annotation['camera_pose'], self.world_to_fat_world)      

                object_pose_msg.poses.append(self.get_ros_pose(location, quat))
                max_min_dict = self.update_coordinate_max_min(max_min_dict, location)
                if frame != 'camera':
                    camera_pose_msg.pose = self.get_ros_pose(camera_location, camera_quat)
                    self.camera_pose_pub.publish(camera_pose_msg)

                print("Location for {} : {}".format(class_name, location))
                print("Rotation for {} : {}\n".format(class_name, RT_transform.quat2euler(get_wxyz_quaternion(quat))))
                    
                # print(location, quat)
                # try:
                #     self.scene_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_scene_image, "bgr8"))
                # except CvBridgeError as e:
                #     print(e)
            self.objects_pose_pub.publish(object_pose_msg)
            self.ros_rate.sleep()
        return max_min_dict

    def visualize_perch_output(self, image_data, annotations, max_min_dict, frame='fat_world'):
        from perch import FATPerch
        print("camera instrinsics : {}".format(self.camera_intrinsics))
        color_img_path = os.path.join(image_directory, image_data['file_name'])
        depth_img_path = color_img_path.replace('.jpg', '.depth.png')
        input_image_files = {
            'input_color_image' : color_img_path,
            'input_depth_image' : depth_img_path
        } 

        if frame == 'fat_world':
            camera_pose = get_camera_pose_in_world(annotations[0]['camera_pose'], None, 'rot')
        if frame == 'world':
            camera_pose = get_camera_pose_in_world(annotations[0]['camera_pose'], self.world_to_fat_world, 'rot')

        camera_pose[:3, 3] /= 100 
        print("camera_pose : {}".format(camera_pose))
        print("max_min_ranges : {}".format(max_min_dict))
        camera_pose = camera_pose.flatten().tolist()
        params = {
            'x_min' : max_min_dict['xmin'],
            'x_max' : max_min_dict['xmax'],
            'y_min' : max_min_dict['ymin'],
            'y_max' : max_min_dict['ymax'],
            'required_object' : '004_sugar_box',
            'table_height' :  max_min_dict['zmin'],
            'use_external_render' : 0, 
            'camera_pose': camera_pose,
            'reference_frame_': frame
        }
        camera_params = {
            'camera_width' : 960,
            'camera_height' : 540,
            'camera_fx' : self.camera_intrinsics['fx'],
            'camera_fy' : self.camera_intrinsics['fy'],
            'camera_cx' : self.camera_intrinsics['cx'],
            'camera_cy' : self.camera_intrinsics['cy'],
            'camera_znear' : 0.1,
            'camera_zfar' : 20,
        }
        fat_perch = FATPerch(params=params, input_image_files=input_image_files, camera_params=camera_params)

    def visualize_model_output(self, image_data, use_thresh=False):
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
        )
        
        img = cv2.imread(img_path)
        # composite, result, img_list = coco_demo.run_on_opencv_image(img)
        composite, result, rotation_list = coco_demo.run_on_opencv_image(img, use_thresh=use_thresh)
        # print(rotation_list)
        # plt.imshow(composite); plt.axis('off')
        # cv2.imwrite('composite.png', composite)
        
        top_viewpoint_ids = rotation_list['top_viewpoint_ids']
        top_inplane_rotation_ids = rotation_list['top_inplane_rotation_ids']
        labels = rotation_list['labels']

        fig = plt.figure(1, (8., 8.), dpi=6000)
        plt.axis("off")
        plt.tight_layout()
        if use_thresh == False:
            grid_size = len(top_viewpoint_ids)+1
            grid = ImageGrid(fig, 111,  
                        nrows_ncols=(1, grid_size),
                        axes_pad=0.1, 
                        )
            grid[0].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
            grid[0].axis("off")
        else:
            grid_size = top_viewpoint_ids[0,:].shape[0]*top_inplane_rotation_ids[0,:].shape[0]+1
            grid = ImageGrid(fig, 111,  
                        nrows_ncols=(top_viewpoint_ids.shape[0], grid_size),
                        axes_pad=0.1, 
                        )
        

        print("Predicted top_viewpoint_ids : {}".format(top_viewpoint_ids))
        print("Predicted top_inplane_rotation_ids : {}".format(top_inplane_rotation_ids))
        print(labels)
        img_list = []

        if use_thresh == False:
            for i in range(len(top_viewpoint_ids)):
                viewpoint_id = top_viewpoint_ids[i]
                inplane_rotation_id = top_inplane_rotation_ids[i]
                label = labels[i]
                fixed_transform = self.fixed_transforms_dict[label]
                theta, phi = get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
                inplane_rotation_angle = get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
                xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
                print("Recovered rotation : {}".format(xyz_rotation_angles))
                rgb_gl, depth_gl = render_pose(
                    label, fixed_transform, self.camera_intrinsics, xyz_rotation_angles, [0,0,100]
                )
                grid[i+1].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
                grid[i+1].axis("off")
        else:
            grid_i = 0
            for box_id in range(top_viewpoint_ids.shape[0]):
                grid[grid_i].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
                grid[grid_i].axis("off")
                grid_i += 1
                label = labels[box_id]
                render_machine = self.get_renderer(label)
                for viewpoint_id in top_viewpoint_ids[box_id, :]:
                    for inplane_rotation_id in top_inplane_rotation_ids[box_id, :]:
                        fixed_transform = self.fixed_transforms_dict[label]
                        theta, phi = get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
                        inplane_rotation_angle = get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
                        xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
                        print("{}. Recovered rotation : {}".format(grid_i, xyz_rotation_angles))
                        rgb_gl, depth_gl = self.render_pose(
                            label, render_machine, xyz_rotation_angles, [0,0,100]
                        )
                        grid[grid_i].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
                        grid[grid_i].axis("off")
                        grid_i += 1
        # plt.savefig('model_output.eps', format='eps', dpi=6000)
        plt.savefig(
            'model_output_{}.png'.format(image_data['file_name'].replace('/','_').replace('.jpg', '')), 
            dpi=1000, bbox_inches = 'tight', pad_inches = 0
        )
        plt.show()

    

if __name__ == '__main__':
    
    # coco_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/coco_results.pth')
    # all_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/predictions.pth')

    image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'

    fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory)
    image_data, annotations = fat_image.get_random_image()
    # fat_image.visualize_image_annotations(image_data, annotations)
    fat_image.visualize_model_output(image_data, use_thresh=True)
    # max_min_dict = fat_image.visualize_pose_ros(image_data, annotations, frame='world')

    # fat_image.visualize_perch_output(image_data, annotations, max_min_dict, frame='world')





