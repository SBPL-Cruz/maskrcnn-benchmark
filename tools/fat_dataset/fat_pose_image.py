import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
from PIL import Image
import numpy as np
# import torch
import json
import sys
from tqdm import tqdm, trange

from pycocotools.coco import COCO
import skimage.io as io
import pylab
from convert_fat_coco import *
from mpl_toolkits.axes_grid1 import ImageGrid

from lib.utils.mkdir_if_missing import mkdir_if_missing
from lib.render_glumpy.render_py import Render_Py
from lib.pair_matching import RT_transform
import pcl
from pprint import pprint

ROS_PYTHON2_PKG_PATH = ['/opt/ros/kinetic/lib/python2.7/dist-packages', '/usr/local/lib/python2.7/dist-packages/']
ROS_PYTHON3_PKG_PATH = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/install/lib/python3/dist-packages'
# ROS_PYTHON3_PKG_PATH = '/home/jessy/projects/ros_python3_ws/install/lib/python3/dist-packages'

class FATImage:
    def __init__(self, 
            coco_annotation_file=None, coco_image_directory=None, 
            depth_factor=1000, 
            model_dir='/media/aditya/A69AFABA9AFA85D9/Datasets/YCB_Video_Dataset/',
            model_mesh_in_mm=False,
            model_mesh_scaling_factor=1,
            models_flipped=False
        ):

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
        self.camera_intrinsic_matrix = \
            np.array([[self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']], 
                     [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']], 
                     [0, 0, 1]])
        self.depth_factor = depth_factor

        self.world_to_fat_world = {}
        self.world_to_fat_world['location'] = [0,0,0]
        self.world_to_fat_world['quaternion_xyzw'] = [0.853, -0.147, -0.351, -0.357]
        self.model_dir = model_dir
        self.model_params = {
            'mesh_in_mm' : model_mesh_in_mm,
            'mesh_scaling_factor' : model_mesh_scaling_factor,
            'flipped' : models_flipped
        }
        self.rendered_root_dir = os.path.join(self.model_dir, "rendered")
        mkdir_if_missing(self.rendered_root_dir)

        self.search_resolution_translation = 0.08
        self.search_resolution_yaw = 0.3926991

        # This matrix converts camera frame (X pointing out) to camera optical frame (Z pointing out) 
        # Multiply by this matrix to convert camera frame to camera optical frame
        # Multiply by inverse of this matrix to convert camera optical frame to camera frame
        self.cam_to_body = np.array([[0, 0, 1, 0],
                                     [-1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, 0, 1]])
        # self.world_to_fat_world['quaternion_xyzw'] = [0.7071, 0, 0, -0.7071]

    def get_random_image(self, name=None, required_objects=None):
        # image_data = self.example_coco.loadImgs(self.image_ids[np.random.randint(0, len(self.image_ids))])[0]
        if name is not None:
            found = False
            print("Tying to get image from DB : {}".format(name))
            for i in range(len(self.image_ids)):
                image_data = self.example_coco.loadImgs(self.image_ids[i])[0]
                if image_data['file_name'] == name:
                    found = True
                    break
            if found == False:
                raise Exception("File not found")
        else:
            image_data = self.example_coco.loadImgs(self.image_ids[7000])[0]
        # image_data = self.example_coco.loadImgs(self.image_ids[0])[0]
        print(image_data)
        annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
        annotations = self.example_coco.loadAnns(annotation_ids)
        self.example_coco.showAnns(annotations)
        # print(annotations)
        # if required_objects is not None:
        #     filtered_annotations = []
        #     for annotation in annotations:
        #         class_name = self.categories[annotation['category_id']]['name']
        #         if class_name in required_objects:
        #             filtered_annotations.append(annotation)
        #     return image_data, filtered_annotations

        return image_data, annotations
    
    def save_yaw_only_dataset(self, scene='all'):
        print("Processing {} images".format(len(self.image_ids)))
        num_images = len(self.image_ids)
        # num_images = 10
        for i in range(num_images):
            image_data = self.example_coco.loadImgs(self.image_ids[i])[0]
            if scene != 'all' and  image_data['file_name'].startswith(scene) == False:
                continue
            annotation_ids = self.example_coco.getAnnIds(imgIds=image_data['id'], catIds=self.category_ids, iscrowd=None)
            annotations = self.example_coco.loadAnns(annotation_ids)
            yaw_only_objects, _ = fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)



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

    def get_ros_pose(self, location, quat, units='cm'):
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        p = Pose()
        if units == 'cm':
            p.position.x, p.position.y, p.position.z = [i/100 for i in location]
        else:
            p.position.x, p.position.y, p.position.z = [i for i in location]

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

    def get_world_point(self, point) :
        camera_fx_reciprocal_ = 1.0 / self.camera_intrinsics['fx']
        camera_fy_reciprocal_ = 1.0 / self.camera_intrinsics['fy']

        world_point = np.zeros(3)

        world_point[2] = point[2]
        world_point[0] = (point[0] - self.camera_intrinsics['cx']) * point[2] * (camera_fx_reciprocal_)
        world_point[1] = (point[1] - self.camera_intrinsics['cy']) * point[2] * (camera_fy_reciprocal_)

        return world_point

    def get_table_pose(self, depth_img_path, frame):
        # Creates a point cloud in camera frame and calculates table pose using RANSAC

        import rospy
        # from tf.transformations import quaternion_from_euler
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2
        from PIL import Image
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        K_inv = np.linalg.inv(self.camera_intrinsic_matrix)
        points_3d = np.zeros((depth_image.shape[0]*depth_image.shape[1], 4), dtype=np.float32)
        count = 0
        cloud = pcl.PointCloud_PointXYZRGB()
        depth_image_pil = np.asarray(Image.open(depth_img_path), dtype=np.float16)
        for x in range(depth_image.shape[1]):
            for y in range(depth_image.shape[0]):
                # point = np.array([x,y,1])
                # t_point = np.matmul(K_inv, point)
                # print("point : {},{}".format(t_point, depth_image[y,x]))
                # print("point : {},{}".format(t_point, depth_image_pil[y,x]/65536 * 10))
                # points_3d[count, :] = t_point[:2].tolist() + \
                #                         [(depth_image[y,x]/self.depth_factor)] +\
                #                         [255 << 16 | 255 << 8 | 255]

                point = np.array([x,y,depth_image[y,x]/self.depth_factor])
                w_point = self.get_world_point(point)
                points_3d[count, :] = w_point.tolist() + \
                                        [255 << 16 | 255 << 8 | 255]
                count += 1

        cloud.from_array(points_3d)
        seg = cloud.make_segmenter()
        # Optional
        seg.set_optimize_coefficients (True)
        # Mandatory
        seg.set_model_type (pcl.SACMODEL_PLANE)
        seg.set_method_type (pcl.SAC_RANSAC)
        seg.set_distance_threshold (0.05)
        # ros_msg = self.xyzrgb_array_to_pointcloud2(
        #     points_3d[:,:3], points_3d[:,3], rospy.Time.now(), frame
        # )
        # print(ros_msg)
        # pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients)
        # pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        inliers, model = seg.segment()

        # if inliers.size
        # 	return
        # end
        # print (model)
        angles = []

        # projection on x,y axis to get yaw
        yaw = np.arctan(model[1]/model[0])
        # pitch = np.arcsin(model[2]/np.linalg.norm(model[:3]))

        # projection on z,y axis to get pitch
        pitch = np.arctan(model[2]/model[1])+np.pi/2
        # pitch = np.arctan(model[2]/model[1])
        roll = 0

        # r is probably for tait-brian angles meaning we can use roll pitch yaw
        # x in sequence means angle with that x-axis excluded (vector projected in other two axis)
        q = get_xyzw_quaternion(RT_transform.euler2quat(roll,pitch,yaw, 'ryxz').tolist())
        # for i in range(3):
        #     angle = model[i]/np.linalg.norm(model[:3])
        #     angles.append(np.arccos(angle))
        # print(inliers)
        inlier_points = points_3d[inliers]
        # ros_msg = self.xyzrgb_array_to_pointcloud2(
        #     inlier_points[:,:3], inlier_points[:,3], rospy.Time.now(), frame
        # )
        ros_msg = self.xyzrgb_array_to_pointcloud2(
            points_3d[:,:3], points_3d[:,3], rospy.Time.now(), frame
        )
        location = np.mean(inlier_points[:,:3], axis=0) * 100
        # for i in inliers:
        #     inlier_points.append(points_3d[inliers,:])

        # inlier_points = np.array(inlier_points)
        # q_rot = 
        print("Table location : {}".format(location))
        return ros_msg, location, q

    def xyzrgb_array_to_pointcloud2(self, points, colors, stamp=None, frame_id=None, seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array
        of points.
        '''
        from sensor_msgs.msg import PointCloud2, PointField

        msg = PointCloud2()
        # assert(points.shape == colors.shape)
        colors = np.zeros(points.shape)
        buf = []

        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id
        if seq: 
            msg.header.seq = seq
        if len(points.shape) == 3:
            msg.height = points.shape[1]
            msg.width = points.shape[0]
        else:
            N = len(points)
            xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
            msg.height = 1
            msg.width = N

        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1)
        ]
        msg.is_bigendian = False
        msg.point_step = 24
        msg.row_step = msg.point_step * N
        msg.is_dense = True
        msg.data = xyzrgb.tostring()

        return msg 

    def get_camera_pose_relative_table(self, depth_img_path, type='quat', cam_to_body=None):
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
            sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import rospy
        # if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
        #     sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        # if '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/install/lib/python3/dist-packages' not in sys.path:
        #     sys.path.append('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/ros_python3_ws/install/lib/python3/dist-packages')
        # # These packages need to be python3 specific, tf is built using python3
        # import tf2_ros
        # from geometry_msgs.msg import TransformStamped
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        table_pose_msg = PoseStamped()
        table_pose_msg.header.frame_id = 'camera'
        table_pose_msg.header.stamp = rospy.Time.now()
        scene_cloud, table_location, table_quat = self.get_table_pose(depth_img_path, 'camera')
        table_pose_msg.pose = self.get_ros_pose(
            table_location, 
            table_quat,
        )
        
        camera_pose_table = {
            'location_worldframe' : table_location,
            'quaternion_xyzw_worldframe': table_quat
        }

        print("Table pose wrt to camera : {}".format(camera_pose_table))
        camera_pose_matrix = np.zeros((4,4))
        camera_rotation = RT_transform.quat2mat(get_wxyz_quaternion(camera_pose_table['quaternion_xyzw_worldframe']))
        camera_pose_matrix[:3, :3] = camera_rotation
        camera_location = [i for i in camera_pose_table['location_worldframe']]
        camera_pose_matrix[:, 3] = camera_location + [1]
        
        print("table height : {}".format(table_location))

        # Doing inverse gives us location of camera in table frame
        camera_pose_matrix = np.linalg.inv(camera_pose_matrix)

        # Convert optical frame to body for PERCH
        if cam_to_body is not None:
            camera_pose_matrix = np.matmul(camera_pose_matrix, np.linalg.inv(cam_to_body))

        if type == 'quat':
            quat = RT_transform.mat2quat(camera_pose_matrix[:3, :3]).tolist()
            camera_pose_table = {
                'location_worldframe' : camera_pose_matrix[:3,3],
                'quaternion_xyzw_worldframe':get_xyzw_quaternion(quat)
            }
            return table_pose_msg, scene_cloud, camera_pose_table
        elif type == 'rot':
            return table_pose_msg, scene_cloud, camera_pose_matrix


    def visualize_pose_ros(
            self, image_data, annotations, frame='camera', camera_optical_frame=True, num_publish=10, write_poses=False, ros_publish=True
        ):
        print("ROS visualizing")
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
            sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import rospy
        import rospkg
        import rosparam
        from geometry_msgs.msg import Pose, PoseStamped, PoseArray, Quaternion
        from sensor_msgs.msg import Image, PointCloud2
        for python2_path in ROS_PYTHON2_PKG_PATH:
            if python2_path in sys.path:
                sys.path.remove(python2_path)
        if ROS_PYTHON3_PKG_PATH not in sys.path:
            sys.path.append(ROS_PYTHON3_PKG_PATH)
        # These packages need to be python3 specific, cv2 is imported from environment, cv_bridge is built using python3
        import cv2
        from cv_bridge import CvBridge, CvBridgeError

        rospy.init_node('fat_pose')
        self.ros_rate = rospy.Rate(5)
        self.objects_pose_pub = rospy.Publisher('fat_image/objects_pose', PoseArray, queue_size=1, latch=True)
        self.camera_pose_pub = rospy.Publisher('fat_image/camera_pose', PoseStamped, queue_size=1, latch=True)
        self.scene_color_image_pub = rospy.Publisher("fat_image/scene_color_image", Image)
        self.table_pose_pub = rospy.Publisher("fat_image/table_pose", PoseStamped, queue_size=1, latch=True)
        self.scene_cloud_pub = rospy.Publisher("fat_image/scene_cloud", PointCloud2, queue_size=1, latch=True)
        self.bridge = CvBridge()
            
        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        cv_scene_color_image = cv2.imread(color_img_path)
        # cv2.imshow(cv_scene_color_image)
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

        cam_to_body = self.cam_to_body if camera_optical_frame == False else None

        if frame == 'camera' or frame =='table':
            depth_img_path = self.get_depth_img_path(color_img_path)
            print("depth_img_path : {}".format(depth_img_path))
            table_pose_msg, scene_cloud, camera_pose_table = self.get_camera_pose_relative_table(depth_img_path)

        # while not rospy.is_shutdown():
        rendered_pose_list_out = {}
        transformed_annotations = []
        for i in range(num_publish):
            yaw_only_objects = []
            count = 0
            for annotation in annotations:
                class_name = self.categories[annotation['category_id']]['name']
                
                if frame == 'camera':
                    location, quat = annotation['location'], annotation['quaternion_xyzw']

                if frame == 'table':
                    location, quat = get_object_pose_in_world(annotation, camera_pose_table)
                    camera_location, camera_quat = camera_pose_table['location_worldframe'], camera_pose_table['quaternion_xyzw_worldframe']  

                if frame == 'fat_world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'])
                    camera_location, camera_quat = get_camera_pose_in_world(annotation['camera_pose'], None, type='quat', cam_to_body=cam_to_body)      

                if frame == 'world':
                    location, quat = get_object_pose_in_world(annotation, annotation['camera_pose'], self.world_to_fat_world)
                    camera_location, camera_quat = get_camera_pose_in_world(
                                                        annotation['camera_pose'], self.world_to_fat_world, type='quat', cam_to_body=cam_to_body
                                                    )      
                object_pose_ros = self.get_ros_pose(location, quat)
                object_pose_msg.poses.append(object_pose_ros)
                max_min_dict = self.update_coordinate_max_min(max_min_dict, location)
                if (frame == 'camera' or frame == 'table') and ros_publish:
                    self.table_pose_pub.publish(table_pose_msg)
                    self.scene_cloud_pub.publish(scene_cloud)

                if frame != 'camera' and ros_publish:
                    camera_pose_msg.pose = self.get_ros_pose(camera_location, camera_quat)
                    self.camera_pose_pub.publish(camera_pose_msg)

                print("Location for {} : {}".format(class_name, location))
                rotation_angles = RT_transform.quat2euler(get_wxyz_quaternion(quat), 'rxyz')
                print("Rotation Eulers for {} : {}".format(class_name, rotation_angles))
                print("ROS Pose for {} : {}".format(class_name, object_pose_ros))
                print("Rotation Quaternion for {} : {}\n".format(class_name, quat))
                if np.all(np.isclose(np.array(rotation_angles[:2]), np.array([-np.pi/2, 0]), atol=0.1)):
                    yaw_only_objects.append({'annotation_id' : annotation['id'],'class_name' : class_name})
                
                if i == 0:
                    transformed_annotations.append({
                        'location' : location,
                        'quaternion_xyzw' : quat,
                        'category_id' : self.category_names.index(class_name),
                        'id' : count
                    })
                    count += 1
                try:
                    self.scene_color_image_pub.publish(self.bridge.cv2_to_imgmsg(cv_scene_color_image, "bgr8"))
                except CvBridgeError as e:
                    print(e)

                if class_name not in rendered_pose_list_out:
                    rendered_pose_list_out[class_name] = []

                # rendered_pose_list_out[class_name].append(location.tolist() + list(rotation_angles))
                rendered_pose_list_out[class_name].append(location + quat)

            if ros_publish:
                self.objects_pose_pub.publish(object_pose_msg)
            self.ros_rate.sleep()
        # pprint(rendered_pose_list_out)
        if write_poses:
            for label, poses in rendered_pose_list_out.items():
                rendered_dir = os.path.join(self.rendered_root_dir, label)
                mkdir_if_missing(rendered_dir)
                pose_rendered_file = os.path.join(
                    rendered_dir,
                    "poses.txt",
                )
                poses = np.array(poses)
                # Convert to meters for PERCH
                poses[:,:3] /= 100
                np.savetxt(pose_rendered_file, np.around(poses, 4))

        # max_min_dict['ymax'] = max_min_dict['ymin'] + 2 * self.search_resolution_translation
        max_min_dict['ymax'] += 0.10
        max_min_dict['ymin'] -= 0.10
        max_min_dict['xmax'] += 0.10
        max_min_dict['xmin'] -= 0.10
        # max_min_dict['ymax'] = 2.00
        # max_min_dict['ymin'] = -2.00
        # max_min_dict['xmax'] = 2.00
        # max_min_dict['xmin'] = -2.00
        # max_min_dict['zmin'] = table_pose_msg.pose.position.z
        print("Yaw only objects in the image : {}".format(yaw_only_objects))

        return yaw_only_objects, max_min_dict, transformed_annotations


    def get_depth_img_path(self, color_img_path):
        return color_img_path.replace(os.path.splitext(color_img_path)[1], '.depth.png')

    def get_renderer(self, class_name):
        width = 960
        height = 540
        # K = np.array([[self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']], 
        #             [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']], 
        #             [0, 0, 1]])
        # K = self.camera_intrinsic_matrix
        ZNEAR = 0.1
        ZFAR = 20
        model_dir = os.path.join(self.model_dir, "models", class_name)
        render_machine = Render_Py(model_dir, self.camera_intrinsic_matrix, width, height, ZNEAR, ZFAR)
        return render_machine

    def get_object_pose_with_fixed_transform(self, class_name, location, rotation_angles, type, use_fixed_transform=True):
        # Location in cm
        # Add fixed transform to given object transform so that it can be applied to a model
        object_world_transform = np.zeros((4,4))
        object_world_transform[:3,:3] = RT_transform.euler2mat(rotation_angles[0],rotation_angles[1],rotation_angles[2])
        object_world_transform[:,3] = [i/100 for i in location] + [1]

        if use_fixed_transform:
            fixed_transform = np.transpose(np.array(self.fixed_transforms_dict[class_name]))
            fixed_transform[:3,3] = [i/100 for i in fixed_transform[:3,3]]
            total_transform = np.matmul(object_world_transform, fixed_transform)
        else:
            total_transform = object_world_transform

        if type == 'quat':
            quat = RT_transform.mat2quat(total_transform[:3, :3]).tolist()
            return total_transform[:3,3], get_xyzw_quaternion(quat)
        elif type == 'rot':
            return total_transform
        elif type == 'euler':
            return total_transform[:3,3], RT_transform.mat2euler(total_transform[:3,:3])

    def render_pose(self, class_name, render_machine, rotation_angles, location):
        # Takes rotation and location in camera frame for object and renders and image for it
        # Expects location in cm

        # fixed_transform = np.transpose(np.array(self.fixed_transforms_dict[class_name]))
        # fixed_transform[:3,3] = [i/100 for i in fixed_transform[:3,3]]
        # object_world_transform = np.zeros((4,4))
        # object_world_transform[:3,:3] = RT_transform.euler2mat(rotation_angles[0],rotation_angles[1],rotation_angles[2])
        # object_world_transform[:,3] = location + [1]

        # total_transform = np.matmul(object_world_transform, fixed_transform)
        total_transform = self.get_object_pose_with_fixed_transform(class_name, location, rotation_angles, 'rot')
        pose_rendered_q = RT_transform.mat2quat(total_transform[:3,:3]).tolist() + total_transform[:3,3].flatten().tolist()
        
        rgb_gl, depth_gl = render_machine.render(
            pose_rendered_q[:4], np.array(pose_rendered_q[4:])
        )
        rgb_gl = rgb_gl.astype("uint8")

        depth_gl = (depth_gl * 1000).astype(np.uint16)
        return rgb_gl, depth_gl
   
    def render_perch_poses(self, max_min_dict, required_object, camera_pose, render_dir=None):
        # Renders equidistant poses in 3D discretized space with both color and depth images
        if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
            sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
        import cv2
        if render_dir is None:
            render_dir = self.rendered_root_dir

        render_machine = self.get_renderer(required_object)
        idx = 0
        rendered_dir = os.path.join(render_dir, required_object)
        mkdir_if_missing(rendered_dir)
        rendered_pose_list_out = []
        for x in np.arange(max_min_dict['xmin'], max_min_dict['xmax'], self.search_resolution_translation):
            # for y in np.arange(max_min_dict['ymin'], max_min_dict['ymax'], self.search_resolution_translation):
            y = (max_min_dict['ymin'] + max_min_dict['ymax'])/2
            for theta in np.arange(0, 2 * np.pi, self.search_resolution_yaw):
                # original_point = np.array([x, y, (max_min_dict['zmin']+max_min_dict['zmin'])/2-0.1913/2, 1])
                # original_point = np.array([x, y, (max_min_dict['zmin']+max_min_dict['zmin'])/2, 1])
                original_point = [x, y, max_min_dict['zmin']]

                # subtract half height of object so that base is on the table
                # TODO take from database, right now this is for mustard bottle
                # new_point = np.copy(original_point)
                # new_point[2] += 0.1913
                object_world_transform = np.zeros((4,4))
                object_world_transform[:3,:3] = RT_transform.euler2mat(theta, 0, 0)
                object_world_transform[:,3] = [i*100 for i in original_point] + [1]

                if camera_pose is not None:
                    # Doing in a frame where z is up
                    total_transform = np.matmul(np.linalg.inv(camera_pose), object_world_transform)
                else:
                    # Doing in camera frame
                    total_transform = object_world_transform
                    # Make it far from camera to so we can see everything
                    total_transform[2, 3] = max_min_dict['zmax']*100
                    
                rgb_gl, depth_gl = self.render_pose(
                    required_object, render_machine, 
                    RT_transform.mat2euler(total_transform[:3,:3]), 
                    total_transform[:3,3].flatten().tolist()
                )
                image_file = os.path.join(
                    rendered_dir,
                    "{}-color.png".format(idx),
                )
                depth_file = os.path.join(
                    rendered_dir,
                    "{}-depth.png".format(idx),
                )
                cv2.imwrite(image_file, rgb_gl)
                cv2.imwrite(depth_file, depth_gl)

                rendered_pose_list_out.append(object_world_transform[:,3].tolist() + [0,0,theta])
                idx += 1

        pose_rendered_file = os.path.join(
            rendered_dir,
            "poses.txt",
        )
        np.savetxt(pose_rendered_file, np.around(rendered_pose_list_out, 4))
    
    


    def visualize_perch_output(self, image_data, annotations, max_min_dict, frame='fat_world', 
            use_external_render=0, required_object='004_sugar_box', camera_optical_frame=True,
            use_external_pose_list=0, model_poses_file=None, use_centroid_shifting=0, predicted_mask_path=None
        ):
        from perch import FATPerch
        print("camera instrinsics : {}".format(self.camera_intrinsics))
        print("max_min_ranges : {}".format(max_min_dict))
        
        cam_to_body = self.cam_to_body if camera_optical_frame == False else None
        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        # depth_img_path = color_img_path.replace(os.path.splitext(color_img_path)[1], '.depth.png')
        depth_img_path = self.get_depth_img_path(color_img_path)
        print("depth_img_path : {}".format(depth_img_path))
        # Get camera pose for PERCH and rendering objects if needed
        if frame == 'fat_world':
            camera_pose = get_camera_pose_in_world(annotations[0]['camera_pose'], None, 'rot', cam_to_body=cam_to_body)
            camera_pose[:3, 3] /= 100 
        if frame == 'world':
            camera_pose = get_camera_pose_in_world(annotations[0]['camera_pose'], self.world_to_fat_world, 'rot', cam_to_body=cam_to_body)
            camera_pose[:3, 3] /= 100 
        if frame == 'table':
            _, _, camera_pose = self.get_camera_pose_relative_table(depth_img_path, type='rot', cam_to_body=cam_to_body)
            camera_pose[:3, 3] /= 100 
            
        if frame == 'camera':
            camera_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            if cam_to_body is not None:
                camera_pose = np.matmul(camera_pose, np.linalg.inv(cam_to_body))

        # camera_pose[:3, 3] /= 100 

        print("camera_pose : {}".format(camera_pose))

        # Prepare data to send to PERCH
        input_image_files = {
            'input_color_image' : color_img_path,
            'input_depth_image' : depth_img_path,
        } 
        if predicted_mask_path is not None:
            input_image_files['predicted_mask_image'] = predicted_mask_path

        # Render poses if necessary
        if use_external_render == 1:
            self.render_perch_poses(max_min_dict, required_object, camera_pose)

        if use_external_pose_list == 1:
            models_root = os.path.join(self.model_dir, 'aligned_cm')
        else:
            models_root = os.path.join(self.model_dir, 'models')

        camera_pose = camera_pose.flatten().tolist()
        params = {
            'x_min' : max_min_dict['xmin'],
            'x_max' : max_min_dict['xmax'],
            'y_min' : max_min_dict['ymin'],
            'y_max' : max_min_dict['ymax'],
            # 'x_min' : max_min_dict['xmin'],
            # 'x_max' : max_min_dict['xmax'] + self.search_resolution_translation,
            # 'y_min' : max_min_dict['ymin'],
            # 'y_max' : max_min_dict['ymin'] + 2 * self.search_resolution_translation,
            'required_object' : required_object,
            # 'table_height' :  max_min_dict['zmin'],
            'table_height' :  0.010,
            'use_external_render' : use_external_render, 
            'camera_pose': camera_pose,
            'reference_frame_': frame,
            'search_resolution_translation': self.search_resolution_translation,
            'search_resolution_yaw': self.search_resolution_yaw,
            'image_debug' : 1,
            'use_external_pose_list': use_external_pose_list,
            'depth_factor': self.depth_factor,
            'shift_pose_centroid': use_centroid_shifting,
            'use_icp': 1
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
        fat_perch = FATPerch(
            params=params, 
            input_image_files=input_image_files, 
            camera_params=camera_params,
            object_names=self.category_names,
            output_dir_name=self.get_clean_name(image_data['file_name']),
            models_root=models_root,
            model_params=self.model_params
        )
        perch_annotations = fat_perch.run_perch_node(model_poses_file)
        return perch_annotations
    
    def get_clean_name(self, name):
        return name.replace('.jpg', '').replace('.png', '').replace('/', '_').replace('.', '_')

    def reject_outliers(self, data, m = 2.):
        d = np.abs(data - np.mean(data))
        mdev = np.std(d)
        s = d/mdev if mdev else 0.
        return data[s<m]

    def visualize_model_output(self, image_data, use_thresh=False, use_centroid=True):

        from maskrcnn_benchmark.config import cfg
        from dipy.core.geometry import cart2sphere, sphere2cart
        # plt.figure()
        # img_path = os.path.join(image_directory, image_data['file_name'])
        # image = io.imread(img_path)
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
            topk_rotations=9
        )
        color_img_path = os.path.join(self.coco_image_directory, image_data['file_name'])
        color_img = cv2.imread(color_img_path)
        composite, mask_list, rotation_list, centroids_2d, boxes, overall_binary_mask \
                = coco_demo.run_on_opencv_image(color_img, use_thresh=use_thresh)

        # depth_img_path = color_img_path.replace('.jpg', '.depth.png')
        depth_img_path = self.get_depth_img_path(color_img_path)
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
        predicted_mask_path = os.path.join(os.path.dirname(depth_img_path), os.path.splitext(os.path.basename(color_img_path))[0] + '.predicted_mask.png')
        cv2.imwrite(predicted_mask_path, overall_binary_mask)

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
            # grid_size = top_viewpoint_ids[0,:].shape[0]*top_inplane_rotation_ids[0,:].shape[0]+1
            grid_size = top_inplane_rotation_ids[0,:].shape[0]+1
            grid = ImageGrid(fig, 111,  
                        nrows_ncols=(top_viewpoint_ids.shape[0], grid_size),
                        axes_pad=0.1, 
                        )
        print("Camera matrix : {}".format(self.camera_intrinsic_matrix))
        K_inv = np.linalg.inv(self.camera_intrinsic_matrix)

        print("Predicted top_viewpoint_ids : {}".format(top_viewpoint_ids))
        print("Predicted top_inplane_rotation_ids : {}".format(top_inplane_rotation_ids))
        print("Predicted boxes : {}".format(boxes))
        print("Predicted labels : {}".format(labels))
        print("Predicted maks path : {}".format(predicted_mask_path))
        img_list = []
        annotations = []
        
        depth_range = []
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
                # plt.figure()
                # plt.imshow(mask_list[box_id])
                object_depth_mask = depth_image[mask_list[box_id] > 0]/self.depth_factor
                object_depth_mask = object_depth_mask.flatten()
                object_depth_mask = self.reject_outliers(object_depth_mask)
                object_depth = np.mean(object_depth_mask)
                min_depth = np.min(object_depth_mask)
                max_depth = np.max(object_depth_mask)
                object_rotation_list = []
                # plt.show()
                grid[grid_i].imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
                grid[grid_i].scatter(centroids_2d[box_id][0], centroids_2d[box_id][1], s=1)
                grid[grid_i].axis("off")
                grid_i += 1
                label = labels[box_id]
                render_machine = self.get_renderer(label)
                # rendered_dir = os.path.join(self.rendered_root_dir, label)
                # mkdir_if_missing(rendered_dir)
                # rendered_pose_list_out = []
                # for viewpoint_id in top_viewpoint_ids[box_id, :]:
                #     for inplane_rotation_id in top_inplane_rotation_ids[box_id, :]:
                for viewpoint_id, inplane_rotation_id in zip(top_viewpoint_ids[box_id, :],top_inplane_rotation_ids[box_id, :]):
                        fixed_transform = self.fixed_transforms_dict[label]
                        theta, phi = get_viewpoint_rotations_from_id(self.viewpoints_xyz, viewpoint_id)
                        inplane_rotation_angle = get_inplane_rotation_from_id(self.inplane_rotations, inplane_rotation_id)
                        xyz_rotation_angles = [phi, theta, inplane_rotation_angle]
                        # centroid = np.matmul(K_inv, np.array(centroids_2d[box_id].tolist() + [1]))

                        centroid_world_point = self.get_world_point(np.array(centroids_2d[box_id].tolist() + [object_depth]))
                        print("{}. Recovered rotation, centroid : {}, {}".format(grid_i, xyz_rotation_angles, centroid_world_point))
                        
                        rgb_gl, depth_gl = self.render_pose(
                            label, render_machine, xyz_rotation_angles, (centroid_world_point*100).tolist()
                        )
                        # rotated_centeroid_2d = np.flip(np.mean(np.argwhere(rgb_gl[:,:,0] > 0), axis=0))
                        # shifted_centeroid_2d =  centroids_2d[box_id] - (rotated_centeroid_2d - centroids_2d[box_id])
                        # shifted_centroid = np.matmul(K_inv, np.array(rotated_centeroid_2d.tolist() + [1]))

                        # print("{}. Recovered rotation, shifted centroid : {}, {}".format(grid_i, xyz_rotation_angles, rotated_centeroid_2d))
                        # centroid = shifted_centroid
                        # print("Center after rotation : {}".format())
                        grid[grid_i].imshow(cv2.cvtColor(rgb_gl, cv2.COLOR_BGR2RGB))
                        grid[grid_i].scatter(centroids_2d[box_id][0], centroids_2d[box_id][1], s=1)
                        # grid[grid_i].scatter(shifted_centeroid_2d[0], shifted_centeroid_2d[1], s=1)
                        grid[grid_i].axis("off")
                        grid_i += 1
                        
                        quaternion =  get_xyzw_quaternion(RT_transform.euler2quat(phi, theta, inplane_rotation_angle).tolist())
                        if use_centroid: 
                            # Collect final annotations with centroid
                            annotations.append({
                                'location' : (centroid_world_point*100).tolist(),
                                'quaternion_xyzw' : quaternion,
                                'category_id' : self.category_names.index(label),
                                'id' : grid_i
                            })
                        else :
                            # Collect rotations only for this object
                            object_rotation_list.append(quaternion)

                if use_centroid == False:
                    # Add predicted rotations in depth range
                    for _, depth in enumerate(np.arange(min_depth, max_depth, 0.02)):
                        centre_world_point = self.get_world_point(np.array(centroids_2d[box_id].tolist() + [depth]))
                        for quaternion in object_rotation_list:
                            annotations.append({
                                'location' : (centre_world_point*100).tolist(),
                                'quaternion_xyzw' : quaternion,
                                'category_id' : self.category_names.index(label),
                                'id' : grid_i
                            })
                            
        model_poses_file = 'model_output_{}.png'.format(self.get_clean_name(image_data['file_name']))
        plt.savefig(
            model_poses_file, 
            dpi=1000, bbox_inches = 'tight', pad_inches = 0
        )


        # plt.show()
        return labels, annotations, model_poses_file, predicted_mask_path

    def compare_clouds(self, annotations_1, annotations_2, f):
        from plyfile import PlyData, PlyElement

        for i in range(len(annotations_1)):
            annotation_1 = annotations_1[i]
            annotation_2 = annotations_2[i]

            object_name = self.category_names[annotation_1['category_id']]

            file_path = os.path.join(self.model_dir, "models", object_name, "textured.ply")
            # cloud = pywavefront.Wavefront(file_path)
            cloud = PlyData.read(file_path).elements[0].data
            cloud = np.transpose(np.vstack((cloud['x'], cloud['y'], cloud['z'])))
            # print(cloud)
            # cloud = pcl.load_XYZI(file_path)
            # cloud = np.asarray(cloud)
            cloud = np.hstack((cloud, np.ones((cloud.shape[0], 1))))
            # print(cloud)
            print("Locations : {}, {}".format(annotation_1['location'], annotation_2['location']))
            print("Quaternions : {}, {}".format(annotation_1['quaternion_xyzw'], annotation_2['quaternion_xyzw']))

            total_transform_1 = self.get_object_pose_with_fixed_transform(
                object_name, annotation_1['location'], RT_transform.quat2euler(get_wxyz_quaternion(annotation_1['quaternion_xyzw'])), 'rot',
                use_fixed_transform=False
            )
            # scaling_transform = np.zeros((4,4))
            # scaling_transform[3,3] = 1
            # scaling_transform[0,0] = 0.0275
            # scaling_transform[1,1] = 0.0275
            # scaling_transform[2,2] = 0.0275
            # total_transform_1 =  scaling_transform * total_transform_1
            # print(total_transform_1)
            # transformed_cloud_1 = np.matmul(total_transform_1, np.transpose(cloud))
            transformed_cloud_1 = np.matmul(cloud, total_transform_1)
            # print(transformed_cloud_1)

            l = transformed_cloud_1[:,3]
            # transformed_cloud_1 = np.divide(transformed_cloud_1[:,:3], transformed_cloud_1[:,3])
            transformed_cloud_1 = transformed_cloud_1[:,:3]/l[:, np.newaxis]

            total_transform_2 = self.get_object_pose_with_fixed_transform(
                object_name, annotation_2['location'], RT_transform.quat2euler(get_wxyz_quaternion(annotation_2['quaternion_xyzw'])), 'rot',
                use_fixed_transform=False
            )
            # transformed_cloud_2 = np.matmul(total_transform_2, np.transpose(cloud))
            transformed_cloud_2 = np.matmul(cloud, total_transform_2)
            l = transformed_cloud_2[:,3]
            transformed_cloud_2 = transformed_cloud_2[:,:3]/l[:, np.newaxis]
            # print(transformed_cloud_2)

            mean_dist = np.linalg.norm(transformed_cloud_1-transformed_cloud_2, axis=1)
            mean_dist = np.sum(mean_dist)/cloud.shape[0]
            # print(mean_dist)
            #/cloud.shape[0]
            print("Average pose distance (in m) : {}".format(mean_dist))
            f.write("{} {}\n".format(object_name, mean_dist))



def run_multiple():
    
    image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_2018.json'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'

    fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory, depth_factor=10000)

    # Running on model and PERCH
    # image_data, annotations = fat_image.get_random_image(name='kitchen_0/000002.left.jpg')
    f = open('accuracy.txt', "w")

    # for category_name in fat_image.category_names:
    # category_name = "035_power_drill"
    # category_name = "019_pitcher_base"
    # category_name = "024_bowl"
    
    # Get Image
    # image_data, annotations = fat_image.get_random_image(name='{}_16k/kitchen_4/000005.left.jpg'.format(category_name))
    image_data, annotations = fat_image.get_random_image(name='kitchen_4/000015.left.jpg')

    # Look at for labels 0 and 1
    # can be improved by further restricted segmentation mask
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000040.left.jpg')
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000060.left.jpg')
    # wrong localization for soup can
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000070.left.jpg')
    # can try with all possible combinations
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000100.left.jpg')
    # Box doesnt look right in this
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000005.left.jpg')
    # 

    # TODO
    # restrict segmentation
    # move in x,y in hypothesis
    # try all possible combinations of rotation and viewpoint, increase topk number

    # fat_image.compare_clouds(annotations, annotations, f)

    # Visualize ground truth in ros
    # yaw_only_objects, max_min_dict_gt, transformed_annotations = fat_image.visualize_pose_ros(
    #     image_data, annotations, frame='camera', camera_optical_frame=False, num_publish=1, write_poses=False, ros_publish=False
    # )
    
    # Run model to get multiple poses for each object
    labels, model_annotations, model_poses_file, predicted_mask_path = fat_image.visualize_model_output(image_data, use_thresh=True, use_centroid=False)

    # Convert model output poses to table frame and save them to file so that they can be read by perch
    _, max_min_dict, _ = fat_image.visualize_pose_ros(
        # image_data, model_annotations, frame='table', camera_optical_frame=False, num_publish=1, write_poses=True, ros_publish=False
        image_data, model_annotations, frame='camera', camera_optical_frame=False, num_publish=1, write_poses=True, ros_publish=False
    )

    # Run perch on written poses
    perch_annotations = fat_image.visualize_perch_output(
        image_data, model_annotations, max_min_dict, frame='camera', 
        use_external_render=0, required_object=[labels[0], labels[1]],
        camera_optical_frame=False, use_external_pose_list=1,
        model_poses_file=model_poses_file, use_centroid_shifting=1,
        predicted_mask_path=predicted_mask_path
    )

    # # # Compare Poses by applying to model and computing distance
    f.write("{} ".format(image_data['file_name']))
    fat_image.compare_clouds(transformed_annotations, perch_annotations, f)


    f.close()

   

if __name__ == '__main__':
    
    # coco_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/coco_results.pth')
    # all_predictions = torch.load('/media/aditya/A69AFABA9AFA85D9/Cruzr/code/fb_mask_rcnn/maskrcnn-benchmark/inference/fat_pose_2018_val_cocostyle/predictions.pth')

    # image_directory = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_train_pose_2018.json'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Datasets/fat/mixed/extra/instances_fat_val_pose_2018.json'

    # fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory)
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000022.left.jpg')

    ## Running with model only
    # below is error case of table pose
    # image_data, annotations = fat_image.get_random_image(name='kitchen_0/001210.left.jpg')
    # fat_image.visualize_image_annotations(image_data, annotations)
    # fat_image.visualize_model_output(image_data, use_thresh=True, write_poses=True)


    ## Using PERCH only with dataset and find yaw only objects
    # yaw_only_objects, max_min_dict = fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)
    # fat_image.visualize_perch_output(
    #     image_data, annotations, max_min_dict, frame='table', 
    #     use_external_render=0, required_object=['007_tuna_fish_can', '006_mustard_bottle'],
    #     # use_external_render=0, required_object=['007_tuna_fish_can'],
    #     camera_optical_frame=False
    # )
    # fat_image.save_yaw_only_dataset(scene="kitchen_0")


    ## Analyze object rotations about various axis
    # image_data, annotations = fat_image.get_random_image(name='kitchen_4/000000.left.jpg')
    # yaw_only_objects, max_min_dict, _ = fat_image.visualize_pose_ros(image_data, annotations, frame='camera', camera_optical_frame=True)
    # print("max_min_ranges : {}".format(max_min_dict))
    # rendered_root_dir = os.path.join(fat_image.model_dir, "rendered_1")
    # for required_object in fat_image.category_names:
    #     fat_image.render_perch_poses(max_min_dict, required_object, None, render_dir=rendered_root_dir)

    ## Running on PERCH only with synthetic color dataset - YCB
    # Use normalize cost to get best results
    # image_directory = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed'
    # annotation_file = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed/instances_fat_val_pose_2018.json'
    # fat_image = FATImage(coco_annotation_file=annotation_file, coco_image_directory=image_directory, depth_factor=100)
    # image_data, annotations = fat_image.get_random_image(name='NewMap1_reduced_2/000000.left.png')
    # yaw_only_objects, max_min_dict, _ = fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)
    # fat_image.visualize_perch_output(
    #     image_data, annotations, max_min_dict, frame='table', 
    #     use_external_render=0, required_object=['008_pudding_box', '010_potted_meat_can', '009_gelatin_box'],
    #     # use_external_render=0, required_object=['008_pudding_box', '010_potted_meat_can', '009_gelatin_box'],
    #     # use_external_render=0, required_object=['009_gelatin_box', '008_pudding_box', '010_potted_meat_can', '004_sugar_box'],
    #     # use_external_render=0, required_object=['004_sugar_box', '036_wood_block', '009_gelatin_box', '008_pudding_box', '010_potted_meat_can'],
    #     # use_external_render=0, required_object=['009_gelatin_box', '008_pudding_box', '010_potted_meat_can'],
    #     camera_optical_frame=False, use_external_pose_list=0
    # )

    ## Running on PERCH only with synthetic color dataset - shape
    # Use normalize cost to get best results
    image_directory = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed'
    annotation_file = '/media/aditya/A69AFABA9AFA85D9/Cruzr/code/Dataset_Synthesizer/Test/Zed/instances_soda_val_pose_2018.json'
    fat_image = FATImage(
        coco_annotation_file=annotation_file, 
        coco_image_directory=image_directory, 
        depth_factor=100, 
        model_dir='/media/aditya/A69AFABA9AFA85D9/Datasets/SameShape/',
        model_mesh_in_mm=True,
        model_mesh_scaling_factor=0.0275,
        models_flipped=True
    )
    # below doesnt work for any method
    # image_data, annotations = fat_image.get_random_image(name='NewMap1_soda_cans/000014.left.png')

    # image_data, annotations = fat_image.get_random_image(name='NewMap1_soda_cans/000000.left.png')
    # image_data, annotations = fat_image.get_random_image(name='NewMap1_soda_cans/000001.left.png')
    # image_data, annotations = fat_image.get_random_image(name='NewMap1_soda_cans/000033.left.png')
    # image_data, annotations = fat_image.get_random_image(name='NewMap1_soda_cans/000038.left.png')
    # image_data, annotations = fat_image.get_random_image(name='NewMap1_soda_cans/000048.left.png')
    f_runtime = open('runtime.txt', "w")
    f_runtime.write("{} {} {}\n".format('name', 'expands', 'runtime'))
    for img_i in ['14', '20', '25', '33', '38', '48']:
    # for img_i in ['25']:
        required_objects = ['sprite', 'coke', 'pepsi']
        image_name = 'NewMap1_soda_cans/0000{}.left.png'.format(img_i)
        image_data, annotations = fat_image.get_random_image(name=image_name, required_objects=required_objects)
        yaw_only_objects, max_min_dict, transformed_annotations = \
            fat_image.visualize_pose_ros(image_data, annotations, frame='table', camera_optical_frame=False)
        max_min_dict['ymax'] = 1.5
        max_min_dict['ymin'] = -1.5
        max_min_dict['xmax'] = 0.5
        max_min_dict['xmin'] = -0.5
        fat_image.search_resolution_translation = 0.05
        perch_annotations, stats = fat_image.visualize_perch_output(
            image_data, annotations, max_min_dict, frame='table', 
            use_external_render=0, required_object=required_objects,
            # use_external_render=0, required_object=['coke', 'sprite', 'pepsi'],
            # use_external_render=0, required_object=['sprite', 'coke', 'pepsi'],
            camera_optical_frame=False, use_external_pose_list=0
        )
        # f = open('accuracy.txt', "w")
        # f.write("{} ".format(image_data['file_name']))
        # print(transformed_annotations)
        # fat_image.compare_clouds(transformed_annotations, perch_annotations, f)
        # f.close()
        f_runtime.write("{} {} {}\n".format(image_name, stats['expands'], stats['runtime']))
    f_runtime.close()

    ## Run Perch with Model
    # Dont use normalize cost and run with shifting centroid
    # run_multiple()