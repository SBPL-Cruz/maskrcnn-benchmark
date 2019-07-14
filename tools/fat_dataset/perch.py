import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
import rospkg
import rosparam
import subprocess
import roslaunch
import os
import numpy as np
from shutil import copy

class FATPerch():
    # Runs perch on single image
    MPI_BIN_ROOT = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install/bin"
    # OUTPUT_POSES_FILE = 'output_perch_poses.txt'
    # OUTPUT_STATS_FILE = 'output_perch_stats.txt'

    # Symmetry info for Yaw axis
    # 1 means semi-symmetry and 2 means full symmetry
    SYMMETRY_INFO = {
        "004_sugar_box" : 1,
        "008_pudding_box" : 1,
        "009_gelatin_box" : 1,
        "010_potted_meat_can" : 1,
    }

    def __init__(
            self, params=None, input_image_files=None, camera_params=None, object_names=None, output_dir_name=None,
            models_root=None
        ):
        self.PERCH_EXEC = subprocess.check_output("catkin_find sbpl_perception perch_fat".split(" ")).decode("utf-8").rstrip().lstrip()
        rospack = rospkg.RosPack()
        self.PERCH_ROOT = rospack.get_path('sbpl_perception')
        PERCH_ENV_CONFIG = "{}/config/pr2_env_config.yaml".format(self.PERCH_ROOT)
        PERCH_PLANNER_CONFIG = "{}/config/pr2_planner_config.yaml".format(self.PERCH_ROOT)
        # PERCH_YCB_OBJECTS = "{}/config/roman_objects.xml".format(self.PERCH_ROOT)

        # aligned_cm textured.ply models are color and have axis according to FAT dataset - use when running perch with network
        # MODELS_ROOT = "{}/data/YCB_Video_Dataset/aligned_cm".format(self.PERCH_ROOT)

        # modes textured.ply models are color models with original YCB axis - use when using perch without network
        self.object_names = object_names

        self.load_ros_param_from_file(PERCH_ENV_CONFIG)
        self.load_ros_param_from_file(PERCH_PLANNER_CONFIG)
        self.use_external_pose_list = params['use_external_pose_list']

        self.set_ros_param_from_dict(params)
        
        self.set_ros_param_from_dict(input_image_files)
        self.set_ros_param_from_dict(camera_params)
        self.set_object_model_params(object_names, models_root)
        self.output_dir_name = output_dir_name
        # self.launch_ros_node(PERCH_YCB_OBJECTS)
        # self.run_perch_node(PERCH_EXEC)

    def load_ros_param_from_file(self, param_file_path):
        command = "rosparam load {}".format(param_file_path)
        print(command)
        subprocess.call(command, shell=True)
    
    def set_ros_param(self, param, value):
        command = 'rosparam set {} "{}"'.format(param, value)
        print(command)
        subprocess.call(command, shell=True)

    def set_ros_param_from_dict(self, params):
        for key, value in params.items():
            self.set_ros_param(key, value)

    def launch_ros_node(self, launch_file):
        command = 'roslaunch {}'.format(launch_file)
        print(command)
        subprocess.call(command, shell=True)

    def set_object_model_params(self, object_names, models_root):
        self.set_ros_param('mesh_in_mm', False)

        params = []
        for object_name in object_names:
            params.append([
                object_name,
                os.path.join(models_root, object_name, 'textured.ply'),
                False,
                False,
                0 if self.use_external_pose_list == 1 else self.SYMMETRY_INFO[object_name],
                0.06,
                1
            ])
        self.set_ros_param('model_bank', params)
        
    def run_perch_node(self, model_poses_file):
        command = "{}/mpirun -n 4 {} {}".format(self.MPI_BIN_ROOT, self.PERCH_EXEC, self.output_dir_name)
        print("Running command : {}".format(command))
        # print(subprocess.check_output(command.split(" ")))
        # output = subprocess.check_output(command, shell=True).decode("utf-8")
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        out, _ = p.communicate()
        out = out.decode("utf-8")
        print(out)
        f = open(os.path.join(self.PERCH_ROOT, 'visualization', self.output_dir_name, 'log.txt'), "w")
        f.write(out)
        f.close()

        # Get annotations from output of PERCH to get accuracy
        annotations = []
        f = open(os.path.join(self.PERCH_ROOT, 'visualization', self.output_dir_name, 'output_poses.txt'), "r")
        lines = f.readlines()
        for i in np.arange(0, len(lines), 3):
            location = list(map(float, lines[i+1].rstrip().split()))
            annotations.append({
                            'location' : [location[0] * 100, location[1] * 100, location[2] * 100],
                            'quaternion_xyzw' : list(map(float, lines[i+2].rstrip().split())),
                            'category_id' : self.object_names.index(lines[i].rstrip()),
                            'id' : i
                        })
        f.close()
        if model_poses_file is not None:
            copy(model_poses_file, os.path.join(self.PERCH_ROOT, 'visualization', self.output_dir_name))
        return annotations

