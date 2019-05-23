import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' not in sys.path:
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
import rospkg
import rosparam
import subprocess
import roslaunch

class FATPerch():
    MPI_BIN_ROOT = "/media/aditya/A69AFABA9AFA85D9/Cruzr/code/openmpi-4.0.0/install/bin"
    OUTPUT_POSES_FILE = 'output_perch_poses.txt'
    OUTPUT_STATS_FILE = 'output_perch_stats.txt'

    def __init__(self, params=None, input_image_files=None, camera_params=None):
        PERCH_EXEC = subprocess.check_output("catkin_find sbpl_perception perch_fat".split(" ")).decode("utf-8").rstrip().lstrip()
        rospack = rospkg.RosPack()
        PERCH_ROOT = rospack.get_path('sbpl_perception')
        PERCH_ENV_CONFIG = "{}/config/pr2_env_config.yaml".format(PERCH_ROOT)
        PERCH_PLANNER_CONFIG = "{}/config/pr2_planner_config.yaml".format(PERCH_ROOT)
        PERCH_YCB_OBJECTS = "{}/config/roman_objects.xml".format(PERCH_ROOT)

        self.load_ros_param_from_file(PERCH_ENV_CONFIG)
        self.load_ros_param_from_file(PERCH_PLANNER_CONFIG)

        self.set_ros_param_from_dict(params)
        self.set_ros_param_from_dict(input_image_files)
        self.set_ros_param_from_dict(camera_params)

        self.launch_ros_node(PERCH_YCB_OBJECTS)
        self.run_perch_node(PERCH_EXEC)

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

    def run_perch_node(self, PERCH_EXEC):
        command = "{}/mpirun -n 1 {} {} {}".format(self.MPI_BIN_ROOT, PERCH_EXEC, self.OUTPUT_POSES_FILE, self.OUTPUT_STATS_FILE)
        print("Running command : {}".format(command))
        # print(subprocess.check_output(command.split(" ")))
        # output = subprocess.check_output(command, shell=True).decode("utf-8")
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        out, err = p.communicate()
        out = out.decode("utf-8")
        print(out)
