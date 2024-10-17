
import sys
sys.path.append('/home/ge_orin/catkin_ws/src/tb3_control/src')
import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from nav_msgs.msg import Odometry
import numpy as np
from tb3_control import dp_ga_function
import updation

import argparse
from utils.config import *
from agents import *
from policy.env import pos


def get_config():
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        '--config',
        metavar='config_json_file',
        default='/home/ge_orin/catkin_ws/src/tb3_control/src/configs/Configs.json')
    arg_parser.add_argument('--mode', type=str, default='test')
    arg_parser.add_argument('--log_time_trained', type=str, default='0')
    arg_parser.add_argument('--num_agents', type=int, default=10)
    arg_parser.add_argument('--map_w', type=int, default=20)
    arg_parser.add_argument('--map_density', type=int, default=1)
    arg_parser.add_argument('--map_type', type=str, default='map')
    arg_parser.add_argument('--trained_num_agents', type=int, default=10)
    arg_parser.add_argument('--trained_map_w', type=int, default=20)
    arg_parser.add_argument('--trained_map_density', type=int, default=1)
    arg_parser.add_argument('--trained_map_type', type=str, default='map')
    arg_parser.add_argument('--nGraphFilterTaps', type=int, default=3)
    arg_parser.add_argument('--numInputFeatures', type=int, default=128)
    arg_parser.add_argument('--numInputFeaturesAP', type=int, default=32)
    arg_parser.add_argument('--num_testset', type=int, default=4500)
    arg_parser.add_argument('--load_num_validset', type=int, default=200)
    arg_parser.add_argument('--test_epoch', type=int, default=0)
    arg_parser.add_argument('--lastest_epoch', action='store_true', default=False)
    arg_parser.add_argument('--best_epoch', action='store_true', default=False)
    arg_parser.add_argument('--con_train', action='store_true', default=False)
    arg_parser.add_argument('--test_general', action='store_true', default=False)
    arg_parser.add_argument('--log_anime', action='store_true', default=True)
    arg_parser.add_argument('--rate_maxstep', type=int, default=2)
    arg_parser.add_argument('--vary_ComR_FOV', action='store_true', default=False)
    arg_parser.add_argument('--commR', type=int, default=7)
    arg_parser.add_argument('--dynamic_commR', action='store_true', default=False)
    arg_parser.add_argument('--symmetric_norm', action='store_true', default=False)
    arg_parser.add_argument('--id_env', type=int, default=None)
    arg_parser.add_argument('--guidance', type=str, default='Project_G')
    arg_parser.add_argument('--data_set', type=str, default='')
    arg_parser.add_argument('--action_select', type=str, default='soft_max')
    arg_parser.add_argument('--list_agents', nargs='+', type=int)
    arg_parser.add_argument('--list_map_w', nargs='+', type=int)
    arg_parser.add_argument('--list_num_testset', nargs='+', type=int)
    arg_parser.add_argument('--nAttentionHeads', type=int, default=0)
    arg_parser.add_argument('--AttentionConcat', action='store_true', default=False)
    arg_parser.add_argument('--batch_numAgent', action='store_true', default=False)
    arg_parser.add_argument('--no_ReLU', action='store_true', default=False)
    arg_parser.add_argument('--tb_ExpName', type=str, default='GNN_Resnet_3Block_distGSO_baseline_128')
    arg_parser.add_argument('--attentionMode', type=str, default='GAT_modified')
    arg_parser.add_argument('--use_dropout', action='store_true', default=False)
    arg_parser.add_argument('--GSO_mode', type=str, default='dist_GSO')
    arg_parser.add_argument('--GNNGAT', action='store_true', default=False)
    arg_parser.add_argument('--CNN_mode', type=str, default="Default")
    arg_parser.add_argument('--shuffle_testSet', action='store_true', default=False)
    arg_parser.add_argument('--test_on_ValidSet', action='store_true', default=False)
    arg_parser.add_argument('--default_actionSelect', action='store_true', default=False)
    arg_parser.add_argument('--extra_policy', type=str, default='one')

    args, unknown = arg_parser.parse_known_args()

    config = process_config(args)
    return config
x_1 = 4.0
y_1 = 0.0

x_2 = 4.0
y_2 = 0.0
def odomcallback_1(odom_1):
    global x_1, y_1, init_di_1
    x_1 = odom_1.pose.pose.position.x
    y_1 = odom_1.pose.pose.position.y
    init_di_1 = odom_1.pose.pose.position.z
    
def odomcallback_2(odom_2):
    global x_2, y_2, init_di_2
    x_2 = odom_2.pose.pose.position.x
    y_2 = odom_2.pose.pose.position.y
    init_di_2 = odom_2.pose.pose.position.z

def planner():
    config = get_config()
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    
    obstacle_y = np.array([9,10,11,10,11,12,13,14,14,14,14])
    obstacle_x = np.array([7,7,7,11,11,11,11,11,10,9,8]) 
    target_x_1 = 4
    target_y_1 = 4
    target_x_2 = 10
    target_y_2 = 10
    agent.update_obs(obstacle_y, obstacle_x)
    agent.update_target(target_x_1, target_y_1, target_x_2, target_y_2)

    global Positon
    pub = rospy.Publisher('waypoints', Float32MultiArray, queue_size=10)
    rospy.init_node('planner', anonymous=True)
    rospy.Subscriber('odometry_1', Odometry, odomcallback_1)
    rospy.Subscriber('odometry_2', Odometry, odomcallback_2)
    rate = rospy.Rate(8) # 10hz
    path_p = []
    length_o = 0
    count = 0
    while not rospy.is_shutdown():   
        rospy.Subscriber('odometry_2', Odometry, odomcallback_2)    
        rospy.Subscriber('odometry_1', Odometry, odomcallback_1) 
        waypoints_sub = agent.run(x_2, y_2, x_1, y_1)

        if waypoints_sub.shape[0] >= 8:
                waypoints_subb=waypoints_sub.copy()
                count+=1
        #print(waypoints_sub)
        waypoints = Float32MultiArray(data=waypoints_subb)
        pub.publish(waypoints)
        print('waypoints from planner', waypoints)
        rate.sleep()
        #nowx, nowy = pos(x_1, y_1, max_len=4.4, numcell=20)
        #if nowx == target_x and nowy == target_y:
        #    break

if __name__ == '__main__':
    try:
        planner()
    except rospy.ROSInterruptException:
        pass
