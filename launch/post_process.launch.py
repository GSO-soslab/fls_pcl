import os
import yaml
import pathlib
from launch import LaunchDescription
import launch.actions
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    ld = LaunchDescription()

    param_config = os.path.join(
        get_package_share_directory('fls_pcl'),
        'config',
        'fls_params.yaml'
    )
    
    node = Node(
        package='fls_pcl',
        executable='fls_pcl.py',
        name='fls_pcl_node',
        namespace="alpha_rise",
        output='screen',
        parameters=[param_config]
    )

    path = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('alpha_rise_bringup'), 'launch','bringup_path.launch.py')]),
        launch_arguments = {'arg_robot_name': 'alpha_rise'}.items()  
    )

    # Vehicle description
    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('alpha_rise_bringup'), 
            'launch/include/description.launch.py')]),
        launch_arguments={
            'robot_name': 'alpha_rise',
            'description_delay': '0.0'
        }.items()  
    )

    rviz_config_dir = os.path.join( get_package_share_directory('alpha_rise_description'), 'rviz', 'config_post.rviz' )

    rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', [rviz_config_dir]],
        )
    
        # Bag file path (change this to the full path or make it configurable)
    bag_file_path = '/home/tony/bags/whale_rock_10_10/rosbag2_2025_10_10-18_09_27/rosbag2_2025_10_10-18_09_28'

    # ROS2 bag play command
    bag_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag_file_path,
            '--rate', '10.0',
            '--clock'
        ],
        output='screen'
    )

    ld.add_action(node)
    ld.add_action(path)
    ld.add_action(description)
    ld.add_action(rviz)
    ld.add_action(bag_play)



    return ld