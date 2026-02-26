import os
import yaml
import pathlib
from launch import LaunchDescription
import launch.actions
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable
from launch.actions import DeclareLaunchArgument

def generate_launch_description():

    ld = LaunchDescription()

    param_config = os.path.join(
        get_package_share_directory('fls_pcl'),
        'config',
        'fls_params.yaml'
    )

    fls_pcl_node = Node(
        package='fls_pcl',
        executable='fls_pcl.py',
        name='fls_pcl_node',
        namespace="alpha_rise",
        output='screen',
        parameters=[param_config]
    )

    fls_voxel_node = Node(
        package='fls_pcl',
        executable='fls_voxels.py',
        name='fls_voxel_node',
        namespace="alpha_rise",
        output='screen',
        parameters=[param_config]
    )

    ld.add_action(fls_pcl_node)
    ld.add_action(fls_voxel_node)

    return ld
