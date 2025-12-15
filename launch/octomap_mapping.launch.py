from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    param_config = os.path.join(
        get_package_share_directory('fls_pcl'),
        'config',
        'octomap_mapping.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            namespace="alpha_rise",
            output='screen',
            parameters=[
                param_config
            ],
            remappings=[
                ('cloud_in', '/alpha_rise/fls/pointcloud/post')
            ]
        )
    ])
