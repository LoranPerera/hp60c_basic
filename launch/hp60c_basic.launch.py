from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    # Launch the camera driver
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                os.path.expanduser('~'),
                '/home/ros2/cam_ws/src/ascamera/launch/hp60c.launch.py'
            )
        )
    )

    # Launch our basic node
    basic_node = Node(
        package='hp60c_basic',
        executable='camera_node',
        name='hp60c_basic',
        output='screen'
    )

    return LaunchDescription([
        camera_launch,
        basic_node,
    ])
