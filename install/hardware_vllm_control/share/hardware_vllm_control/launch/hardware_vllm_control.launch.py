from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hardware_vllm_control',
            executable='image_saver',
            name='image_saver',
            output='screen',
        ),

        Node(
            package='hardware_vllm_control',
            executable='thrust_control',
            name='thrust_control',
            output='screen',
        ),

        Node(
            package='hardware_vllm_control',
            executable='gpt_command',
            name='gpt_command',
            output='screen',
        ),
    ])
