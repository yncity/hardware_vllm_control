from setuptools import find_packages, setup

package_name = 'hardware_vllm_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ioes',
    maintainer_email='ioes@todo.todo',
    description='Hardware-side ROS2 nodes for image capture, GPT navigation, and motor thrust control.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # image capture
            'image_saver = hardware_vllm_control.image_saver:main',

            # motor thrust control (pigpio)
            'thrust_control = hardware_vllm_control.thrust_control:main',

            # GPT-based navigation command
            'gpt_command = hardware_vllm_control.gpt_command:main',
            
            # manual keyboard control
            'manual_control = hardware_vllm_control.manual_control:main',
        ],
    },
)
