from setuptools import setup
import os
from glob import glob

package_name = 'hp60c_basic'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nl',
    description='Minimal HP60C depth camera package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = hp60c_basic.camera_node:main',
            'fingertip_3d = hp60c_basic.fingertip_3d:main',
            'fingertip_pose = hp60c_basic.fingertip_pose:main',
            'fingertip_yolo = hp60c_basic.fingertip_yolo:main',
        ],
    },
)
