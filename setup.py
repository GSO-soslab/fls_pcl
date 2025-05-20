from setuptools import setup
import os
from glob import glob

package_name = 'fls_pcl'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Ensure the 'launch' files are correctly handled
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*'))),
        # Ensure the 'config' YAML files are correctly handled
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tony Jacob',
    maintainer_email='tony.jacob@uri.edu ',
    description='Converts FLS image to pointclouds',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fls_pcl_node = fls_pcl.fls_pcl:main',
        ],
    },
)
