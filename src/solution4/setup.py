from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'solution4'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.onnx')),
     ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mffin53',
    maintainer_email='mffin53@todo.todo',
    description=(
        'Emirates Robotics Competition 2026 Phase 1 entry: a Library Assistant '
        'Robot for TIAGo Pro that identifies the target shelf column from its '
        'overhead digit marker, locates the requested book by colour, and '
        'navigates between the shelf and the collection bin.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'book_target_node = solution4.book_target_node:main',
            'nav_node = solution4.nav_node:main',
            'manipulation_node = solution4.manipulation_node:main',
            'collect_digits = solution4.collect_digits:main',
        ],
    },
)
