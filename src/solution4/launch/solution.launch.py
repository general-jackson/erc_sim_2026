from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    shelf_column_arg = DeclareLaunchArgument(
        'shelf_column_number',
        default_value='1',
        description='Target shelf column number'
    )
    
    book_colour_arg = DeclareLaunchArgument(
        'book_colour',
        default_value='red',
        description='Target book color'
    )

    node1 = Node(
        package='solution4',
        executable='book_target_node',
        name='node1_search',
        output='screen',
        parameters=[{
            'shelf_column_number': LaunchConfiguration('shelf_column_number'),
            'book_colour': LaunchConfiguration('book_colour'),
        }]
    )

    node2 = Node(
        package='solution4',
        executable='nav_node',
        name='node2_navigation',
        output='screen'
    )

    # Grasps the book once node 2 reports REACHED_SHELF, then delivers it to
    # the collection bin.
    node3 = Node(
        package='solution4',
        executable='manipulation_node',
        name='node3_manipulation',
        output='screen',
        parameters=[{
            'book_colour': LaunchConfiguration('book_colour'),
        }]
    )

    return LaunchDescription([
        shelf_column_arg,
        book_colour_arg,
        node1,
        node2,
        node3
    ])