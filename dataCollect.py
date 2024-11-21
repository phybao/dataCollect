#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import numpy as np
import pandas as pd

class dataCollect(Node):
    def __init__(self):
        super().__init__('datacollect_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10)
        self.get_logger().info('Data Collecting Node Started')

    def lidar_callback(self, msg):
        # Convert lidar data to Cartesian coordinates and store valid points
        lidarXY = []
        for i, range_value in enumerate(msg.ranges):
            if msg.range_min <= range_value <= msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment
                x = range_value * math.cos(angle)
                y = range_value * math.sin(angle)
                lidarXY.append([x, y])

        # Convert to numpy array and remove NaN or inf values
        lidarXY = np.array(lidarXY)
        lidarXY = lidarXY[~np.isnan(lidarXY).any(axis=1)]

        # Check if there's valid data
        if lidarXY.shape[0] > 0:
            # Save valid scan data to CSV and unsubscribe
            df = pd.DataFrame(lidarXY, columns=['x', 'y'])
            df.to_csv('lidar.csv', index=False)
            self.get_logger().info('Valid scan data saved to lidar.csv')
            
            # Unsubscribe to stop receiving further data
            self.subscription = None
            self.destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = dataCollect()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
