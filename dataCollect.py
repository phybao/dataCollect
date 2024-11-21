#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math
from sklearn.cluster import DBSCAN
import csv

def trilaterate(cx1, cy1, d1, cx2, cy2, d2):
    """
    Trilateration calculation to find the position of the LIDAR sensor
    given two circle centers (cx1, cy1) and (cx2, cy2) and their respective
    radii (distances) d1 and d2.
    Returns the coordinates (x, y) of the LIDAR position.
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    distance = math.sqrt(dx**2 + dy**2)

    if distance > d1 + d2 or distance < abs(d1 - d2) or distance == 0:
        return None, None

    x = (d1**2 - d2**2 + distance**2) / (2 * distance)
    y = math.sqrt(max(0, d1**2 - x**2))  # Ensure no negative values inside sqrt

    lidar_x = cx1 + x * dx / distance - y * dy / distance
    lidar_y = cy1 + x * dy / distance + y * dx / distance

    return lidar_x, lidar_y

def fit_circle(x, y):
    """
    Fit a circle to given x and y points using the Kasa method.
    Returns the center (cx, cy) and radius r of the circle.
    """
    A = np.c_[x, y, np.ones(len(x))]
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2, c[1] / 2
    radius = np.sqrt(c[2] + cx**2 + cy**2)
    return cx, cy, radius

class LocalizationNode(Node):
    def __init__(self):
        super().__init__('Localization_node')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)
        self.csv_file = 'lidar_positions.csv'
        self.get_logger().info('Localization Node Started')

        # Prepare CSV file
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['LIDAR_X', 'LIDAR_Y'])

    def lidar_callback(self, msg):
        ranges = msg.ranges
        lidarXY = []

        for i, range in enumerate(ranges):
            if msg.range_min <= range <= msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment
                x = range * math.cos(angle)
                y = range * math.sin(angle)
                lidarXY.append([x, y])

        lidarXY = np.array(lidarXY)
        lidarXY = lidarXY[~np.isnan(lidarXY).any(axis=1)]
        lidarXY = lidarXY[~np.isinf(lidarXY).any(axis=1)]

        if lidarXY.shape[0] == 0:
            self.get_logger().info("No valid lidar data.")
            return

        model = DBSCAN(eps=0.1, min_samples=5)
        labels = model.fit_predict(lidarXY)

        unique_labels = np.unique(labels)
        centers = []
        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = lidarXY[labels == label]
            if 10 <= len(cluster_points) <= 50:
                cx, cy, radius = fit_circle(cluster_points[:, 0], cluster_points[:, 1])
                if 0.02 <= radius <= 0.3:
                    centers.append((cx, cy, radius))

        if len(centers) >= 2:
            centers.sort(key=lambda c: c[2], reverse=True)
            (cx1, cy1, r1), (cx2, cy2, r2) = centers[:2]
            lidar_x, lidar_y = trilaterate(cx1, cy1, r1, cx2, cy2, r2)

            if lidar_x is not None and lidar_y is not None:
                self.get_logger().info(f"LIDAR Position: ({lidar_x:.2f}, {lidar_y:.2f})")

                # Save to CSV
                with open(self.csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([lidar_x, lidar_y])

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
