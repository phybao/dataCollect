#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import math
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import numpy.linalg as la
from matplotlib.patches import Circle

# TO DO: WRITE YOUR CODE HERE
def trilaterate(cx1, cy1, d1, cx2, cy2, d2):
    """
    Trilateration calculation to find the position of the LIDAR sensor
    given two circle centers (cx1, cy1) and (cx2, cy2) and their respective
    radii (distances) d1 and d2.
    Returns the coordinates (x, y) of the LIDAR position.
    """
    # Step 1: Calculate the distance between the two centers
    dx = cx2 - cx1
    dy = cy2 - cy1
    distance = math.sqrt(dx**2 + dy**2)

    # Step 2: Check if a solution is possible
    if distance > d1 + d2 or distance < abs(d1 - d2) or distance == 0:
        # No intersection
        return None, None


    #set global frame at the c1 center
    # Step 3: Calculate the point where the line through the circle intersections crosses the line between the circle centers
    x = (d1**2 - d2**2 + distance**2) / (2 * distance)
    y = - math.sqrt(d1**2 - x**2)

    intersection_x = x
    intersection_y = y

    # For simplicity, return one of the intersection points (or both if needed)
    lidar_x = intersection_x
    lidar_y = intersection_y

    return lidar_x, lidar_y

def fit_circle(x, y):
    """
    Fit a circle to given x and y points using the Kasa method.
    Returns the center (cx, cy) and radius r of the circle.
    """
    A = np.c_[x, y, np.ones(len(x))]
    b = x**2 + y**2
    c, _, _, _ = la.lstsq(A, b, rcond=None)
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
        self.get_logger().info('Localization Node Started')

    def lidar_callback(self, msg):
        ranges = msg.ranges
        lidarXY = []

        # Convert lidar data to Cartesian coordinates
        for i, range in enumerate(ranges):
            if msg.range_min <= range <= msg.range_max:
                angle = msg.angle_min + i * msg.angle_increment
                x = range * math.cos(angle)
                y = range * math.sin(angle)
                lidarXY.append([x, y])

        # Convert to numpy array
        lidarXY = np.array(lidarXY)
        
        # Remove any rows with NaN or inf values
        lidarXY = lidarXY[~np.isnan(lidarXY).any(axis=1)]
        lidarXY = lidarXY[~np.isinf(lidarXY).any(axis=1)]

        # Check if there's data to process
        if lidarXY.shape[0] == 0:
            print("No valid lidar data.")
            return

        # Apply DBSCAN clustering
        # IMPORTANT: Adjust 'eps' and 'min_samples' parameters to get the expected results
        ESP = 0.1
        SAMPLES = 5
        model = DBSCAN(eps=ESP, min_samples=SAMPLES)
        db = model.fit(lidarXY)
        labels = db.labels_

        # Plot setup
        plt.figure(figsize=(10, 10))

        # Generate unique colors for each cluster label
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab10', len(unique_labels))  # Choose a color map and use the number of unique labels

        # Plot each cluster in a unique color
        for label in unique_labels:
            if label == -1:
                # Noise points (label -1)
                color = 'k'  # Black color for noise points
                cluster_points = lidarXY[labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=color, s=10, label='Noise')
            else:
                # Assign a unique color to each label
                color = colors(label / len(unique_labels))
                cluster_points = lidarXY[labels == label]
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], s=15, label=f'Cluster {label}')

        # Filter clusters by size and create dictionary of clusters
        ls, cs = np.unique(labels, return_counts=True)
        dic = dict(zip(ls, cs))

        # Print the size of each cluster label
        for label, size in dic.items():
            if label == -1:
                print(f"Label {label} (Noise): {size} points")
            else:
                print(f"Label {label}: {size} points")

        # Only select landmark clusters
        # IMPORTANT: Adjust MIN_POINT and MAX_POINT to get the expected landmarks
        MIN_POINT = 10
        MAX_POINT = 50
        idx = [i for i, label in enumerate(labels) if dic[label] < MAX_POINT and dic[label] > MIN_POINT and label >= 0]
        clusters = {label: [i for i in idx if db.labels_[i] == label] for label in np.unique(db.labels_[idx])}

        # Fit a circle to each cluster and visualize
        centers = []
        for label, group_idx in clusters.items():
            group_idx = np.array(group_idx)
            x_coords = lidarXY[group_idx, 0]
            y_coords = lidarXY[group_idx, 1]

            # Fit circle to the cluster points
            cx, cy, radius = fit_circle(x_coords, y_coords)

            # Check if the radius is within the desired range
            # IMPORTANT: Adjust MIN_RADIUS and MAX_RADIUS to your landmarks
            MIN_RADIUS = 0.02
            MAX_RADIUS = 0.3
            if MIN_RADIUS <= radius <= MAX_RADIUS:
                d = math.sqrt(cx**2 + cy**2)
                centers.append((cx, cy, radius, d))  
                self.get_logger().info(f"Cluster {label}: Center=({cx:.2f}, {cy:.2f}), Radius={radius:.2f}, Distance={d:.2f}")

                # Plot cluster points with a unique color
                plt.scatter(x_coords, y_coords, s=15, label=f'Cluster {label}')

                # Plot the fitted circle
                circle = Circle((cx, cy), radius, color=np.random.rand(3,), fill=False, linewidth=2, linestyle='--')
                plt.gca().add_patch(circle)

            else:
                self.get_logger().info(f"Cluster {label}: Rejected due to radius {radius:.2f} outside range.")

        # Sort centers by radius in descending order
        centers.sort(key=lambda center:center[2], reverse=True)

        # Ensure we have exactly two circles to perform trilateration
        if len(centers) >= 2:
            (cx1, cy1, r1, d1), (cx2, cy2, r1, d2) = centers[:2]
            self.get_logger().info(f"d1={d1:.2f} - d2={d2:.2f}")

            # NOTE: Here we have two input options
            # 1: Using centers of the fitted circles (convenient) - trilaterate(cx1, cy1, d1, cx2, cy2, d2)
            # 2: Using landmark positions (more accurate): trilaterate(px1, py1, d1, px2, py2, d2)            
            lidar_x, lidar_y = trilaterate(cx1, cy1, d1, cx2, cy2, d2)

            if lidar_x is not None:
                self.get_logger().info(f"LIDAR Position: ({lidar_x:.2f}, {lidar_y:.2f})")

            else:
                self.get_logger().info("Trilateration failed: No intersection.")

        # Add labels and legend
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('LIDAR Clusters and Fitted Circles')
        plt.legend(loc='upper right')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

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
