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
    y = -math.sqrt(d1**2 - x**2)

    return x, y

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
        super().__init__('localization_node')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)
        self.get_logger().info('Localization Node Started')
        self.csv_file = open('lidar_positions.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Lidar_X', 'Lidar_Y'])

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

        ESP = 0.1
        SAMPLES = 5
        model = DBSCAN(eps=ESP, min_samples=SAMPLES)
        db = model.fit(lidarXY)
        labels = db.labels_

        ls, cs = np.unique(labels, return_counts=True)
        dic = dict(zip(ls, cs))

        MIN_POINT = 10
        MAX_POINT = 50
        idx = [i for i, label in enumerate(labels) if dic[label] < MAX_POINT and dic[label] > MIN_POINT and label >= 0]
        clusters = {label: [i for i in idx if db.labels_[i] == label] for label in np.unique(db.labels_[idx])}

        centers = []
        for label, group_idx in clusters.items():
            group_idx = np.array(group_idx)
            x_coords = lidarXY[group_idx, 0]
            y_coords = lidarXY[group_idx, 1]

            cx, cy, radius = fit_circle(x_coords, y_coords)

            MIN_RADIUS = 0.02
            MAX_RADIUS = 0.3
            if MIN_RADIUS <= radius <= MAX_RADIUS:
                d = math.sqrt(cx**2 + cy**2)
                centers.append((cx, cy, radius, d))

        centers.sort(key=lambda center: center[2], reverse=True)

        if len(centers) >= 2:
            (cx1, cy1, r1, d1), (cx2, cy2, r2, d2) = centers[:2]
            lidar_x, lidar_y = trilaterate(cx1, cy1, d1, cx2, cy2, d2)

            if lidar_x is not None:
                self.get_logger().info(f"LIDAR Position: ({lidar_x:.2f}, {lidar_y:.2f})")
                self.csv_writer.writerow([lidar_x, lidar_y])

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

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
