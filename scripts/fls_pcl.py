#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
from cv_bridge import CvBridge
import cv2

class FLS_PCL:
    def __init__(self):
        #Publisher info
        self.pub_pcl = rospy.Publisher("/alpha_rise/fls/pointcloud", PointCloud2, queue_size=1)
        self.bridge = CvBridge()
        
        self.pointcloud_msg = PointCloud2()
        rospy.Subscriber("/alpha_rise/fls/data/image", Image, self.image_CB, queue_size=1)
        # rospy.Subscriber("/alpha_rise/fls/data/display", Image, self.display_CB, queue_size=1)

    
    def image_CB(self, msg):
        #GrayScale Image with bins x beams 
        current = self.bridge.imgmsg_to_cv2(msg)
        rows, columns = current.shape
        self.n_bins, self.n_beams = rows, columns
        edge_list, sensor_frame = self.highest_intensity_coordinates_per_column(current)
        edge_image = np.zeros((rows, columns))
        
        for row, col, intensity in edge_list:
            ##Gfaussian Noise removbal
            edge_image[row, col] = intensity

        # print(sensor_frame[500],sensor_frame[500][0])
        fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32,1),
        # PointField('rgb', 16, PointField.INT32,1),
        ]
        self.pointcloud_msg.fields = fields
        self.pointcloud_msg.height = self.n_beams
        self.pointcloud_msg.width = self.n_beams
        h = Header()
        h.frame_id = 'alpha_rise/fls_link_sf'
        h.stamp = msg.header.stamp
        self.pointcloud_msg.header = h
        self.pointcloud_msg.point_step = 4 * (len(fields))  # Each point occupies 16 bytes
 
        # # Populate the point data
        num_points = self.pointcloud_msg.width * self.pointcloud_msg.height
        self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points
        self.pointcloud_msg.is_dense = True  # All points are valid

        self.points = np.zeros(((self.n_beams), (self.n_beams), len(fields)), dtype=np.float32)
        # print(self.points.shape)

        for i in range(len(sensor_frame)): #n_bins #rows.
            for j in range(len(sensor_frame)):
                self.points[i][j][0] = sensor_frame[i][0]
                self.points[i][j][1] =  sensor_frame[i][1]
                self.points[i][j][3] = sensor_frame[i][2]

        self.pointcloud_msg.data = self.points.tobytes()        
        # print(self.points.shape)
        # # # # print(self.pointcloud_msg.header)
        self.pub_pcl.publish(self.pointcloud_msg)

    def highest_intensity_coordinates_per_column(self, image):
        """
        Retrieve the coordinates of the highest intensity index per column for all rows in an image.
        
        Parameters:
        image_path (str): Path to the image file.
        
        Returns:
        List[Tuple[int, int]]: A list of tuples containing (row_index, column_index) for the highest intensity per column.
        """
        # Load the image in grayscale
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError("Image not found or unable to load.")
        
        # Initialize a list to hold the coordinates of the highest intensity per column
        image_highest_coordinates = []
        sensor_frame_coordinates = []
        delta_theta = np.radians(70) / self.n_beams 
        # Loop over each column to find the maximum intensity
        for col in range(image.shape[1]):
            # Get the maximum value and its index in the current column
            max_index = np.argmax(image[:, col])  # Row index of the maximum value
            theta = np.radians(-70/2) + col * delta_theta
            sensor_frame_x = (40/self.n_bins) * (self.n_bins - max_index) * np.cos(theta)
            sensor_frame_y = (40/self.n_bins) * (self.n_bins - max_index) * np.sin(theta)
            if image[max_index, col] > 10:
                image_highest_coordinates.append((max_index, col, image[max_index, col]))  # Append coordinates as (row, column)
                sensor_frame_coordinates.append((sensor_frame_x, sensor_frame_y, image[max_index, col]))

        return image_highest_coordinates, sensor_frame_coordinates

    
if __name__ == "__main__":
    rospy.init_node("FLS_to_PCL")
    FLS_PCL()
    rospy.spin()