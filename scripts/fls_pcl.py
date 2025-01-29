#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
from cv_bridge import CvBridge
import cv2
from math import nan

class FLS_PCL:
    def __init__(self):
        #Params
        self.horizontal_beamwidth = rospy.get_param('/alpha_rise/fls_params/horizontal_beamwidth', 70) #degrees
        self.max_range = rospy.get_param('/alpha_rise/fls_params/max_range', 40)  #m
        self.intensity_threshold = rospy.get_param('/alpha_rise/fls_params/intensity_threshold', 10)
        self.range_threshold = rospy.get_param('/alpha_rise/fls_params/range_threshold', 10) #m

        self.bridge = CvBridge()

        rospy.Subscriber("/alpha_rise/fls/data/image", Image, self.image_CB, queue_size=1)
        self.frame_id = rospy.get_param("/alpha_rise/fls_params/frame_id", "alpha_rise/fls_link_sf")
        #Publisher info
        self.pub_pcl = rospy.Publisher("/alpha_rise/fls/pointcloud", PointCloud2, queue_size=1)
        self.pub_fls_edge_image = rospy.Publisher("/alpha_rise/fls/data/image/edge", Image, queue_size=1)

        # Populate PointCloud2 message
        self.pointcloud_msg = PointCloud2()
        self.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32,1),
        # PointField('rgb', 16, PointField.INT32,1),
        ]
        self.pointcloud_msg.fields = self.fields
        self.pointcloud_msg.point_step = 4 * (len(self.fields))  # Each point occupies 16 bytes
        self.pointcloud_msg.is_dense = True  # All points are valid
        
    def image_CB(self, msg):
        #GrayScale Image with bins x beams 
        current = self.bridge.imgmsg_to_cv2(msg)
        rows, columns = current.shape
        self.n_bins, self.n_beams = rows, columns
        edge_list, sensor_frame = self.highest_intensity_coordinates_per_column_optimized(current)
        edge_image = np.zeros((rows, columns))
        
        if len(edge_list) > 0: 
            rows, cols, intensities = edge_list[:, 0], edge_list[:, 1], edge_list[:, 2]
            edge_image[rows, cols] = intensities
            
            #Convert image to view in image_view
            edge_image = cv2.normalize(edge_image, None, 0, 255, cv2.NORM_MINMAX)
            edge_image = edge_image.astype(np.uint8)
            self.pub_fls_edge_image.publish(self.bridge.cv2_to_imgmsg(edge_image, encoding="mono8"))
    
            self.pointcloud_msg.height = self.n_beams
            self.pointcloud_msg.width = self.n_beams
            h = Header()
            h.frame_id = self.frame_id
            h.stamp = msg.header.stamp
            self.pointcloud_msg.header = h
    
            # # Populate the point data
            num_points = self.pointcloud_msg.width * self.pointcloud_msg.height
            self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points

            self.points = np.zeros(((self.pointcloud_msg.height), (self.pointcloud_msg.width), len(self.fields)), dtype=np.float32)
        
            for i in range(len(sensor_frame)): 
                for j in range(len(sensor_frame)):
                    #Range threhold
                    if sensor_frame[i][0] > 10:                    
                        #X
                        self.points[i][j][0] = sensor_frame[i][0]
                        #Y
                        self.points[i][j][1] = sensor_frame[i][1]
                        #Intensity
                        self.points[i][j][3] = sensor_frame[i][2]
                    else:
                        #X
                        self.points[i][j][0] = nan
                        #Y
                        self.points[i][j][1] = nan
                        #Intensity
                        self.points[i][j][3] = nan

            # for i,j in product(range(len(sensor_frame)),range(len(sensor_frame))): 
            #     self.points[i][j][0] = sensor_frame[i][0]
            #     #Y
            #     self.points[i][j][1] = sensor_frame[i][1]
            #     #Intensity
            #     self.points[i][j][3] = sensor_frame[i][2]

            self.pointcloud_msg.data = self.points.tobytes()        
            self.pub_pcl.publish(self.pointcloud_msg)
        else:
            rospy.loginfo_throttle(3,"No measurements")

    def highest_intensity_coordinates_per_column_optimized(self, image):
        """
        Retrieve the highest intensity index per column and convert into sensor frame.

        Parameters:
        image (np.array): Image array

        Returns:
        Tuple[List[Tuple[int, int, int]], List[Tuple[float, float, int]]]: 
        - A list of tuples (row_index, column_index, intensity_value) for the highest intensity per column.
        - A list of tuples (x, y, intensity_value) in the sensor frame.
        """
        if image is None:
            raise ValueError("Image not found or unable to load.")

        # Precompute constants
        delta_theta = np.radians(self.horizontal_beamwidth) / self.n_beams
        base_theta = np.radians(-self.horizontal_beamwidth / 2)
        range_factor = self.max_range / self.n_bins

        # Find max intensity row indices for each column
        max_intensity_rows = np.argmax(image, axis=0)  # (1D array of row indices)
        
        # Get intensity values at those max locations
        max_intensity_values = image[max_intensity_rows, np.arange(image.shape[1])]

        # Threshold filtering
        valid_mask = max_intensity_values > self.intensity_threshold

        # Compute angles
        theta_values = base_theta + np.arange(image.shape[1]) * delta_theta

        # Compute sensor frame coordinates
        r_values = range_factor * (self.n_bins - max_intensity_rows)
        sensor_x = r_values * np.cos(theta_values)
        sensor_y = r_values * np.sin(theta_values)

        # Apply the threshold mask
        image_highest_coordinates = list(zip(max_intensity_rows[valid_mask], 
                                            np.where(valid_mask)[0], 
                                            max_intensity_values[valid_mask]))


        sensor_frame_coordinates = list(zip(sensor_x[valid_mask], 
                                            sensor_y[valid_mask], 
                                            max_intensity_values[valid_mask]))

        return np.array(image_highest_coordinates), np.array(sensor_frame_coordinates)
    
if __name__ == "__main__":
    rospy.init_node("FLS_to_PCL")
    FLS_PCL()
    rospy.spin()