#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
from math import nan, sqrt

class FLS_PCL(Node):
    def __init__(self):
        super().__init__('fls_pcl_node')

        # Declare and read parameters
        self.declare_parameter('horizontal_beamwidth', Parameter.Type.INTEGER)
        self.declare_parameter('max_range',Parameter.Type.INTEGER)
        self.declare_parameter('intensity_threshold',Parameter.Type.INTEGER)
        self.declare_parameter('range_threshold',Parameter.Type.INTEGER)
        self.declare_parameter('beam_skip_count',Parameter.Type.INTEGER)
        self.declare_parameter('frame_id',Parameter.Type.STRING)
        self.declare_parameter('sub_topic',Parameter.Type.STRING)


        self.horizontal_beamwidth = self.get_parameter('horizontal_beamwidth').value
        self.max_range = self.get_parameter('max_range').value
        self.intensity_threshold = self.get_parameter('intensity_threshold').value
        self.range_threshold = self.get_parameter('range_threshold').value
        self.beam_skip_count = self.get_parameter('beam_skip_count').value
        self.frame_id = self.get_parameter('frame_id').value
        sub_topic = self.get_parameter('sub_topic').value


        # CV bridge
        self.bridge = CvBridge()

        # Publishers
        self.pub_pcl = self.create_publisher(PointCloud2, '/alpha_rise/fls/pointcloud', 10)
        self.pub_fls_edge_image = self.create_publisher(Image, '/alpha_rise/fls/data/image/edge', 10)

        # Subscriber
        self.create_subscription(Image,sub_topic,self.image_CB,10)


        # Populate PointCloud2 message
        self.pointcloud_msg = PointCloud2()
        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        self.pointcloud_msg.fields = self.fields
        self.pointcloud_msg.point_step = 4 * (len(self.fields))  # Each point occupies 16 bytes
        self.pointcloud_msg.is_dense = True  # All points are valid
        
    def image_CB(self, msg):
        #GrayScale Image with bins x beams 
        current = self.bridge.imgmsg_to_cv2(msg)
        rows, columns = current.shape
        self.n_bins, self.n_beams = rows, columns
        edge_list, sensor_frame = self.extract_highest_intensity_per_beam(current)
        edge_image = np.zeros((rows, columns))

        self.pointcloud_msg.height = 1
        self.pointcloud_msg.width = self.n_beams
        h = Header()
        h.frame_id = self.frame_id
        self.pointcloud_msg.header = h

        # # Populate the point data
        num_points = self.pointcloud_msg.width * self.pointcloud_msg.height
        self.pointcloud_msg.row_step = self.pointcloud_msg.point_step * num_points

        #n_beams * (x,y,z,i)
        self.points = np.zeros(((self.pointcloud_msg.width), len(self.fields)), dtype=np.float32)
        if len(edge_list) > 0: 
            self.get_logger().info("Scanning", throttle_duration_sec = 3)
            for i in range(len(sensor_frame)): 
                    #Range threhold
                    if i % self.beam_skip_count == 0:
                        #Check for intensity threshold to remove noise.
                        if sensor_frame[i][2] > self.intensity_threshold:
                            #Check for measurements past a range
                            if sqrt(sensor_frame[i][0]**2 + sensor_frame[i][1]**2) > self.range_threshold:                    
                                #X
                                self.points[i][0] = sensor_frame[i][0]
                                #Y
                                self.points[i][1] = sensor_frame[i][1]
                                #Intensity
                                self.points[i][3] = sensor_frame[i][2]
                            else:
                                #X
                                self.points[i][0] = nan
                                #Y
                                self.points[i][1] = nan
                                #Z
                                self.points[i][2] = nan
                                #Intensity
                                self.points[i][3] = nan
                        else:
                            #X
                            self.points[i][0] = nan
                            #Y
                            self.points[i][1] = nan
                            #Z
                            self.points[i][2] = nan
                            #Intensity
                            self.points[i][3] = nan
                    else:
                        #X
                        self.points[i][0] = nan
                        #Y
                        self.points[i][1] = nan
                        #Z
                        self.points[i][2] = nan
                        #Intensity
                        self.points[i][3] = nan
            
            rows, cols, intensities = edge_list[:, 0], edge_list[:, 1], edge_list[:, 2]
            edge_image[rows, cols] = intensities
            
                
        else:
            self.get_logger().warn("No measurements", throttle_duration_sec = 3)
            self.points[:][:][:] = nan
        
        #Convert image to view in image_view
        edge_image = cv2.normalize(edge_image, None, 0, 255, cv2.NORM_MINMAX)
        edge_image = edge_image.astype(np.uint8)
        self.pub_fls_edge_image.publish(self.bridge.cv2_to_imgmsg(edge_image, encoding="mono8"))

        self.pointcloud_msg.data = self.points.tobytes()        
        self.pub_pcl.publish(self.pointcloud_msg)

    def extract_highest_intensity_per_beam(self, image):
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
        valid_mask = max_intensity_values > 0#self.intensity_threshold

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
    
def main():
    rclpy.init()
    node = FLS_PCL()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()