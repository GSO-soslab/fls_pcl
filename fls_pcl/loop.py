#!/usr/bin/env python3

#Author: Tony Jacob
#Part of RISE Project. 
#Tries to estimate whether a circumnavigation has been complete & also estimates iceberg velocity.
#tony.jacob@uri.edu

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, Image
from nav_msgs.msg import Odometry
import numpy as np
from cv_bridge import CvBridge
from math import nan
import cv2

class Loop(Node):
    def __init__(self):
        super().__init__('loop_checker')

        self.create_subscription(Image, "/alpha_rise/costmap/image", self.costmap_image_callback, 10)
        self.create_subscription(Odometry, "/alpha_rise/odometry/filtered", self.odometry_callback, 10)
        self.map_pub = self.create_publisher(Image, "/alpha_rise/costmap/global", 10)
        self.map_pub_list = self.create_publisher(Image, "/alpha_rise/list", 10)

        self.map = np.zeros((500, 500), dtype=np.uint8)
        
        self.bridge = CvBridge()
        self.odom_cb_flag = 0
        self.prev_cx_centered = 0
        self.prev_cy_centered = 0
        self.list_of_local_maps = []

    def odometry_callback(self, msg):
        self.vx_x = msg.pose.pose.position.x
        self.vx_y = msg.pose.pose.position.y

        """
        ------->x IMAGE
        |       x
        |    _|
        |    y ODOM
        v y
        """
        self.vx_x_image = -round(self.vx_y)
        self.vx_y_image = -round(self.vx_x)
        self.odom_cb_flag = 1
        
    def costmap_image_callback(self, msg):
        if self.odom_cb_flag:
            costmap_cv_image = self.bridge.imgmsg_to_cv2(msg)
            # print(costmap_cv_image.shape, flush=True)
            self.map = self.create_global_map(self.map, costmap_cv_image, (self.vx_x_image, self.vx_y_image))
            self.map_pub.publish(self.bridge.cv2_to_imgmsg(self.map))
            # self.sift_method(costmap_cv_image)
            self.pub_list(costmap_cv_image)
            # self.cv2_template_matching(costmap_cv_image)

    def pub_list(self, local_image):
        if len(self.list_of_local_maps)  > 4:
            index, best_image = self.most_similar_image_akaze(local_image, self.list_of_local_maps, len(self.list_of_local_maps)//2)
            # index, best_image = self.most_similar_image(local_image, self.list_of_local_maps)
            
            if index != None:
                # mix= np.hstack((best_image, local_image))
                mix = self.bridge.cv2_to_imgmsg(best_image, encoding="rgb8")
                self.map_pub_list.publish(mix)
                print(index, flush=True)

    def most_similar_image_akaze(self, query_img, img_list, oldest_n):
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(query_img, None)

        # AKAZE uses binary descriptors, so Hamming distance is still valid
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        best_index = -1
        best_matches_count = 0
        best_matches = None
        best_kp2 = None

        if oldest_n <= 0:
            return None, None  # invalid parameter

        end_idx = min(oldest_n, len(img_list))  # In case oldest_n > total images

        for i in range(end_idx):
            img = img_list[i]
            kp2, des2 = akaze.detectAndCompute(img, None)
            if des2 is None:
                continue

            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 100]

            if len(good_matches) > best_matches_count:
                best_matches_count = len(good_matches)
                best_index = i
                best_matches = good_matches
                best_kp2 = kp2

        if best_index == -1 or best_matches_count < 2:  # Minimum 3 matches
            return None, None

        match_img = cv2.drawMatches(query_img, kp1, img_list[best_index], best_kp2, best_matches, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        return best_index, match_img
    
    def most_similar_image_orb(self, query_img, img_list, oldest_n):
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(query_img, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            best_index = -1
            best_matches_count = 0
            best_matches = None
            best_kp2 = None

            if oldest_n <= 0:
                return None, None  # invalid parameter

            end_idx = min(oldest_n, len(img_list))  # in case oldest_n > len(img_list)

            for i in range(end_idx):
                img = img_list[i]
                kp2, des2 = orb.detectAndCompute(img, None)
                if des2 is None:
                    continue

                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                good_matches = [m for m in matches if m.distance < 90]

                if len(good_matches) > best_matches_count:
                    best_matches_count = len(good_matches)
                    best_index = i
                    best_matches = good_matches
                    best_kp2 = kp2

            if best_index == -1 or best_matches_count < 2:
                return None, None

            match_img = cv2.drawMatches(query_img, kp1, img_list[best_index], best_kp2, best_matches, None,
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            return best_index, match_img

    def create_global_map(self, large_img, small_img, small_center_coords):
        large_image_copy = large_img.copy()
        lh, lw = large_img.shape[:2]
        sh, sw = small_img.shape[:2]
        
        # Coordinates where the center of the small image should go (origin at large image center)
        cx_centered, cy_centered = small_center_coords
        curr_distance = np.sqrt((cx_centered - self.prev_cx_centered)**2 + (cy_centered - self.prev_cy_centered)**2)
        # Convert to top-left origin pixel coordinates
        cx = lw // 2 + cx_centered
        cy = lh // 2 + cy_centered 

        # Compute top-left corner for placing the small image
        x_start = cx - sw // 2
        y_start = cy - sh // 2
        x_end = x_start + sw
        y_end = y_start + sh

        # Ensure bounds
        if x_start < 0 or y_start < 0 or x_end > lw or y_end > lh:
            raise ValueError("Small image goes out of bounds of large image at given center.")

        mask = small_img > 0
        large_img[y_start:y_end, x_start:x_end][mask] = small_img[mask]

        if curr_distance > 20: #m
            large_image_copy = large_img
            self.list_of_local_maps.append(small_img)
            
            print(f"updated; {len(self.list_of_local_maps)}", flush=True)
            self.prev_cx_centered = cx_centered
            self.prev_cy_centered = cy_centered

        # Insert small image
        # large_img[y_start:y_end, x_start:x_end] = small_img
        # large_img[y_start:y_end, x_start:x_end] = cv2.bitwise_or(large_img[y_start:y_end, x_start:x_end], small_img)

        
        return large_image_copy

    
def main():
    rclpy.init()
    node = Loop()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()