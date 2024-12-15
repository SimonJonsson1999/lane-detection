import numpy as np
import cv2
from BaseDetector import BaseDetector
from utils import load_config, display, display_points_on_image, visualize_histogram

class SlidingWindowDetector(BaseDetector):

    def __init__(self, config):
        """
        Initializes the LaneLineDetector with parameters for edge detection, 
        region of interest, and Hough transform.
        
        Parameters:
            config (dict): Configuration dictionary containing configuration parameters
        """
        self.config = config
        self.height = 0
        self.width = 0

    def pre_process(self, img):
        print(f"Starting Preprocessing")
        self.height = img.shape[0]
        print(f"image height = {self.height}")
        self.width = img.shape[1]
        print(f"image width = {self.width}")
        mask = self._pre_processing(img)
        polygon = self._get_polygon()
        masked_img = self._region_of_interest(mask, polygon)
        return masked_img
    
    def get_lines(self, img, mask):
        src_points = self._get_src_points()
        dst_points = self._get_dst_points()
        transformed_mask = self._transform_perspective(mask, src_points, dst_points, (self.width, self.height))
        left_base, right_base= self._get_base(transformed_mask)
        left_indexes, right_indexes = self.sliding_window(mask, left_base, right_base)
        
    def add_lines(self, img, lines):
        pass


    def _pre_processing(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray_img_blur = cv2.GaussianBlur(gray_img, (5,5), 0)
        white_mask = cv2.threshold(gray_img_blur, 200, 255, cv2.THRESH_BINARY)[1]
        lower_yellow = np.array([0,100,100])
        upper_yellow = np.array([210,255,255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return mask


    def _transform_perspective(self, img, src_points, dst_points, dst_size):
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        perspective_transformed_img = cv2.warpPerspective(img, matrix, dst_size)
        return perspective_transformed_img
    
    def _get_polygon(self):
        polygon = np.float32([
                                [int(self.width * 0.45), int(self.height * 0.55)],  # Top-left 0.62 -> 0.55
                                [int(self.width * 0.58), int(self.height * 0.55)],  # Top-right 0.62 -> 0.55
                                [int(0.95 * self.width), int(self.height * 0.94)],  # Bottom-right
                                [int(self.width * 0.15), int(self.height * 0.94)]   # Bottom-left
                                ])
        return polygon
    
    def _region_of_interest(self, img, polygon):
        mask = np.zeros_like(img)
        vertices = np.array([polygon], dtype=np.int32)
        ignore_mask_color = (255,) * img.shape[2] if len(img.shape) > 2 else 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    def _get_src_points(self):
        points = self._get_polygon()
        return points
    
    def _get_dst_points(self):
        points = np.float32([
                                [0, 0],                   # Top-left
                                [self.width, 0],          # Top-right
                                [self.width, self.height], # Bottom-right
                                [0, self.height]          # Bottom-left
                                ])
        return points
    
    def _get_base(self, mask):
        histogram = np.sum(mask[self.height // 2:, :], axis=0)
        visualize_histogram(histogram)
        midpoint = int(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        return left_base, right_base
    
    def sliding_window(self, mask, left_base, right_base, n_windows=50, margin=100, minpix=50):
        left_indexes, right_indexes = np.array([]), np.array([])
        window_height = int(self.height/n_windows)
        y, x = mask.nonzero()
        left_x_current = left_base
        right_x_current = right_base
        for window in range(n_windows):
            y_low = self.height - (window+1)*window_height
            y_high = self.height - window*window_height
            x_left_low = left_x_current - margin
            x_left_high = left_x_current + margin
            x_right_low = right_x_current - margin
            x_right_high = right_x_current + margin

            left_indices = ((y >= y_low) & (y < y_high) & (x >= x_left_low) & (x < x_left_high)).nonzero()[0]
            right_indices = ((y >= y_low) & (y < y_high) & (x >= x_right_low) & (x < x_right_high)).nonzero()[0]
            np.append(left_indexes, left_indices)
            np.append(right_indexes, right_indices)
            if len(left_indices) > minpix:
                left_x_current = int(np.mean(x[left_indices]))
            if len(right_indices) > minpix:
                right_x_current = int(np.mean(x[right_indices]))
            if left_indexes:
                left_indexes = np.concatenate(left_indexes)
            if right_indexes:
                right_indexes = np.concatenate(right_indexes)
            # leftx = x[left_lane_indices]
            # lefty = y[left_lane_indices]
            # rightx = x[right_lane_indices]
            # righty = y[right_lane_indices]
        return left_indexes, right_indexes
