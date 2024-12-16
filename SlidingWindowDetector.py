import numpy as np
import cv2
from BaseDetector import BaseDetector
from utils import load_config, display, display_points_on_image, visualize_histogram, fill_lane

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
        self.src_points = []

    def pre_process(self, img):
        self.height = img.shape[0]
        self.width = img.shape[1]
        mask = self._pre_processing(img)
        self.src_points = self.pick_points(img) if len(self.src_points) == 0 else self.src_points
        masked_img = self._region_of_interest(mask, self.src_points)
        return masked_img
    
    def get_lines(self, img, mask):
        self.src_points = self.pick_points(img) if len(self.src_points) == 0 else self.src_points
        dst_points = self._get_dst_points()
        # display(mask)
        transformed_mask = self._transform_perspective(mask, self.src_points, dst_points, (self.width, self.height))
        kernel = np.ones((11,11), np.uint8)
        opening = cv2.morphologyEx(transformed_mask, cv2.MORPH_CLOSE, kernel)
        # display(transformed_mask)
        left_base, right_base= self._get_base(transformed_mask)
        left_fit, right_fit, debug_img = self._sliding_window(transformed_mask, left_base, right_base)
        # display(debug_img)
        pts_left, pts_right = self._find_points(left_fit, right_fit)
        original_pts_left = self._transform_points(pts_left, dst_points, self.src_points)
        original_pts_right = self._transform_points(pts_right,dst_points, self.src_points)
        
        return (original_pts_left, original_pts_right)
    
    def add_lines(self, img, lines):
        (pts_left, pts_right) = lines

        filled_lane = fill_lane(img, pts_left, pts_right)
        # result = self._replace_lane_area(img, filled_lane_normal)
        return filled_lane

    def _replace_lane_area(self, original_img, filled_lane):
        """
        Replace the lane area in the original image with the filled lane.
        :param original_img: The original image.
        :param filled_lane: The transformed filled lane image in normal coordinates.
        :return: The combined image.
        """
        lane_mask = cv2.cvtColor(filled_lane, cv2.COLOR_BGR2GRAY)
        _, lane_mask_binary = cv2.threshold(lane_mask, 1, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(lane_mask_binary)
        original_without_lane = cv2.bitwise_and(original_img, original_img, mask=inv_mask)
        result = cv2.add(original_without_lane, filled_lane)
        return result


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

    def pick_points(self, img):
        """
        Function to let the user pick the source points interactively
        in the order: top-left, top-right, bottom-right, bottom-left.
        """
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Image", img)
                self.src_points.append((x, y))
                if len(self.src_points) == 4:
                    print("Points selected:", self.src_points)
                    cv2.destroyAllWindows()

        
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Return the selected points
        return np.array(self.src_points, dtype=np.float32)
    def _transform_perspective(self, img, src_points, dst_points, dst_size):
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        perspective_transformed_img = cv2.warpPerspective(img, matrix, dst_size)
        return perspective_transformed_img
    
    def _transform_points(self, pts, src, dst):
        matrix = cv2.getPerspectiveTransform(src, dst)
        np_pts = np.array([pt[0] for pt in pts], dtype=np.float32).reshape((-1, 1, 2))
        transformed_pts = cv2.perspectiveTransform(pts, matrix)
        return transformed_pts
    
    # def _get_polygon(self):
    #     polygon = np.float32([
    #                             [int(self.width * 0.45), int(self.height * 0.62)],  # Top-left 
    #                             [int(self.width * 0.58), int(self.height * 0.62)],  # Top-right
    #                             [int(0.95 * self.width), int(self.height * 0.94)],  # Bottom-right
    #                             [int(self.width * 0.15), int(self.height * 0.94)]   # Bottom-left
    #                             ])
    #     return polygon
    
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
        # visualize_histogram(histogram)
        midpoint = int(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        return left_base, right_base
    
    def _sliding_window(self, mask, left_base, right_base, n_windows=50, margin=100, minpix=50, alpha=0.7):
        out_img = np.dstack((mask, mask, mask)) * 255
        left_lane_indices , right_lane_indices  = [], []
        window_height = self.height // n_windows
        y, x = mask.nonzero()
        left_x_current = left_base
        right_x_current = right_base

        for window in range(n_windows):
            y_low = self.height - (window + 1) * window_height
            y_high = self.height - window * window_height
            x_left_low = left_x_current - margin
            x_left_high = left_x_current + margin
            x_right_low = right_x_current - margin
            x_right_high = right_x_current + margin
            cv2.rectangle(out_img, (x_left_low, y_low), (x_left_high, y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (x_right_low, y_low), (x_right_high, y_high), (0, 255, 0), 2)
            cv2.circle(out_img, (right_x_current, (y_low + y_high) // 2), 5, (0, 0, 255), -1)
            cv2.circle(out_img, (left_x_current, (y_low + y_high) // 2), 5, (0, 0, 255), -1)
            good_left_indices = ((y >= y_low) & (y < y_high) & (x >= x_left_low) & (x < x_left_high)).nonzero()[0]
            good_right_indices  = ((y >= y_low) & (y < y_high) & (x >= x_right_low) & (x < x_right_high)).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)
            
            if len(good_left_indices) > minpix:
                current_mean_left = np.mean(x[good_left_indices])
                left_x_current = int(alpha * current_mean_left + (1 - alpha) * left_x_current)

            if len(good_right_indices) > minpix:
                current_mean_right = np.mean(x[good_right_indices])
                right_x_current = int(alpha * current_mean_right + (1 - alpha) * right_x_current)
        
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)
        leftx, lefty = x[left_lane_indices], y[left_lane_indices]
        rightx, righty = x[right_lane_indices], y[right_lane_indices]
        left_fit = (np.polyfit(lefty, leftx, 2), int(np.min(lefty))) if len(lefty) > 0 else None
        right_fit = (np.polyfit(righty, rightx, 2), int(np.min(righty))) if len(righty) > 0 else None
        out_img[lefty, leftx] = [255, 0, 0]  
        out_img[righty, rightx] = [0, 0, 255]  

        return left_fit, right_fit, out_img
    
    def _find_points(self, left_fit, right_fit):
        min_value = min(left_fit[1], right_fit[1])
        left_y = np.linspace(min_value, self.height-1,self.height-min_value)
        right_y = np.linspace(min_value, self.height-1,self.height-min_value)
        left_fitx = left_fit[0][0]*left_y**2 + left_fit[0][1]*left_y + left_fit[0][2]
        right_fitx = right_fit[0][0]*right_y**2 + right_fit[0][1]*right_y + right_fit[0][2]
        pts_left = np.array([np.transpose(np.vstack([left_fitx, left_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_y])))])
        return pts_left, pts_right
