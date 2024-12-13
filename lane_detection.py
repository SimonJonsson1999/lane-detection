import numpy as np
import cv2
import matplotlib.pyplot as plt


class LaneLineDetector(object):
    def __init__(self, config):
        self.config = config
        self.kernel_size = self.config['canny_edge'].get('kernel_size', 5)
        self.low_threshold = self.config['canny_edge'].get('low_threshold', 180)
        self.high_threshold = self.config['canny_edge'].get('high_threshold', 240)
        self.apex_factor = self.config['region_of_interest'].get('apex_factor', 0.85)

        
        self.rho = self.config['hough_transform'].get('rho', 1)
        self.theta = self.config['hough_transform'].get('theta', np.pi / 180)
        self.threshold = self.config['hough_transform'].get('threshold', 2)
        self.min_line_length = self.config['hough_transform'].get('min_line_length', 4)
        self.max_line_gap = self.config['hough_transform'].get('max_line_gap', 5)

    def detect(self, img):
        region_of_interest = self._get_region_of_interest(img)
        canny = self._canny(img)
        masked_canny = self._mask_canny(canny, region_of_interest)
        lines = self._get_hough_lines(masked_canny)
        line_image = np.copy(img)*0
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        return lines_edges

    def _get_region_of_interest(self, img):
        x_size = img.shape[1]
        y_size = img.shape[0]
        left_bottom = (0, y_size)
        right_bottom = (x_size, y_size)
        apex = (x_size/2, y_size/(2 * self.apex_factor))
        region_of_interest = (left_bottom, apex, right_bottom)
        return region_of_interest
    
    def _get_hough_lines(self, masked_edges, rho=1, theta=np.pi/180, threshold=2, min_line_length=4, max_line_gap=5):
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                                       min_line_length, max_line_gap)
        return lines
    
    def _canny(self, img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray_img,(self.kernel_size, self.kernel_size),0)
        edges = cv2.Canny(blur_gray, self.low_threshold, self.high_threshold)
        return edges
    
    def _mask_canny(self, canny, region_of_interest):
        mask = np.zeros_like(canny)   
        ignore_mask_color = 255
        vertices = np.array([[region_of_interest[0], region_of_interest[1], region_of_interest[2],]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        masked_canny = cv2.bitwise_and(canny, mask)
        return masked_canny
    
