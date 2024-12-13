import numpy as np
import cv2
import matplotlib.pyplot as plt


class LaneLineDetector(object):
    """
    Detects lane lines in an input image using image processing techniques such as 
    Canny edge detection, region masking, and Hough Line Transform.
    """
    def __init__(self, config):
        """
        Initializes the LaneLineDetector with parameters for edge detection, 
        region of interest, and Hough transform.
        
        Parameters:
            config (dict): Configuration dictionary containing configuration parameters
        """
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
        """
        Detects lane lines in the input image.
        
        Parameters:
            img (ndarray): Input image (BGR format).

        Returns:
            ndarray: Output image with detected lanes drawn on it.
        """
        region_of_interest = self._get_region_of_interest(img)
        canny = self._canny(img)
        masked_canny = self._mask_canny(canny, region_of_interest)
        lines = self._get_hough_lines(masked_canny)
        lane_lines = self._create_lane_lines(img, lines)
        detected_image = self._draw_lane_lines(img, lane_lines)
        # print(f"Number of lines: {len(lines)}")
        # print(f"left_lane: {left_lane}")
        # print(f"right_lane: {right_lane}")
        
        return detected_image

    def _get_region_of_interest(self, img):
        """
        Defines the triangular region of interest where lanes are expected.

        Parameters:
            img (ndarray): Input image.

        Returns:
            tuple: A triangle defined by three vertices (left bottom, apex, right bottom).
        """
        x_size = img.shape[1]
        y_size = img.shape[0]
        left_bottom = (0, y_size)
        right_bottom = (x_size, y_size)
        apex = (x_size/2, y_size/(2 * self.apex_factor))
        region_of_interest = (left_bottom, apex, right_bottom)
        return region_of_interest
    
    def _get_hough_lines(self, masked_edges, rho=1, theta=np.pi/180, threshold=2, min_line_length=4, max_line_gap=5):
        """
        Detects line segments using the Hough Line Transform.

        Parameters:
            masked_edges (ndarray): Masked edges image.
            rho (float): Distance resolution of the accumulator in pixels.
            theta (float): Angular resolution of the accumulator in radians.
            threshold (int): Minimum number of intersections in Hough accumulator.
            min_line_length (int): Minimum line length to detect.
            max_line_gap (int): Maximum allowed gap between line segments.

        Returns:
            list: Detected line segments represented as [x1, y1, x2, y2].
        """
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                                       min_line_length, max_line_gap)
        return lines
    
    def _canny(self, img):
        """
        Applies Canny edge detection to the input image.
        
        Parameters:
            img (ndarray): Input image (BGR format).

        Returns:
            ndarray: Edges detected in the image.
        """
        gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray_img,(self.kernel_size, self.kernel_size),0)
        edges = cv2.Canny(blur_gray, self.low_threshold, self.high_threshold)
        return edges
    
    def _mask_canny(self, canny, region_of_interest):
        """
        Masks the Canny edges image to focus on the region of interest.

        Parameters:
            canny (ndarray): Canny edges image.
            region_of_interest (tuple): Triangle region defined by three vertices.

        Returns:
            ndarray: Masked edges image.
        """
        mask = np.zeros_like(canny)   
        ignore_mask_color = 255
        vertices = np.array([[region_of_interest[0], region_of_interest[1], region_of_interest[2],]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        masked_canny = cv2.bitwise_and(canny, mask)
        return masked_canny
    
    def _average_lines(self, lines):
        """
        Averages the detected line segments to estimate left and right lanes.
        
        Parameters:
            lines (list): List of detected line segments [x1, y1, x2, y2].

        Returns:
            tuple: (left_lane, right_lane) where each lane is a tuple (slope, intercept).
                   Returns None for a lane if no lines are found for it.
        """
        left_lines = []
        left_lengths = []
        right_lines = []
        right_lengths = []
        
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y2 - slope*x2
                length = np.sqrt( (y2 - y1)**2 + (x2 - x1)**2)
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_lengths.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_lengths.append((length))
        left_lane  = np.dot(left_lengths,  left_lines) / np.sum(left_lengths)  if len(left_lengths) > 0 else None
        right_lane = np.dot(right_lengths, right_lines) / np.sum(right_lengths) if len(right_lengths) > 0 else None
        return left_lane, right_lane
    
    def _find_line_points(self, y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        x1 = (y1 - intercept)/slope
        x2 = (y2 - intercept)/slope
        
        return ((x1, y1), (x2, y2))
    
    def _create_lane_lines(self, img, lines):
        left_lane, right_lane = self._average_lines(lines)
        y1 = img.shape[0]
        y2 = 0.65*y1
        
        left_line  = self._find_line_points(y1, y2, left_lane)
        right_line = self._find_line_points(y1, y2, right_lane)
        return left_line, right_line
    
    
    def _draw_lane_lines(self, img, lines, color=[255, 0, 0], thickness=10):
        line_image = np.zeros_like(img)
        for line in lines:
            if line is None:  # Skip if no line
                continue
            pt1, pt2 = line
            if pt1 is None or pt2 is None:  # Check for invalid points
                continue
            # Convert points to integers
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            cv2.line(line_image, pt1, pt2, color, thickness)
        return cv2.addWeighted(img, 1.0, line_image, 1.0, 0.0)
    # def _draw_lane_lines(self, img, lines, color=[255,0,0], thickness=10):
    #     line_image = np.zeros_like(img)
    #     for line in lines:
    #         cv2.line(line_image, *line, color, thickness)
    #     return cv2.addWeighted(img, 1.0, line_image, 1.0, 0.0)