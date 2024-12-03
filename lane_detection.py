import numpy as np
import cv2
import matplotlib.pyplot as plt


class LaneLineDetector(object):
    def __init__(self):
        self.kernel_size = 5
        self.low_threshold = 180
        self.high_threshold = 240
        self.apex_factor = 0.9

    def detect(self, img):
        left_bottom, right_bottom, apex = self._get_region_of_interest(img)
        gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray_img,(self.kernel_size, self.kernel_size),0)
        edges = cv2.Canny(blur_gray, self.low_threshold, self.high_threshold)

        mask = np.zeros_like(edges)   
        ignore_mask_color = 255
        vertices = np.array([[left_bottom,apex, right_bottom,]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 2     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 4 #minimum number of pixels making up a line
        max_line_gap = 5    # maximum gap in pixels between connectable line segments
        line_image = np.copy(img)*0 # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges)) 

        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

        return lines_edges

    def _get_region_of_interest(self, img):
        x_size = img.shape[1]
        y_size = img.shape[0]
        left_bottom = (0, y_size)
        right_bottom = (x_size, y_size)
        apex = (x_size/2, y_size/(2 * self.apex_factor))
        return left_bottom, right_bottom, apex
    
    def _detect_line(self):
        pass
