import numpy as np
import cv2

class BaseDetector(object):

    def detect(self, img):
        mask = self.pre_process(img)
        lines = self.get_lines(img, mask)
        filled_lane, line_image = self.add_lines(img, lines)
        return line_image, filled_lane, mask
    
    def pre_process(self, img):
        raise NotImplementedError
    
    def get_lines(self, img, mask):
        raise NotImplementedError
    
    def add_lines(self, img, lines):
        raise NotImplementedError
    
    def _draw_lane_lines(self, img, lines, color=[255, 0, 0], thickness=10):
        """
        Draws the detected lane lines on the image.

        Parameters:
            img (ndarray): Input image.
            lines (tuple): Left and right lane lines.
            color (list): RGB color for the lines (default is blue).
            thickness (int): Line thickness (default is 10).

        Returns:
            ndarray: Image with drawn lane lines.
        """
        line_image = np.zeros_like(img)
        for line in lines:
            if line is None: 
                continue
            pt1, pt2 = line
            if pt1 is None or pt2 is None:  
                continue
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            cv2.line(line_image, pt1, pt2, color, thickness)
        return cv2.addWeighted(img, 1.0, line_image, 1.0, 0.0)