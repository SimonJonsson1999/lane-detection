import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_config(config_path="config.yml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def display(image, name="test"):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_points_on_image(image, points, name="test"):
    polygon_points_np = np.array(points, np.int32)
    polygon_points_np = polygon_points_np.reshape((-1, 1, 2))
    cv2.polylines(image, [polygon_points_np], isClosed=True, color=(0, 0, 255), thickness=20)

    cv2.namedWindow("Polygon", cv2.WINDOW_NORMAL)
    cv2.imshow("Polygon", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_histogram(histogram, name="Histogram Visualization"):
    plt.figure(figsize=(10, 5))
    plt.plot(histogram, label="Histogram")
    plt.xlabel("Column Index")
    plt.ylabel("Sum of Values")
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.show()

def fill_lane(img, left_points, right_points, color=[0,255,0]):
    pts = np.hstack((left_points, right_points))
    # img = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    cv2.fillPoly(img, np.int_([pts]), color)
    return img

def draw_lane_through_points(img, pts, color=[0, 0, 255], thickness=10):
    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    return cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)

