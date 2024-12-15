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

