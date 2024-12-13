
import matplotlib.image as mpimg
from lane_detection import LaneLineDetector
from utils import load_config
import cv2


def main():
    config = load_config("config.yml")
    lane_line_detector = LaneLineDetector(config)
    video_path = config['test_video']['path']
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  
        frame = lane_line_detector.detect(frame)
        cv2.imshow('Lane Line Detection', frame)
        cv2.waitKey(500)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()