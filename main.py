import matplotlib.image as mpimg
from CannyHoughDetector import CannyHoughDetector
from SlidingWindowDetector import SlidingWindowDetector
from utils import load_config
import cv2


def main():
    config = load_config("config.yml")
    
    lane_line_detector = SlidingWindowDetector(config)
    # lane_line_detector = CannyHoughDetector(config)
    video_path = config['test_video']['path']
    output_path = config['test_video'].get('output', None)
    show_video = config['test_video'].get('show_video', True)
    
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  
        frame = lane_line_detector.detect(frame)
        if show_video:
            cv2.imshow('Lane Line Detection', frame)
            # cv2.waitKey(0)
        if output_path:
            out_video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    if output_path:
        out_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
