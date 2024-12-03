
import matplotlib.image as mpimg
from lane_detection import LaneLineDetector
import cv2


def main():

    lane_line_detector = LaneLineDetector()
    video_path = 'test_videos/solidYellowLeft.mp4'
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Error: Could not open video.")
        return
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break  # Exit loop when video ends or fails to read

        # Detect lane lines on the current frame
        frame = lane_line_detector.detect(frame)

        # Display the result
        cv2.imshow('Lane Line Detection', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()