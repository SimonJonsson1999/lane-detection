# Lane-Line Detector

## Project Description
A lane-line detector is essential for self-driving cars and serves as an excellent way to learn and showcase computer vision techniques. In this project, I explore different methods to detect lane lines. The techniques implemented include:

### Detection Methods
1. **CannyHoughDetector**  
    - Uses the Canny edge algorithm to detect edges, followed by a transformation into Hough space to identify straight lines.  
    - Performs reasonably well for straight driving but struggles with curved roads.

2. **SlidingWindowDetector**  
    - Applies thresholding and performs a perspective transform on the region-of-interest (ROI).  
    - Uses a sliding window to detect lane lines. Points on the lines are then used to fit a polynomial describing the lane-line, allowing detection of curves.  
    - Lane lines are detected in the transformed space and later mapped back to pixel space, where they are displayed on the original image.

-
## TODO
- [x] Finish basic lane-line detection algorithm
- [x] Create the video and save it instead of showing at run time
- [x] Calculate one line from all hough lines
- [ ] Perspective transform and sliding window approach (https://www.youtube.com/watch?v=ApYo6tXcjjQ&t=214s)
- [ ] Make predicted line lengths shorter when necessary (curves)
- [ ] Add Car detector
- [ ] Find the drivable field as the area between the lane-lines
- [ ] Find data-set for deep-learning approach
- [ ] Implement deep-learning approach
- [ ] Calculate FPS when running algortihm



## Installation
Step-by-step instructions on how to set up the lane-line detector locally:
1. Clone the repository: `git clone git@github.com:SimonJonsson1999/lane-detection.git`
2. Navigate to the project directory: `cd lane-line-detector`
3. Install dependencies: `pip install -r requirements.txt`

## Usage
Details on how to use the lane-line detector effectively:
1. Prepare your input data (images or video files).
2. Run the detection script: `python main.py`

## Results


