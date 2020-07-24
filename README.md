## Project title
Ball Tracker, a ball detection and tracking pipeline using a pre-trained YOLO v3 object detector and OpenCV Tracking API

## Motivation
I build this project for learning computer vision and also for fun. Building a detection and tracking pipeline from scratch
is something I have always wanted to do since I started studying OpenCV. Also, I feel like there's something really profound
in the act of drawing bounding boxes on images, sort of like a technology that comes out of a sci-fi or video games.

## Tech/framework used
- Python
- OpenCV
- Darknet YOLO v3

## Features
- Accurate object detection using YOLO v3.
- Fast and optimized tracking with OpenCV Tracking API
- Ease of customizing detection/tracking settings
- Easier to run

## How to use?
First and foremost, download the [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and put in in the `models/` folder.
Then simply run `python3 main.py [flag] [path_to_video]`

## Limitations
- Inference takes too long, resulting in jagged video feed. This is because I'm using a detector that's built for 80 classes to detect only 1 class.
- Lacking the ability to write and save the video that contains the bounding boxes. 
- Lacking the ability to detect and track multiple objects

## What's next?
- Build my own neural net for object detection.
- Implement a Kalman filter to track the detected object and compare the performance with OpenCV Tracking API, cuz why not?
- Implement multi-object detection and tracking.

## Credits
Big thanks to @spmallick, @jrosebr1 for providing great learning resources. Huge credit to @pjreddie for his work on the YOLO v3 object detector.
The YOLO v3 model used in this project can be found [here](https://pjreddie.com/darknet/yolo/)
