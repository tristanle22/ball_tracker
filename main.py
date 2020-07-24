import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from detect_and_track import DetectAndTrack

def main():
  parser = argparse.ArgumentParser(description='Object detection + tracking using YOLO in OpenCV')
  parser.add_argument('--image', help='Path to image file.')
  parser.add_argument('--video', help='Path to video file.')
  args = parser.parse_args()

  if (args.image):
    if not os.path.isfile(args.image):
      print("Input image file ", args.image, " doesn't exist")
      return
    cap = cv2.VideoCapture(args.image)
  elif (args.video):
    if not os.path.isfile(args.video):
      print("Input video file ", args.video, " doesn't exist")
      return
    cap = cv2.VideoCapture(args.video)
  else:
    cap = cv2.VideoCapture(0)

  config_file = "./detect_and_track_config.json"
  tracker = DetectAndTrack(config_file, cap)
  tracker.detect_and_track()

if __name__ == '__main__':
  main()
