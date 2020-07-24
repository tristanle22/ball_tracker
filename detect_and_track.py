import numpy as np
import cv2
import json
import time

class DetectAndTrack:
  """ 
  An object detection + tracking pipeline using YOLO v3

  Atributes
  ---------
  cap: cv2.VideoCapture
    The video capture which contains the object to track
  net: cv2.dnn.Net
    The pre-trained model used for inference
  classes: list
    The list of the model's class names
  cur_bbox: list
    The list that describe the bounding box of the object: [left, right, width, height]
  objectness_threshold: float
    The objectness threshold. Describe how likely a region contains an object
  confidence_threshold: float
    The confidence threshold. Describe how likely an object belongs to the chosen class
  nms_threshold: float
    The Non-maxima suppression threshold. Used for removing overlapping bounding boxes
  input_width: int
    The width of the input for inference
  input_height: int
    The height of the input for inference
  output_layers_names: list
    The list contains the network's output layers' names
  object_class_id: int
    The index of the object class to be detected (default 32 'sports_ball')
  lost_track_duration_threshold: int
    The threshold for allowable number of frames which the object couldn't be tracked
  tracker_type: str
    The type of tracker to use. More info here: https://docs.opencv.org/3.4/d9/df8/group__tracking.html
  max_tracking_duration: int
    The max number of frames to track before the object has to be detected again
  tracker_box_color: tuple
    The color in BGR of bounding boxes created by the tracker
  detector_box_color: tuple
    The color in BGR of bounding boxes created by the detector
  
  Methods
  -------
  update_config(config_file)
    Update the attributes with the settings from the configuration file

  create_tracker()
    Create a tracker based on self.tracker_type

  get_output_names()
    Get the network's output layers' name

  draw_predictions(frame, conf, draw_detection)
    Draw the bounding box obtained from the detector/tracker

  postprocess(frame, outs)
    Extract the bounding box and confidence score of the object from inference outputs
  
  detect(frame)
    Detect the object in a frame

  track(frame, tracker)
    Track the object in a frame
  """

  def __init__(self, config_file, video_capture):
    """
    Parameters
    ----------
    config_file: str
      The path to the configuration file
    video_capture: cv2.VideoCapture
      The video capture that contains the object 
    """
    self.cap = video_capture
    self.net = None
    self.classes = []
    self.cur_bbox = [0,0,0,0] # [left, right, width, height]

    # Detection settings
    self.objectness_threshold = 0.5
    self.confidence_threshold = 0.90
    self.nms_threshold = 0.4
    self.input_width = 416
    self.input_height = 416
    self.output_layers_names = []
    self.object_class_id = 32

    # Tracking settings
    self.lost_track_duration_threshold = 10
    self.tracker_type = ""
    self.max_tracking_duration = 90
    
    # Annotation settings
    self.tracker_box_color = (178,255,50)
    self.detector_box_color = (255,178,50)

    # Update settings
    self.update_config(config_file)

  def update_config(self, config_file):
    """Update the attributes with the settings found in the configuration file.

    The configuration file MUST be a json file and follow the pre-existing format

    Parameters
    ----------
    config_file:
      Path to the configuration file
    """

    with open(config_file,'r') as f:
      settings = json.load(f)

      model_configuration = settings['detector_settings']['model']['config_file']
      model_weights = settings['detector_settings']['model']['weight_file']
      class_file = settings['detector_settings']['model']['class_file']
      self.net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
      with open(class_file, 'rt') as f:
        self.classes = f.read().rstrip('\n').split('\n')
      
      self.objectness_threshold = settings['detector_settings']['objectness_threshold']
      self.confidence_threshold = settings['detector_settings']['confidence_threshold']
      self.nms_threshold = settings['detector_settings']['nms_threshold']
      self.input_width = settings['detector_settings']['input_width']
      self.input_height = settings['detector_settings']['input_height']
      self.object_class_id = settings['detector_settings']['object_class_id']

      self.tracker_type = settings['tracker_settings']['tracker_type']
      self.lost_track_duration_threshold = settings['tracker_settings']['lost_track_duration_threshold']
      self.max_tracking_duration = settings['tracker_settings']['max_tracking_duration']  
    
    self.get_output_names()

  def create_tracker(self):
    """ Create an OpenCV tracker object.

    Returns:
    --------
    bool
      The flag that indicates whether the tracker was created successfully
    cv2.Tracker
      The OpenCV tracker
    """
    if self.tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
        ok = True
    elif self.tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
        ok = True
    elif self.tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
        ok = True
    elif self.tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
        ok = True
    elif self.tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
        ok = True
    elif self.tracker_type == 'GOTURN': #TODO
        tracker = cv2.TrackerGOTURN_create()
        ok = True
    elif self.tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
        ok = True
    elif self.tracker_type == "MOSSE":
        tracker = cv2.TrackerMOSSE_create()
        ok = True
    else:
        tracker = None
        ok = False
    return (ok, tracker)

  def get_output_names(self):
    """ Obtain the network's output layers' name and store it in self.output_layers_names

    This information is required for inference.
    """
    
    layer_names = self.net.getLayerNames()
    self.output_layers_names = [layer_names[i[0]-1] for i in self.net.getUnconnectedOutLayers()]

  def draw_predictions(self, frame, conf, draw_detection):
    """ Draw the bounding boxes and annotate the object

    Parameters
    ----------
    frame: np.ndarray
      The frame to be drawn on
    conf: float
      The confidence score of how likely the object belongs to the selected class
    draw_detection: bool
      The flag that indicates to draw the detector or tracker's bounding box
    """

    if draw_detection:
      cv2.rectangle(frame, 
                    pt1=(self.cur_bbox[0], self.cur_bbox[1]), 
                    pt2=(self.cur_bbox[0]+self.cur_bbox[2], self.cur_bbox[1]+self.cur_bbox[3]), 
                    color=self.detector_box_color, 
                    thickness=3)
      if conf > 0:
        label = '%.2f' % conf
        label = '{}:{}'.format(self.classes[self.object_class_id],label)
      else:
        label = 'Cannot detect object'
    else:
      cv2.rectangle(frame, 
                    pt1=(self.cur_bbox[0], self.cur_bbox[1]), 
                    pt2=(self.cur_bbox[0]+self.cur_bbox[2], self.cur_bbox[1]+self.cur_bbox[3]), 
                    color=self.tracker_box_color, 
                    thickness=3)
      if conf > 0:
        label = self.classes[self.object_class_id]
      else:
        label = 'Fail to track object'    

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, thickness=1)
    top = max(self.cur_bbox[1], label_size[1])
    cv2.rectangle(frame, pt1=(self.cur_bbox[0], top - round(1.5*label_size[1])), 
                         pt2=(self.cur_bbox[0] + round(1.5*label_size[0]), top + base_line), 
                         color=(255, 255, 255), 
                         thickness=-1)
    cv2.putText(frame, label, (self.cur_bbox[0], top), cv2.FONT_HERSHEY_DUPLEX, 
                fontScale=0.75,
                color=(0,0,0),
                thickness=1)

  def postprocess(self, frame, outs):
    """ Extract the most confident bounding box and
        associated confidence from the inference's output. Perform Non-maxima suppression to remove overlaps.

    Parameters
    ----------
    frame: numpy.ndarray
      The frame used for obtaining dimensions, which is used to scale the bounding box's whereabout
    outs: list
      The inference output. Contains 3 layers of different scales. 
      Each layer is a list of bounding boxes which has confidence score corresponds to each and very
      single id in the class

    Returns
    -------
    bool
      The flag which indicates whether the desired object was found or not
    float
      The confidence score of how likely the detected object belongs to the desired class
    """

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    confidences = []
    boxes = []
    object_class_offset = 5

    outs = np.concatenate(outs, axis=0)
    detections = outs[outs[:,4] > self.objectness_threshold]
    detections = detections[detections[:,self.object_class_id + object_class_offset] > self.confidence_threshold]

    for bbox in detections:
      center_x = int(bbox[0] * frameWidth)
      center_y = int(bbox[1] * frameHeight)
      width = int(bbox[2] * frameWidth)
      height = int(bbox[3] * frameHeight)
      left = int(center_x - width / 2)
      bottom = int(center_y - height / 2)
      confidences.append(float(bbox[self.object_class_id + object_class_offset]))
      boxes.append([left, bottom, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.confidence_threshold)
    if len(confidences) == 0 and len(indices) == 0:
      self.cur_bbox = [0,0,0,0]
      return (False, 0)

    max_conf_index = confidences.index(max(confidences))
    self.cur_bbox = boxes[indices[max_conf_index, 0]]
    return (True, confidences[max_conf_index])

  def detect(self, frame):
    """ Detect the object in a frame using inference.

    Parameters
    ----------
    frame: numpy.ndarray
      The frame that might/might not contain the object
    
    Returns
    -------
    tuple
      Returns the output of postprocess()  
    """

    blob = cv2.dnn.blobFromImage(frame, 
                                 scalefactor=1/255,
                                 size=(self.input_width,self.input_height),
                                 mean=[0,0,0],
                                 swapRB=1,
                                 crop=False)
    self.net.setInput(blob)
    outs = self.net.forward(self.output_layers_names)

    return self.postprocess(frame, outs)

  def track(self, frame, tracker):
    """ Track the object in a frame. 

    Save the object's bounding box into the class attribute self.cur_bbox

    Parameters
    ----------
    frame: numpy.ndarray
      The frame in which new location of the object is to be determined
    tracker: cv2.Tracker
      The OpenCV tracker to be used for finding the object
    
    Returns
    -------
    bool
      True of the object was found successfully, False otherwise.
    """

    ok, bbox = tracker.update(frame)
    if not ok:
      self.cur_bbox = [0,0,0,0]
      return False

    self.cur_bbox = [round(x) for x in bbox]
    return True
  
  def detect_and_track(self):
    """ The implementation of the detection-and-track pipeline.

    The flow is as follow: Detect object, then track the object until tracking fails or the tracking duration ends.
    Detect the object again and repeat. Execution ends when run out of frames or when user hit ESC.

    Returns
    -------
    bool
      True if the user ends execution or run out of frames to track. False if fail to create/initialize tracker
    """
    
    lost_track_count = 0
    while True:
      ret, frame = self.cap.read()
      if not ret:
        print("Fail to read first frame!")
        return True

      # start = time.time()
      detected, conf = self.detect(frame)
      # end = time.time()
      # print("Detection run time: {} ms".format(1000*(end-start)))

      self.draw_predictions(frame, conf, True)
      cv2.imshow("Frame", frame)
      if cv2.waitKey(0) & 0xFF == 27:
        return True

      if not detected:
        print("Fail to detect object!")
      else:
        # Create new tracker
        create_state, tracker = self.create_tracker()
        if not create_state:
          print("Fail to create tracker!")
          return False

        # Initialize the new tracker
        init_status = tracker.init(frame, tuple(self.cur_bbox))
        if not init_status:
          print("Fail to initialize new tracker!")
          return False

        for i in range(90):
          track_confidence = 0
          ret, frame = self.cap.read()
          if not ret:
            print("Fail to read frame to track!")
            return True

          track_ok = self.track(frame, tracker)
          if not track_ok:
            track_confidence = -1
            lost_track_count += 1
            if lost_track_count > self.lost_track_duration_threshold:
              print("Lost track of the object. Cannot recover!")
              break
          else:
            track_confidence = 100
            lost_track_count = 0

          self.draw_predictions(frame, track_confidence, False)
          cv2.imshow("Frame", frame)
          if cv2.waitKey(0) & 0xFF == 27:
            return True

    return True
    