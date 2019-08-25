import matplotlib
#%matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches as patches 
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import os
import os.path
import datetime
from imutils.video import VideoStream
import argparse
import imutils
import time
import pickle
from os.path import isfile, join
from skimage import io
import skimage
import imageio


def read_video():
    """
    This function takes the first sample video of the worm moving and reads it using cv2.VideoCapture.
    
    Arguments
    --------
    No arguments are taken: the function itself includes a path to the video, in this case w060.avi
    
    Returns
    ------
    It returns nothing as well but if implemented in a function it will be able to read the video
    
    """
    #read the video
    path2video = 'w060.avi'
    # read video
    video = cv2.VideoCapture(path2video)
    return video

def tracker_pickling(fileroi, slicedfile):
    """
    Uses MIL tracking to automate creation of an roi with the worm video and saves those positions
    onto a pickled list.
    
    ----------
    Arguments: none
    
    ----------
    Output: pickled list
    """
    file = fileroi
    roi_list = []
    sliced_roi_list = []
    if os.path.exists(file):
        with open(fileroi,"rb") as al:
            roi_list = pickle.load() 
    tracker = cv2.TrackerMIL_create()
    tracker_name = str(tracker).split()[0][1:] 
    
    # Read video
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("w060.avi")
    
    # Read first frame.
    ret, frame = cap.read()
    
    # Special function allows us to draw on the very first frame our desired ROI
    roi = cv2.selectROI(frame, False)
    
    # Initialize tracker with first frame and bounding box
    ret = tracker.init(frame, roi)
    
    while True:
        # Read a new frame
        ret, frame = cap.read()
        
        # Update tracker
        success, roi = tracker.update(frame)
        
        # roi variable is a tuple of 4 floats
        # We need each value and we need them as integers
        (x,y,w,h) = tuple(map(int,roi))
        roi_list.append((x,y,w,h))

        sliced_roi_list.append(((slice(y,h)), (slice(x,w))))
        
        # Draw Rectangle as Tracker moves
        if ret:
            # Tracking success
            p1 = (x, y)
            p2 = (x+w, y+h)
            cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        else :
            # Tracking failure
            cv2.putText(frame, "Failure to Detect Tracking!!", (100,200), \
                        cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
    
        # Display tracker type on frame
        cv2.putText(frame, tracker_name, (20,400), cv2.FONT_HERSHEY_SIMPLEX, \
                    1, (0,255,0),3);
    
        # Display result
        cv2.imshow(tracker_name, frame)
    
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 :
            break
            
    cap.release()
    cv2.destroyAllWindows()    
    with open(fileroi,"wb") as al:
        pickle.dump(roi_list,al)
    with open(slicedfile,"wb") as sliced:
        pickle.dump(sliced_roi_list,sliced)
    
        
def unpickled():   
    """
    This function allows users to see and define the plotted region of interest
    coordinates that have been plotted using plot_images()
    
    Arguments
    --------
    None
    
    Returns 
    -------
    roi_list: the pickled list of roi coordinates
    """
    pickle_off = open("mil_tracked","rb")
    roi_list = pickle.load(pickle_off)
    return roi_list

def sliced_unpickler(file):
    """
    This function allows users to see the plotted roi coordinates plotted using
    mil_tracker. This utilzes a pickled list of slice objects to more accurately
    replicate previous solutions that worked.
    
    Arguments
    ----------
    None
    
    Returns
    ---------
    sliced_roi: pickled list of roi coordinates in slice form
    """
    pickle_off = open(file,"rb")
    roi_list = pickle.load(pickle_off)
    return roi_list
    
    

def runthrough_mil_tracked(end_frames):
   video = cv2.VideoCapture("w060.avi")
   for i in range (end_frames):
        fig, ax = plt.subplots()
        frame = video.read()[1][:,:,1]
        ax.imshow(frame[sliced_unpickler()[i]])
        plt.ginput(10000, timeout=0, show_clicks=False)
        plt.close()