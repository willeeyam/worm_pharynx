import matplotlib
%matplotlib
from matplotlib import pyplot as plt
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
    path2video = '../data/w060.avi'
    # read video
    video = cv2.VideoCapture(path2video)
    return video
    
def zoom2roi(ax):
    """
    Identify coordinates of zoomed-in/moved axis in interactive mode

    Arguments
    ---------
    ax: matplotlib axis
        zoomed-in/moved axis

    Returns
    ---------
    zoom_coords: tuple of slice objects
        coordinates of zoomed in box, use as im[zoom_coords]

    """    
    # get x and y limits
    xlim = [int(x) for x in ax.get_xlim()]
    ylim = [int(y) for y in ax.get_ylim()]

    # make and return slice objects
    return (slice(ylim[1],ylim[0]), slice(xlim[0],xlim[1]))
    

#Plots the images
def plot_images(start_num, num_Frames):
    """
    This function allows users to manually plot coordinates that will 
    zoom in to regions of interest. Then these selected 
    roi_coordinates will then be saved into a pickle. Previous frames 
    that have had new roi_coordinates already set will be overridden
    
    Arguments
    --------
    start_num: the frame at which the plotting begins
    num_Frames: the number of frames that will be plotted
    
    Returns
    -------
    pickled file 'roi_list'
    """
    %matplotlib 
    counter = 0
   #Loops the video to the frame that is desired
    while(counter < start_num-1):
        frame = read_video().read()[1][:,:,0]
        counter +=1
    #Pickling List
    file = "roi_list"
    roi_list = []
    if os.path.exists(file):
        with open("roi_list","rb") as al:
            roi_list = pickle.load(al)
        # Find number of frames in video
        while(counter < num_Frames):
            frame = read_video().read()[1][:,:,0]
            fig, ax = plt.subplots()
            ax.imshow(frame)
            #save_frames(frame)
            plt.ginput(10000, timeout=0, show_clicks=False)
            roicoords = zoom2roi(ax)
            if(counter in range (len(roi_list))):
                roi_list[counter] = roicoords
            else:
                roi_list.append(roicoords)
            #save_roicoords(roicoords)
            #roi_frame = np.array(plt.imshow(frame[roicoords]))
            counter+= 1
            plt.close()      
    with open("roi_list","wb") as al:
        pickle.dump(roi_list,al)

       # with open(file,'rb') as al:
        #    roi_list = pickle.load(al)
        
#Unpickles the Coordinate List
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
    pickle_off = open("roi_list","rb")
    roi_list = pickle.load(pickle_off)
    return roi_list
    
#Tracker for Pickled List
def pickle_tracker():
    """
    This function simply takes the unpickled() function and makes it 
    easier to display, giving the user information on coordinate numbers
    
    Arguments 
    -------
    None
    
    Returns
    ------
    ROI Coord Number:[i](ycoord1,ycoord2,None)(xcoord1,xcoord2,None)
    """
    unpickled()
    tracker_count = 0
    return_statement = []
    for i in range (tracker_count, len(unpickled())  ):
        tracker_count = tracker_count + 1
        return_statement.append(("ROI Coord Number: " + str(tracker_count) + " "+ str(unpickled()[i])))
    return return_statement
    
#Plots the Regions of Interest
def plot_roi(start_frames, end_frames):
    """
    This function plots images from the video, resized so they fit within the ROI coordinates
    
    Arguments
    -------
    start_frames: which frame the function starts displaying first
    end_frames: which frame the function ends
    
    Returns
    ------
    Plots of frames that can be navigated using alt-click
    """
    #reads up until start_frames if start_frames !=0
    video = read_video()
    for i in range (start_frames, end_frames):
        fig, ax = plt.subplots()
        frame = video.read()[1][:,:,1]
        ax.imshow(frame[remodeled_roi_list()[i]])
        plt.ginput(10000, timeout=0, show_clicks=False)
        plt.close()
        
def find_max_frames():
    """
    This function runs through all the ROI coordinates and it finds 
    the maximum x-length and y-length present throughout all the frames
    
    Argumetns
    -------
    None
    
    Returns
    ------
    max_xvalue: maximum x-size distance-wise
    max_yvalue: maximum y-size distance-wise
    """
    max_yvalue = -1
    max_xvalue = -1
    for i in range (0,len(unpickled())):
        if((unpickled()[i][1].stop - unpickled()[i][1].start) >= max_xvalue):
            max_xvalue = unpickled()[i][1].stop - unpickled()[i][1].start

        else:
            max_xvalue = max_xvalue

        if((unpickled()[i][0].stop - unpickled()[i][0].start) >= max_yvalue):
            max_yvalue = unpickled()[i][0].stop - unpickled()[i][0].start
            
        else:
            max_yvalue = max_yvalue

    return max_xvalue, max_yvalue
    
def resize_frames(num_frames):
    """
    Takes the array from ROI_list and then modifies each coordinate value such that all of the coordinates are 
    adjusted to ensure every frame in the list is the same shape. This is done by comparing the difference between 
    larget frame's x and y dimensions with the x and y dimensions of every frame.
    
    Arguments
    --------
    num_frames: number of frames that the resize function will affect, usually is len(roi_list)
    
    Returns
    -------
    A pickled list named 'remodeled_roi_list' 
    """
    counter = 0
    x_standard, y_standard = find_max_frames()
    # creates a new list to store the true roi_coords in
    file = "remodeled_roi_list"
    remodeled_roi_list = []
    if os.path.exists(file):
        with open("remodeled_roi_list","rb") as al:
            remodeled_roi_list = pickle.load(al)

    for i in range (0,num_frames):
        if(unpickled()[i][1].stop-unpickled()[i][1].start!=x_standard): ##Check first sign...see if it's pickled_roi_list[i][1].start+ or - the rest
            new_xval = x_standard-(unpickled()[i][1].stop - unpickled()[i][1].start)
        if(unpickled()[i][0].stop-unpickled()[i][0].start!=y_standard):
            new_yval = y_standard-(unpickled()[i][0].stop-unpickled()[i][0].start)
             # make and return slice objects
        counter = counter
        if(counter in range (len(remodeled_roi_list))):
            remodeled_roi_list[counter]= (slice(unpickled()[i][0].start, unpickled()[i][0].stop+new_yval), slice(unpickled()[i][1].start, unpickled()[i][1].stop + new_xval))
        else:
            remodeled_roi_list.append((slice(unpickled()[i][0].start, unpickled()[i][0].stop+new_yval), slice(unpickled()[i][1].start, unpickled()[i][1].stop + new_xval)))
    with open("remodeled_roi_list","wb") as al:
        pickle.dump(remodeled_roi_list,al)

def remodeled_roi_list():
    """
    This function calls the pickle opening function and assigns the variable remodeled_roi_list
    to equal the pickled file "remodeled_roi_list"
    
    Arguments
    -------
    None
    
    Returns
    ------
    remodeled_roi_list = ["remodeled_roi_list"]
    """
    with open("remodeled_roi_list","rb") as al:
        remodeled_roi_list = pickle.load(al)
    return remodeled_roi_list

#Tracker for Pickled List
def remodeled_pickle_tracker():
    """
    This function returns a list of ROI coordinates that have already been resized so that 
    each frame is the same shape in a string value. 
    
    Arguments
    --------
    None
    
    Returns
    -------
    "ROI Coord Number:" +[i] + remodeled_roi_list[i]
    """
    pickle_off = open("remodeled_roi_list","rb")
    remodeled_roi_list = pickle.load(pickle_off)
    tracker_count = 0
    return_statement = []
    for i in range (tracker_count, len(remodeled_roi_list)):
        tracker_count = tracker_count + 1
        return_statement.append(("ROI Coord Number: " + str(tracker_count) + " "+ str(remodeled_roi_list[i])))
    return return_statement
    
def save_array(num_frames):
    """
    This function saves the slice objects collected from the remodeled
    roi list and converts them to an array of images that can be then 
    concatenated to a stack
    
    Arguments
    --------
    num_frames: number of frames it saves
    remodeled_roi_list
    
    Returns
    -------
    Array of images
    """
    remodeled_array=[]
    frame = read_video().read()[1][:,:,1]


    for i in range (num_frames):
        frame = read_video().read()[1][:,:,1]
        remodeled_array = frame[remodeled_roi_list()[i]]
    return remodeled_array
    
plot_roi(30)





