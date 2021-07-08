# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:53:30 2021

@author: ahall
"""

import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd


SOURCE_DIR = "..\\processed_data\\LABELLED"
DEST_DIR = "..\\processed_data\\LINE_DETECTION"

#image processing functions
def preprocess (img_array):
    blur_image = False
    dilate_image = False
    edge_detection = "threshold"
    
    #convolutional blurring (optional)
    
    
    if(blur_image == True):
        blurred = cv2.medianBlur(img_array,3)
        
        
    else:
        blurred = img_array
        
    #erode image
    if(dilate_image == True):
        kernel = np.ones((2,2), np.uint8)
        blurred = cv2.dilate(blurred, kernel, iterations=1)


    if(edge_detection == "canny"):
        edges = cv2.Canny(blurred,75,150) #edge filter
        
    else:
        edges = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #invert the image
        edges = abs(255-edges[1]) #the edges object output above includes the threshold figure - ned to take the image array only
    
    return edges


def get_lines(img_array, img_name):
   
    
    top_lines = cv2.HoughLinesP(img_array,rho=1,theta=np.pi/60,threshold=10, minLineLength=4, maxLineGap=2) 
    #top_lines = cv2.HoughLines(img_array, 1, np.pi/180,200)
    
    
    df = pd.DataFrame()
    
    #draw the lines
    if(top_lines is not None):
        for line in top_lines:
            for x1,y1,x2,y2 in line:
                data = pd.DataFrame.from_records({"image":img_name, "x1":x1,"y1":y1,"x2":x2,"y2":y2}, index=[0])
                df = pd.concat([df,data], axis=0)
                df = df.reset_index()
                del df['index']

                #lines_array.append( np.array([x1,y1,x2,y2]))


    return df


if __name__ == "__main__":

    lines_array = pd.DataFrame()

    for image in os.listdir(SOURCE_DIR):
    
        img = cv2.imread(os.path.join(SOURCE_DIR,image), cv2.IMREAD_GRAYSCALE)
        
        edges = preprocess(img)
        
        image_lines = get_lines(edges, image)
        lines_array = pd.concat([lines_array,image_lines], axis=0)
        
        
        results_array = np.ones([256, 256])
        results_array = results_array*255
        
        for line in range(len(image_lines)):
            cv2.line(results_array, (image_lines['x1'][line], image_lines['y1'][line]), (image_lines['x2'][line], image_lines['y2'][line]) , 0,1)
        
        
        cv2.imwrite(os.path.join(DEST_DIR,image),results_array)
        
    lines_array.to_csv("line_points.csv")
        
