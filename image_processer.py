# -*- coding: utf-8 -*-
"""
Script to apply filter mask to images, slice and augment them; then save them into training and testing sets
"""

import numpy as np
import cv2
import os
import random
from PIL import Image
import glob
from skimage import morphology
# from shutil import copyfile

random.seed(57)

X_SIZE = 256 # 120
Y_SIZE = 256 # 90
SOURCE_DIR = "C:\\Users\\ahall\\Documents\\projects\\clast-analysis\\raw_data\\LABELLED"  # noqa E501
TEMP_DIR = "C:\\Users\\ahall\\Documents\\projects\\clast-analysis\\processed_data\\temp"
DEST_DIR = "C:\\Users\\ahall\\Documents\\projects\\clast-analysis\\processed_data\\LABELLED"  # noqa E501


REMOVE_SMALL_PORES = False
MIN_PORE_SIZE = 30
ROTATE_IMGS = False
TEST_PROP = 1
FILTER_COLOR = "black"


#%%




def remove_small_pores(image_mask, lower_pore_limit):
    """removes small pwoers from binary mask"""

    filtered_img = morphology.remove_small_holes(image_mask, area_threshold = lower_pore_limit)
    return filtered_img



# returns the binary image
def convert_to_binary(input_file, size_x, size_y):
    input_img = cv2.imread(input_file)
    #input_img = cv2.resize(input_img, (size_x, size_y))

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    
    if(FILTER_COLOR == 'black'):
    #black mask
        mask = cv2.inRange(hsv, (0, 0, 0), (255, 255, 100))
        
    if(FILTER_COLOR == 'none'):
        # blue mask
        return input_img
        
    
    
    inv_mask = cv2.bitwise_not(mask)

    #inv_mask = mask <= 0
    #blue = np.zeros_like(input_img, np.uint8)
    #blue[inv_mask] = input_img[inv_mask]


    #thresh = 64
    #grey = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    #binary = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)[1]

    return inv_mask


def slice_images(
        source_directory,
        temp_directory,
        fileName,
        height,
        width):
    k = 0
    im = Image.open(os.path.join(source_directory, fileName))
    imgwidth = im.size[0]
    imgheight = im.size[1]
    for i in range(0, imgheight, height):
        for j in range(0, imgwidth, width):
            box = (j, i, j + width, i + height)

            try:
                a = im.crop(box)
                if(i + height < imgheight and j + width < imgwidth):
                    
                    imgName = fileName.split(".", 1)[0]

                    a.save(
                        os.path.join(
                            temp_directory,
                            (imgName +
                             "IMG-%s.jpg" %
                             k)))

            except BaseException:
                pass
            k += 1


# processes all the raw images in a given directory
def process_all_images(
        source_directory,
        temp_directory,
        dest,
        size_x,
        size_y,
        test_prop,
        remove_pores,
        min_pore_size,
        rotate):
    

    i = 0
    # for root, dirs, files in os.walk(source_directory):
    for image in os.listdir(source_directory):
        
        print(i)
        
        files = glob.glob(os.path.join(temp_directory,"*"))
        for f in files:
            os.remove(f)
        
        
        slice_images(
            source_directory,
            temp_directory,
            image,
            size_y,
            size_x)
        
    
        for image in os.listdir(temp_directory):
            
            rand = random.uniform(0, 1)
            
            try:
                processed_img = convert_to_binary(os.path.join(
                    temp_directory, image), size_x, size_y)
                if(remove_pores):
                    processed_img = remove_small_pores(processed_img, min_pore_size)
                    processed_img = processed_img*255                    

                if rand <= test_prop:
                    cv2.imwrite(os.path.join(dest, image),
                                processed_img)
                else:


                    cv2.imwrite(os.path.join(dest, image),
                                processed_img)
                        
            except BaseException:
                continue
        
        i = i + 1
        

        
        
        

if __name__ == "__main__":


    process_all_images(SOURCE_DIR, 
                       TEMP_DIR, 
                       DEST_DIR, X_SIZE, 
                       Y_SIZE, TEST_PROP, 
                       REMOVE_SMALL_PORES, MIN_PORE_SIZE, 
                       ROTATE_IMGS)
    

        
    


