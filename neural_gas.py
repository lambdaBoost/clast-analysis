# -*- coding: utf-8 -*-
"""
Created on Wed May  1 10:22:48 2019

@author: ahall
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
from neupy import algorithms
import pickle


STOP_CONDITION = 0.00001 # min change in training error for early stopping
MIN_EPOCHS = 5 # number of epochs before early stopping becomes active
MAX_EPOCHS = 25
MAX_NODES = 10000

#read in input image
img = cv2.imread('C:\\Users\\ahall\\Documents\\projects\\clast-analysis\\processed_data\\RAW\\N13920IMG-33.jpg')

#convert to binary
thresh = 128
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(grey, thresh, 255, cv2.THRESH_BINARY)[1]

plt.imshow(binary, cmap='Greys_r')




#create plot and set dimensions
f,ax = plt.subplots(1,2)
f.set_figheight(10)
f.set_figwidth(20)


# Convert to binary and apply canny edge detection
# This part can be improved

edges = cv2.Canny(binary,100,200)

#plot edge detection image and original image
ax[0].imshow(img,cmap='Greys_r')
ax[1].imshow(edges,cmap='Greys_r')


#%%
# Convert each white pixel to a separate data point
# We will use the data points in order to learn
# the topological structure of the image
def image_to_data(img):
    data = []
    for (x, y), value in np.ndenumerate(img):
        if value == 255:
            data.append([y, -x])
    return data


data = image_to_data(edges)

#  Image scaling 
# Less data points = less accurate topology but faster processing
scale_factor = 0.001   
data = scale_factor * np.array(data)
len_data = len(data)
print(f"Number of data points: {len_data}")

plt.figure(figsize=(8, 9))
plt.plot(*np.array(data).T / scale_factor ,marker=',',lw=0, linestyle="", alpha=1);

#%%
#Plot function
def draw_image(graph, show=True):
    for node_1, node_2 in graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        line, = plt.plot(*weights.T, color='black')
        plt.setp(line, linewidth=0.2, color='black')

    plt.xticks([], [])
    plt.yticks([], [])
    
    if show:
        plt.show()
        

# GNG initialization function
def create_gng(max_nodes, step=0.2, n_start_nodes=2, max_edge_age=50):
    return algorithms.GrowingNeuralGas(
        n_inputs=2,
        n_start_nodes=n_start_nodes,

        shuffle_data=True,
        verbose=True,

        step=step,
        neighbour_step=0.005,

        max_edge_age=max_edge_age,
        max_nodes=max_nodes,

        n_iter_before_neuron_added=10,
        after_split_error_decay_rate=0.5,
        error_decay_rate=0.995,
        min_distance_for_update=0.01,
    )
    
#%%
    

    
gng = create_gng(max_nodes=MAX_NODES)
    
#set epochs for training and plot in each iteration
for epoch in range(MAX_EPOCHS):
    gng.train(data, epochs=1)
        
    # Plot images - takes ages so comment out unless necessary
    #plt.figure(figsize=(16, 18))
    #draw_image(gng.graph)
    #plt.savefig(str('results\\GNG' + str(epoch) + '.png') )
    
    #early stopping
    if(epoch > MIN_EPOCHS and ( gng.errors.train[-2] -gng.errors.train[-1] < STOP_CONDITION)):
        break

plt.figure(figsize=(16, 18))
draw_image(gng.graph)
#plt.savefig(str('results\\GNG' + str(epoch) + '.png') )       
pickle.dump(gng,open("results\\gng.p","wb"))
