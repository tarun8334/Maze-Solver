import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from prep_maze import *
from extract_maze import *

# function to shift the gate points inside the maze
def gate_shift(gt, img_shape):

    gate = gt
    h,w = img_shape

    temp = gate[0][0]
    gate[0][0] = gate[0][1]
    gate[0][1] = temp

    temp = gate[1][0]
    gate[1][0] = gate[1][1]
    gate[1][1] = temp
    
   # shifting the gate points inside the maze based on in which quadrant they lie 
    if(gate[0][0] > h/2):
        gate[0][0] -= 3
    elif(gate[0][0] <= h/2):
        gate[0][0] += 3
    
    if(gate[1][0] > h/2):
        gate[1][0] -= 3
    elif(gate[1][0] <= h/2):
        gate[1][0] += 3

    if(gate[0][1] > w/2):
        gate[0][1] -= 3
    elif(gate[0][1] <= w/2):
        gate[0][1] += 3

    if(gate[1][1] > w/2):
        gate[1][1] -= 3
    elif(gate[1][1] <= w/2):
        gate[1][1] += 3

    return gate


# function to find the gate points
def gate_find(preprocess_img, corners):

    # adding border to the image
    prep_img = cv.copyMakeBorder(preprocess_img, 10, 10, 10, 10, cv.BORDER_CONSTANT, 0)
    cn1 = corners[0] + 10
    cn2 = corners[1] + 10
    cn3 = corners[2] + 10
    cn4 = corners[3] + 10

    # equation of line correspoding the corner point 1 and 2 
    dcn = np.abs(cn2 - cn1)
    if dcn[0] > dcn[1]:
        x = np.linspace(cn1[0], cn2[0], dcn[0])
        y = np.linspace(cn1[1], cn2[1], dcn[0])
    else:
        x = np.linspace(cn1[0], cn2[0], dcn[1])
        y = np.linspace(cn1[1], cn2[1], dcn[1])
    x = x.astype(int)
    y = y.astype(int)
    l1 = np.array([x, y]).T
   
   # equation of line correspoding the corner point 2 and 3
    dcn = np.abs(cn3 - cn2)
    if dcn[0] > dcn[1]:
        x = np.linspace(cn2[0], cn3[0], dcn[0])
        y = np.linspace(cn2[1], cn3[1], dcn[0])
    else:
        x = np.linspace(cn2[0], cn3[0], dcn[1])
        y = np.linspace(cn2[1], cn3[1], dcn[1])
    x = x.astype(int)
    y = y.astype(int)
    l2 = np.array([x, y]).T

    # equation of line correspoding the corner point 3 and 4
    dcn = np.abs(cn4 - cn3)
    if dcn[0] > dcn[1]:
        x = np.linspace(cn3[0], cn4[0], dcn[0])
        y = np.linspace(cn3[1], cn4[1], dcn[0])
    else:
        x = np.linspace(cn3[0], cn4[0], dcn[1])
        y = np.linspace(cn3[1], cn4[1], dcn[1])
    x = x.astype(int)
    y = y.astype(int)
    l3 = np.array([x, y]).T

   # equation of line correspoding the corner point 4 and 1
    dcn = np.abs(cn1 - cn4)
    if dcn[0] > dcn[1]:
        x = np.linspace(cn4[0], cn1[0], dcn[0])
        y = np.linspace(cn4[1], cn1[1], dcn[0])
    else:
        x = np.linspace(cn4[0], cn1[0], dcn[1])
        y = np.linspace(cn4[1], cn1[1], dcn[1])
    x = x.astype(int)
    y = y.astype(int)
    l4 = np.array([x, y]).T

    k = 3
    # To be changed to larger values if dimensions of image are large
    gate1 = []
    gate2 = []
    gate3 = []
    gate4 = []
    # appending the values corresponding to each line in which the neighbourhood of k*k has no maze wall point
    for i in range(0,len(l1)):
        if np.max(prep_img[l1[i][1]-k:l1[i][1]+k, l1[i][0]-k:l1[i][0]+k]) == 0:
            gate1.append(l1[i])
    for i in range(0,len(l2)):
        if np.max(prep_img[l2[i][1]-k:l2[i][1]+k, l2[i][0]-k:l2[i][0]+k]) == 0:
            gate2.append(l2[i])
    for i in range(0,len(l3)):
        if np.max(prep_img[l3[i][1]-k:l3[i][1]+k, l3[i][0]-k:l3[i][0]+k]) == 0:
            gate3.append(l3[i])
    for i in range(0,len(l4)):
        if np.max(prep_img[l4[i][1]-k:l4[i][1]+k, l4[i][0]-k:l4[i][0]+k]) == 0:
            gate4.append(l4[i])
    
   
    gate1 = np.array(gate1)
    gate2 = np.array(gate2)
    gate3 = np.array(gate3)
    gate4 = np.array(gate4)
    gate = []
   
   # checking whether there is a probable gate points and finding the mean point of all of them.
    if len(gate1) > 0:
        gate1 = [np.mean(gate1[:,0]).astype(np.uint)-10, np.mean(gate1[:,1]).astype(np.uint)-10]
        gate.append(gate1)
    if len(gate2) > 0:    
        gate2 = [np.mean(gate2[:,0]).astype(np.uint)-10, np.mean(gate2[:,1]).astype(np.uint)-10]
        gate.append(gate2)
    if len(gate3) > 0:
        gate3 = [np.mean(gate3[:,0]).astype(np.uint)-10, np.mean(gate3[:,1]).astype(np.uint)-10]
        gate.append(gate3)
    if len(gate4) > 0:
        gate4 = [np.mean(gate4[:,0]).astype(np.uint)-10, np.mean(gate4[:,1]).astype(np.uint)-10]
        gate.append(gate4)

    gate = np.array(gate,dtype=int)
  # using gate shift to shift the gate
    gate = gate_shift(gate,preprocess_img.shape)

    return gate