import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from extract_maze import *
from gate_find import *
from path_find import *
from solution_overlay import *

# eneter relative path of the image
path  = '../data/input/test14.png'

# Code which integrates all the implemented functions to display the solution

org_img = cv2.imread(path)
org_img = cv2.cvtColor(org_img,cv2.COLOR_BGR2RGB)

preprocess_img = preprocess(org_img)
extracted_maze,corners = extract_maze(preprocess_img)
gates = gate_find(preprocess_img,corners)

# For hand drawn test case use the k value in Gate_find as k = 7

path_res = final_path(extracted_maze,gates[0],gates[1])
final_res = sol_overlay(org_img,path_res,preprocess_img)

plt.imshow(final_res)
plt.show()
