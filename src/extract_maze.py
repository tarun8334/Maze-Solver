import numpy as np
import cv2
from matplotlib import pyplot as plt
from prep_maze import *

# function to make  a binary mask with inputas corners
def make_mask(img_shape,cor):
    points = np.array(cor)
    mask = np.zeros(img_shape)
   
   # mask is made using the below cv2 inbuilt function.
    cv2.fillPoly(mask, pts=[points], color = 255)
    mask = (mask/255).astype('uint8')

    return mask
# function to resolve conflict between two corners
# if two corners are too close to each other, then the corner which is closer to the border of the image is chosen
def resolve_conflict(pt1,pt2,img_shape):
    h,w = img_shape
    
    dist1=0
    dist2=0

    if(pt1[0] < h/2):
        dist1 += pt1[0]
        dist2 += pt2[0]
    else:
        dist1 += (h-pt1[0])
        dist2 += (h-pt2[0])

    if(pt1[1] < w/2):
        dist1 += pt1[1]
        dist2 += pt2[1]
    else:
        dist1 += (w-pt1[1])
        dist2 += (w-pt2[1])

    if(dist1 < dist2):
        return pt1
    else:
        return pt2

# Function to arrange points in clockwise order

def order_points_new(pts):
    pts = np.array(pts)
    # Sorting the array in the x co-ordinates
    xsort = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xsort[:2, :]
    rightMost = xsort[2:, :]

    # Sorting the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost


    # Sorting the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    return np.array([tl, tr, br, bl], dtype=int)

# function for labelling the image by using bfs
def connectedComp(img):
    img = img.copy()
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,0)
    h = img.shape[0]
    w = img.shape[1]
    l_img = np.zeros((h,w))
    vis = np.zeros(h*w,dtype = 'int')
    label = 1
    #  bfs algorithm
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(img[i][j]==255 and l_img[i][j] == 0):
                queue = []
                queue.append(i*w+j)
                vis[i*w+j] = 1
                while len(queue) != 0:
                    temp = queue.pop(0)
                    x = int(temp/w)
                    y = temp%w
                    l_img[x][y] = label
                    for l in range(-1,2):
                        for m in range(-1,2):
                            if(vis[(x+l)*w +y+m] == 0 and img[x+l][y+m] == 255):
                                queue.append((x+l)*w +y+m)
                                vis[(x+l)*w +y+m] = 1
                label += 1
    l_img = l_img[1:h-1,1:w-1]
    # returning label image
    return l_img


# extracting corner points from the top two labelled components
def corner_extraction(list_point):
    
    list_pointx = np.array(list_point.copy())
    list_pointy = np.array(list_point.copy())

    # array sorting in x co-ordinates 
    arr1inds = list_pointx[:,0].argsort()
    list_pointx[:,1] = list_pointx[:,1][arr1inds[::1]]
    list_pointx[:,0] = list_pointx[:,0][arr1inds[::1]]

    # array sorting in y co-ordinates 
    arr1inds = list_pointy[:,1].argsort()
    list_pointy[:,1] = list_pointy[:,1][arr1inds[::1]]
    list_pointy[:,0] = list_pointy[:,0][arr1inds[::1]]

    # finding the minimum and maximum x cordinates
    xmin = list_pointx[0][0]
    xmax = list_pointx[list_pointx.shape[0]-1][0]

    # finding the minimum and maximum y cordinates
    ymin = list_pointy[0][1]
    ymax = list_pointy[list_pointy.shape[0]-1][1]

   # appending the values corresponding to the minimum and maximum y co ordinates
   # list is then sorrted to find points corresponding to the x co-ordinates
    temp_xmin = []
    for x in list_pointx:
        if(x[0] > xmin):
            break

        temp_xmin.append(x[1])

    temp_xmin.sort()
    xmin_y1 = temp_xmin[0]
    xmin_y2 = temp_xmin[len(temp_xmin)-1]

    # x max 
    # same as done above for x maximum
    temp_xmax = []
    for x in range(list_pointx.shape[0]-1,-1,-1):
        if(list_pointx[x][0] < xmax):
            break

        temp_xmax.append(list_pointx[x][1])

    temp_xmax.sort()
    xmax_y1 = temp_xmax[0]
    xmax_y2 = temp_xmax[len(temp_xmax)-1]

    # y min
    # same as done above for y min
    temp_ymin = []
    for y in list_pointy:
        if(y[1] > ymin):
            break

        temp_ymin.append(y[0])

    temp_ymin.sort()
    ymin_x1 = temp_ymin[0]
    ymin_x2 = temp_ymin[len(temp_ymin)-1]

    # y max
    # same as done above for y max
    temp_ymax = []
    for y in range(list_pointy.shape[0]-1,-1,-1):
        if(list_pointy[y][1] < ymax):
            break

        temp_ymax.append(list_pointy[y][0])

    temp_ymax.sort()
    ymax_x1 = temp_ymax[0]
    ymax_x2 = temp_ymax[len(temp_ymax)-1]

    corners = []
   # appending all the corner points found.
    corners.append([xmin,xmin_y1])
    corners.append([xmin,xmin_y2])
    corners.append([xmax,xmax_y1])
    corners.append([xmax,xmax_y2])
    corners.append([ymin_x1,ymin])
    corners.append([ymin_x2,ymin])
    corners.append([ymax_x1,ymax])
    corners.append([ymax_x2,ymax])

    return corners

# Manhattan distance
def dist(pt1, pt2):
    return np.abs(pt1[0]-pt2[0]) + np.abs(pt1[1]-pt2[1])

# function for finding the unique corner points of the image
def unique_corners(corners, img_shape):

    crns = []

    crns.append(corners[0])

    for c1 in corners:
        flag=0
        pt = c1
        for i in range(len(crns)):
            # the corner points which are less than 50 manhattan unit distance away are selected only once.
            if(dist(c1,crns[i])<=50):
                flag=1
                # conflict resolve
                pt = resolve_conflict(c1,crns[i], img_shape)
                crns.pop(i)
                crns.append(pt)
        
        if(flag==0):
            crns.append(pt)
    
    return crns
# final function to output the extracted maze by calling ab0ve functions in the below function.
def extract_maze(preprocessed_img):
   
    binary_img = preprocessed_img.copy()
   # finding the labelled image
    label_img = connectedComp(binary_img)

    h,w = label_img.shape
    new = np.zeros((h,w),dtype = 'uint8')

    for i in range(h):
        for j in range(w):
            if label_img[i][j] == 1:
                new[i][j] = 255

    num_of_labels = int(np.max(label_img))

    # finding the count of all labels 
    count_label = np.zeros((num_of_labels+1,2),dtype = int)
    for i in range(1,num_of_labels+1):
        count_label[i][1] = np.count_nonzero(label_img == i)
        count_label[i][0] = i

    # sorting the labels in descending order of their count
    arr1inds = count_label[:,1].argsort()
    count_label[:,0] = count_label[:,0][arr1inds[::-1]]
    count_label[:,1] = count_label[:,1][arr1inds[::-1]]
   
    # finding the top two labels
    top1,top2 = count_label[0][0],count_label[1][0]

    points_list = list()
   # finding the points corresponding to the top two labels
    for i in range(h):
        for j in range(w):
            if label_img[i][j] == top1 or label_img[i][j] == top2:
                temp = [i,j]
                points_list.append(temp)
    
   # finding the corner points of the image
    corner = corner_extraction(points_list)
    refined_corner = unique_corners(corner, [h,w])

  # interchanging the y and x co-ordinates of the corner points
    for i in range(4):
        temp = refined_corner[i][0]
        refined_corner[i][0] = refined_corner[i][1]
        refined_corner[i][1] = temp
    
    refined_corner = order_points_new(refined_corner)    
    mask = make_mask(label_img.shape,refined_corner)   # mask obtained from corner 
    extracted_img = preprocessed_img.copy()  

    # extracting the maze from the image using mask value          
    for i in range(h):
        for j in range(w):
            if mask[i][j] == 1:
                extracted_img[i][j] = 255 - preprocessed_img[i][j]
            else:
                extracted_img[i][j] = 0

    return extracted_img,refined_corner


# Verified all working fine

# img = cv2.imread('/home/prash/Documents/DIP_Project_new/data/inputs/noisy.jpg')

# pre = preprocess(img)
# extract_maze(pre)
# plt.imshow(ext_img,cmap = 'gray')
# plt.show()