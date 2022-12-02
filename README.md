# Maze Solver

## Brief Description
Our project is about tracing the shortest path from a input maze image and then applying the concepts of Digital Image Processing and Game Theory to obtain a final Output


## Setup and Installation
This project expects the following libraries to be present on the users system along with Python3 version 3.8.10

These libraries can be installed via pip3

* > **_OpenCV:_** a library of programming functions mainly aimed at real-time computer vision <br>
* > **_Numpy:_** for dealing with large, multi-dimensional arrays and matrices <br>
* > **_MatplotLib:_** a plotting library

Version details are present in requirements.txt

## How to download and run the code
* Download and extract the zip file from github repo <br>
* Now open the src folder and run the maze_solver.py file to get the output for one of the test cases
* To run for a custom input file, firstly paste the file in data/input/
* Now change the file name in line 13n of maze_solver from '../data/input/test14.png' to '../data/input/\<newfilename\>'
* Now run the maze_solve.py to obtain the result

## Sample outputs
* Success Case <br>
![test14_res](https://user-images.githubusercontent.com/82945846/204342411-28a68916-11d2-4cde-ae3d-a22eebf961c9.png) <br>

* Failure Case <br>
![test9_res](https://user-images.githubusercontent.com/82945846/204343034-d9407109-585c-4b07-ab11-4a72b1d9c900.png)

## Repository Structure
* data
  * input (Contain all the test cases)
  * output (Contains output of all the test cases)
* docs
  * Research Paper (Paranjpe_Saied_Maze_Solver[5190])
  * Team11_Flodd_Fill.pptx (Final presentation slides)
* src
  * extract_maze.py
  * gate_find.py
  * maze_solve.py
  * path_find.py
  * prep_maze.py
  * solution_overlay.py
* results (Contains the results of intermediate outputs)
  * PreProcessing.png
  * Path.png
  * Gates.png
  * Final_Sol.png
  * ExtractedMaze.png
 

## Biblography:
[Research Paper](https://stacks.stanford.edu/file/druid:yt916dh6570/Paranjpe_Saied_Maze_Solver.pdf) <br>
[OpenCV fillPoly](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga311160e71d37e3b795324d097cb3a7dc)
