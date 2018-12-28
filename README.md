# VUT FIT POVa 2018 Pedestrian Tracking

[![Python version](https://img.shields.io/badge/Python-3-blue.svg?style=flat-square)](https://www.python.org/)

> **Task:** Implement a vision system capable of tracking multiple pedestrians from one or multiple stationary cameras. Evaluate the accuracy (how long can it track a person) on an existing dataset.
>
> Mandatory scored parts of the project are:
>   - The solution (program, ...) and achieved results - 12b
>   - Project report - 6b
>    - Final presentation (on stage) - 12b
>
> **The project report** has to include task definition, short review of existing previous solutions or other relevant information sources, description of your solution, experiments and evaluation results. It is not enough to provide a solution to a task - you should evaluate how well it works. The evaluation can be  quantitative (e.g. segmentation accuracy on a standard dataset) or qualitative (e.g. user feedback). 
>
> **How to submit the projects:** One \*.zip file including all source codes, project report, and **presentation slides**. If the file size limit is a problem, contact us. Please include examples of data and instructions how to use your code and tools. 
>
> **Project presentations** are scheduled for Thursday 2018-01-04 14:00 in M103.

## Installation

1) **Python 3.7** is required.
2) Please create a virtual environment for this project.
3) With activated virtual environment, run: `pip install -r requirements.txt` in the project folder.
4) Download *OpenPose* model from `http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel`
   and save it to `openpose/pose/coco`.

## Run

~~~
./main.py
~~~

## Documentation

* `./report.pdf`
* `./slides.pdf`
