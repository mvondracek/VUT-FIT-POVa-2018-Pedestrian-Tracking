# VUT FIT POVa 2018 Pedestrian Tracking

| ⚠ This project is deprecated and isn't maintained. |
|----------------------------------------------------|

[![Python version](https://img.shields.io/badge/Python-3-blue.svg?style=flat-square)](https://www.python.org/)

> [Computer Vision (POVa)](https://www.fit.vutbr.cz/study/courses/index.php.en?id=12895)<br/>
> [Faculty of Information Technology (FIT)](http://www.fit.vutbr.cz/.en)<br/>
> [Brno University of Technology (BUT)](https://www.vutbr.cz/en/)

We have implemented a computer vision system intended for **tracking pedestrians in observed scene**. Our system is capable of detecting people in images from **two cameras**. Detected bodies are then matched together based on similarities in their histograms to make a pair of images of the same person. Detected person is then located in 3D space using triangulation, which uses depth planes in 3D space and their intersection. Located frames of people are then tracked to form **path in space over time**.

**Team** *in alphabetical order*: Lukáš Petrovič [@flaxh](https://github.com/flaxh), Filip Šťastný [@xstast24](https://github.com/xstast24), Martin Vondráček [@mvondracek](https://github.com/mvondracek), 

![Tracking visualisation](https://raw.githubusercontent.com/mvondracek/VUT-FIT-POVa-2018-Pedestrian-Tracking/master/doc/s3_single_3fps.png)

*Path visualisation of the tracked person (green) in an observed scene*

![Tracking with video](./doc/main.gif)
*Scene observed by two cameras (left) and the result of pedestrian tracking (right)*

## Installation

1) **Python 3.7** is required.
2) Please create a virtual environment for this project.
3) With activated virtual environment, run: `pip install -r requirements.txt` in the project folder.
4) Download *OpenPose* model from `http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel`
   and save it to `openpose/pose/coco`.
5) Please download testing data from `https://github.com/mvondracek/VUT-FIT-POVa-2018-Pedestrian-Tracking`

## Run

~~~
./main.py
~~~

## Documentation

* [project report](./doc/VUT_FIT_POVa_2018_Pedestrian_Tracking_report.pdf)
* presentation slides
  ([pptx with animations](./doc/VUT_FIT_POVa_2018_Pedestrian_Tracking_presentation.pptx),
  [pdf](./doc/VUT_FIT_POVa_2018_Pedestrian_Tracking_presentation.pdf))
* [supplemental materials](./doc/)
