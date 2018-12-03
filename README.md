# Multi-Target Tracking

Estimate object trajectories in an automotive context, using KITTI dataset, with Mask-RCNN.
A very detailed description of the implamentation is available in the `relazione.pdf` file (but in italian).

![TestImage1](./images/image1.png)
## Usage
Three python scripts, each with different functions:
 * Kitti-dets : Calcute tracks with predicted boxes and accuracy with motmetrics library.
 * Kitti-mask : Calcute tracks with masks.
 * kitti-features : Calcute tracks with features vectors.


## Prerequisites
The implementation has been done with the programming language Python 3.Project dependencies are the following Python modules. 

Requirement | Version Used
------------| ------------
Python | 3.6
matplotlib | 2.2.2  
numpy |  1.14.5 
motmetrics | 1.1.3
opencv-python | 3.4.3.18 

The Mask-RCNN code is based on the following GitHub repository:

- [fedebecat/kitti_playground](https://github.com/fedebecat/kitti_playground)
