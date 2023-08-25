# CarPlate Detection Project Source Code
## Introduction
This is my implementation for Shanghai JiaoTong Univeristy Computer Vision(AU335) Project. The requirement of the project is given in "课程大作业.pdf".  
The complete explanation and result of the related project is written in "陈敬彦_519030990031_计算机视觉课程报告.pdf” using Chinese.  
上海交通大学计算机视觉(AU335)课程大作业相关代码与报告

## Environment Implementation
- Python 3.10
- NumPy 1.24.2
- EasyOCR 1.7.0
- os
- glob

These libraries can be downloaded using pip or anaconda.

## Notes
Filepaths in line 7 to 9 are required to modify to use your own images and save the resized images in your desired filepath, else the code won't be able to run successfully.
1. path: filepath used to save processed images, including resized images, grayscaled images and images of each character detected in each CarPlate image
2. images: input to detect the carplate number in each original image
3. images_resize: filepath used to get every resized images
