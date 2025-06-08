"""takes an image and outputs its shape"""

import sys
import os
import cv2
import matplotlib.pyplot as plt

image_path = "/workspace/recordings/extracted_images_from_recordings/color_0000.png" #REPLACE WITH YOUR OWN

image = cv2.imread(image_path)

if image is None:
    print(f"Failed to load image: {image_path}")
else: 
    height, width = image.shape[:2]
    print(f"Image Dimensions: Width = {width}, Height = {height}")