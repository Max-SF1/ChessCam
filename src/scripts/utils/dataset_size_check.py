"""given a path to a dataset folder, checks its current size"""
import os

directory_path = r"/workspace/recordings/Dataset_images_nulls_occlusions" #either the clean one or the nulls & occlusions one. 

lst = os.listdir(directory_path) # your directory path
number_files = len(lst)
print(f"there are {number_files} images in dataset") 