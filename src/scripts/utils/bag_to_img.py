"""bag-to-images: 
a pyrealsense2 based document converter. .bag -> mp4 """

import os
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

# ===== CONFIGURATION =====
bag_file = "recordings/2025-08-09_18-03-57_001.bag"  # Path to your .bag file
output_folder = "my_chess_images"              # Folder to save images
save_every_n_frames = 240                         # Change this to save fewer/more frames

# ===== CREATE OUTPUT FOLDER =====
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_path = os.path.join(output_folder, timestamp)
os.makedirs(output_path, exist_ok=True)

# ===== SETUP PIPELINE =====
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, bag_file, repeat_playback=False)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)

frame_count = 0
saved_count = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame_count += 1
        if frame_count % save_every_n_frames == 0:
            color_image = np.asanyarray(color_frame.get_data())
            filename = os.path.join(output_path, f"frame_{saved_count:06}.png")
            cv2.imwrite(filename, color_image)
            saved_count += 1
            print(f"[INFO] Saved: {filename}")
except Exception as e:
    print("[INFO] Done reading .bag file or error occurred:", e)
finally:
    pipeline.stop()
    print(f"[INFO] Extracted {saved_count} images to: {output_path}")
