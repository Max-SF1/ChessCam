"""
Intel RealSense Recording Script
Press 's' to start/stop recording (multiple sessions allowed)
Press 'q' to quit.
IMPORTANT - ONLY WORKS WHEN THE OPENCV WINDOW IS IN FOCUS!!! IF TERMINAL IS IN FOCUS, IT WON'T WORK! 
"""
from datetime import datetime

import os
import cv2
import pyrealsense2 as rs
import numpy as np


def get_new_recording_filename():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("recordings", exist_ok=True)
    counter = 1
    while True:
        name = f"{timestamp}_{counter:03}.bag"
        path = os.path.join("recordings", name)
        if not os.path.exists(path):
            return path
        counter += 1

def start_pipeline(record_to_bag=None, width=1280, height=720, fps=30): #1280 × 720 ?
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    if record_to_bag:
        config.enable_record_to_file(record_to_bag)

    pipeline.start(config)
    return pipeline


def main():
    print("Press 's' to start/stop recording, 'q' to quit.")

    recording = False
    width, height, fps = 640, 480, 30
    pipeline = start_pipeline(width=width, height=height, fps=fps)
    align = rs.align(rs.stream.color)

    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03),
                cv2.COLORMAP_JET
            )

            cv2.imshow("Color Stream", color_image)
            cv2.imshow("Depth Stream", depth_colormap)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                pipeline.stop()
                recording = not recording
                if recording:
                    filename = get_new_recording_filename()
                    print(f"[INFO] Recording to: {filename}")
                    pipeline = start_pipeline(
                        record_to_bag=filename, 
                        width=width, 
                        height=height, 
                        fps=fps)
                else:
                    print("[INFO] Stopped recording.")
                    pipeline = start_pipeline(width=width, height=height, fps=fps)

            elif key == ord('q'):
                print("[INFO] Exiting.")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 