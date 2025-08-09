"""
"""
from datetime import datetime
import os
import cv2
import pyrealsense2 as rs
import numpy as np
def get_new_image_folder():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join("recordings", timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder

def start_pipeline(width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)
    return pipeline

def main():
    print("Press 's' to start/stop saving images, 'q' to quit.")

    saving = False
    frame_count = 0
    save_folder = None

    width, height, fps = 1280, 720, 30
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

            # If saving flag is on, save the current color frame as an image
            if saving:
                filename = os.path.join(save_folder, f"frame_{frame_count:06}.png")
                cv2.imwrite(filename, color_image)
                frame_count += 1

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                saving = not saving
                if saving:
                    save_folder = get_new_image_folder()
                    frame_count = 0
                    print(f"[INFO] Started saving images to: {save_folder}")
                else:
                    print("[INFO] Stopped saving images.")

            elif key == ord('q'):
                print("[INFO] Exiting.")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
