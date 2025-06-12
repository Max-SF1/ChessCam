"""trying to run online inference with tracker"""
from ultralytics import YOLO 
import os
import cv2
import pyrealsense2 as rs
import numpy as np

############################ LOAD MODEL AND EXPORT WITH TENSORRT ##################################################################
# model = YOLO("/workspace/src/scripts/runs/detect/train5/weights/best.pt")
# model.export(format="engine",
#              imgsz = 1280, 
#              data= "/workspace/src/scripts/datasets/Scacchi-MachineLearning-YoloV5-2/data.yaml") 
###################################################################################################################################

tensorrt_model = YOLO("/workspace/src/scripts/runs/detect/train5/weights/best.engine")

############ DEBUG: CHECK LOADED AND EXPORTED MODEL CLASSES: ###################
# print(model.names)
# print(tensorrt_model.names)
# Initialize RealSense pipeline
###############################################################################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)


try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO tracking on the frame
        results = tensorrt_model.track(source=frame, persist=True)

        # Render the results and display
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv11 Online Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
