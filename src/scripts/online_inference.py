"""trying to run online inference with tracker"""
import os
import cv2
from ultralytics import YOLO 
import pyrealsense2 as rs
import numpy as np


##### KEYS ####
# b - draws bounding box 
# p - draws localization point 
# q - quits :) 



######################### CV2 DRAWING FUNCTIONS ###################################################################################
# mouse callback function - reference: 
def draw_circle(event,x,y,flags,param):
    """draws a circle, leave me alone pylint! """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print(f"x = {x}, y = {y} ") prints coordinates of click
        corners.append((x,y))
        if len(corners) == 5: 
           corners.pop(0)
####################################################################################################################################

# ############################ LOAD MODEL AND EXPORT WITH TENSORRT ##################################################################
# # model = YOLO("/workspace/src/scripts/runs/detect/train5/weights/best.pt")
# # model.export(format="engine",
# #              imgsz = 1280,
# #              data= "/workspace/src/scripts/datasets/Scacchi-MachineLearning-YoloV5-2/data.yaml") 
# ###################################################################################################################################


tensorrt_model = YOLO("/workspace/src/scripts/runs/detect/train5/weights/best.engine")

# ############ DEBUG: CHECK LOADED AND EXPORTED MODEL CLASSES: ###################
# # print(model.names)
# # print(tensorrt_model.names)
# # Initialize RealSense pipeline
# ###############################################################################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

drawbb = False # parameter for displaying bounding boxes
draw_localization_pt = False 

corners = []

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO tracking on the frame
        results = tensorrt_model.track(source=frame, persist=True, verbose = False ) #removes all the printing of updates to the terminal.


        key = cv2.waitKey(1) & 0xFF  # Capture key once
        if key == ord('b'):
            drawbb = not drawbb
        elif key == ord('p'):
            draw_localization_pt = not draw_localization_pt 
        elif key == ord('q'):
            break

        # Render the results and display
        if drawbb:
            annotated_frame = results[0].plot(line_width=2, font_size=5)
        else: 
            annotated_frame = frame
        
        # take predictions and project them into a single point for localization 
        if draw_localization_pt:
            for box in results[0].boxes.xyxy.cpu():
                bb_height = (box[1]-box[3]).numpy() # abs if it doesn't work out 
                cv2.circle(annotated_frame,(int((box[0].numpy()+box[2].numpy())/2),int(box[3].numpy() + bb_height*0.25)),5,(0,255,0),-1)
        
        for corner in corners:
            cv2.circle(annotated_frame,(corner[0],corner[1]),10,(255,0,0),-1)

        # print(annotated_frame.shape) annotated frame is 720,1280,3 (RGB image)
        cv2.namedWindow("YOLOv11 Online Inference")
        cv2.setMouseCallback("YOLOv11 Online Inference",draw_circle)
        cv2.imshow("YOLOv11 Online Inference", annotated_frame)   

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


