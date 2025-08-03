"""trying to run online inference with tracker"""
import os
import cv2
from ultralytics import YOLO 
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
import pyrealsense2 as rs
import numpy as np
from utils.supporting_structs import Piece  #to clean up code, piece has been moved to a new file. 




##### KEYS ####
# b - draws bounding box
# p - draws localization point
# q - quits :)
# c - to re-localize corners. 
# remember, cv2 window has to be in focus for the program to work. 



# ############################ LOAD MODEL AND EXPORT WITH TENSORRT ###############
# model = YOLO("/workspace/src/scripts/runs/detect/train5/weights/best.pt")
# model.export(format="engine",
#              imgsz = 1280,
#              data= "/workspace/src/scripts/datasets/Scacchi-MachineLearning-YoloV5-2/data.yaml") 
# ##############################################################################

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


###FLAGS###
drawbb = False # parameter for displaying bounding boxes
draw_localization_pt = False 
first_localization = True 

corners = []
#corners given by cv2's find corners are ordered row by row, left to right in every row. 
#matching_corners should be ordered the same way.
matching_corners_for_homography = np.array([
     [ [x, y] ] for y in range(8,1,-1) for x in range(8,1,-1) 
], dtype=np.float32)
corners_found = False #will update to true once corners have been found at least once
homography_matrix = []

pieces = []
try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        # Run YOLO tracking on the frame
        results = tensorrt_model.track(source=frame, persist=True, verbose = False, tracker="botsort.yaml" ) #removes all the printing of updates to the terminal.
        #unmodified botsort tracker added 
        if corners_found == False: #find corners: 
            ret, corners = cv2.findChessboardCorners(frame, (7,7), None)
            if ret:
                corners_found = True
        else:
            cv2.drawChessboardCorners(frame, (7,7), corners, ret)
            homography_matrix, _= cv2.findHomography(corners,matching_corners_for_homography)
            homography_matrix = np.array(homography_matrix)



        key = cv2.waitKey(1) & 0xFF  # Capture key once
        if key == ord('b'):
            drawbb = not drawbb
        elif key == ord('p'):
            draw_localization_pt = not draw_localization_pt 
        elif key == ord('q'):
            break
        elif key == ord('c'): # if you moved the table and want to redraw the corners. 
            corners_found = False 

        # Render the results and display
        if drawbb:
            annotated_frame = results[0].plot(line_width=2, font_size=5)
        else: 
            annotated_frame = frame
        
        # take predictions and project them into a single point for localization 
        if key == ord('l'):
            pieces = [] #reset the list. 
            for bounding_box in results[0].boxes:
                pieces.append(Piece(bounding_box,tensorrt_model))
                        #development tip: print(bounding_box) is really cool if you want to find the attributes of a bounding box! 
        if draw_localization_pt == True: 
            for piece in pieces:
                loc = piece.point_loc()
                if draw_localization_pt:
                    cv2.circle(annotated_frame,loc,5,(0,255,0),-1)
                piece.point_loc_on_ideal_board(homography_matrix)
        else: #take the detections, associate them with kalman filters, and update filter based on measurements. 
            for bounding_box in results[0].boxes:

                
        cv2.namedWindow("YOLOv11 Online Inference")
        cv2.imshow("YOLOv11 Online Inference", annotated_frame)   

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


