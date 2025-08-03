"""trying to run online inference with tracker"""
import os
import cv2
from ultralytics import YOLO 
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
import pyrealsense2 as rs
import numpy as np
from utils.supporting_structs import Piece, PieceManager, Homography, Piece_Associator




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
manager = PieceManager([])
homography = Homography()
ret = False 

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
        manager.associate_pieces(results[-1].boxes.xywh)
        if not homography.check_initialization(): #find corners: 
            ret, corners = cv2.findChessboardCorners(frame, (7,7), None)
            print(corners)
            if ret:
                homography.initialized = True
        else:
            cv2.drawChessboardCorners(frame, (7,7), corners, ret)
            homography.compute_homography(corners)



        key = cv2.waitKey(1) & 0xFF  # Capture key once
        if key == ord('b'):
            drawbb = not drawbb
        elif key == ord('p'):
            draw_localization_pt = not draw_localization_pt 
        elif key == ord('q'):
            break
        elif key == ord('c'): # if you moved the table and want to redraw the corners. 
            homography.initialized = False 

        # Render the results and display
        if drawbb:
            annotated_frame = results[0].plot(line_width=2, font_size=5)
        else: 
            annotated_frame = frame
        
        # take predictions and project them into a single point for localization 
        if key == ord('l'):
            manager.dump_pieces()
            for bounding_box in results[0].boxes:
                manager.append_piece(Piece(bounding_box,tensorrt_model))
                #tip: print(bounding_box) is really cool if you want to find the attributes of a bounding box! 

        manager.time_update() #time-update piece kalman filters 
        manager.display_active_pieces()
           
        if draw_localization_pt: 
            locations = manager.get_point_locations()
            for loc in locations: 
                cv2.circle(annotated_frame,loc,5,(0,255,0),-1)

        if manager.pieces:  
            for piece in manager.pieces: 
                x, y, w, h = piece.mean[:4]
                top_left = (int(x+w/2), int(y+h/2))
                bottom_right = (int(x - w/2), int(y - h/2))
                cv2.rectangle(annotated_frame, top_left, bottom_right, color=(0, 255, 0), thickness=2)


        #take the detections, associate them with kalman filters, and update filter based on measurements. 
            # for bounding_box in results[0].boxes:
                
                
        cv2.namedWindow("YOLOv11 Online Inference")
        cv2.imshow("YOLOv11 Online Inference", annotated_frame)   

finally:
    pipeline.stop()
    cv2.destroyAllWindows()


