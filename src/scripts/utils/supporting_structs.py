from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
import numpy as np 
import cv2

class Piece:
    """a class that tracks each piece in the game, given a kalman filter for said piece."""
    def __init__(self, piece_bbox,model):

        self.piece_bbox= piece_bbox
        class_id = int(piece_bbox.cls)  # class ID (tensor to int)
        self.piece_type = model.names[class_id] #get the class string
        self.kf = KalmanFilterXYWH()
        measurement = np.array(piece_bbox.xywh[0].cpu().numpy())
        self.mean, self.covariance = self.kf.initiate(measurement)

    # def display(self):
    #     """displays piece attributes"""
    #     print(f"{self.piece_type}, loc: {self.piece_location}, proj loc:{self.proj_piece_location}")

    def predict(self):
        """displays the piece parameter prediction"""
        predicted_mean, predicted_covariance = self.kf.predict(self.mean, self.covariance)
        return predicted_mean, predicted_covariance
    def point_loc(self):
        """returns the location of a point on the board, for drawing the point."""
        box_xyxy = self.piece_bbox.xyxy.cpu()
        bb_height = (box_xyxy[0][1]-box_xyxy[0][3]).numpy() # abs if it doesn't work out 
        localization_pt = (
            int((box_xyxy[0][0].numpy() + box_xyxy[0][2].numpy()) / 2), #point x position
            int(box_xyxy[0][3].numpy() + bb_height * 0.25) #point y position
        )
        return localization_pt
    def point_loc_on_ideal_board(self,H,display=True):
         """projects the point representing the function to the ideal """
         localization_pt = self.point_loc()
         localization_pt_formatted = np.array([[localization_pt]], dtype=np.float32)

         if display: 
            projected  = cv2.perspectiveTransform(localization_pt_formatted, H)
            loc_number, loc_letter = projected[0][0]
            print(f"the projected location is {projected} for piece type {self.piece_type}")
            print("piece spotted at square ", int(loc_number),chr(int(loc_letter + 64)))



######################### CV2 DRAWING FUNCTIONS (not used atm.) ##################################
# mouse callback function - reference: 
# def draw_circle(event,x,y):
#     """draws a circle, leave me alone pylint! """
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         # print(f"x = {x}, y = {y} ") prints coordinates of click
#         corners.append((x,y))
#         if len(corners) == 5: 
#            corners.pop(0)
# and to call the function inside main loop we used:
#  cv2.setMouseCallback("YOLOv11 Online Inference",draw_circle)
##################################################################################