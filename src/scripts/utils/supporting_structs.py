from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
import numpy as np 
import cv2
from typing import List 
from ultralytics.engine.results import Results
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
#corners given by cv2's find corners are ordered row by row, left to right in every row. 
#matching_corners should be ordered the same way.


class Homography: 
    """Holds homography matrix and applies homographic transforms."""
    def __init__(self):
        self.homography_matrix = []
        self.initialized = False 
        self.matching_corners_for_homography = np.array([[ [x, y] ] for y in range(8,1,-1) for x in range(8,1,-1) 
                                 ], dtype=np.float32)

    def check_initialization(self):
        return self.initialized
    def compute_homography(self,corners):
        # cv2.drawChessboardCorners(frame, (7,7), corners, ret)
        self.homography_matrix, _= cv2.findHomography(corners,self.matching_corners_for_homography,)
        self.homography_matrix = np.array(self.homography_matrix)

    def transform_point_location_to_ideal_board(self, locations):
        """Projects the point for center of piece location to the ideal board """
        localization_pt_formatted = np.array([[locations]], dtype=np.float32)

        if not self.homography_matrix:
            print("no initialized homography matrix found :( ")
        else: 
            projected  = cv2.perspectiveTransform(localization_pt_formatted, self.homography_matrix )
            loc_number, loc_letter = projected[0][0]
            print(f"the projected location is {projected} for piece type {self.piece_type}")
            print("piece spotted at square ", int(loc_number),chr(int(loc_letter + 64)))



class Piece:
    """A class that tracks each piece in the game, given a kalman filter for said piece."""
    def __init__(self, piece_bbox,model):
        self.piece_bbox= piece_bbox
        class_id = int(piece_bbox.cls)  # class ID (tensor to int)
        self.piece_type = model.names[class_id] #get the class string
        self.kf = KalmanFilterXYWH()
        measurement = np.array(piece_bbox.xywh[0].cpu().numpy())
        self.mean, self.covariance = self.kf.initiate(measurement)

    def display(self):
        """displays piece attributes"""
        print(f"{self.piece_type}, loc: {self.mean}")
    
    def time_update(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)

    
    def measurement_update(self,piece_bbox):
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, piece_bbox)


class PieceManager():
    """Piece Manager handles all functionality related to displaying the pieces, 
    updating their Kalman Filters. """
    def __init__(self,pieces): 
        self.pieces = pieces 
        self.associator = Piece_Associator()
    def dump_pieces(self):
        self.pieces = []
    def append_piece(self,piece):
        self.pieces.append(piece)
    def time_update(self):
        for piece in self.pieces:
            piece.time_update()
    def display_active_pieces(self):
        if not self.pieces:
            print("pieces empty :(")
        else:
            for piece in self.pieces: 
                print(f"{piece.mean=}", f"{piece.covariance=}", f"{piece.piece_type=}")
    def get_point_locations(self):
        if not self.pieces:
            print("pieces empty :( inside get_point_locations()")
        locations = []
        for piece in self.pieces:
            x,y,w,h = piece.mean[:4] #the x,y,w,h coordinates of the predicted bounding box. 
            localization_pt = [
                int(x),              # center x
                int(y + h * 0.25)    # 25% above the bottom edge
            ]
            locations.append(localization_pt)
        return locations
    
    def associate_pieces(self,results):
        """
        Calls on the associator to associate pieces with current detections. 
        Updates pieces accordingly.
        
        results = (N,4) tensor([[x_i,y_i,w_i,h_i], [], ..., []])
        where N is the number of objects.
        
        """
        res = results.cpu().numpy().tolist()
        self.associator.associate(res,self.pieces)

       
        

class Piece_Associator():
    """Associates the most recent detections with the pieces' Kalman Filters."""
    def __init__(self):
        pass 
    def associate(self, results: List[np.ndarray], pieces: List['Piece']) -> None:

        if not pieces:
            return

        for i, piece in enumerate(pieces):
            print(f"[Piece {i}] mean: {piece.mean}, type: {piece.piece_type}")

        # Extract piece states (e.g. [x, y, w, h])
        piece_positions = np.array([piece.mean[:4] for piece in pieces])
        detection_positions = np.array(results)
        cost_matrix = cdist(piece_positions, detection_positions)
        # Hungarian matching
        piece_indices, det_indices = linear_sum_assignment(cost_matrix)

        print("Matched indices:", list(zip(piece_indices, det_indices)))

        # Update each matched piece with corresponding detection
        for p_idx, d_idx in zip(piece_indices, det_indices):
            pieces[p_idx].measurement_update(detection_positions[d_idx])




        




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