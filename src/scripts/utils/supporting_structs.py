from typing import List 
import numpy as np 
import cv2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH
from utils import chess_supporting_structs
#corners given by cv2's find corners are ordered row by row, left to right in every row. 
#matching_corners should be ordered the same way.

import numpy as np


class Probation:
    """
    Attaches to a new pawn, makes sure to kill the pawn if it doesn't get an association
    Consistently. 
    """
    def __init__(self, probation_thresh = 5):
        self.probation_score = 0
        self.on_probation = True
        self.probation_thresh = probation_thresh

    def __end_probation(self):
        """Deactivate object functionality when probation period has elapsed."""
        self.on_probation = False 
    def update_score(self):
        """Increase probation score by 1, if reaches the threshold object gets off of probation."""
        self.probation_score = self.probation_score + 1 
        if self.probation_score >= self.probation_thresh:
            self.__end_probation()


def column_major_to_row_major(corners, rows, cols):
    """
    To get CV2's findChessboardcorners to be consistently row-major.
    Reorders points from column-major (top-to-bottom, right-to-left)
    to row-major (left-to-right, top-to-bottom).

    Parameters:
        corners: np.ndarray of shape (N, 1, 2)
        rows: int — number of rows in grid
        cols: int — number of columns in grid

    Returns:
        np.ndarray of shape (N, 1, 2) in row-major order
    """
    # Flatten to (N, 2)
    points = corners.reshape(-1, 2)

    # Reshape to (cols, rows, 2) assuming column-major order
    grid = points.reshape(cols, rows, 2)

    # Swap axes to get (rows, cols, 2), i.e., row-major
    grid_row_major = np.transpose(grid[::-1], (1, 0, 2))

    # Return to original shape (N, 1, 2)
    return grid_row_major.reshape(-1, 1, 2)

def get_piece_class_mapping():
    """A dictionary that maps from class ID to the Piece type string."""
    return {
        0: 'black-bishop',
        1: 'black-king',
        2: 'black-knight',
        3: 'black-pawn',
        4: 'black-queen',
        5: 'black-rook',
        6: 'white-bishop',
        7: 'white-king',
        8: 'white-knight',
        9: 'white-pawn',
        10: 'white-queen',
        11: 'white-rook'
    }

def fix_column_major(corners):
    """
    Convert Column major order to row major.
    if the grid of corners is given in a row-major order do nothing.
    """
    if (corners[1][0][1]-corners[0][0][1] > 20): 
        #we are in column major order 
        return column_major_to_row_major(corners,rows=7,cols= 7)
    else: 
        return corners 


class Homography:
    """Holds homography matrix and applies homographic transforms."""
    def __init__(self, homography_matrix=None):
        if homography_matrix is not None:

            self.homography_matrix = np.array(homography_matrix, dtype=np.float32)
            self.initialized = True
        else:
            self.homography_matrix = None
            self.initialized = False

        self.matching_corners_for_homography = np.array(
            [[[x, y]] for y in range(8, 1, -1) for x in range(8, 1, -1)],
            dtype=np.float32
        )
    def out_of_bounds_correction(self, loc):
        """
        Checks if a Detection's projected location is out of bounds, corrects it up to a certain threshold. 
        
        returns a threshold_exceeded boolean if the piece is waaaay out of bounds. (probably false detection)
        
        """
        valid_location = True 
        if loc < 9 and loc > 1 :
            return loc, valid_location
        if loc > 9 and loc < 10: 
            loc = 8
            return loc, valid_location
        else: 
            valid_location = False 
            return loc, valid_location

    def check_initialization(self):
        """returns True if the Homography matrix has been computed, and false otherwise."""
        return self.initialized
    def compute_homography(self,corners):
        """Computes the matrix H using cv2's FindHomography, given the corners found by FindChessboardCorners."""
        # cv2.drawChessboardCorners(frame, (7,7), corners, ret)
        self.homography_matrix, _= cv2.findHomography(corners,self.matching_corners_for_homography,)
        self.homography_matrix = np.array(self.homography_matrix)
        print(self.homography_matrix)
        self.initialized =True

    def transform_point_location_to_ideal_board(self, locations):
        """
            Takes the piece's "center of mass", and prints the Chessboard square to which the piece belongs.

            Args:
            center of mass: [x,y] - computed with [x',int(y'+0.25*h)] where x' and y' are the xy-coords of the center of the bounding 
            box.

            Returns: None.
            either prints that no Homography matrix exists, or prints the square to which the piece belongs. 
            ----
            note: make sure that the chessboard corners were found in the correct orientation.
        """
        localization_pt_formatted = np.array([[locations]], dtype=np.float32)

        if not self.initialized:
            print("no initialized homography matrix found :( ")
        else: 
            projected  = cv2.perspectiveTransform(localization_pt_formatted, self.homography_matrix )
            loc_number, loc_letter = projected[0][0]
            #avoid out of bounds errors. 
            print(loc_letter, loc_number)
            loc_letter, validity_l = self.out_of_bounds_correction(loc_letter)
            loc_number, validity_n = self.out_of_bounds_correction(loc_number)
            validity = validity_n and validity_l
            return loc_number,loc_letter, validity



class Piece:
    """A class that tracks each piece in the game, given a kalman filter for said piece."""
    def __init__(self, piece_bbox, class_id):
        self.probation = Probation()
        self.piece_bbox= piece_bbox
        class_mapping = get_piece_class_mapping()
        self.piece_type = class_mapping[int(class_id)] #get the class string
        self.kf = KalmanFilterXYWH()
        measurement = np.array(piece_bbox)
        self.mean, self.covariance = self.kf.initiate(measurement)

    def display(self):
        """Displays piece attributes"""
        print(f"{self.piece_type}, loc: {self.mean}")
    
    def time_update(self):
        """Updates KF through time. (progresses dynamical model)"""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)

    
    def measurement_update(self,piece_bbox):
        """Updates KF with measurement. Also updates Probation."""
        self.probation.update_score()
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, piece_bbox)


class PieceManager():

    """Piece Manager handles all functionality related to displaying the pieces, 
    updating their Kalman Filters. """

    def __init__(self,pieces,time_to_live = 4):
        self.pieces = pieces
        self.piece_scores = [] 
        self.associator = PieceAssociator()
        self.gamestate = chess_supporting_structs.GameState(self)
        self.time_to_live = time_to_live
    def dump_pieces(self):
        """kill all your pieces"""
        self.pieces = []
    
    def kill_invalid_pieces(self, validity_array):
        """Kill a specific piece."""
        if not self.pieces or not validity_array:
            return     
        # Keep only those pieces and scores where score <= time_to_live
        filtered = [(p, s) for p, s, validity in zip(self.pieces, self.piece_scores, validity_array) if validity is True]

        # Unzip filtered list back to separate lists
        if filtered:
            self.pieces, self.piece_scores = zip(*filtered)
            # zip(*) returns tuples, convert to lists:
            self.pieces = list(self.pieces)
            self.piece_scores = list(self.piece_scores)
        else:
            # If none remain, empty both lists
            self.pieces = []
            self.piece_scores = []


    def kill_unassociated_pieces(self):
        """Examines all pieces and discards those that have not been associated in more than time_to_live frames."""
        if not self.pieces:
            return     
        # Keep only those pieces and scores where score <= time_to_live
        filtered = [(p, s) for p, s in zip(self.pieces, self.piece_scores) if s <= self.time_to_live]

        # Unzip filtered list back to separate lists
        if filtered:
            self.pieces, self.piece_scores = zip(*filtered)
            # zip(*) returns tuples, convert to lists:
            self.pieces = list(self.pieces)
            self.piece_scores = list(self.piece_scores)
        else:
            # If none remain, empty both lists
            self.pieces = []
            self.piece_scores = []


    def append_piece(self,piece):
        self.pieces.append(piece)
        self.piece_scores.append(0)
    def time_update(self):
        for piece in self.pieces:
            piece.time_update()
    def display_active_pieces(self):
        if not self.pieces:
            print("manager does not handle any pieces.")
        else:
            for piece in self.pieces: 
                print(f"{piece.mean=}", f"{piece.covariance=}", f"{piece.piece_type=}")

    def get_point_locations(self):
        """Returns "center of mass" point for all the pieces. """
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
        
        results = (N,4) tensor([[x_i,y_i,w,h], [], ..., []])
        where N is the number of objects. 

        for un-associated detections, we'd like to *create new pieces*! 
        ---
        (ultimately only uses x,y for matching but w,h are needed for the kf update.)
        
        """
        results_bbox = results[-1].boxes.xywh
        class_ids = results[-1].boxes.cls
        res = results_bbox.cpu().numpy().tolist()
        self.associator.associate(self,res, class_ids)

class PieceAssociator():
    """Associates the most recent detections with the pieces' Kalman Filters."""
    def __init__(self):
        pass

    def associate(self,  manager: PieceManager, results: List[np.ndarray], class_ids,  c_threshold = 20) -> None: #used to be 40.23 something 
        """Associates between the detections and the trackable objects. """
        #### edge cases #### 
        if not manager.pieces:
            for r_idx, result in enumerate(results):
                manager.append_piece(Piece(results[r_idx], class_ids[r_idx]))
            return

        if not results:
            # No detections, so treat all current pieces as unmatched
            validity_array = [True] * len(manager.piece_scores)
            for p_idx, piece in enumerate(manager.pieces):
                manager.piece_scores[p_idx] += 1
                if piece.probation.on_probation:
                    validity_array[p_idx] = False
            manager.kill_invalid_pieces(validity_array)
            return

        # Extract piece states (e.g. [x, y])
        piece_positions = np.array(manager.get_point_locations())
        reduced_res = []
        for result in results: 
            x,y,w,h = result 
            result_center_of_mass = [
                int(x),              # center x
                int(y + h * 0.25)    # 25% above the bottom edge
            ]
            reduced_res.append(result_center_of_mass)
            
        detection_positions = np.array(reduced_res)
        cost_matrix = cdist(piece_positions, detection_positions)
        piece_indices, det_indices = linear_sum_assignment(cost_matrix)
        matched_piece_indices = set()
        matched_det_indices = set()
        validity_array = [True] * len(manager.piece_scores)
        all_piece_indices = set(range(cost_matrix.shape[0]))
        all_det_indices = set(range(cost_matrix.shape[1]))
        for p_idx, d_idx in zip(piece_indices, det_indices):
            if cost_matrix[p_idx,d_idx] < c_threshold:
                matched_piece_indices.add(p_idx)
                matched_det_indices.add(d_idx)
                manager.pieces[p_idx].measurement_update(results[d_idx])
                manager.piece_scores[p_idx] = 0 #we had a detection, we can zero out the score.
        unmatched_piece_indices = all_piece_indices - matched_piece_indices
        unmatched_det_indices = all_det_indices - matched_det_indices
        # print("-"*30)
        # print(f"{matched_piece_indices=}, {unmatched_det_indices=}, {unmatched_piece_indices=}")
        # print(f"{len(manager.pieces)=}")
        # print("-"*30)
        for p_idx in unmatched_piece_indices:
                manager.piece_scores[p_idx] += 1
                if manager.pieces[p_idx].probation.on_probation:
                    validity_array[p_idx] = False
        for d_idx in unmatched_det_indices:
                manager.append_piece(Piece(results[d_idx], class_ids[d_idx]))
                validity_array.append(True) #new valid piece has been introduced.
        manager.kill_invalid_pieces(validity_array)






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