from typing import List 
import numpy as np 
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def test_associate(results: List[np.ndarray], pieces: List['Piece'], c_threshold = 86.023) -> None:
    """Associates between the detections and the trackable objects. """
    if not pieces:
            return
    
    piece_positions = pieces
    reduced_res = [row[:2] for row in results] #convert x,y,w,h rows to x,y
    detection_positions = np.array(reduced_res)
    cost_matrix = cdist(piece_positions, detection_positions)
    # Hungarian matching
    piece_indices, det_indices = linear_sum_assignment(cost_matrix)

    # print("Matched indices:", list(zip(piece_indices, det_indices)))

    # Update each matched piece with corresponding detection
    for p_idx, d_idx in zip(piece_indices, det_indices):
        if cost_matrix[p_idx,d_idx] < c_threshold:
            print(f"{pieces[p_idx]} is updated by {results[d_idx]}")

        
pieces = [
    np.array([0.0, 0.0]),
    np.array([5.0, 5.0]),
    np.array([10.0, 10.0]),
]

# Create a list of detection results as numpy arrays with (x, y, w, h)
results = [
    np.array([0.1, 0.1, 1.0, 1.0]),
    np.array([5.2, 400.9, 1.0, 1.0]),
    np.array([105.5, 100.5, 1.0, 1.0]),
]
test_associate(results,pieces)
