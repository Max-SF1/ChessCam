import chess


def to_square_index(file_index, rank):
    """
    file_index: 1-8 (a=1, b=2, ..., h=8) (letters)
    rank: 1-8
    returns: square index for python-chess
    """
    print(f"{file_index=}, {rank=}")
    file_index = 9-int(file_index)
    rank = 9-int(rank) 
    file_letter = chr(int(file_index) + 96)
    square_str = file_letter + str(int(rank))
    print(f"{square_str=}")
    return chess.parse_square(square_str)

class ChessClassMapper:
    """
    maps between class labels and the class/type definition in the python chess library. 
    """
    def __init__(self):
        self.mapping = {
            'black-bishop':  (chess.BLACK, chess.BISHOP),
            'black-king':    (chess.BLACK, chess.KING),
            'black-knight':  (chess.BLACK, chess.KNIGHT),
            'black-pawn':    (chess.BLACK, chess.PAWN),
            'black-queen':   (chess.BLACK, chess.QUEEN),
            'black-rook':    (chess.BLACK, chess.ROOK),
            'white-bishop':  (chess.WHITE, chess.BISHOP),
            'white-king':    (chess.WHITE, chess.KING),
            'white-knight':  (chess.WHITE, chess.KNIGHT),
            'white-pawn':    (chess.WHITE, chess.PAWN),
            'white-queen':   (chess.WHITE, chess.QUEEN),
            'white-rook':    (chess.WHITE, chess.ROOK),
        }
        # Create reverse mapping automatically
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def get_color_and_piece(self, class_str):
        """Returns (color, piece_type) tuple for the given class string.
        
        Raises KeyError if the class_str is not found.
        """
        if class_str not in self.mapping:
            raise KeyError(f"Unknown class string: {class_str}")
        return self.mapping[class_str]
    def reverse_map_color_and_piece(self, color, type):
        """returns Piece type string from color and type"""
        return self.reverse_mapping[(color,type)]


class PieceTracker: 
    """Tracks the changes and number of pieces in the current board, and future board."""
    def __init__(self):
        self.curr_white_piece_squares = set()
        self.curr_black_piece_squares = set()
        self.next_white_piece_squares = set()
        self.next_black_piece_squares = set()
        self.candidate_board = chess.Board()
    def update_sets(self):
        """
        Takes the next-move sets and makes them current move, 
        resets next-move sets. 
        """
        self.curr_black_piece_squares = self.next_black_piece_squares
        self.curr_white_piece_squares = self.next_white_piece_squares
        self.next_white_piece_squares = set()
        self.next_black_piece_squares = set()
    def add_square_idx_to_appropriate_set(self, color, square_index,first_move):
        """Adds a valid piece's square index to the appropriate color set. """
        if first_move: 
            if color == chess.WHITE:
                self.curr_white_piece_squares.add(square_index)
            if color == chess.BLACK:
                self.curr_white_piece_squares.add(square_index)
        else: 
            if color == chess.WHITE:
                self.next_white_piece_squares.add(square_index)
            if color == chess.BLACK:
                self.next_white_piece_squares.add(square_index)
    def discern_movement_type(self):
        ###NEED TO ADD PROMOTION LOGIC. 
        """based on set size-comparisons, discern the move type that just occured."""
        move = chess.Move.null()
        black_same = False
        white_same = False
        if len(self.curr_black_piece_squares) == len(self.next_black_piece_squares):
            black_same = True 
        if len(self.curr_white_piece_squares) == len(self.next_white_piece_squares):
            white_same = True 
        if black_same and white_same:
            print("we just had a movement")
            if self.curr_black_piece_squares == self.next_black_piece_squares:
                # The pieces that changed were White's
                (from_square,) = self.curr_white_piece_squares - self.next_white_piece_squares
                (to_square,) = self.next_white_piece_squares - self.curr_white_piece_squares
                move = chess.Move(from_square, to_square)
                print("white moved from", self.curr_white_piece_squares - self.next_white_piece_squares,
                    "to", self.next_white_piece_squares - self.curr_white_piece_squares)

            if self.curr_white_piece_squares == self.next_white_piece_squares:
                #the pieces that changed were Black's
                (from_square,) = self.curr_black_piece_squares - self.next_black_piece_squares
                (to_square,) =  self.next_black_piece_squares -self.curr_black_piece_squares
                move = chess.Move(from_square, to_square) 
                print("black moved from", self.curr_black_piece_squares - self.next_black_piece_squares,
                    "to", self.next_black_piece_squares - self.curr_black_piece_squares)
        
        if black_same and not white_same:
            print("black ate white")
        if white_same and not black_same:
            print("white ate black")
        return move 


class GameState():
    """ Gamestate holds the current board configuration, as well as who makes the current moves. """
    def __init__(self,manager):
        self.tracker = PieceTracker()
        self.mapper = ChessClassMapper()
        self.board = chess.Board.empty() 
        self.board.turn = False 
        self.manager = manager 
        self.first_move = True 
        self.true_board = chess.Board()

    def set_true_piece_type(self,piece, sq_index):
        """
        Changes Piece type to the one that should be there in Chess' starting configuration. 
        """ 
        py_chess_piece_object = self.true_board.piece_at(sq_index) #get the pychess piece object, extract attributes for our own piece object.
        piece.piece_type = self.mapper.reverse_map_color_and_piece(py_chess_piece_object.color,py_chess_piece_object.piece_type)

    def initialize_board(self,homography):
        """
        Initialize_board goes over the initial detections, converts pieces to board representation and adds them to the game.
        """
        print("called initialize")
        if self.first_move is True: 
            self.board.turn = chess.WHITE
            self.board.clear() 
            square_indices = [] #py-chess gives every square a number
            validity_array = []
            locs = self.manager.get_point_locations()
            for loc in locs: 
                loc_number,loc_letter, validity = homography.transform_point_location_to_ideal_board(loc)
                validity_array.append(validity)
                if validity:
                    loc_letter, loc_number = int(loc_letter),int(loc_number)
                    #fixing the orientation
                    square_indices.append(to_square_index(loc_letter,loc_number))
                else: 
                    square_indices.append(3) #junk value just to append something.
            for piece,sq_index,idx in zip(self.manager.pieces,square_indices,range(len(square_indices))): 
                if validity_array[idx] is False:
                    continue #we don't want to project invalid pieces
                if self.first_move is True:
                    print(locs[idx],)
                    self.set_true_piece_type(piece, sq_index)
                color, role  = self.mapper.get_color_and_piece(piece.piece_type)
                self.tracker.add_square_idx_to_appropriate_set(color=color,square_index=sq_index, first_move=self.first_move)
                self.board.set_piece_at(sq_index, chess.Piece(role, color))
            self.manager.kill_invalid_pieces(validity_array)
            self.first_move = False
        else:
            square_indices = [] #py-chess gives every square a number
            validity_array = []
            locs = self.manager.get_point_locations()
            for loc in locs: 
                loc_number,loc_letter, validity = homography.transform_point_location_to_ideal_board(loc)
                validity_array.append(validity)
                if validity:
                    loc_letter, loc_number = int(loc_letter),int(loc_number) 
                    #fixing the orientation
                    square_indices.append(to_square_index(loc_letter,loc_number))
                else: 
                    square_indices.append(3) #junk value just to append something.
            for piece,sq_index,idx in zip(self.manager.pieces,square_indices,range(len(square_indices))):
                if validity_array[idx] is False:
                    continue #we don't want to project invalid pieces
                print(locs[idx])
                color, role  = self.mapper.get_color_and_piece(piece.piece_type)
                self.tracker.add_square_idx_to_appropriate_set(color=color,square_index=sq_index, first_move=self.first_move)

            move = self.tracker.discern_movement_type()
            self.board.push(move=move)
            self.tracker.update_sets()          
            self.manager.kill_invalid_pieces(validity_array)

        #note: board won't update for now. 
        print(self.board)