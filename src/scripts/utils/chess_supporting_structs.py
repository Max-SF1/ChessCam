import chess


def to_square_index(file_index, rank):
    """
    file_index: 1-8 (a=1, b=2, ..., h=8) (letters)
    rank: 1-8
    returns: square index for python-chess
    """
    file_letter = chr(int(file_index) + 96)
    square_str = file_letter + str(int(rank))
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

    def get_color_and_piece(self, class_str):
        """Returns (color, piece_type) tuple for the given class string.
        
        Raises KeyError if the class_str is not found.
        """
        if class_str not in self.mapping:
            raise KeyError(f"Unknown class string: {class_str}")
        return self.mapping[class_str]



# board = chess.Board.empty()  # start with an empty board

# # Place a white pawn on e2
# board.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.BLACK))

# # Place a black knight on g8
# board.set_piece_at(chess.G8, chess.Piece(chess.KNIGHT, chess.BLACK))



class GameState():
    def __init__(self,manager):
        self.mapper = ChessClassMapper()
        self.board = chess.Board.empty() 
        self.board.turn = False 
        self.manager = manager 
    def initialize_board(self,homography):
        """
        Initialize_board goes over the initial detections, converts pieces to board representation and adds them to the game.
        """
        self.board.turn = not self.board.turn
        if self.board.turn:
            print("White's turn")
        else:
            print("Black's turn")
        self.board = chess.Board.empty()
        indices = []
        locs = self.manager.get_point_locations()
        for loc in locs: 
            loc_number,loc_letter = homography.transform_point_location_to_ideal_board(loc)
            indices.append(to_square_index(loc_letter,loc_number))
        for piece,index in zip(self.manager.pieces,indices): 
            color, role  = self.mapper.get_color_and_piece(piece.piece_type)
            # print(color, role)
            self.board.set_piece_at(index, chess.Piece(role, color))
        print(self.board)