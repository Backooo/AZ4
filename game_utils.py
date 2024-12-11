from typing import Callable, Optional, Any
from enum import Enum
import numpy as np

BOARD_COLS = 7
BOARD_ROWS = 6
BOARD_SHAPE = (6, 7)
INDEX_HIGHEST_ROW = BOARD_ROWS - 1
INDEX_LOWEST_ROW = 0

BoardPiece = np.int8  # The data type (dtype) of the board pieces
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = BoardPiecePrint(' ')
PLAYER1_PRINT = BoardPiecePrint('X')
PLAYER2_PRINT = BoardPiecePrint('O')

PlayerAction = np.int8  # The column to be played

class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0

class MoveStatus(Enum):
    IS_VALID = 1
    WRONG_TYPE = 'Input does not have the correct type (PlayerAction).'
    OUT_OF_BOUNDS = 'Input is out of bounds.'
    FULL_COLUMN = 'Selected column is full.'

class SavedState:
    pass

GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape BOARD_SHAPE and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)


def pretty_print_converter(elem: np.int8) -> str:
    """
    Converts the elements of the board to the pretty printed version.
    """
    if elem == NO_PLAYER:
        return NO_PLAYER_PRINT
    elif elem == PLAYER1:
        return PLAYER1_PRINT
    elif elem == PLAYER2:
        return PLAYER2_PRINT

def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] of the array should appear in the lower-left in the printed string representation. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    board_delimiter = "|==============|\n"
    board_numbers = "|0 1 2 3 4 5 6 |"
    board_string = board_delimiter
    
    for row in range(BOARD_ROWS -1,-1,-1):
        board_string += "|"
        for col in range(BOARD_COLS):
            board_string += pretty_print_converter(board[row][col])+" "
        board_string += "|\n"
        
    board_string += board_delimiter + board_numbers
    return board_string
# "|==============|\n|              |\n|              |\n|              |\n|              |\n|              |\n|              |\n|==============|\n|0 1 2 3 4 5 6 |"



def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    board = np.full(BOARD_SHAPE, NO_PLAYER, dtype=BoardPiece)
    
    lines = pp_board.strip().split('\n')
    
    for row in range(BOARD_ROWS):
        line = lines[row + 1]
        
        for col in range(BOARD_COLS):
            char = line[1 + col * 2] 
            
            if char == PLAYER1_PRINT:
                board[BOARD_ROWS - 1 - row, col] = PLAYER1
            elif char == PLAYER2_PRINT:
                board[BOARD_ROWS - 1 - row, col] = PLAYER2
                
    return board

def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Sets board[i, action] = player, where i is the lowest open row. The input 
    board should be modified in place, such that it's not necessary to return 
    something.
    """
    for row in range(BOARD_ROWS):
        if set_board_piece(board, row, action, player):
            break
        
        
def set_board_piece(board: np.ndarray, row: int, col: int, player: BoardPiece) -> bool:
    """
    Sets the board piece to the player-element if the BoardPiece is empty.
    Returns True if the piece was set.
    """
    if board[row, col] == NO_PLAYER:
        board[row, col] = player
        return True
    return False


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    """
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Right, Down, Down-Right, Down-Left
    
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] == player:
                for d_row, d_col in directions:
                    counter = 0
                    for k in range(4):
                        x, y = row + k * d_row, col + k * d_col
                        if 0 <= x < BOARD_ROWS and 0 <= y < BOARD_COLS and board[x, y] == player:
                            counter += 1
                        else:
                            break
                    if counter == 4:
                        return True
    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    
    if np.all(board != NO_PLAYER):
        return GameState.IS_DRAW
    
    return GameState.STILL_PLAYING


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Returns a MoveStatus indicating whether a move is accepted as a valid move 
    or not, and if not, why.
    The provided column must be of the correct type (PlayerAction).
    Furthermore, the column must be within the bounds of the board and the
    column must not be full.
    """
    
    if not isinstance(column, (int, np.integer)):
        return MoveStatus.WRONG_TYPE
        
    if column < 0 or column >= BOARD_COLS:
        return MoveStatus.OUT_OF_BOUNDS

    if board[INDEX_HIGHEST_ROW, column] != NO_PLAYER:
        return MoveStatus.FULL_COLUMN

    return MoveStatus.IS_VALID