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
    Initialize the game board with empty slots.

    Returns:
        np.ndarray: A 6x7 ndarray of type BoardPiece, initialized to NO_PLAYER (0).
    """
    return np.full(BOARD_SHAPE, NO_PLAYER, BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Convert the game board to a human-readable string representation.

    Args:
        board (np.ndarray): The current game board.

    Returns:
        str: A string representation of the board suitable for printing.
    """
    wall = "|"
    columns = "0 1 2 3 4 5 6"
    dividers = "============="
    ndarray_print = np.full(BOARD_SHAPE, NO_PLAYER_PRINT, BoardPiecePrint)

    ndarray_print_string = wall + dividers + wall + "\n"

    for array in board:
        for element in array:
            if element == PLAYER1:
                ndarray_print[np.where(board == element)] = PLAYER1_PRINT
            if element == PLAYER2:
                ndarray_print[np.where(board == element)] = PLAYER2_PRINT

    for array in np.flip(ndarray_print):
        ndarray_print_string += wall + " ".join(map(str, np.flip(array))) + wall + "\n"

    return ndarray_print_string + wall + dividers + wall + "\n" + wall + columns + wall


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Convert a string representation of the board back to an ndarray.

    Args:
        pp_board (str): A string representation of the game board.

    Returns:
        np.ndarray: A 6x7 ndarray representing the board.
    """
    ndarray_as_array = np.full(BOARD_SHAPE, NO_PLAYER, BoardPiece)
    ndarray_as_list = pp_board.splitlines()
    ndarray_as_list.pop(0)
    ndarray_as_list.pop()
    ndarray_as_list.pop()

    for i in range(len(ndarray_as_list)):
        ndarray_as_list[i] = ndarray_as_list[i][1:-1]
        ndarray_as_list[i] = ndarray_as_list[i][::2]

    for string in ndarray_as_list:
        for char in string:
            if char == "X":
                ndarray_as_array[5 - ndarray_as_list.index(string), string.index(char)] = PLAYER1
            if char == "O":
                ndarray_as_array[5 - ndarray_as_list.index(string), string.index(char)] = PLAYER2

    return ndarray_as_array


def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece):
    """
    Apply a player's action to the board in the specified column.

    Args:
        board (np.ndarray): The current game board.
        action (PlayerAction): The column where the player wants to place their piece.
        player (BoardPiece): The player making the move (PLAYER1 or PLAYER2).
    """
    for array in board:
        if array[action] == NO_PLAYER:
            array[action] = player
            break


def connected_four(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Check if a player has four connected pieces in a row, column, or diagonal.

    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The player to check for a winning connection.

    Returns:
        bool: True if the player has four connected pieces, False otherwise.
    """
    columns_top_positions_player = []

    for i in range(BOARD_COLS):
        for j in range(BOARD_ROWS):
            if board[BOARD_ROWS - 1 - j, i] == NO_PLAYER:
                continue
            if board[BOARD_ROWS - 1 - j, i] == player:
                columns_top_positions_player.append((BOARD_ROWS - 1 - j, i))
                break
            break

    for (row, column) in columns_top_positions_player:
        connected_stones_row = 0
        connected_stones_column = 0
        for i in range(BOARD_COLS):
            if board[row, i] == player:
                connected_stones_row += 1
                if connected_stones_row >= 4:
                    return True
                continue
            connected_stones_row = 0

        if row >= 3:
            for i in range(BOARD_ROWS - (BOARD_ROWS - 1 - row)):
                if board[row - i, column] == player:
                    connected_stones_column += 1
                    continue
                break
            if connected_stones_column >= 4:
                return True

        diagonal_NE = diagonal_NW = diagonal_SE = diagonal_SW = 0
        Stopp_NE = Stopp_NW = Stopp_SE = Stopp_SW = True

        for i in range(BOARD_COLS):
            if Stopp_SW and row - i >= 0 and column - i >= 0 and board[row - i, column - i] == player:
                diagonal_SW += 1
            else:
                Stopp_SW = False
            if Stopp_NE and row + i < BOARD_ROWS and column + i < BOARD_COLS and board[row + i, column + i] == player:
                diagonal_NE += 1
            else:
                Stopp_NE = False
            if Stopp_SE and row - i >= 0 and column + i < BOARD_COLS and board[row - i, column + i] == player:
                diagonal_SE += 1
            else:
                Stopp_SE = False
            if Stopp_NW and row + i < BOARD_ROWS and column - i >= 0 and board[row + i, column - i] == player:
                diagonal_NW += 1
            else:
                Stopp_NW = False

        if diagonal_SW + diagonal_NE - 1 >= 4 or diagonal_SE + diagonal_NW - 1 >= 4:
            return True
    return False


def check_end_state(board: np.ndarray, player: BoardPiece) -> GameState:
    """
    Check the current state of the game.

    Args:
        board (np.ndarray): The current game board.
        player (BoardPiece): The player making the move.

    Returns:
        GameState: The game state indicating if it's a win, draw, or still playing.
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    if np.any(board[BOARD_ROWS - 1] == NO_PLAYER):
        return GameState.STILL_PLAYING
    return GameState.IS_DRAW


def check_move_status(board: np.ndarray, column: Any) -> MoveStatus:
    """
    Validate a player's move.

    Args:
        board (np.ndarray): The current game board.
        column (Any): The column to be checked.

    Returns:
        MoveStatus: Indicates if the move is valid or the type of error.
    """
    if not isinstance(column, PlayerAction):
        return MoveStatus.WRONG_TYPE

    if not (int(column) >= 0 and int(column) < BOARD_COLS):
        return MoveStatus.OUT_OF_BOUNDS

    if not board[BOARD_ROWS - 1, int(column)] == NO_PLAYER:
        return MoveStatus.FULL_COLUMN

    return MoveStatus.IS_VALID
