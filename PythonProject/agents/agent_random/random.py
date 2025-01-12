import numpy as np
import random
from typing import Optional, Tuple
from game_utils import BoardPiece, PlayerAction, SavedState, BOARD_COLS, check_move_status, MoveStatus

def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates a random valid move for the given player.

    This function chooses a column at random and checks if it is a valid move. 
    If the column is full or invalid, it will retry until a valid move is found.

    Args:
        board (np.ndarray): The current game board state.
        player (BoardPiece): The player making the move (either PLAYER1 or PLAYER2).
        saved_state (Optional[SavedState]): An optional saved state for the game, not used in this function.

    Returns:
        Tuple[PlayerAction, Optional[SavedState]]: A tuple containing the randomly chosen valid column 
        (as `PlayerAction`) and the unchanged saved state.
    """
    action = PlayerAction(random.randint(0, BOARD_COLS - 1))
    while not check_move_status(board, action) == MoveStatus.IS_VALID:
        action = PlayerAction(random.randint(0, BOARD_COLS - 1))
    return tuple([action, saved_state])